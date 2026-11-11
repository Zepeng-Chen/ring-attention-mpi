#include "attn/core.hpp"
#include "attn/distributed.hpp"

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

#ifndef ATTN_GIT_COMMIT
#define ATTN_GIT_COMMIT "unknown"
#endif

namespace {
struct Options {
    attn::Shape shape;
    attn::Mask mask;
    attn::KernelOptions kernel;
    attn::Partition partition = attn::Partition::contiguous;
    attn::FetchOrder order = attn::FetchOrder::staggered;
    std::string impl = "ring", mask_name = "none", partition_name = "contiguous";
    std::string data = "synthetic", cluster = "local", purpose = "development";
    std::string kernel_name = "tuned", oref;
    bool mha_control = false;
    double tolerance_scale = 0;   // 0: absolute 1e-5; x > 0: x times the largest reference magnitude
    double warmup_seconds = 0;    // keep warming up until this much wall time has passed
    std::string csv, rank_csv, trace_csv;
    std::uint64_t seed = 42;
    float alpha = 1;
    int reps = 3, warmup = 1;
    std::size_t validate = 16;
    bool help = false;
};
std::int64_t integer(const std::string& s) {
    std::size_t end = 0;
    const auto x = std::stoll(s, &end);
    if (end != s.size()) throw std::invalid_argument("invalid integer: " + s);
    return x;
}
int small_integer(const std::string& s) {
    const auto x = integer(s);
    if (x < 0 || x > std::numeric_limits<int>::max()) throw std::invalid_argument("integer out of range: " + s);
    return static_cast<int>(x);
}
Options parse(int argc, char** argv) {
    Options o;
    bool nq_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") { o.help = true; continue; }
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--N") o.shape.n = integer(value);
        else if (key == "--Nq") { o.shape.nq = integer(value); nq_set = true; }
        else if (key == "--d") o.shape.d = small_integer(value);
        else if (key == "--dv") o.shape.dv = small_integer(value);
        else if (key == "--H") o.shape.heads = small_integer(value);
        else if (key == "--Hkv") o.shape.kv_heads = small_integer(value);
        else if (key == "--impl") o.impl = value;
        else if (key == "--block-q") o.kernel.block_q = small_integer(value);
        else if (key == "--block-k" || key == "--block") o.kernel.block_k = small_integer(value);
        else if (key == "--threads") o.kernel.threads = small_integer(value);
        else if (key == "--poll-tiles") o.kernel.poll_tiles = small_integer(value);
        else if (key == "--kernel") {
            o.kernel_name = value;
            if (value == "basic") o.kernel.kernel = attn::Kernel::basic;
            else if (value == "tuned") o.kernel.kernel = attn::Kernel::tuned;
            else throw std::invalid_argument("unknown kernel: " + value);
        }
        else if (key == "--mask") {
            o.mask_name = value;
            if (value == "none") o.mask.kind = attn::MaskKind::none;
            else if (value == "causal") o.mask.kind = attn::MaskKind::causal;
            else if (value == "window") o.mask.kind = attn::MaskKind::window;
            else if (value == "sink-window") o.mask.kind = attn::MaskKind::sink_window;
            else throw std::invalid_argument("unknown mask: " + value);
        } else if (key == "--window") o.mask.window = integer(value);
        else if (key == "--sink") o.mask.sink = integer(value);
        else if (key == "--partition") {
            o.partition_name = value;
            if (value == "contiguous") o.partition = attn::Partition::contiguous;
            else if (value == "zigzag") o.partition = attn::Partition::zigzag;
            else throw std::invalid_argument("unknown partition: " + value);
        } else if (key == "--fetch-order") {
            if (value == "staggered") o.order = attn::FetchOrder::staggered;
            else if (value == "ascending") o.order = attn::FetchOrder::ascending;
            else if (value == "reverse") o.order = attn::FetchOrder::reverse;
            else throw std::invalid_argument("unknown fetch order: " + value);
        } else if (key == "--seed") {
            const auto n = integer(value);
            if (n < 0) throw std::invalid_argument("seed must be nonnegative");
            o.seed = static_cast<std::uint64_t>(n);
        } else if (key == "--alpha") {
            std::size_t end = 0;
            o.alpha = std::stof(value, &end);
            if (end != value.size()) throw std::invalid_argument("invalid alpha");
        } else if (key == "--validate") o.validate = static_cast<std::size_t>(small_integer(value));
        else if (key == "--reps") o.reps = small_integer(value);
        else if (key == "--warmup") o.warmup = small_integer(value);
        else if (key == "--warmup-seconds") {
            std::size_t end = 0;
            o.warmup_seconds = std::stod(value, &end);
            if (end != value.size() || !std::isfinite(o.warmup_seconds) || o.warmup_seconds < 0) throw std::invalid_argument("invalid warm-up time");
        }
        else if (key == "--data") o.data = value;
        else if (key == "--cluster") o.cluster = value;
        else if (key == "--purpose") o.purpose = value;
        else if (key == "--csv") o.csv = value;
        else if (key == "--rank-csv") o.rank_csv = value;
        else if (key == "--trace-csv") o.trace_csv = value;
        else if (key == "--oref") o.oref = value;
        else if (key == "--mha-control") o.mha_control = small_integer(value) != 0;
        else if (key == "--tolerance-scale") {
            std::size_t end = 0;
            o.tolerance_scale = std::stod(value, &end);
            if (end != value.size() || !std::isfinite(o.tolerance_scale) || o.tolerance_scale < 0) throw std::invalid_argument("invalid tolerance scale");
        }
        else throw std::invalid_argument("unknown option: " + key);
    }
    if (!nq_set) o.shape.nq = o.shape.n;
    o.shape.validate(); o.mask.validate();
    if (o.reps < 1 || o.kernel.threads < 1 || o.kernel.block_q < 1 || o.kernel.block_k < 1 ||
        !std::isfinite(o.alpha) || o.alpha <= 0) throw std::invalid_argument("invalid run configuration");
    if (o.data != "synthetic" && o.data.rfind("file:", 0) != 0) throw std::invalid_argument("data must be synthetic or file:<directory>");
    if (o.data != "synthetic" && o.alpha != 1) throw std::invalid_argument("alpha applies only to synthetic data");
    return o;
}

std::string quote(const std::string& s) {
    std::string result = "\"";
    for (char c : s) {
        if (c == '\0') continue;      // some MPI libraries count the terminator in the version length
        if (c == '"') result += '"';
        result += (c == '\n' || c == '\r') ? ' ' : c;
    }
    return result + "\"";
}
std::string now() {
    const std::time_t time = std::time(nullptr);
    std::ostringstream out; out << std::put_time(std::gmtime(&time), "%Y-%m-%dT%H:%M:%SZ"); return out.str();
}
double peak_rss_mb() {
    rusage usage{}; getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return static_cast<double>(usage.ru_maxrss) / (1024 * 1024);
#else
    return static_cast<double>(usage.ru_maxrss) / 1024;
#endif
}
// Proportional set size: private pages plus this process's share of shared
// pages. Unlike the peak resident size it does not count a shared-memory
// window once per process. Linux only; returns 0 where unavailable.
double pss_mb() {
    std::ifstream in("/proc/self/smaps_rollup");
    std::string word;
    while (in >> word)
        if (word == "Pss:") { double kb = 0; in >> kb; return kb / 1024; }
    return 0;
}
double max_value(double value) {
    double result = 0; attn::mpi_check(MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD), "max reduction"); return result;
}
double mean_value(double value, int ranks) {
    double result = 0; attn::mpi_check(MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), "mean reduction"); return result / ranks;
}
std::uint64_t reduce_count(std::uint64_t value, MPI_Op op) {
    std::uint64_t result = 0; attn::mpi_check(MPI_Allreduce(&value, &result, 1, MPI_UINT64_T, op, MPI_COMM_WORLD), "count reduction"); return result;
}

void append(const std::string& path, const std::string& header, const std::string& row) {
    if (path.empty()) { std::cout << row << '\n'; return; }
    const std::filesystem::path p(path);
    if (!p.parent_path().empty()) std::filesystem::create_directories(p.parent_path());
    const bool empty = !std::filesystem::exists(p) || std::filesystem::file_size(p) == 0;
    if (!empty) {
        std::ifstream existing(path); std::string first; std::getline(existing, first);
        if (first != header) throw std::runtime_error("CSV schema mismatch: " + path);
    }
    std::ofstream file(path, std::ios::app);
    if (!file) throw std::runtime_error("cannot write " + path);
    if (empty) file << header << '\n';
    file << row << '\n';
    file.flush();
    file.close();
    if (!file) throw std::runtime_error("CSV write failed: " + path);
}

int execute(const Options& options, int rank, int ranks, int thread_level, const std::string& command) {
    // With --mha-control the K/V heads are replicated after loading, so the
    // attention itself runs as multi-head attention with H key/value heads.
    Options o = options;
    const attn::Shape loaded = options.shape;
    if (options.mha_control) o.shape.kv_heads = o.shape.heads;
    const bool serial = o.impl == "S0" || o.impl == "S0-fp32" || o.impl == "S1";
    if (serial && ranks != 1) throw std::invalid_argument("serial implementations require exactly one MPI rank");
    if (serial && o.kernel.threads != 1) throw std::invalid_argument("S0/S0-fp32/S1 are single-thread baselines; use --impl ring with one rank for OpenMP");
#ifndef _OPENMP
    if (o.kernel.threads != 1) throw std::invalid_argument("OpenMP was disabled at build time");
#else
    omp_set_dynamic(0);
    int actual_threads = 0;
#pragma omp parallel num_threads(o.kernel.threads)
    {
#pragma omp single
        actual_threads = omp_get_num_threads();
    }
    if (actual_threads != o.kernel.threads)
        throw std::runtime_error("OpenMP team size differs from --threads; check OMP_THREAD_LIMIT and allocation");
#endif
    MPI_Comm shared;
    attn::mpi_check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared), "node grouping");
    int local_rank = 0; MPI_Comm_rank(shared, &local_rank);
    const int nodes = static_cast<int>(reduce_count(local_rank == 0 ? 1 : 0, MPI_SUM));
    MPI_Comm_free(&shared);
    const double data_start = MPI_Wtime();
    auto qp = attn::positions(o.shape.nq, ranks, rank, o.partition);
    auto kp = attn::positions(o.shape.n, ranks, rank, o.partition);
    const auto q = o.data == "synthetic" ? attn::make_queries(o.shape, qp, o.seed, o.alpha)
                                        : attn::read_queries(o.shape, qp, o.data.substr(5));
    auto load_kv = [&](const std::vector<std::int64_t>& rows) {
        auto block = o.data == "synthetic" ? attn::make_kv(loaded, rows, o.seed)
                                           : attn::read_kv(loaded, rows, o.data.substr(5));
        return options.mha_control ? attn::replicate_kv_heads(block, loaded) : block;
    };
    const auto kv = load_kv(kp);
    const double datagen = max_value(MPI_Wtime() - data_start);

    // Sample globally, so --validate is a total budget independent of rank count.
    const auto samples = attn::sample_rows(static_cast<std::size_t>(o.shape.nq), o.validate);
    std::vector<std::size_t> selected;
    attn::Queries sampled;
    for (std::size_t i = 0; i < qp.size(); ++i)
        if (std::binary_search(samples.begin(), samples.end(), static_cast<std::size_t>(qp[i]))) {
            selected.push_back(i); sampled.pos.push_back(qp[i]);
            sampled.q.insert(sampled.q.end(), q.q.begin() + i * o.shape.heads * o.shape.d,
                             q.q.begin() + (i + 1) * o.shape.heads * o.shape.d);
        }
    std::vector<double> expected;
    if (!selected.empty()) {
        const auto all_pos = attn::positions(o.shape.n, 1, 0, attn::Partition::contiguous);
        const auto all_kv = load_kv(all_pos);
        expected = attn::reference64(sampled, all_kv, o.shape, o.mask);
    }
    // Tolerance: 1e-5 absolute for the order-one synthetic outputs. Real model
    // tensors have their own magnitude, so --tolerance-scale x uses x times the
    // largest reference magnitude instead (stated with every such result).
    double reference_scale = 0;
    for (double x : expected) reference_scale = std::max(reference_scale, std::abs(x));
    reference_scale = max_value(reference_scale);
    const double atol = o.tolerance_scale > 0 ? o.tolerance_scale * reference_scale : 1e-5;
    // Optional external reference (e.g. PyTorch's attention output for real Q/K/V).
    std::vector<float> external;
    if (!o.oref.empty()) external = attn::read_tensor(o.oref, qp, o.shape.nq, o.shape.heads, o.shape.dv);
    std::unique_ptr<attn::DistributedAttention> distributed;
    if (!serial) distributed = std::make_unique<attn::DistributedAttention>(
        o.shape, q, kv, o.mask, o.partition, attn::parse_implementation(o.impl),
        o.kernel, o.order, !o.trace_csv.empty());
    const double setup = max_value(distributed ? distributed->setup_seconds() : 0);
    attn::OnlineState state(q.pos.size(), o.shape);
    std::vector<float> output;
    const std::uint64_t serial_pairs = serial ? attn::count_pairs(qp, kp, o.shape, o.mask) : 0;
    auto run = [&]() {
        if (distributed) return distributed->run();
        MPI_Barrier(MPI_COMM_WORLD);
        attn::RunMetrics metrics;
        const double start = MPI_Wtime();
        if (o.impl == "S1") {
            state.reset();
            metrics.useful_pairs = attn::process_block(q, kv, o.shape, o.mask, o.kernel, state);
            output = attn::finalize(state, o.shape);
        } else if (o.impl == "S0-fp32") output = attn::reference32(q, kv, o.shape, o.mask);
        else {
            const auto reference = attn::reference64(q, kv, o.shape, o.mask);
            output.assign(reference.begin(), reference.end());
        }
        metrics.total = MPI_Wtime() - start;
        metrics.compute = metrics.total;
        if (o.impl != "S1") metrics.useful_pairs = serial_pairs;
        return metrics;
    };
    // Untimed warm-up: at least --warmup repetitions and at least --warmup-seconds
    // of sustained load, so that the measured repetitions run at the processor's
    // steady clock (it stays at the base frequency for the first seconds of load).
    {
        const double begin = MPI_Wtime();
        for (int done = 0;; ++done) {
            const double elapsed = max_value(MPI_Wtime() - begin);
            if (done >= o.warmup && elapsed >= o.warmup_seconds) break;
            run();
        }
    }

    char mpi_text[MPI_MAX_LIBRARY_VERSION_STRING]; int mpi_length = 0;
    MPI_Get_library_version(mpi_text, &mpi_length);
    const std::string mpi_version(mpi_text, static_cast<std::size_t>(mpi_length));
    char host[MPI_MAX_PROCESSOR_NAME]; int host_length = 0; MPI_Get_processor_name(host, &host_length);
    const std::string hostname(host, static_cast<std::size_t>(host_length));
    const std::string header = "timestamp,cluster,nodes,ranks,threads,gpus,impl,partition,mask,N,Nq,d,dv,H,Hkv,block_q,block_k,data,seed,alpha,rep,t_total_max,t_total_mean,t_compute_max,t_compute_mean,t_wait_max,t_wait_mean,t_datagen,bytes_recv_max,useful_flops,gflops,peak_rss_mb_max,val_rows,max_abs_err,max_rel_err,nan_inf,compiler,mpi_version,git_commit,purpose,validation,t_setup_max,t_post_max,t_other_max,thread_level,window,sink,fetch_order,poll_tiles,command,peak_rss_scope,kernel,mha_control,oref_max_abs_err,oref_abs_max,ref_abs_max,atol,pss_mb_max,pss_mb_total,warmup_seconds";
    if (rank == 0 && o.csv.empty()) std::cout << header << '\n';
    bool failed = false;
    for (int rep = 0; rep < o.reps; ++rep) {
        const auto metrics = run();
        const auto& actual = distributed ? distributed->output() : output;
        std::vector<float> sampled_output;
        for (auto i : selected)
            sampled_output.insert(sampled_output.end(), actual.begin() + i * o.shape.heads * o.shape.dv,
                                   actual.begin() + (i + 1) * o.shape.heads * o.shape.dv);
        auto error = attn::compare(sampled_output, expected, atol);
        // Check every output for nonfinite values, even if reference rows are sampled.
        error.nonfinite = 0;
        for (float x : actual) if (!std::isfinite(x)) ++error.nonfinite;
        const double abs_error = max_value(error.max_abs), rel_error = max_value(error.max_rel);
        // Against the external reference: largest absolute difference over all
        // local outputs, and the reference's own largest magnitude for scale.
        double oref_error = 0, oref_scale = 0;
        for (std::size_t i = 0; i < external.size(); ++i) {
            oref_error = std::max(oref_error, std::abs(static_cast<double>(actual[i]) - external[i]));
            oref_scale = std::max(oref_scale, std::abs(static_cast<double>(external[i])));
        }
        oref_error = max_value(oref_error); oref_scale = max_value(oref_scale);
        const auto nonfinite = reduce_count(error.nonfinite, MPI_SUM);
        const auto failures = reduce_count(error.failed, MPI_SUM);
        failed = failed || nonfinite || failures;
        const double total = max_value(metrics.total), total_mean = mean_value(metrics.total, ranks);
        const double compute = max_value(metrics.compute), compute_mean = mean_value(metrics.compute, ranks);
        const double wait = max_value(metrics.wait), wait_mean = mean_value(metrics.wait, ranks);
        const double post = max_value(metrics.post), other = max_value(metrics.other);
        const auto bytes = reduce_count(metrics.bytes_recv, MPI_MAX);
        const auto pairs = reduce_count(metrics.useful_pairs, MPI_SUM);
        const double flops = 2.0 * (o.shape.d + o.shape.dv) * static_cast<double>(pairs);
        const double rss = max_value(peak_rss_mb());
        const double pss_now = pss_mb();
        const double pss_max = max_value(pss_now), pss_total = mean_value(pss_now, ranks) * ranks;
        const auto validation = nonfinite || failures ? "failed" : (o.validate ? "passed" : "not_requested");
        if (rank == 0) {
            std::ostringstream line; line << std::setprecision(17);
            line << now() << ',' << quote(o.cluster) << ',' << nodes << ',' << ranks << ',' << o.kernel.threads << ",0,"
                 << quote(o.impl) << ',' << o.partition_name << ',' << o.mask_name << ',' << o.shape.n << ',' << o.shape.nq << ','
                 << o.shape.d << ',' << o.shape.dv << ',' << o.shape.heads << ',' << o.shape.kv_heads << ','
                 << o.kernel.block_q << ',' << o.kernel.block_k << ',' << quote(o.data) << ',' << o.seed << ',' << o.alpha << ',' << rep << ','
                 << total << ',' << total_mean << ',' << compute << ',' << compute_mean << ',' << wait << ',' << wait_mean << ',' << datagen << ','
                 << bytes << ',' << flops << ',' << flops / total / 1e9 << ',' << rss << ',' << samples.size() << ','
                 << abs_error << ',' << rel_error << ',' << nonfinite << ',' << quote(__VERSION__) << ',' << quote(mpi_version) << ','
                 << quote(ATTN_GIT_COMMIT) << ',' << quote(o.purpose) << ',' << validation << ',' << setup << ',' << post << ',' << other << ','
                 << thread_level << ',' << o.mask.window << ',' << o.mask.sink << ',' << static_cast<int>(o.order) << ',' << o.kernel.poll_tiles << ','
                 << quote(command) << ",process_including_setup_and_validation,"
                 << (o.impl == "S0" || o.impl == "S0-fp32" ? "reference" : o.kernel_name) << ','
                 << (options.mha_control ? 1 : 0) << ',';
            if (!o.oref.empty()) line << oref_error;
            line << ',';
            if (!o.oref.empty()) line << oref_scale;
            line << ',' << reference_scale << ',' << atol << ',' << pss_max << ',' << pss_total << ',' << o.warmup_seconds;
            append(o.csv, header, line.str());
        }
        // One file per rank avoids unsafe concurrent append and preserves exclusive timing intervals.
        if (!o.rank_csv.empty()) {
            std::ostringstream line; line << std::setprecision(17) << rep << ',' << rank << ',' << quote(hostname) << ','
                << metrics.total << ',' << metrics.compute << ',' << metrics.wait << ',' << metrics.post << ',' << metrics.other << ','
                << metrics.bytes_recv << ',' << metrics.useful_pairs;
            append(o.rank_csv + ".rank" + std::to_string(rank) + ".csv",
                   "rep,rank,host,total,compute,wait,post,other,bytes_recv,useful_pairs", line.str());
        }
        if (distributed && !o.trace_csv.empty())
            for (const auto& t : distributed->trace()) {
                std::ostringstream line; line << std::setprecision(17) << rep << ',' << rank << ',' << t.step << ',' << t.owner << ','
                    << t.post_begin << ',' << t.post_end << ',' << t.compute_begin << ',' << t.compute_end << ','
                    << t.wait_begin << ',' << t.wait_end << ',' << t.observed_complete;
                append(o.trace_csv + ".rank" + std::to_string(rank) + ".csv",
                       "rep,rank,step,owner,post_begin,post_end,compute_begin,compute_end,wait_begin,wait_end,observed_complete", line.str());
            }
    }
    return failed ? 2 : 0;
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    try {
        if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED unavailable");
        const auto o = parse(argc, argv);
        if (o.help) {
            if (!rank) std::cout << "attn_bench --impl S0|S0-fp32|S1|allgather|iallgather|ring|overlap|rma_get|rma_put|local\n"
                "  --N 128 [--Nq N] --d 64 --dv 64 --H 1 --Hkv 1 --threads 1\n"
                "  --mask none|causal|window|sink-window --window 128 --sink 4\n"
                "  --partition contiguous|zigzag --block-q 16 --block-k 64 --kernel tuned|basic\n"
                "  --fetch-order staggered|ascending|reverse --poll-tiles 0\n"
                "  --data synthetic|file:<dir> --seed 42 --alpha 1 --validate 16\n"
                "  --warmup 1 --warmup-seconds 0 --reps 3 --csv <path> --rank-csv <prefix> --trace-csv <prefix>\n"
                "  --cluster local --purpose development|pilot|final\n"
                "  --mha-control 0|1 (replicate K/V heads: matched multi-head control)\n"
                "  --oref <file> (external reference output, raw float32 [Nq][H][dv])\n"
                "  --tolerance-scale x (validate with x * largest reference magnitude instead of 1e-5)\n";
            MPI_Finalize(); return 0;
        }
        std::ostringstream command; for (int i = 0; i < argc; ++i) { if (i) command << ' '; command << argv[i]; }
        const int result = execute(o, rank, ranks, provided, command.str());
        MPI_Finalize(); return result;
    } catch (const std::exception& error) {
        std::cerr << "rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1); return 1;
    }
}
