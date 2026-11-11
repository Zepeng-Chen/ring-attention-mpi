// MPI + CUDA driver: one GPU per rank, K/V blocks travel around the ring.
//   --kernel k0|k1|k2          GPU kernel variant (see include/attn/gpu.hpp)
//   --comm host|aware          host-staged MPI buffers + explicit copies, or
//                              CUDA-aware MPI on device pointers
//   --schedule overlap|blocking  exchange the next block while the kernel runs,
//                              or only after it finished
// With one rank this is the single-GPU kernel benchmark. Output is validated
// against the independent fp64 reference on sampled query rows.
#include "attn/core.hpp"
#include "attn/gpu.hpp"

#include <mpi.h>
#if defined(OPEN_MPI)
#include <mpi-ext.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void mpi_check(int status, const char* what) {
    if (status == MPI_SUCCESS) return;
    char text[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, text, &length);
    throw std::runtime_error(std::string(what) + ": " + std::string(text, static_cast<std::size_t>(length)));
}
struct Options {
    attn::Shape shape;
    attn::Mask mask;
    attn::Partition partition = attn::Partition::contiguous;
    std::string kernel = "k2", comm = "host", schedule = "overlap";
    std::string mask_name = "none", partition_name = "contiguous", data = "synthetic";
    std::string cluster = "local", purpose = "development", csv, rank_csv;
    std::uint64_t seed = 42;
    float alpha = 1;
    int reps = 3, warmup = 1;
    std::size_t validate = 16, k1_budget_mb = 2048;
    double tolerance_scale = 0;   // 0: absolute 1e-5; x > 0: x times the largest reference magnitude
    // Functional test only: let the ranks of a node share its GPUs (rank r uses
    // device r mod #GPUs). Timings of such a run say nothing about scaling.
    bool share_gpu = false;
};
// Numeric options are read strictly, as in the CPU driver: the whole token must be
// a number, and real values must be finite.
std::int64_t integer(const std::string& s) {
    std::size_t end = 0;
    const auto x = std::stoll(s, &end);
    if (end != s.size()) throw std::invalid_argument("invalid integer: " + s);
    return x;
}
int small_integer(const std::string& s) {
    const auto x = integer(s);
    if (x < 0 || x > 0x7fffffff) throw std::invalid_argument("integer out of range: " + s);
    return static_cast<int>(x);
}
double real(const std::string& s) {
    std::size_t end = 0;
    const double x = std::stod(s, &end);
    if (end != s.size() || !std::isfinite(x)) throw std::invalid_argument("invalid number: " + s);
    return x;
}
Options parse(int argc, char** argv) {
    Options o;
    bool nq_set = false;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i], value = argv[i + 1];
        if (key == "--N") o.shape.n = integer(value);
        else if (key == "--Nq") { o.shape.nq = integer(value); nq_set = true; }
        else if (key == "--d") o.shape.d = small_integer(value);
        else if (key == "--dv") o.shape.dv = small_integer(value);
        else if (key == "--H") o.shape.heads = small_integer(value);
        else if (key == "--Hkv") o.shape.kv_heads = small_integer(value);
        else if (key == "--kernel") o.kernel = value;
        else if (key == "--comm") o.comm = value;
        else if (key == "--schedule") o.schedule = value;
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
        } else if (key == "--seed") o.seed = static_cast<std::uint64_t>(small_integer(value));
        else if (key == "--alpha") o.alpha = static_cast<float>(real(value));
        else if (key == "--validate") o.validate = static_cast<std::size_t>(small_integer(value));
        else if (key == "--reps") o.reps = small_integer(value);
        else if (key == "--warmup") o.warmup = small_integer(value);
        else if (key == "--data") o.data = value;
        else if (key == "--cluster") o.cluster = value;
        else if (key == "--purpose") o.purpose = value;
        else if (key == "--csv") o.csv = value;
        else if (key == "--rank-csv") o.rank_csv = value;
        else if (key == "--k1-budget-mb") o.k1_budget_mb = static_cast<std::size_t>(small_integer(value));
        else if (key == "--tolerance-scale") o.tolerance_scale = real(value);
        else if (key == "--share-gpu") o.share_gpu = value == "1";
        else throw std::invalid_argument("unknown option: " + key);
    }
    if (argc % 2 == 0) throw std::invalid_argument("every option needs a value");
    if (!nq_set) o.shape.nq = o.shape.n;
    o.shape.validate();
    o.mask.validate();
    if (o.comm != "host" && o.comm != "aware") throw std::invalid_argument("--comm must be host or aware");
    if (o.schedule != "overlap" && o.schedule != "blocking") throw std::invalid_argument("--schedule must be overlap or blocking");
    if (o.reps < 1 || o.warmup < 0) throw std::invalid_argument("invalid repetition counts");
    if (!std::isfinite(o.tolerance_scale) || o.tolerance_scale < 0) throw std::invalid_argument("--tolerance-scale must be finite and >= 0");
    if (!std::isfinite(o.alpha)) throw std::invalid_argument("--alpha must be finite");
    if (o.data != "synthetic" && o.data.rfind("file:", 0) != 0) throw std::invalid_argument("data must be synthetic or file:<directory>");
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
    std::ostringstream out;
    out << std::put_time(std::gmtime(&time), "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}
void append(const std::string& path, const std::string& header, const std::string& row) {
    if (path.empty()) { std::cout << row << '\n'; return; }
    const std::filesystem::path p(path);
    if (!p.parent_path().empty()) std::filesystem::create_directories(p.parent_path());
    const bool empty = !std::filesystem::exists(p) || std::filesystem::file_size(p) == 0;
    std::ofstream file(path, std::ios::app);
    if (!file) throw std::runtime_error("cannot write " + path);
    if (empty) file << header << '\n';
    file << row << '\n';
    file.flush();
    if (!file) throw std::runtime_error("CSV write failed: " + path);
}
double reduce(double value, MPI_Op op) {
    double result = 0;
    mpi_check(MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, op, MPI_COMM_WORLD), "reduce");
    return result;
}
std::uint64_t reduce(std::uint64_t value, MPI_Op op) {
    std::uint64_t result = 0;
    mpi_check(MPI_Allreduce(&value, &result, 1, MPI_UINT64_T, op, MPI_COMM_WORLD), "reduce");
    return result;
}
int count_int(std::size_t count) {
    if (count > 0x7fffffffu) throw std::overflow_error("MPI count exceeds INT_MAX");
    return static_cast<int>(count);
}

struct Metrics {
    // Per-rank intervals: total wall time, kernel time on the device (events),
    // time blocked in MPI, time blocked waiting for the kernel after MPI
    // returned, and time spent staging blocks through host memory.
    double total = 0, kernel = 0, mpi = 0, gpu_wait = 0, staging = 0;
    std::uint64_t bytes = 0;
};

int execute(const Options& o, int rank, int ranks, const std::string& command) {
    // GPU of this rank: ranks on the same node take the node's devices in order.
    MPI_Comm node;
    mpi_check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node), "node split");
    int local_rank = 0, local_size = 0;
    MPI_Comm_rank(node, &local_rank);
    MPI_Comm_size(node, &local_size);
    const int nodes = static_cast<int>(reduce(static_cast<std::uint64_t>(local_rank == 0 ? 1 : 0), MPI_SUM));
    MPI_Comm_free(&node);
    const int devices = attn::gpu::device_count();
    if (devices < 1) throw std::runtime_error("no CUDA device visible");
    if (local_size > devices && !o.share_gpu)
        throw std::runtime_error("more ranks on a node than GPUs; one GPU per rank is required (--share-gpu 1 for a functional test)");
    const int device_id = local_rank % devices;
    // Number of distinct devices in use (equals the rank count unless GPUs are shared).
    const int gpus_used = static_cast<int>(reduce(static_cast<std::uint64_t>(local_rank < devices ? 1 : 0), MPI_SUM));
    const bool aware = o.comm == "aware", overlap = o.schedule == "overlap";
#if defined(OPEN_MPI) && defined(MPIX_CUDA_AWARE_SUPPORT)
    if (aware && ranks > 1 && MPIX_Query_cuda_support() != 1)
        throw std::runtime_error("this MPI library reports no CUDA-aware support");
#else
    if (aware && ranks > 1) throw std::runtime_error("cannot verify CUDA-aware MPI support in this build");
#endif

    const auto qp = attn::positions(o.shape.nq, ranks, rank, o.partition);
    const auto kp = attn::positions(o.shape.n, ranks, rank, o.partition);
    const bool synthetic = o.data == "synthetic";
    const auto q = synthetic ? attn::make_queries(o.shape, qp, o.seed, o.alpha) : attn::read_queries(o.shape, qp, o.data.substr(5));
    const auto kv = synthetic ? attn::make_kv(o.shape, kp, o.seed) : attn::read_kv(o.shape, kp, o.data.substr(5));

    // Sampled fp64 reference rows (each uses every key), as in the CPU driver.
    const auto samples = attn::sample_rows(static_cast<std::size_t>(o.shape.nq), o.validate);
    std::vector<std::size_t> selected;
    attn::Queries sampled;
    for (std::size_t i = 0; i < qp.size(); ++i)
        if (std::binary_search(samples.begin(), samples.end(), static_cast<std::size_t>(qp[i]))) {
            selected.push_back(i);
            sampled.pos.push_back(qp[i]);
            sampled.q.insert(sampled.q.end(), q.q.begin() + i * o.shape.heads * o.shape.d,
                             q.q.begin() + (i + 1) * o.shape.heads * o.shape.d);
        }
    const auto all_pos = attn::positions(o.shape.n, 1, 0, attn::Partition::contiguous);
    std::vector<double> expected;
    if (!selected.empty()) {
        const auto all_kv = synthetic ? attn::make_kv(o.shape, all_pos, o.seed) : attn::read_kv(o.shape, all_pos, o.data.substr(5));
        expected = attn::reference64(sampled, all_kv, o.shape, o.mask);
    }
    const std::uint64_t pairs = reduce(attn::count_pairs(qp, all_pos, o.shape, o.mask), MPI_SUM);
    // Same tolerance rule as the CPU driver: 1e-5 absolute for order-one synthetic
    // outputs, or a stated multiple of the largest reference magnitude for real tensors.
    double reference_scale = 0;
    for (double x : expected) reference_scale = std::max(reference_scale, std::abs(x));
    reference_scale = reduce(reference_scale, MPI_MAX);
    const double atol = o.tolerance_scale > 0 ? o.tolerance_scale * reference_scale : 1e-5;

    std::vector<int> kcounts(ranks), vcounts(ranks);
    std::size_t max_rows = 0;
    for (int r = 0; r < ranks; ++r) {
        const auto rows = attn::positions(o.shape.n, ranks, r, o.partition).size();
        max_rows = std::max(max_rows, rows);
        kcounts[r] = count_int(rows * o.shape.kv_heads * o.shape.d);
        vcounts[r] = count_int(rows * o.shape.kv_heads * o.shape.dv);
    }
    attn::gpu::Device* device = attn::gpu::create(device_id, o.shape, o.mask, attn::gpu::parse_variant(o.kernel),
                                                  q, max_rows, o.k1_budget_mb << 20);
    std::vector<float> output(qp.size() * static_cast<std::size_t>(o.shape.heads) * o.shape.dv);
    const int left = (rank + ranks - 1) % ranks, right = (rank + 1) % ranks;

    auto run = [&]() {
        mpi_check(MPI_Barrier(MPI_COMM_WORLD), "barrier");
        Metrics metrics;
        const double started = MPI_Wtime();
        attn::gpu::reset(device);
        // Step 0 uses the local shard: stage it once.
        double t0 = MPI_Wtime();
        if (!kv.k.empty()) std::memcpy(attn::gpu::k_host(device, 0), kv.k.data(), kv.k.size() * sizeof(float));
        if (!kv.v.empty()) std::memcpy(attn::gpu::v_host(device, 0), kv.v.data(), kv.v.size() * sizeof(float));
        attn::gpu::set_block(device, 0, kp);
        attn::gpu::upload(device, 0);
        attn::gpu::sync_copy(device);
        metrics.staging += MPI_Wtime() - t0;
        int current = 0;
        for (int step = 0; step < ranks; ++step) {
            const int owner = (rank - step + ranks) % ranks;
            if (step == ranks - 1) {
                attn::gpu::launch(device, current);
                t0 = MPI_Wtime();
                metrics.kernel += attn::gpu::sync_compute(device);
                metrics.gpu_wait += MPI_Wtime() - t0;
                break;
            }
            const int next = 1 - current, next_owner = (owner + ranks - 1) % ranks;
            float* recv_k = aware ? attn::gpu::k_device(device, next) : attn::gpu::k_host(device, next);
            float* recv_v = aware ? attn::gpu::v_device(device, next) : attn::gpu::v_host(device, next);
            float* send_k = aware ? attn::gpu::k_device(device, current) : attn::gpu::k_host(device, current);
            float* send_v = aware ? attn::gpu::v_device(device, current) : attn::gpu::v_host(device, current);
            std::array<MPI_Request, 4> requests{};
            auto post = [&] {
                mpi_check(MPI_Irecv(recv_k, kcounts[next_owner], MPI_FLOAT, left, 0, MPI_COMM_WORLD, &requests[0]), "Irecv K");
                mpi_check(MPI_Irecv(recv_v, vcounts[next_owner], MPI_FLOAT, left, 1, MPI_COMM_WORLD, &requests[1]), "Irecv V");
                mpi_check(MPI_Isend(send_k, kcounts[owner], MPI_FLOAT, right, 0, MPI_COMM_WORLD, &requests[2]), "Isend K");
                mpi_check(MPI_Isend(send_v, vcounts[owner], MPI_FLOAT, right, 1, MPI_COMM_WORLD, &requests[3]), "Isend V");
            };
            auto wait_mpi = [&] {
                const double begin = MPI_Wtime();
                mpi_check(MPI_Waitall(4, requests.data(), MPI_STATUSES_IGNORE), "Waitall");
                metrics.mpi += MPI_Wtime() - begin;
            };
            auto wait_kernel = [&] {
                const double begin = MPI_Wtime();
                metrics.kernel += attn::gpu::sync_compute(device);
                metrics.gpu_wait += MPI_Wtime() - begin;
            };
            auto stage_next = [&] {
                const double begin = MPI_Wtime();
                attn::gpu::set_block(device, next, attn::positions(o.shape.n, ranks, next_owner, o.partition));
                if (!aware) attn::gpu::upload(device, next);
                attn::gpu::sync_copy(device);
                metrics.staging += MPI_Wtime() - begin;
            };
            attn::gpu::launch(device, current);       // asynchronous
            if (overlap) {
                post();
                wait_mpi();
                stage_next();                           // copy stream; the kernel may still run
                wait_kernel();
            } else {
                wait_kernel();
                post();
                wait_mpi();
                stage_next();
            }
            metrics.bytes += (static_cast<std::uint64_t>(kcounts[next_owner]) + static_cast<std::uint64_t>(vcounts[next_owner])) * sizeof(float);
            current = next;
        }
        attn::gpu::finalize(device, output.data());
        metrics.total = MPI_Wtime() - started;
        return metrics;
    };
    for (int rep = 0; rep < o.warmup; ++rep) run();

    char host[MPI_MAX_PROCESSOR_NAME];
    int host_length = 0;
    MPI_Get_processor_name(host, &host_length);
    char mpi_text[MPI_MAX_LIBRARY_VERSION_STRING];
    int mpi_length = 0;
    MPI_Get_library_version(mpi_text, &mpi_length);
    const std::string header = "timestamp,cluster,nodes,ranks,gpus,kernel,comm,schedule,partition,mask,N,Nq,d,dv,H,Hkv,data,seed,alpha,rep,"
        "t_total_max,t_total_mean,t_kernel_max,t_kernel_mean,t_mpi_max,t_mpi_mean,t_gpu_wait_max,t_staging_max,bytes_recv_max,"
        "useful_flops,gflops,val_rows,max_abs_err,max_rel_err,nan_inf,validation,gpu_name,mpi_version,purpose,command,ref_abs_max,atol";
    bool failed = false;
    for (int rep = 0; rep < o.reps; ++rep) {
        const auto metrics = run();
        std::vector<float> sampled_output;
        for (auto i : selected)
            sampled_output.insert(sampled_output.end(), output.begin() + i * o.shape.heads * o.shape.dv,
                                  output.begin() + (i + 1) * o.shape.heads * o.shape.dv);
        auto error = attn::compare(sampled_output, expected, atol);
        error.nonfinite = 0;
        for (float x : output) if (!std::isfinite(x)) ++error.nonfinite;
        const double abs_error = reduce(error.max_abs, MPI_MAX), rel_error = reduce(error.max_rel, MPI_MAX);
        const auto nonfinite = reduce(error.nonfinite, MPI_SUM), failures = reduce(error.failed, MPI_SUM);
        failed = failed || nonfinite || failures;
        const double total = reduce(metrics.total, MPI_MAX);
        const double flops = 2.0 * (o.shape.d + o.shape.dv) * static_cast<double>(pairs);
        if (rank == 0) {
            std::ostringstream line;
            line << std::setprecision(17) << now() << ',' << quote(o.cluster) << ',' << nodes << ',' << ranks << ',' << gpus_used << ','
                 << o.kernel << ',' << o.comm << ',' << o.schedule << ',' << o.partition_name << ',' << o.mask_name << ','
                 << o.shape.n << ',' << o.shape.nq << ',' << o.shape.d << ',' << o.shape.dv << ',' << o.shape.heads << ',' << o.shape.kv_heads << ','
                 << quote(o.data) << ',' << o.seed << ',' << o.alpha << ',' << rep << ','
                 << total << ',' << reduce(metrics.total, MPI_SUM) / ranks << ','
                 << reduce(metrics.kernel, MPI_MAX) << ',' << reduce(metrics.kernel, MPI_SUM) / ranks << ','
                 << reduce(metrics.mpi, MPI_MAX) << ',' << reduce(metrics.mpi, MPI_SUM) / ranks << ','
                 << reduce(metrics.gpu_wait, MPI_MAX) << ',' << reduce(metrics.staging, MPI_MAX) << ','
                 << reduce(metrics.bytes, MPI_MAX) << ',' << flops << ',' << flops / total / 1e9 << ',' << samples.size() << ','
                 << abs_error << ',' << rel_error << ',' << nonfinite << ','
                 << (nonfinite || failures ? "failed" : (o.validate ? "passed" : "not_requested")) << ','
                 << quote(attn::gpu::name(device)) << ',' << quote(std::string(mpi_text, static_cast<std::size_t>(mpi_length))) << ','
                 << quote(o.purpose) << ',' << quote(command) << ',' << reference_scale << ',' << atol;
            if (o.csv.empty() && rep == 0) std::cout << header << '\n';
            append(o.csv, header, line.str());
        } else {
            // Keep the reductions collective on every rank.
            reduce(metrics.total, MPI_SUM); reduce(metrics.kernel, MPI_MAX); reduce(metrics.kernel, MPI_SUM);
            reduce(metrics.mpi, MPI_MAX); reduce(metrics.mpi, MPI_SUM); reduce(metrics.gpu_wait, MPI_MAX);
            reduce(metrics.staging, MPI_MAX); reduce(metrics.bytes, MPI_MAX);
        }
        if (!o.rank_csv.empty()) {
            std::ostringstream line;
            line << std::setprecision(17) << rep << ',' << rank << ',' << quote(std::string(host, static_cast<std::size_t>(host_length))) << ','
                 << device_id << ',' << metrics.total << ',' << metrics.kernel << ',' << metrics.mpi << ',' << metrics.gpu_wait << ','
                 << metrics.staging << ',' << metrics.bytes;
            append(o.rank_csv + ".rank" + std::to_string(rank) + ".csv",
                   "rep,rank,host,gpu,total,kernel,mpi,gpu_wait,staging,bytes_recv", line.str());
        }
    }
    attn::gpu::destroy(device);
    return failed ? 2 : 0;
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    try {
        const auto options = parse(argc, argv);
        std::ostringstream command;
        for (int i = 0; i < argc; ++i) { if (i) command << ' '; command << argv[i]; }
        const int result = execute(options, rank, ranks, command.str());
        MPI_Finalize();
        return result;
    } catch (const std::exception& error) {
        std::cerr << "rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
