// Statistics of real attention rows (experiment E11), computed in double
// precision with the two-pass row algorithm of the fp64 reference.
//
// For sampled query rows i and every head h, with exact causal probabilities
// p_j (j <= i):
//   entropy            -sum_j p_j log p_j              (nats)
//   max_prob, argmax_distance = i - argmax_j p_j
//   mass_first, mass_sink     probability on key 0 / on the first `sink` keys
//   mass_w             probability on the last w keys            (locality)
//   err_w              || O_window(w) - O_exact || / || O_exact ||   where the
//                      window output renormalises over the last w keys
//   err_sw             same for the sink+window mask (first `sink` keys kept)
//   abs_w, abs_sw      the same two differences without the division
//
// Two edge cases are handled explicitly. (1) If every weight of a window
// underflows relative to the maximum of the whole row, the window is evaluated
// again with its own maximum, so its renormalised output stays defined. (2) If
// the exact output is the zero vector the relative error is 0 when the
// approximation is also exact and infinite otherwise; the absolute error is
// always reported.
//
// Queries are independent here, so ranks simply split the sampled rows; the
// distributed ring is exercised on the same files by attn_bench.
#include "attn/core.hpp"

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<std::int64_t> parse_list(const std::string& text) {
    std::vector<std::int64_t> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) values.push_back(std::stoll(item));
    std::sort(values.begin(), values.end());
    if (values.empty() || values.front() <= 0) throw std::invalid_argument("windows must be positive");
    return values;
}
} // namespace

int main(int argc, char** argv) {
    // Only the main thread calls MPI; the OpenMP threads below only compute.
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED is not available");
        attn::Shape shape;
        std::string directory, csv, label = "data";
        std::vector<std::int64_t> windows = parse_list("16,64,256,1024,4096");
        std::int64_t sink = 4;
        std::size_t sample = 1024;
        for (int i = 1; i + 1 < argc; i += 2) {
            const std::string key = argv[i], value = argv[i + 1];
            if (key == "--data") directory = value;
            else if (key == "--N") shape.n = shape.nq = std::stoll(value);
            else if (key == "--H") shape.heads = std::stoi(value);
            else if (key == "--Hkv") shape.kv_heads = std::stoi(value);
            else if (key == "--d") shape.d = std::stoi(value);
            else if (key == "--dv") shape.dv = std::stoi(value);
            else if (key == "--rows") sample = static_cast<std::size_t>(std::stoull(value));
            else if (key == "--windows") windows = parse_list(value);
            else if (key == "--sink") sink = std::stoll(value);
            else if (key == "--csv") csv = value;
            else if (key == "--label") label = value;
            else throw std::invalid_argument("unknown option: " + key);
        }
        if (directory.empty() || csv.empty()) throw std::invalid_argument("--data <dir> and --csv <file> are required");
        shape.validate();
        if (sink < 0) throw std::invalid_argument("sink must be nonnegative");
        const std::size_t n = static_cast<std::size_t>(shape.n);
        const auto chosen = attn::sample_rows(n, sample ? sample : n);
        std::vector<std::int64_t> mine;                       // this rank's sampled rows
        for (std::size_t s = static_cast<std::size_t>(rank); s < chosen.size(); s += static_cast<std::size_t>(ranks))
            mine.push_back(static_cast<std::int64_t>(chosen[s]));
        const auto all = attn::positions(shape.n, 1, 0, attn::Partition::contiguous);
        const auto kv = attn::read_kv(shape, all, directory);
        const auto q = attn::read_tensor(directory + "/q.bin", mine, shape.n, shape.heads, shape.d);

        const int group = shape.heads / shape.kv_heads;
        const std::size_t nw = windows.size();
        const std::size_t width = 8 + 5 * nw;                 // doubles per (row, head) record
        // Below this sum of weights (relative to the row maximum) a window is
        // re-evaluated with its own maximum; above it the direct formula is accurate.
        const double tiny_mass = 1e-280;
        const double infinity = std::numeric_limits<double>::infinity();
        std::vector<double> records(mine.size() * static_cast<std::size_t>(shape.heads) * width);
        const double scale = 1.0 / std::sqrt(static_cast<double>(shape.d));
        const auto dv = static_cast<std::size_t>(shape.dv);

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::vector<double> score(n), weight(n), accumulator(dv), exact(dv), snapshots(nw * dv), sink_sum(dv);
            std::vector<double> snapshot_mass(nw), window_output(dv), union_output(dv), part(dv), sink_part(dv);
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
            for (std::int64_t r = 0; r < static_cast<std::int64_t>(mine.size()); ++r) {
                const auto i = static_cast<std::size_t>(mine[static_cast<std::size_t>(r)]);
                for (int h = 0; h < shape.heads; ++h) {
                    const int kh = h / group;
                    const float* qr = q.data() + (static_cast<std::size_t>(r) * shape.heads + h) * shape.d;
                    // Pass 1: scores and their maximum.
                    double maximum = -std::numeric_limits<double>::infinity();
                    std::size_t argmax = 0;
                    for (std::size_t j = 0; j <= i; ++j) {
                        const float* kr = kv.k.data() + (j * shape.kv_heads + kh) * shape.d;
                        double dot = 0;
                        for (int a = 0; a < shape.d; ++a) dot += static_cast<double>(qr[a]) * static_cast<double>(kr[a]);
                        score[j] = dot * scale;
                        if (score[j] > maximum) { maximum = score[j]; argmax = j; }
                    }
                    double z = 0, weighted_log = 0;
                    for (std::size_t j = 0; j <= i; ++j) {
                        const double shifted = score[j] - maximum;
                        weight[j] = std::exp(shifted);
                        z += weight[j];
                        weighted_log += weight[j] * shifted;
                    }
                    // Pass 2: accumulate from the nearest key outwards, taking a
                    // snapshot whenever a window boundary is crossed.
                    std::fill(accumulator.begin(), accumulator.end(), 0.0);
                    double partial = 0;
                    std::size_t next = 0;
                    for (std::size_t distance = 1; distance <= i + 1; ++distance) {
                        const std::size_t j = i + 1 - distance;
                        const float* vr = kv.v.data() + (j * shape.kv_heads + kh) * shape.dv;
                        const double w = weight[j];
                        partial += w;
                        for (std::size_t a = 0; a < dv; ++a) accumulator[a] += w * static_cast<double>(vr[a]);
                        while (next < nw && (distance == static_cast<std::size_t>(windows[next]) || (distance == i + 1 && distance < static_cast<std::size_t>(windows[next])))) {
                            std::copy(accumulator.begin(), accumulator.end(), snapshots.begin() + static_cast<std::ptrdiff_t>(next * dv));
                            snapshot_mass[next] = partial;
                            ++next;
                        }
                    }
                    for (std::size_t a = 0; a < dv; ++a) exact[a] = accumulator[a] / z;
                    double norm = 0;
                    for (double x : exact) norm += x * x;
                    norm = std::sqrt(norm);
                    // Sink keys: the first `sink` positions (only those <= i).
                    const std::size_t sink_count = std::min<std::size_t>(static_cast<std::size_t>(sink), i + 1);
                    std::fill(sink_sum.begin(), sink_sum.end(), 0.0);
                    double sink_mass = 0;
                    for (std::size_t j = 0; j < sink_count; ++j) {
                        const float* vr = kv.v.data() + (j * shape.kv_heads + kh) * shape.dv;
                        sink_mass += weight[j];
                        for (std::size_t a = 0; a < dv; ++a) sink_sum[a] += weight[j] * static_cast<double>(vr[a]);
                    }
                    // Sum of exp(score - own maximum) and of the correspondingly weighted values
                    // over the keys [begin, end): a softmax of that range alone.
                    auto range_sums = [&](std::size_t begin, std::size_t end, std::vector<double>& sum, double& own_max) {
                        own_max = -infinity;
                        for (std::size_t j = begin; j < end; ++j) own_max = std::max(own_max, score[j]);
                        std::fill(sum.begin(), sum.end(), 0.0);
                        double total = 0;
                        for (std::size_t j = begin; j < end; ++j) {
                            const float* vr = kv.v.data() + (j * shape.kv_heads + kh) * shape.dv;
                            const double w = std::exp(score[j] - own_max);
                            total += w;
                            for (std::size_t a = 0; a < dv; ++a) sum[a] += w * static_cast<double>(vr[a]);
                        }
                        return total;
                    };
                    auto relative = [&](double absolute) { return norm > 0 ? absolute / norm : (absolute > 0 ? infinity : 0.0); };
                    double* record = records.data() + (static_cast<std::size_t>(r) * shape.heads + h) * width;
                    record[0] = static_cast<double>(i);
                    record[1] = h;
                    record[2] = std::log(z) - weighted_log / z;            // entropy
                    record[3] = 1.0 / z;                                   // largest probability
                    record[4] = static_cast<double>(i - argmax);
                    record[5] = weight[0] / z;
                    record[6] = sink_mass / z;
                    record[7] = norm;
                    for (std::size_t w = 0; w < nw; ++w) {
                        const double* snap = snapshots.data() + w * dv;
                        const double mass = snapshot_mass[w];
                        // The window is [first, i]; the sink is [0, sink_count). When the
                        // two regions touch or overlap their union is the whole causal row,
                        // so the sink+window output is exact.
                        const std::size_t span = std::min<std::size_t>(static_cast<std::size_t>(windows[w]), i + 1);
                        const std::size_t first = i + 1 - span;
                        const bool disjoint = first > sink_count;
                        double probability = mass / z;
                        if (mass >= tiny_mass) {
                            for (std::size_t a = 0; a < dv; ++a) window_output[a] = snap[a] / mass;
                            if (disjoint) for (std::size_t a = 0; a < dv; ++a) union_output[a] = (snap[a] + sink_sum[a]) / (mass + sink_mass);
                        } else {
                            // The window's weights underflow relative to the row maximum.
                            double window_max = 0, sink_max = 0;
                            const double window_total = range_sums(first, i + 1, part, window_max);
                            for (std::size_t a = 0; a < dv; ++a) window_output[a] = part[a] / window_total;
                            probability = std::exp(window_max - maximum + std::log(window_total) - std::log(z));
                            if (disjoint && sink_mass >= tiny_mass) {
                                // The sink carries all the weight of the union that double precision can see.
                                for (std::size_t a = 0; a < dv; ++a) union_output[a] = (snap[a] + sink_sum[a]) / (mass + sink_mass);
                            } else if (disjoint) {
                                const double sink_total = sink_count ? range_sums(0, sink_count, sink_part, sink_max) : 0.0;
                                const double top = sink_count ? std::max(window_max, sink_max) : window_max;
                                const double fw = std::exp(window_max - top), fs = sink_count ? std::exp(sink_max - top) : 0.0;
                                for (std::size_t a = 0; a < dv; ++a)
                                    union_output[a] = (fw * part[a] + (sink_count ? fs * sink_part[a] : 0.0)) / (fw * window_total + fs * sink_total);
                            }
                        }
                        double window_error = 0, sink_error = 0;
                        for (std::size_t a = 0; a < dv; ++a) {
                            window_error += (window_output[a] - exact[a]) * (window_output[a] - exact[a]);
                            if (disjoint) sink_error += (union_output[a] - exact[a]) * (union_output[a] - exact[a]);
                        }
                        window_error = std::sqrt(window_error);
                        sink_error = std::sqrt(sink_error);
                        record[8 + 5 * w] = probability;
                        record[9 + 5 * w] = relative(window_error);
                        record[10 + 5 * w] = relative(sink_error);
                        record[11 + 5 * w] = window_error;
                        record[12 + 5 * w] = sink_error;
                    }
                }
            }
        }

        // Gather fixed-width records on rank 0 and write one CSV.
        std::vector<int> counts(static_cast<std::size_t>(ranks)), offsets(static_cast<std::size_t>(ranks));
        const int local = static_cast<int>(records.size());
        MPI_Gather(&local, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<double> gathered;
        if (rank == 0) {
            int total = 0;
            for (int r = 0; r < ranks; ++r) { offsets[static_cast<std::size_t>(r)] = total; total += counts[static_cast<std::size_t>(r)]; }
            gathered.resize(static_cast<std::size_t>(total));
        }
        MPI_Gatherv(records.data(), local, MPI_DOUBLE, gathered.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::ofstream out(csv);
            if (!out) throw std::runtime_error("cannot write " + csv);
            out.precision(10);
            out << "label,N,sink,row,head,entropy,max_prob,argmax_distance,mass_first,mass_sink,output_norm";
            for (auto w : windows) out << ",mass_w" << w << ",err_w" << w << ",err_sw" << w << ",abs_w" << w << ",abs_sw" << w;
            out << '\n';
            for (std::size_t record = 0; record * width < gathered.size(); ++record) {
                const double* x = gathered.data() + record * width;
                out << label << ',' << shape.n << ',' << sink << ',' << static_cast<long long>(x[0]) << ',' << static_cast<int>(x[1]);
                for (std::size_t c = 2; c < width; ++c) out << ',' << x[c];
                out << '\n';
            }
            std::cout << "wrote " << gathered.size() / width << " (row, head) records to " << csv << '\n';
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
