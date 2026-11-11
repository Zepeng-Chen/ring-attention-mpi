// What does one step of the blocking ring cost when nothing is computed?
//
// Every rank does what M2 does between two blocks: MPI_Sendrecv of a key block
// (send to the right-hand neighbour, receive from the left-hand one), then the same
// for a value block, alternating between two buffers. All ranks do it at the same
// time, as in a real run, and nothing else. The two-rank ping-pong of attn_netbench
// measures one message between two ranks of an idle node; this measures the
// exchange of the ring itself, with every rank sending and receiving at once.
//
//   --rows 16,64,256   shard sizes c in rows (a block has c*d numbers; several values)
//   --d 64 --dv 64     head dimensions of the key and the value block
//   --budget 0.3       seconds measured per shard size
//   --csv FILE         append one line per shard size (default: standard output)
//
// One CSV line per shard size: time of one ring step (both exchanges), as the
// maximum and the mean over ranks.
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(int status, const char* what) {
    if (status == MPI_SUCCESS) return;
    char text[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, text, &length);
    throw std::runtime_error(std::string(what) + ": " + std::string(text, static_cast<std::size_t>(length)));
}
long long integer(const std::string& s) {
    std::size_t end = 0;
    const long long x = std::stoll(s, &end);
    if (end != s.size() || x <= 0) throw std::invalid_argument("expected a positive integer: " + s);
    return x;
}
double positive(const std::string& s) {
    std::size_t end = 0;
    const double x = std::stod(s, &end);
    if (end != s.size() || !std::isfinite(x) || x <= 0) throw std::invalid_argument("expected a positive finite number: " + s);
    return x;
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    try {
        std::vector<long long> rows{16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
        long long d = 64, dv = 64;
        double budget = 0.3;
        std::string csv, cluster = "local";
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc) throw std::invalid_argument("missing value for " + std::string(argv[i]));
            const std::string key = argv[i], value = argv[i + 1];
            if (key == "--rows") {
                rows.clear();
                std::stringstream list(value);
                std::string item;
                while (std::getline(list, item, ',')) rows.push_back(integer(item));
                if (rows.empty()) throw std::invalid_argument("--rows needs at least one value");
            }
            else if (key == "--d") d = integer(value);
            else if (key == "--dv") dv = integer(value);
            else if (key == "--budget") budget = positive(value);
            else if (key == "--csv") csv = value;
            else if (key == "--cluster") cluster = value;
            else throw std::invalid_argument("unknown option: " + key);
        }
        if (ranks < 2) throw std::invalid_argument("attn_ringbench needs at least two ranks");
        const int left = (rank + ranks - 1) % ranks, right = (rank + 1) % ranks;

        // Number of nodes (for the record).
        char name[MPI_MAX_PROCESSOR_NAME] = {0};
        int length = 0;
        MPI_Get_processor_name(name, &length);
        std::vector<char> names(static_cast<std::size_t>(ranks) * MPI_MAX_PROCESSOR_NAME);
        check(MPI_Allgather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, names.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                            MPI_COMM_WORLD), "MPI_Allgather names");
        int nodes = 0;
        for (int r = 0; r < ranks; ++r) {
            bool seen = false;
            for (int s = 0; s < r && !seen; ++s)
                seen = std::string(names.data() + static_cast<std::size_t>(r) * MPI_MAX_PROCESSOR_NAME) ==
                       std::string(names.data() + static_cast<std::size_t>(s) * MPI_MAX_PROCESSOR_NAME);
            if (!seen) ++nodes;
        }

        struct Line { long long rows; std::size_t bytes; long long steps; double worst, mean; };
        std::vector<Line> lines;
        for (long long c : rows) {
            if (c * std::max(d, dv) > 0x7fffffffLL) throw std::overflow_error("block exceeds the MPI count limit");
            const int nk = static_cast<int>(c * d), nv = static_cast<int>(c * dv);
            // Two buffers per tensor, as in the ring: the block received in one step is sent in the next.
            std::vector<float> k[2] = {std::vector<float>(nk, static_cast<float>(rank)), std::vector<float>(nk, -1.0f)};
            std::vector<float> v[2] = {std::vector<float>(nv, static_cast<float>(rank)), std::vector<float>(nv, -1.0f)};
            int current = 0;
            auto step = [&] {
                const int next = 1 - current;
                check(MPI_Sendrecv(k[current].data(), nk, MPI_FLOAT, right, 0, k[next].data(), nk, MPI_FLOAT, left, 0,
                                   MPI_COMM_WORLD, MPI_STATUS_IGNORE), "MPI_Sendrecv K");
                check(MPI_Sendrecv(v[current].data(), nv, MPI_FLOAT, right, 1, v[next].data(), nv, MPI_FLOAT, left, 1,
                                   MPI_COMM_WORLD, MPI_STATUS_IGNORE), "MPI_Sendrecv V");
                current = next;
            };
            // Warm up (connections, page faults), calibrate, then measure; every rank uses the same step counts.
            long long steps = 64, done = 0;
            double worst = 0, sum = 0;
            for (int pass = 0; pass < 3; ++pass) {
                check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
                const double start = MPI_Wtime();
                for (long long i = 0; i < steps; ++i) step();
                const double elapsed = MPI_Wtime() - start;
                done += steps;
                check(MPI_Allreduce(&elapsed, &worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD), "MPI_Allreduce max");
                check(MPI_Allreduce(&elapsed, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), "MPI_Allreduce sum");
                if (pass == 0) steps = 256;
                if (pass == 1) {
                    const double per = std::max(worst / static_cast<double>(steps), 1e-9);
                    steps = static_cast<long long>(std::min(2000000.0, std::max(256.0, budget / per)));
                }
            }
            // After `done` steps every block has moved `done` ranks to the right.
            const float expected = static_cast<float>(((rank - done) % ranks + ranks) % ranks);
            if (k[current].front() != expected || k[current].back() != expected || v[current].front() != expected ||
                v[current].back() != expected)
                throw std::runtime_error("ring exchange delivered the wrong block");
            lines.push_back({c, static_cast<std::size_t>(nk + nv) * sizeof(float), steps, worst / static_cast<double>(steps),
                             sum / static_cast<double>(ranks) / static_cast<double>(steps)});
        }

        if (rank == 0) {
            std::ofstream file;
            bool header = true;
            if (!csv.empty()) {
                header = !std::filesystem::exists(csv) || std::filesystem::file_size(csv) == 0;
                file.open(csv, std::ios::app);
                if (!file) throw std::runtime_error("cannot write CSV: " + csv);
            }
            std::ostream& out = csv.empty() ? std::cout : file;
            out << std::setprecision(17);
            if (header)
                out << "cluster,ranks,nodes,rows,d,dv,bytes_per_step,steps,seconds_per_step_max,seconds_per_step_mean\n";
            for (const auto& line : lines)
                out << cluster << ',' << ranks << ',' << nodes << ',' << line.rows << ',' << d << ',' << dv << ',' << line.bytes << ','
                    << line.steps << ',' << line.worst << ',' << line.mean << '\n';
            out.flush();
            if (!out) throw std::runtime_error("CSV output failed: " + csv);
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
