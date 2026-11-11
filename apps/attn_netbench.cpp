// Hand-written MPI microbenchmarks for the performance model:
//   pingpong : two-sided latency and bandwidth between rank 0 and a peer
//   rget     : one-sided MPI_Rget latency and bandwidth (peer waits inside MPI)
//   progress : does a large transfer complete while the *other* side computes
//              without calling MPI?  Tested for Isend/Irecv and for Rget.
// One CSV line per (mode, peer, message size). Timings are per one-way message.
// Progress rows are single sender/target-side diagnostics, with the origin in
// MPI_Wait. They do not establish application overlap or exact transfer timing.
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
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

// Busy computation with no MPI calls; returns a value so it is not removed.
double spin(double seconds) {
    const auto start = std::chrono::steady_clock::now();
    double x = 1.0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds)
        for (int i = 0; i < 4096; ++i) x = x * 1.0000001 + 1e-9;
    return x;
}

enum Tag { tag_data = 1, tag_control = 2 };
struct Control { int mode, bytes, iterations, stop; };   // sent from rank 0 to the peer
enum Mode { pingpong = 0, rget = 1, progress_send = 2, progress_rget = 3 };

struct Row { std::string mode; int peer, same_node; std::size_t bytes; int iterations; double seconds; };

std::string csv_quote(const std::string& value) {
    std::string result = "\"";
    for (char c : value) {
        if (c == '\0') continue;      // some MPI libraries count the terminator in the version length
        if (c == '"') result += '"';
        result += (c == '\n' || c == '\r') ? ' ' : c;
    }
    return result + '"';
}

double positive_seconds(const std::string& value) {
    std::size_t end = 0;
    const double result = std::stod(value, &end);
    if (end != value.size() || !std::isfinite(result) || result <= 0)
        throw std::invalid_argument("time budget must be finite and positive: " + value);
    return result;
}

void verify_payload(const std::vector<char>& data, std::size_t bytes, char expected) {
    if (!std::all_of(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(bytes),
                     [=](char value) { return value == expected; }))
        throw std::runtime_error("MPI microbenchmark payload mismatch");
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    try {
        std::size_t max_bytes = std::size_t(1) << 24;
        double budget = 0.2, busy = 0.2;
        std::string csv, cluster = "local";
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc) throw std::invalid_argument("missing value for " + std::string(argv[i]));
            const std::string key = argv[i], value = argv[i + 1];
            if (key == "--max-bytes") {
                std::size_t end = 0;
                const auto bytes = std::stoull(value, &end);
                if (value.empty() || value.front() == '-' || end != value.size() || bytes > (1ULL << 30))
                    throw std::invalid_argument("--max-bytes must be an integer from 0 to 1 GiB");
                max_bytes = static_cast<std::size_t>(bytes);
            }
            else if (key == "--budget") budget = positive_seconds(value);
            else if (key == "--busy") busy = positive_seconds(value);
            else if (key == "--csv") csv = value;
            else if (key == "--cluster") cluster = value;
            else throw std::invalid_argument("unknown option: " + key);
        }
        if (ranks < 2) throw std::invalid_argument("attn_netbench needs at least two ranks");
        if (max_bytes > (std::size_t(1) << 30)) throw std::invalid_argument("--max-bytes is limited to 1 GiB");

        // Which ranks share a node with rank 0?
        char name[MPI_MAX_PROCESSOR_NAME] = {0};
        int length = 0;
        MPI_Get_processor_name(name, &length);
        std::vector<char> names(static_cast<std::size_t>(ranks) * MPI_MAX_PROCESSOR_NAME);
        check(MPI_Allgather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, names.data(), MPI_MAX_PROCESSOR_NAME,
                            MPI_CHAR, MPI_COMM_WORLD), "MPI_Allgather names");
        auto same_node = [&](int peer) {
            return std::strncmp(names.data(), names.data() + static_cast<std::size_t>(peer) * MPI_MAX_PROCESSOR_NAME,
                                MPI_MAX_PROCESSOR_NAME) == 0;
        };
        // Select one peer of each available node relationship, regardless of
        // rank placement (which may be cyclic across nodes).
        std::vector<int> peers;
        bool have_local = false, have_remote = false;
        for (int peer = 1; peer < ranks; ++peer) {
            const bool local = same_node(peer);
            if ((local && !have_local) || (!local && !have_remote)) {
                peers.push_back(peer);
                if (local) have_local = true;
                else have_remote = true;
            }
        }

        std::vector<char> buffer(std::max<std::size_t>(max_bytes, 1), 1), scratch(std::max<std::size_t>(max_bytes, 1), 0);
        char* window_data = nullptr;
        MPI_Win window = MPI_WIN_NULL;
        check(MPI_Win_allocate(static_cast<MPI_Aint>(std::max<std::size_t>(max_bytes, 1)), 1, MPI_INFO_NULL,
                               MPI_COMM_WORLD, &window_data, &window), "MPI_Win_allocate");
        check(MPI_Win_lock_all(MPI_MODE_NOCHECK, window), "MPI_Win_lock_all");
        std::memset(window_data, 2, std::max<std::size_t>(max_bytes, 1));
        check(MPI_Win_sync(window), "MPI_Win_sync");
        check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");

        std::vector<std::size_t> sizes{0};
        for (std::size_t bytes = 1; bytes <= max_bytes; bytes *= 4) sizes.push_back(bytes);
        std::vector<Row> rows;
        double sink = 0;

        if (rank == 0) {
            for (int peer : peers) {
                auto command = [&](int mode, std::size_t bytes, int iterations, int stop = 0) {
                    Control c{mode, static_cast<int>(bytes), iterations, stop};
                    check(MPI_Send(&c, 4, MPI_INT, peer, tag_control, MPI_COMM_WORLD), "send control");
                };
                for (std::size_t bytes : sizes) {
                    const int count = static_cast<int>(bytes);
                    // --- two-sided ping-pong: calibrate, then measure ---
                    std::fill(scratch.begin(), scratch.begin() + count, 0);
                    int iterations = 8;
                    double elapsed = 0;
                    for (int attempt = 0; attempt < 2; ++attempt) {
                        command(pingpong, bytes, iterations);
                        const double start = MPI_Wtime();
                        for (int i = 0; i < iterations; ++i) {
                            check(MPI_Send(buffer.data(), count, MPI_BYTE, peer, tag_data, MPI_COMM_WORLD), "ping");
                            check(MPI_Recv(scratch.data(), count, MPI_BYTE, peer, tag_data, MPI_COMM_WORLD, MPI_STATUS_IGNORE), "pong");
                        }
                        elapsed = MPI_Wtime() - start;
                        if (attempt == 0) {
                            const double per = std::max(elapsed / iterations, 1e-9);
                            iterations = static_cast<int>(std::min(200000.0, std::max(8.0, budget / per)));
                        }
                    }
                    rows.push_back({"pingpong", peer, same_node(peer), bytes, iterations, elapsed / (2.0 * iterations)});
                    verify_payload(scratch, bytes, 1);

                    // --- one-sided Rget while the peer blocks inside MPI ---
                    std::fill(scratch.begin(), scratch.begin() + count, 0);
                    command(rget, bytes, 0);
                    iterations = 8;
                    for (int attempt = 0; attempt < 2; ++attempt) {
                        const double start = MPI_Wtime();
                        for (int i = 0; i < iterations; ++i) {
                            MPI_Request request;
                            check(MPI_Rget(scratch.data(), count, MPI_BYTE, peer, 0, count, MPI_BYTE, window, &request), "MPI_Rget");
                            check(MPI_Wait(&request, MPI_STATUS_IGNORE), "MPI_Wait rget");
                        }
                        elapsed = MPI_Wtime() - start;
                        if (attempt == 0) {
                            const double per = std::max(elapsed / iterations, 1e-9);
                            iterations = static_cast<int>(std::min(200000.0, std::max(8.0, budget / per)));
                        }
                    }
                    check(MPI_Send(buffer.data(), 0, MPI_BYTE, peer, tag_data, MPI_COMM_WORLD), "release peer");
                    rows.push_back({"rget", peer, same_node(peer), bytes, iterations, elapsed / iterations});
                    verify_payload(scratch, bytes, 2);
                }
                // --- progress tests on the largest sizes only ---
                for (std::size_t bytes : sizes) {
                    if (bytes < (std::size_t(1) << 16)) continue;
                    const int count = static_cast<int>(bytes);
                    // (a) Peer posts Isend and then computes for `busy` seconds without MPI.
                    //     We time how long our receive takes to complete.
                    std::fill(scratch.begin(), scratch.begin() + count, 0);
                    MPI_Request request;
                    check(MPI_Irecv(scratch.data(), count, MPI_BYTE, peer, tag_data, MPI_COMM_WORLD, &request), "MPI_Irecv");
                    command(progress_send, bytes, 0);
                    double start = MPI_Wtime();
                    check(MPI_Wait(&request, MPI_STATUS_IGNORE), "MPI_Wait progress recv");
                    rows.push_back({"busy_sender_recv", peer, same_node(peer), bytes, 1, MPI_Wtime() - start});
                    verify_payload(scratch, bytes, 1);
                    check(MPI_Recv(scratch.data(), 0, MPI_BYTE, peer, tag_data, MPI_COMM_WORLD, MPI_STATUS_IGNORE), "peer done");
                    // (b) Peer computes for `busy` seconds without MPI; we Rget from it.
                    std::fill(scratch.begin(), scratch.begin() + count, 0);
                    command(progress_rget, bytes, 0);
                    check(MPI_Recv(scratch.data(), 0, MPI_BYTE, peer, tag_data, MPI_COMM_WORLD, MPI_STATUS_IGNORE), "peer started");
                    start = MPI_Wtime();
                    check(MPI_Rget(scratch.data(), count, MPI_BYTE, peer, 0, count, MPI_BYTE, window, &request), "MPI_Rget busy");
                    check(MPI_Wait(&request, MPI_STATUS_IGNORE), "MPI_Wait busy rget");
                    rows.push_back({"busy_target_rget", peer, same_node(peer), bytes, 1, MPI_Wtime() - start});
                    verify_payload(scratch, bytes, 2);
                    check(MPI_Recv(scratch.data(), 0, MPI_BYTE, peer, tag_data, MPI_COMM_WORLD, MPI_STATUS_IGNORE), "peer done");
                }
                command(0, 0, 0, 1);
            }
            for (int r = 1; r < ranks; ++r)
                if (std::find(peers.begin(), peers.end(), r) == peers.end()) {
                    Control c{0, 0, 0, 1};
                    check(MPI_Send(&c, 4, MPI_INT, r, tag_control, MPI_COMM_WORLD), "stop idle rank");
                }
        } else {
            for (;;) {
                Control c{};
                check(MPI_Recv(&c, 4, MPI_INT, 0, tag_control, MPI_COMM_WORLD, MPI_STATUS_IGNORE), "recv control");
                if (c.stop) break;
                if (c.mode == pingpong) {
                    for (int i = 0; i < c.iterations; ++i) {
                        check(MPI_Recv(scratch.data(), c.bytes, MPI_BYTE, 0, tag_data, MPI_COMM_WORLD, MPI_STATUS_IGNORE), "peer recv");
                        check(MPI_Send(scratch.data(), c.bytes, MPI_BYTE, 0, tag_data, MPI_COMM_WORLD), "peer send");
                    }
                } else if (c.mode == rget) {
                    // Stay inside MPI so that target-side progress is available.
                    check(MPI_Recv(scratch.data(), 0, MPI_BYTE, 0, tag_data, MPI_COMM_WORLD, MPI_STATUS_IGNORE), "peer wait");
                } else if (c.mode == progress_send) {
                    MPI_Request request;
                    check(MPI_Isend(buffer.data(), c.bytes, MPI_BYTE, 0, tag_data, MPI_COMM_WORLD, &request), "peer isend");
                    sink += spin(busy);
                    check(MPI_Wait(&request, MPI_STATUS_IGNORE), "peer wait isend");
                    check(MPI_Send(buffer.data(), 0, MPI_BYTE, 0, tag_data, MPI_COMM_WORLD), "peer done");
                } else if (c.mode == progress_rget) {
                    check(MPI_Send(buffer.data(), 0, MPI_BYTE, 0, tag_data, MPI_COMM_WORLD), "peer started");
                    sink += spin(busy);
                    check(MPI_Send(buffer.data(), 0, MPI_BYTE, 0, tag_data, MPI_COMM_WORLD), "peer done");
                }
            }
        }
        check(MPI_Win_unlock_all(window), "MPI_Win_unlock_all");
        check(MPI_Win_free(&window), "MPI_Win_free");

        if (rank == 0) {
            char version[MPI_MAX_LIBRARY_VERSION_STRING];
            int version_length = 0;
            MPI_Get_library_version(version, &version_length);
            std::string mpi(version, static_cast<std::size_t>(version_length));
            for (char& c : mpi) if (c == '\n' || c == '\r' || c == '"' || c == ',') c = ' ';
            std::ofstream file;
            if (!csv.empty()) {
                file.open(csv, std::ios::app);
                if (!file) throw std::runtime_error("cannot write CSV: " + csv);
            }
            std::ostream& out = csv.empty() ? std::cout : file;
            out << std::setprecision(17);
            if (csv.empty() || file.tellp() == 0)
                out << "cluster,ranks,mode,peer,same_node,bytes,iterations,seconds,mb_per_s,busy_seconds,mpi_version\n";
            for (const auto& row : rows)
                out << csv_quote(cluster) << ',' << ranks << ',' << row.mode << ',' << row.peer << ',' << row.same_node << ','
                    << row.bytes << ',' << row.iterations << ',' << row.seconds << ','
                    << (row.seconds > 0 ? static_cast<double>(row.bytes) / row.seconds / 1e6 : 0.0) << ','
                    << busy << ",\"" << mpi << "\"\n";
            out.flush();
            if (!csv.empty()) file.close();
            if (!out) throw std::runtime_error("CSV output failed: " + csv);
        }
        if (sink < 0) std::cerr << sink;
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
