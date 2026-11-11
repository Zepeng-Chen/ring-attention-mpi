#include "attn/distributed.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

namespace attn {

void mpi_check(int status, const char* operation) {
    if (status == MPI_SUCCESS) return;
    char text[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, text, &length);
    throw std::runtime_error(std::string(operation) + ": " + std::string(text, static_cast<std::size_t>(length)));
}

const char* implementation_name(Implementation impl) {
    switch (impl) {
    case Implementation::allgather: return "allgather";
    case Implementation::iallgather: return "iallgather";
    case Implementation::ring: return "ring";
    case Implementation::overlap: return "overlap";
    case Implementation::rma_get: return "rma_get";
    case Implementation::rma_put: return "rma_put";
    case Implementation::local: return "local";
    }
    throw std::invalid_argument("unknown distributed implementation");
}

Implementation parse_implementation(const std::string& name) {
    if (name == "allgather" || name == "M1") return Implementation::allgather;
    if (name == "iallgather" || name == "M1b") return Implementation::iallgather;
    if (name == "ring" || name == "M2") return Implementation::ring;
    if (name == "overlap" || name == "M3") return Implementation::overlap;
    if (name == "rma_get" || name == "rma-get" || name == "M7") return Implementation::rma_get;
    if (name == "rma_put" || name == "rma-put" || name == "M8") return Implementation::rma_put;
    if (name == "local" || name == "M0") return Implementation::local;
    throw std::invalid_argument("unknown distributed implementation: " + name);
}

namespace {
int count_int(std::size_t count) {
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("MPI count exceeds INT_MAX; reduce the problem size");
    return static_cast<int>(count);
}
std::size_t row_count(std::int64_t n, int ranks, int rank, Partition partition) {
    const auto chunks = partition == Partition::zigzag ? std::int64_t(2) * ranks : ranks;
    const auto base = n / chunks;
    const auto extra = n % chunks;
    auto result = base + (rank < extra ? 1 : 0);
    if (partition == Partition::zigzag) result += base + (chunks - 1 - rank < extra ? 1 : 0);
    return static_cast<std::size_t>(result);
}
std::size_t tensor_count(std::size_t rows, int heads, int dim) {
    return checked_size(checked_size(rows, static_cast<std::size_t>(heads)), static_cast<std::size_t>(dim));
}
bool any_allowed(const Queries& q, const std::vector<std::int64_t>& keys, const Mask& mask) {
    if (q.pos.empty() || keys.empty()) return false;
    if (mask.kind == MaskKind::none) return true;
    // Partition positions are sorted even for zigzag. Find the last key no later
    // than each query; it is the best window candidate. Sink keys are separate.
    for (auto qp : q.pos) {
        auto after = std::upper_bound(keys.begin(), keys.end(), qp);
        if (after != keys.begin() && mask.allows(qp, *std::prev(after))) return true;
        if (mask.kind == MaskKind::sink_window && mask.allows(qp, keys.front())) return true;
    }
    return false;
}
} // namespace

struct DistributedAttention::Impl {
    Shape shape;
    const Queries& queries;
    const KVBlock& local;
    Mask mask;
    Partition partition;
    Implementation method;
    KernelOptions kernel;
    FetchOrder order;
    bool tracing;
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0, ranks = 1;
    std::vector<int> kcounts, vcounts, koffsets, voffsets;
    std::array<KVBlock, 2> buffer;
    std::vector<float> gathered_k, gathered_v, result;
    OnlineState state;
    MPI_Win window = MPI_WIN_NULL;
    float* window_data = nullptr;
    bool passive_epoch = false;
    std::size_t max_rows = 0, slot_size = 0;
    double setup = 0, started = 0;
    RunMetrics metrics;
    std::vector<StepTrace> traces;

    Impl(const Shape& s, const Queries& q, const KVBlock& kv, const Mask& m,
         Partition p, Implementation impl, KernelOptions options, FetchOrder fetch,
         bool trace_enabled, MPI_Comm communicator)
        : shape(s), queries(q), local(kv), mask(m), partition(p), method(impl),
          kernel(std::move(options)), order(fetch), tracing(trace_enabled),
          state(q.pos.size(), s) {
        const double begin = MPI_Wtime();
        shape.validate(); mask.validate();
        int provided = 0, is_main = 0;
        mpi_check(MPI_Query_thread(&provided), "MPI_Query_thread");
        mpi_check(MPI_Is_thread_main(&is_main), "MPI_Is_thread_main");
        if (provided < MPI_THREAD_FUNNELED || !is_main)
            throw std::invalid_argument("DistributedAttention requires MPI_THREAD_FUNNELED and the MPI main thread");
        if (kernel.block_q <= 0 || kernel.block_k <= 0 || kernel.threads <= 0 || kernel.poll_tiles < 0)
            throw std::invalid_argument("invalid kernel options");
#ifndef _OPENMP
        if (kernel.threads != 1) throw std::invalid_argument("this build has no OpenMP support");
#endif
        if (kernel.progress) throw std::invalid_argument("DistributedAttention owns the MPI progress callback");
        mpi_check(MPI_Comm_dup(communicator, &comm), "MPI_Comm_dup");
        mpi_check(MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN), "MPI_Comm_set_errhandler");
        mpi_check(MPI_Comm_rank(comm, &rank), "MPI_Comm_rank");
        mpi_check(MPI_Comm_size(comm, &ranks), "MPI_Comm_size");
        if (queries.pos != positions(shape.nq, ranks, rank, partition) ||
            local.pos != positions(shape.n, ranks, rank, partition) ||
            queries.q.size() != tensor_count(queries.pos.size(), shape.heads, shape.d) ||
            local.k.size() != tensor_count(local.pos.size(), shape.kv_heads, shape.d) ||
            local.v.size() != tensor_count(local.pos.size(), shape.kv_heads, shape.dv))
            throw std::invalid_argument("local tensors do not match the expected shard");
        kcounts.resize(ranks); vcounts.resize(ranks);
        koffsets.resize(ranks); voffsets.resize(ranks);
        std::size_t ktotal = 0, vtotal = 0;
        for (int r = 0; r < ranks; ++r) {
            const auto rows = row_count(shape.n, ranks, r, partition);
            max_rows = std::max(max_rows, rows);
            kcounts[r] = count_int(tensor_count(rows, shape.kv_heads, shape.d));
            vcounts[r] = count_int(tensor_count(rows, shape.kv_heads, shape.dv));
            if (method == Implementation::allgather || method == Implementation::iallgather ||
                method == Implementation::local) {
                koffsets[r] = count_int(ktotal); voffsets[r] = count_int(vtotal);
            }
            ktotal += static_cast<std::size_t>(kcounts[r]);
            vtotal += static_cast<std::size_t>(vcounts[r]);
        }
        for (auto& b : buffer) {
            b.pos.reserve(max_rows);
            b.k.reserve(tensor_count(max_rows, shape.kv_heads, shape.d));
            b.v.reserve(tensor_count(max_rows, shape.kv_heads, shape.dv));
        }
        result.resize(tensor_count(q.pos.size(), shape.heads, shape.dv));
        if (tracing) traces.reserve(static_cast<std::size_t>(ranks));
        if (method == Implementation::allgather || method == Implementation::iallgather ||
            method == Implementation::local) {
            count_int(ktotal); count_int(vtotal);
            gathered_k.resize(ktotal); gathered_v.resize(vtotal);
        }
        if (method == Implementation::local) {
            // Control: pay for the communication here, outside the timed phase.
            mpi_check(MPI_Allgatherv(local.k.data(), kcounts[rank], MPI_FLOAT, gathered_k.data(),
                      kcounts.data(), koffsets.data(), MPI_FLOAT, comm), "MPI_Allgatherv K (setup)");
            mpi_check(MPI_Allgatherv(local.v.data(), vcounts[rank], MPI_FLOAT, gathered_v.data(),
                      vcounts.data(), voffsets.data(), MPI_FLOAT, comm), "MPI_Allgatherv V (setup)");
        }
        if (method == Implementation::rma_get || method == Implementation::rma_put) {
            slot_size = tensor_count(max_rows, shape.kv_heads, shape.d) + tensor_count(max_rows, shape.kv_heads, shape.dv);
            const auto floats = method == Implementation::rma_get ? local.k.size() + local.v.size() : checked_size(2, slot_size);
            const auto bytes = checked_size(floats, sizeof(float));
            if (bytes > static_cast<std::size_t>(std::numeric_limits<MPI_Aint>::max()))
                throw std::overflow_error("MPI window size overflow");
            mpi_check(MPI_Win_allocate(static_cast<MPI_Aint>(bytes), sizeof(float), MPI_INFO_NULL,
                                      comm, &window_data, &window), "MPI_Win_allocate");
            mpi_check(MPI_Win_set_errhandler(window, MPI_ERRORS_RETURN), "MPI_Win_set_errhandler");
            if (method == Implementation::rma_get) {
                mpi_check(MPI_Win_lock_all(MPI_MODE_NOCHECK, window), "MPI_Win_lock_all");
                passive_epoch = true;
                if (!local.k.empty()) std::copy(local.k.begin(), local.k.end(), window_data);
                if (!local.v.empty()) std::copy(local.v.begin(), local.v.end(), window_data + local.k.size());
                // Publish every owner's local stores before any remote read.
                mpi_check(MPI_Win_sync(window), "MPI_Win_sync");
                mpi_check(MPI_Barrier(comm), "MPI_Barrier publish");
            }
        }
        setup = MPI_Wtime() - begin;
    }

    ~Impl() {
        if (window != MPI_WIN_NULL) {
            if (passive_epoch) MPI_Win_unlock_all(window);
            MPI_Win_free(&window);
        }
        if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
    }

    void prepare(KVBlock& block, int owner) {
        block.pos = positions(shape.n, ranks, owner, partition);
        block.k.resize(static_cast<std::size_t>(kcounts[owner]));
        block.v.resize(static_cast<std::size_t>(vcounts[owner]));
    }

    double now() const { return MPI_Wtime() - started; }

    template<class Action> void waiting(Action action) {
        const double begin = MPI_Wtime(); action(); metrics.wait += MPI_Wtime() - begin;
    }
    template<class Action> void posting(Action action) {
        const double begin = MPI_Wtime(); action(); metrics.post += MPI_Wtime() - begin;
    }

    void compute(const KVBlock& block, MPI_Request* requests = nullptr, int count = 0,
                 StepTrace* step = nullptr) {
        auto options = kernel;
        double polling = 0;
        if (requests && kernel.poll_tiles > 0) {
            options.progress = [&] {
                const double begin = MPI_Wtime();
                int done = 0;
                mpi_check(MPI_Testall(count, requests, &done, MPI_STATUSES_IGNORE), "MPI_Testall");
                polling += MPI_Wtime() - begin;
                if (done && step && step->observed_complete == 0) step->observed_complete = now();
            };
        }
        if (step) step->compute_begin = now();
        const double begin = MPI_Wtime();
        metrics.useful_pairs += process_block(queries, block, shape, mask, options, state);
        metrics.compute += MPI_Wtime() - begin - polling;
        metrics.wait += polling;
        if (step) step->compute_end = now();
    }

    void finish_requests(MPI_Request* requests, int count, StepTrace* step) {
        if (step) step->wait_begin = now();
        waiting([&] { mpi_check(MPI_Waitall(count, requests, MPI_STATUSES_IGNORE), "MPI_Waitall"); });
        if (step) {
            step->wait_end = now();
            if (step->observed_complete == 0) step->observed_complete = step->wait_end;
        }
    }

    void allgather(bool asynchronous) {
        std::array<MPI_Request, 2> requests{MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        StepTrace first{0, rank};
        first.post_begin = now();
        if (asynchronous) {
            posting([&] {
                mpi_check(MPI_Iallgatherv(local.k.data(), kcounts[rank], MPI_FLOAT, gathered_k.data(),
                          kcounts.data(), koffsets.data(), MPI_FLOAT, comm, &requests[0]), "MPI_Iallgatherv K");
                mpi_check(MPI_Iallgatherv(local.v.data(), vcounts[rank], MPI_FLOAT, gathered_v.data(),
                          vcounts.data(), voffsets.data(), MPI_FLOAT, comm, &requests[1]), "MPI_Iallgatherv V");
            });
            first.post_end = now();
            // The receive buffers are inaccessible until both requests finish.
            compute(local, requests.data(), 2, &first);
            finish_requests(requests.data(), 2, &first);
        } else {
            waiting([&] {
                mpi_check(MPI_Allgatherv(local.k.data(), kcounts[rank], MPI_FLOAT, gathered_k.data(),
                          kcounts.data(), koffsets.data(), MPI_FLOAT, comm), "MPI_Allgatherv K");
                mpi_check(MPI_Allgatherv(local.v.data(), vcounts[rank], MPI_FLOAT, gathered_v.data(),
                          vcounts.data(), voffsets.data(), MPI_FLOAT, comm), "MPI_Allgatherv V");
            });
            first.post_end = now();
        }
        if (tracing) traces.push_back(first);
        metrics.bytes_recv = static_cast<std::uint64_t>(gathered_k.size() + gathered_v.size() - local.k.size() - local.v.size()) * sizeof(float);
        for (int owner = 0; owner < ranks; ++owner) {
            if (asynchronous && owner == rank) continue;
            prepare(buffer[0], owner);
            std::copy_n(gathered_k.data() + koffsets[owner], kcounts[owner], buffer[0].k.data());
            std::copy_n(gathered_v.data() + voffsets[owner], vcounts[owner], buffer[0].v.data());
            compute(buffer[0]);
        }
    }

    // Same owner order as the ring (own shard first), no MPI in the timed phase.
    void local_only() {
        for (int step = 0; step < ranks; ++step) {
            const int owner = (rank - step + ranks) % ranks;
            StepTrace trace_step{step, owner};
            prepare(buffer[0], owner);
            std::copy_n(gathered_k.data() + koffsets[owner], kcounts[owner], buffer[0].k.data());
            std::copy_n(gathered_v.data() + voffsets[owner], vcounts[owner], buffer[0].v.data());
            compute(buffer[0], nullptr, 0, &trace_step);
            if (tracing) traces.push_back(trace_step);
        }
    }

    void ring(bool asynchronous) {
        buffer[0] = local;
        int current = 0;
        const int left = (rank + ranks - 1) % ranks, right = (rank + 1) % ranks;
        for (int step_index = 0; step_index < ranks; ++step_index) {
            const int owner = (rank - step_index + ranks) % ranks;
            StepTrace step{step_index, owner};
            if (step_index == ranks - 1) {
                compute(buffer[current], nullptr, 0, &step);
            } else {
                const int next = 1 - current, next_owner = (owner + ranks - 1) % ranks;
                prepare(buffer[next], next_owner);
                if (asynchronous) {
                    std::array<MPI_Request, 4> requests{};
                    step.post_begin = now();
                    posting([&] {
                        mpi_check(MPI_Irecv(buffer[next].k.data(), kcounts[next_owner], MPI_FLOAT, left, 0, comm, &requests[0]), "MPI_Irecv K");
                        mpi_check(MPI_Irecv(buffer[next].v.data(), vcounts[next_owner], MPI_FLOAT, left, 1, comm, &requests[1]), "MPI_Irecv V");
                        mpi_check(MPI_Isend(buffer[current].k.data(), kcounts[owner], MPI_FLOAT, right, 0, comm, &requests[2]), "MPI_Isend K");
                        mpi_check(MPI_Isend(buffer[current].v.data(), vcounts[owner], MPI_FLOAT, right, 1, comm, &requests[3]), "MPI_Isend V");
                    });
                    step.post_end = now();
                    // Reading the send buffer is legal; do not resize/reuse it
                    // until sends complete. Receive goes to the other slot.
                    compute(buffer[current], requests.data(), 4, &step);
                    finish_requests(requests.data(), 4, &step);
                } else {
                    compute(buffer[current], nullptr, 0, &step);
                    step.wait_begin = now();
                    waiting([&] {
                        mpi_check(MPI_Sendrecv(buffer[current].k.data(), kcounts[owner], MPI_FLOAT, right, 0,
                                  buffer[next].k.data(), kcounts[next_owner], MPI_FLOAT, left, 0, comm, MPI_STATUS_IGNORE), "MPI_Sendrecv K");
                        mpi_check(MPI_Sendrecv(buffer[current].v.data(), vcounts[owner], MPI_FLOAT, right, 1,
                                  buffer[next].v.data(), vcounts[next_owner], MPI_FLOAT, left, 1, comm, MPI_STATUS_IGNORE), "MPI_Sendrecv V");
                    });
                    step.wait_end = now();
                    step.observed_complete = step.wait_end;
                }
                metrics.bytes_recv += static_cast<std::uint64_t>(kcounts[next_owner] + static_cast<std::uint64_t>(vcounts[next_owner])) * sizeof(float);
                current = next;
            }
            if (tracing) traces.push_back(step);
        }
    }

    void rma_get() {
        std::vector<int> owners;
        owners.reserve(ranks);
        for (int step = 0; step < ranks; ++step) {
            int owner = (rank - step + ranks) % ranks;
            if (order == FetchOrder::ascending) owner = step;
            if (order == FetchOrder::reverse) owner = (rank + step) % ranks;
            const auto pos = positions(shape.n, ranks, owner, partition);
            if (any_allowed(queries, pos, mask)) owners.push_back(owner);
        }
        if (owners.empty()) return;
        auto fetch = [&](int owner, KVBlock& target, std::array<MPI_Request, 2>& requests) {
            prepare(target, owner);
            requests = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
            if (owner == rank) { target = local; return; }
            posting([&] {
                mpi_check(MPI_Rget(target.k.data(), kcounts[owner], MPI_FLOAT, owner, 0,
                                  kcounts[owner], MPI_FLOAT, window, &requests[0]), "MPI_Rget K");
                mpi_check(MPI_Rget(target.v.data(), vcounts[owner], MPI_FLOAT, owner, kcounts[owner],
                                  vcounts[owner], MPI_FLOAT, window, &requests[1]), "MPI_Rget V");
            });
            metrics.bytes_recv += static_cast<std::uint64_t>(kcounts[owner] + static_cast<std::uint64_t>(vcounts[owner])) * sizeof(float);
        };
        std::array<MPI_Request, 2> initial{};
        fetch(owners.front(), buffer[0], initial);
        finish_requests(initial.data(), 2, nullptr);
        int current = 0;
        for (std::size_t i = 0; i < owners.size(); ++i) {
            StepTrace step{static_cast<int>(i), owners[i]};
            if (i + 1 == owners.size()) {
                compute(buffer[current], nullptr, 0, &step);
            } else {
                const int next = 1 - current;
                std::array<MPI_Request, 2> requests{};
                step.post_begin = now();
                fetch(owners[i + 1], buffer[next], requests);
                step.post_end = now();
                compute(buffer[current], requests.data(), 2, &step);
                finish_requests(requests.data(), 2, &step);
                current = next;
            }
            if (tracing) traces.push_back(step);
        }
    }

    void rma_put() {
        buffer[0] = local;
        int current = 0;
        const int right = (rank + 1) % ranks;
        if (ranks > 1) waiting([&] { mpi_check(MPI_Win_fence(MPI_MODE_NOPRECEDE, window), "MPI_Win_fence open"); });
        for (int i = 0; i < ranks; ++i) {
            const int owner = (rank - i + ranks) % ranks;
            StepTrace step{i, owner};
            if (i == ranks - 1) {
                compute(buffer[current], nullptr, 0, &step);
            } else {
                const int next = 1 - current, next_owner = (owner + ranks - 1) % ranks;
                prepare(buffer[next], next_owner);
                const auto base = static_cast<MPI_Aint>(static_cast<std::size_t>(next) * slot_size);
                step.post_begin = now();
                posting([&] {
                    mpi_check(MPI_Put(buffer[current].k.data(), kcounts[owner], MPI_FLOAT, right,
                                      base, kcounts[owner], MPI_FLOAT, window), "MPI_Put K");
                    mpi_check(MPI_Put(buffer[current].v.data(), vcounts[owner], MPI_FLOAT, right,
                                      base + kcounts[owner], vcounts[owner], MPI_FLOAT, window), "MPI_Put V");
                });
                step.post_end = now();
                compute(buffer[current], nullptr, 0, &step);
                step.wait_begin = now();
                waiting([&] { mpi_check(MPI_Win_fence(i == ranks - 2 ? MPI_MODE_NOSUCCEED : 0, window), "MPI_Win_fence complete"); });
                step.wait_end = now(); step.observed_complete = step.wait_end;
                const float* incoming = window_data + static_cast<std::size_t>(next) * slot_size;
                std::copy_n(incoming, kcounts[next_owner], buffer[next].k.data());
                std::copy_n(incoming + kcounts[next_owner], vcounts[next_owner], buffer[next].v.data());
                metrics.bytes_recv += static_cast<std::uint64_t>(kcounts[next_owner] + static_cast<std::uint64_t>(vcounts[next_owner])) * sizeof(float);
                current = next;
            }
            if (tracing) traces.push_back(step);
        }
    }

    RunMetrics run() {
        mpi_check(MPI_Barrier(comm), "MPI_Barrier before timing");
        metrics = {};
        traces.clear();
        started = MPI_Wtime();
        state.reset();
        switch (method) {
        case Implementation::allgather: allgather(false); break;
        case Implementation::iallgather: allgather(true); break;
        case Implementation::ring: ring(false); break;
        case Implementation::overlap: ring(true); break;
        case Implementation::rma_get: rma_get(); break;
        case Implementation::rma_put: rma_put(); break;
        case Implementation::local: local_only(); break;
        }
        const double begin = MPI_Wtime();
        for (std::size_t i = 0; i < state.l.size(); ++i)
            for (int a = 0; a < shape.dv; ++a)
                result[i * shape.dv + a] = state.l[i] != 0 ? state.acc[i * shape.dv + a] / state.l[i] : 0.0f;
        metrics.compute += MPI_Wtime() - begin;
        metrics.total = MPI_Wtime() - started;
        metrics.other = std::max(0.0, metrics.total - metrics.compute - metrics.wait - metrics.post);
        return metrics;
    }
};

DistributedAttention::DistributedAttention(const Shape& s, const Queries& q, const KVBlock& kv,
        const Mask& mask, Partition partition, Implementation impl, KernelOptions kernel,
        FetchOrder order, bool trace, MPI_Comm comm)
    : impl_(std::make_unique<Impl>(s, q, kv, mask, partition, impl, std::move(kernel), order, trace, comm)) {}
DistributedAttention::~DistributedAttention() = default;
RunMetrics DistributedAttention::run() { return impl_->run(); }
const std::vector<float>& DistributedAttention::output() const { return impl_->result; }
const std::vector<StepTrace>& DistributedAttention::trace() const { return impl_->traces; }
double DistributedAttention::setup_seconds() const { return impl_->setup; }

} // namespace attn
