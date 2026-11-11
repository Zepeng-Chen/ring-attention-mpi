#pragma once

#include "attn/core.hpp"
#include <mpi.h>

#include <memory>

namespace attn {

// `local` is the zero-communication control: every K/V shard is gathered once
// during (untimed) construction, so the timed phase is the same per-rank
// computation as the other variants with no MPI call inside it.
enum class Implementation { allgather, iallgather, ring, overlap, rma_get, rma_put, local };
enum class FetchOrder { staggered, ascending, reverse };
const char* implementation_name(Implementation impl);
Implementation parse_implementation(const std::string& name);

struct RunMetrics {
    double total = 0, compute = 0, wait = 0, post = 0, other = 0;
    std::uint64_t bytes_recv = 0, useful_pairs = 0;
};
struct StepTrace {
    int step = 0, owner = 0;
    // Rank-relative seconds. The request lifetime is not a network-transfer interval.
    double post_begin = 0, post_end = 0, compute_begin = 0, compute_end = 0;
    double wait_begin = 0, wait_end = 0, observed_complete = 0;
};

// Collective construction/destruction on comm. q/local_kv must outlive this object.
// Data/shapes/options must be identical across ranks except for the expected shard.
// MPI must provide at least MPI_THREAD_FUNNELED; call all methods on its main thread.
class DistributedAttention {
public:
    DistributedAttention(const Shape& shape, const Queries& q, const KVBlock& local_kv,
                         const Mask& mask, Partition partition, Implementation impl,
                         KernelOptions kernel = {}, FetchOrder order = FetchOrder::staggered,
                         bool trace = false, MPI_Comm comm = MPI_COMM_WORLD);
    ~DistributedAttention();
    DistributedAttention(const DistributedAttention&) = delete;
    DistributedAttention& operator=(const DistributedAttention&) = delete;
    // Barrier is outside the measured interval. Each call repeats communication,
    // state reset and output finalisation; construction/window setup is excluded.
    RunMetrics run();
    const std::vector<float>& output() const;
    const std::vector<StepTrace>& trace() const;
    double setup_seconds() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void mpi_check(int status, const char* operation);

} // namespace attn
