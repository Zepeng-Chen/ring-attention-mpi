#include "attn/distributed.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Case {
    const char* name;
    attn::Shape shape;
    attn::Mask mask;
    attn::Partition partition;
    float alpha;
};
void require(bool passed, const char* message) {
    if (!passed) throw std::runtime_error(message);
}

void run_tests(int rank, int ranks) {
    const std::vector<Case> cases = {
        {"uneven_gqa", {37, 29, 13, 7, 6, 2}, {}, attn::Partition::contiguous, 1},
        {"causal_stress", {37, 37, 13, 7, 6, 2}, {attn::MaskKind::causal, 5, 4}, attn::Partition::contiguous, 50},
        {"zigzag_causal", {37, 37, 13, 7, 14, 2}, {attn::MaskKind::causal, 5, 4}, attn::Partition::zigzag, 1},
        {"window", {19, 23, 8, 11, 2, 1}, {attn::MaskKind::window, 3, 4}, attn::Partition::contiguous, 1},
        {"sink_window", {19, 23, 8, 11, 2, 1}, {attn::MaskKind::sink_window, 3, 4}, attn::Partition::zigzag, 1},
        {"empty_shards", {3, 2, 7, 5, 1, 1}, {attn::MaskKind::causal, 3, 1}, attn::Partition::zigzag, 1},
        {"N1000_tail", {1000, 17, 9, 5, 1, 1}, {}, attn::Partition::contiguous, 1}
    };
    const std::vector<attn::Implementation> methods = {
        attn::Implementation::allgather, attn::Implementation::iallgather,
        attn::Implementation::ring, attn::Implementation::overlap,
        attn::Implementation::rma_get, attn::Implementation::rma_put,
        attn::Implementation::local
    };
    for (const auto& test : cases) {
        const auto& shape = test.shape;
        const auto q = attn::make_queries(shape, attn::positions(shape.nq, ranks, rank, test.partition), 83, test.alpha);
        const auto local = attn::make_kv(shape, attn::positions(shape.n, ranks, rank, test.partition), 83);
        const auto full = attn::make_kv(shape, attn::positions(shape.n, 1, 0, attn::Partition::contiguous), 83);
        const auto expected = attn::reference64(q, full, shape, test.mask);
        std::uint64_t expected_pairs = 0;
        for (auto qp : q.pos) for (auto kp : full.pos)
            if (test.mask.allows(qp, kp)) expected_pairs += shape.heads;
        for (auto method : methods) {
            const std::vector<attn::FetchOrder> orders = method == attn::Implementation::rma_get
                ? std::vector<attn::FetchOrder>{attn::FetchOrder::staggered, attn::FetchOrder::ascending, attn::FetchOrder::reverse}
                : std::vector<attn::FetchOrder>{attn::FetchOrder::staggered};
            for (auto order : orders) {
                attn::KernelOptions options{4, 7, 1, 2, {}};
#ifdef _OPENMP
                options.threads = 2;
#endif
                attn::DistributedAttention solver(shape, q, local, test.mask, test.partition, method, options, order, true);
                std::vector<float> previous;
                double max_abs = 0;
                for (int rep = 0; rep < 3; ++rep) {
                    const auto stats = solver.run();
                    const auto errors = attn::compare(solver.output(), expected);
                    require(errors.failed == 0, "distributed output differs from fp64");
                    require(stats.useful_pairs == expected_pairs, "useful pair count mismatch");
                    require(std::isfinite(stats.total) && stats.total >= 0, "invalid runtime");
                    require(stats.compute + stats.wait + stats.post <= stats.total + 1e-8, "timing double counts intervals");
                    if (rep > 0) require(previous == solver.output(), "repeated run has stale state or buffer data");
                    previous = solver.output();
                    max_abs = std::max(max_abs, errors.max_abs);
                    if (method == attn::Implementation::local) {
                        require(stats.bytes_recv == 0 && stats.wait == 0, "the control must not communicate while timed");
                    } else if (method != attn::Implementation::rma_get || test.mask.kind == attn::MaskKind::none) {
                        const auto expected_bytes = (full.k.size() + full.v.size() - local.k.size() - local.v.size()) * sizeof(float);
                        // Empty query ranks correctly skip every RMA Get.
                        if (!(method == attn::Implementation::rma_get && q.pos.empty()))
                            require(stats.bytes_recv == expected_bytes, "logical payload mismatch");
                    }
                }
                double global_abs = 0;
                attn::mpi_check(MPI_Reduce(&max_abs, &global_abs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), "MPI_Reduce error");
                if (rank == 0)
                    std::cout << test.name << ',' << attn::implementation_name(method)
                              << ",P=" << ranks << ",order=" << static_cast<int>(order)
                              << ",max_abs=" << global_abs << ",repetitions=3\n";
            }
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        require(provided >= MPI_THREAD_FUNNELED, "MPI thread support insufficient");
        run_tests(rank, ranks);
    } catch (const std::exception& e) {
        std::cerr << "rank " << rank << ": " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return 0;
}
