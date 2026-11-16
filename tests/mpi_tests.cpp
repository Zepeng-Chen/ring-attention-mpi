#include "attn/distributed.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
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
    // The keys on both sides of every shard boundary carry nearly all the weight (see dominate()).
    bool boundary_keys = false;
};
void require(bool passed, const char* message) {
    if (!passed) throw std::runtime_error(message);
}

// The positions that begin or end a run of consecutive positions owned by one rank: the keys on both
// sides of every shard boundary, and the two ends of the sequence.
std::vector<char> shard_boundaries(std::int64_t n, int ranks, attn::Partition partition) {
    std::vector<char> boundary(static_cast<std::size_t>(n), 0);
    for (int r = 0; r < ranks; ++r) {
        const auto pos = attn::positions(n, ranks, r, partition);
        for (std::size_t i = 0; i < pos.size(); ++i)
            if (i == 0 || i + 1 == pos.size() || pos[i - 1] + 1 != pos[i] || pos[i] + 1 != pos[i + 1])
                boundary[static_cast<std::size_t>(pos[i])] = 1;
    }
    return boundary;
}
// Component 0 of every query becomes 6 and component 0 of every boundary key 4, where the synthetic
// values are below 1 in magnitude. With d = 13 a boundary key then scores about 6.7 from this component
// and any other key at most 1.7, so the keys at the shard boundaries take nearly all the weight of every
// row: a key or value that is off by one row at a shard boundary cannot hide behind a small weight.
void dominate(attn::Queries& q, const attn::Shape& s) {
    for (std::size_t i = 0; i < q.pos.size() * static_cast<std::size_t>(s.heads); ++i) q.q[i * s.d] = 6.0f;
}
void dominate(attn::KVBlock& kv, const attn::Shape& s, const std::vector<char>& boundary) {
    for (std::size_t i = 0; i < kv.pos.size(); ++i)
        if (boundary[static_cast<std::size_t>(kv.pos[i])])
            for (int h = 0; h < s.kv_heads; ++h) kv.k[(i * s.kv_heads + h) * s.d] = 4.0f;
}
// What the comparison with fp64 would see if one boundary key were wrong. For every boundary key the
// reference is evaluated again (a) without that key and (b) with its key and value rows replaced by those
// of the neighbouring position, the two typical results of an offset or a count that is off by one.
// Returns the smallest change found: over these cases and over the local rows that may see the key, the
// largest difference of an output component (infinity if this rank has no such row).
double boundary_sensitivity(const attn::Queries& q, const attn::KVBlock& full, const std::vector<double>& expected,
                            const attn::Shape& s, const attn::Mask& mask, const std::vector<char>& boundary) {
    double smallest = std::numeric_limits<double>::infinity();
    const auto row = static_cast<std::size_t>(s.heads) * s.dv;
    const auto kw = static_cast<std::ptrdiff_t>(s.kv_heads) * s.d, vw = static_cast<std::ptrdiff_t>(s.kv_heads) * s.dv;
    const auto count = static_cast<std::ptrdiff_t>(full.pos.size());
    for (std::ptrdiff_t b = 0; b < count && count > 1; ++b) {
        if (!boundary[static_cast<std::size_t>(full.pos[b])]) continue;
        attn::KVBlock without = full, shifted = full;
        without.pos.erase(without.pos.begin() + b);
        without.k.erase(without.k.begin() + b * kw, without.k.begin() + (b + 1) * kw);
        without.v.erase(without.v.begin() + b * vw, without.v.begin() + (b + 1) * vw);
        const std::ptrdiff_t other = b + 1 < count ? b + 1 : b - 1;
        std::copy(full.k.begin() + other * kw, full.k.begin() + (other + 1) * kw, shifted.k.begin() + b * kw);
        std::copy(full.v.begin() + other * vw, full.v.begin() + (other + 1) * vw, shifted.v.begin() + b * vw);
        for (const attn::KVBlock* wrong : {&without, &shifted}) {
            const auto changed = attn::reference64(q, *wrong, s, mask);
            for (std::size_t i = 0; i < q.pos.size(); ++i) {
                if (!mask.allows(q.pos[i], full.pos[b])) continue;
                double largest = 0;
                for (std::size_t a = 0; a < row; ++a)
                    largest = std::max(largest, std::abs(changed[i * row + a] - expected[i * row + a]));
                smallest = std::min(smallest, largest);
            }
        }
    }
    return smallest;
}

void run_tests(int rank, int ranks) {
    const std::vector<Case> cases = {
        {"uneven_gqa", {37, 29, 13, 7, 6, 2}, {}, attn::Partition::contiguous, 1},
        {"causal_stress", {37, 37, 13, 7, 6, 2}, {attn::MaskKind::causal, 5, 4}, attn::Partition::contiguous, 50},
        {"zigzag_causal", {37, 37, 13, 7, 14, 2}, {attn::MaskKind::causal, 5, 4}, attn::Partition::zigzag, 1},
        {"window", {19, 23, 8, 11, 2, 1}, {attn::MaskKind::window, 3, 4}, attn::Partition::contiguous, 1},
        {"sink_window", {19, 23, 8, 11, 2, 1}, {attn::MaskKind::sink_window, 3, 4}, attn::Partition::zigzag, 1},
        {"empty_shards", {3, 2, 7, 5, 1, 1}, {attn::MaskKind::causal, 3, 1}, attn::Partition::zigzag, 1},
        {"N1000_tail", {1000, 17, 9, 5, 1, 1}, {}, attn::Partition::contiguous, 1},
        {"boundary_keys", {61, 47, 13, 7, 6, 2}, {}, attn::Partition::contiguous, 1, true},
        {"boundary_keys_zigzag", {61, 47, 13, 7, 6, 2}, {}, attn::Partition::zigzag, 1, true},
        {"boundary_keys_causal", {61, 61, 13, 7, 6, 2}, {attn::MaskKind::causal, 5, 4}, attn::Partition::contiguous, 1, true},
        {"boundary_keys_causal_zigzag", {61, 61, 13, 7, 6, 2}, {attn::MaskKind::causal, 5, 4}, attn::Partition::zigzag, 1, true}
    };
    // A wrong boundary key must change the result by at least this much, a hundred times the tolerance.
    const double required_sensitivity = 1e-3;
    const std::vector<attn::Implementation> methods = {
        attn::Implementation::allgather, attn::Implementation::iallgather,
        attn::Implementation::ring, attn::Implementation::overlap,
        attn::Implementation::rma_get, attn::Implementation::rma_put,
        attn::Implementation::local
    };
    for (const auto& test : cases) {
        const auto& shape = test.shape;
        auto q = attn::make_queries(shape, attn::positions(shape.nq, ranks, rank, test.partition), 83, test.alpha);
        auto local = attn::make_kv(shape, attn::positions(shape.n, ranks, rank, test.partition), 83);
        auto full = attn::make_kv(shape, attn::positions(shape.n, 1, 0, attn::Partition::contiguous), 83);
        const auto boundary = shard_boundaries(shape.n, ranks, test.partition);
        if (test.boundary_keys) {
            dominate(q, shape); dominate(local, shape, boundary); dominate(full, shape, boundary);
        }
        const auto expected = attn::reference64(q, full, shape, test.mask);
        if (test.boundary_keys) {
            // The test is only worth something if a wrong boundary key would be seen: check that first.
            const double mine = boundary_sensitivity(q, full, expected, shape, test.mask, boundary);
            double smallest = 0;
            attn::mpi_check(MPI_Allreduce(&mine, &smallest, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD), "MPI_Allreduce error");
            if (rank == 0)
                std::cout << test.name << ",sensitivity,P=" << ranks << ",smallest_change=" << smallest
                          << ",required=" << required_sensitivity << '\n';
            require(smallest >= required_sensitivity, "a wrong key at a shard boundary would change the result too little");
        }
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
