#include "attn/core.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
void verify(const std::vector<float>& actual, const std::vector<double>& expected,
            const std::string& name) {
    const auto err = attn::compare(actual, expected);
    std::cout << name << ": max_abs=" << err.max_abs << " max_rel=" << err.max_rel
              << " nonfinite=" << err.nonfinite << " failed=" << err.failed << '\n';
    require(err.failed == 0, name + " differs from independent fp64 reference");
}

void kernels() {
    const attn::Shape s{37, 19, 13, 7, 6, 2};
    auto kp = attn::positions(s.n, 1, 0, attn::Partition::contiguous);
    auto qp = attn::positions(s.nq, 1, 0, attn::Partition::contiguous);
    for (float alpha : {1.0f, 50.0f}) {
        const auto q = attn::make_queries(s, qp, 42, alpha);
        const auto kv = attn::make_kv(s, kp, 42);
        for (auto kind : {attn::MaskKind::none, attn::MaskKind::causal,
                          attn::MaskKind::window, attn::MaskKind::sink_window}) {
            const attn::Mask mask{kind, 5, 4};
            const auto expected = attn::reference64(q, kv, s, mask);
            std::uint64_t count = 0;
            for (auto i : qp) for (auto j : kp) if (mask.allows(i, j)) count += s.heads;
            require(attn::count_pairs(qp, kp, s, mask) == count, "closed-form pair count mismatch");
            for (auto kernel : {attn::Kernel::basic, attn::Kernel::tuned}) {
                const std::string tag = kernel == attn::Kernel::basic ? " basic" : " tuned";
                for (int bk : {1, 8, 37, 64}) {
                    attn::OnlineState st(qp.size(), s);
                    attn::KernelOptions opt{4, bk, 1, 0, {}, kernel};
                    const auto pairs = attn::process_block(q, kv, s, mask, opt, st);
                    require(pairs == count, "useful work counter mismatch");
                    verify(attn::finalize(st, s), expected, "V1/V3/V4 bk=" + std::to_string(bk) + tag);
                }
                attn::OnlineState st(qp.size(), s);
                // Reversed owner order forces masked blocks before valid ones.
                std::uint64_t pairs = 0;
                for (int rank = 4; rank >= 0; --rank) {
                    const auto block = attn::make_kv(s, attn::positions(s.n, 5, rank, attn::Partition::zigzag), 42);
                    attn::KernelOptions opt{3, 7, 1, 2, {}, kernel};
#ifdef _OPENMP
                    opt.threads = 2;
#endif
                    pairs += attn::process_block(q, block, s, mask, opt, st);
                }
                require(pairs == count, "useful work counter mismatch across blocks");
                verify(attn::finalize(st, s), expected, "V8 reversed variable blocks" + tag);
            }
        }
    }
}

void boundaries() {
    for (auto partition : {attn::Partition::contiguous, attn::Partition::zigzag})
        for (int n : {1, 3, 37, 1000})
            for (int ranks : {1, 2, 3, 5, 7, 8}) {
                std::vector<std::int64_t> all;
                for (int r = 0; r < ranks; ++r) {
                    auto p = attn::positions(n, ranks, r, partition);
                    all.insert(all.end(), p.begin(), p.end());
                }
                std::sort(all.begin(), all.end());
                require(all.size() == static_cast<std::size_t>(n), "partition length");
                for (int i = 0; i < n; ++i) require(all[i] == i, "partition coverage");
            }
    const attn::Shape s{1000, 3, 9, 5, 1, 1};
    const auto q = attn::make_queries(s, {0, 1, 2}, 17, 1);
    const auto kv = attn::make_kv(s, attn::positions(1000, 1, 0, attn::Partition::contiguous), 17);
    attn::OnlineState st(q.pos.size(), s);
    attn::process_block(q, kv, s, {}, {2, 64, 1, 0, {}}, st);
    verify(attn::finalize(st, s), attn::reference64(q, kv, s, {}), "V2 N=1000 tail");
    st.reset();
    const auto future = attn::make_kv(s, {10, 11}, 17);
    attn::Mask mask{attn::MaskKind::sink_window, 3, 100};
    attn::process_block(q, future, s, mask, {}, st);
    const auto output = attn::finalize(st, s);
    require(std::all_of(output.begin(), output.end(), [](float x) { return x == 0; }), "fully masked row policy");
    require(!mask.allows(1, 2), "sink mask must remain causal");
    // Unsorted key positions take the per-pair fallback and must stay correct.
    const attn::Shape small{6, 6, 9, 5, 1, 1};
    const auto sq = attn::make_queries(small, {0, 1, 2, 3, 4, 5}, 17, 1);
    const auto shuffled = attn::make_kv(small, {4, 0, 5, 2, 1, 3}, 17);
    const auto ordered = attn::make_kv(small, {0, 1, 2, 3, 4, 5}, 17);
    const attn::Mask causal{attn::MaskKind::causal, 3, 1};
    attn::OnlineState sst(sq.pos.size(), small);
    require(attn::process_block(sq, shuffled, small, causal, {2, 4, 1, 0, {}}, sst) == 21, "unsorted pair count");
    verify(attn::finalize(sst, small), attn::reference64(sq, ordered, small, causal), "unsorted key fallback");
    require(attn::compare({std::nanf("")}, {0}).failed == 1, "NaN must fail validation");
    bool rejected = false;
    try { attn::Shape bad{1, 1, 4, 4, 3, 2}; bad.validate(); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid GQA must be rejected");
}

void gqa() {
    const attn::Shape grouped{31, 13, 8, 11, 14, 2};
    auto mha = grouped;
    mha.kv_heads = mha.heads;
    const auto qp = attn::positions(grouped.nq, 1, 0, attn::Partition::contiguous);
    const auto kp = attn::positions(grouped.n, 1, 0, attn::Partition::contiguous);
    const auto q = attn::make_queries(grouped, qp, 73, 1);
    const auto kv = attn::make_kv(grouped, kp, 73);
    attn::KVBlock repeated{kp, {}, {}};
    for (std::size_t i = 0; i < kp.size(); ++i)
        for (int h = 0; h < mha.heads; ++h) {
            const auto offset = i * grouped.kv_heads + h / 7;
            repeated.k.insert(repeated.k.end(), kv.k.begin() + offset * grouped.d, kv.k.begin() + (offset + 1) * grouped.d);
            repeated.v.insert(repeated.v.end(), kv.v.begin() + offset * grouped.dv, kv.v.begin() + (offset + 1) * grouped.dv);
        }
    attn::Mask mask{attn::MaskKind::causal, 8, 2};
    attn::OnlineState st(q.pos.size(), grouped);
    attn::process_block(q, kv, grouped, mask, {4, 7, 1, 0, {}}, st);
    verify(attn::finalize(st, grouped), attn::reference64(q, repeated, mha, mask), "V5 replicated GQA");
    const auto control = attn::replicate_kv_heads(kv, grouped);
    require(control.k == repeated.k && control.v == repeated.v, "replicate_kv_heads differs from manual replication");
    attn::OnlineState mha_state(q.pos.size(), mha);
    attn::process_block(q, control, mha, mask, {4, 7, 1, 0, {}}, mha_state);
    require(attn::finalize(mha_state, mha) == attn::finalize(st, grouped), "matched MHA control must be bit-identical to GQA");
}

void extreme_weight_tails() {
    // All inputs are finite. Raising tiny weights to exp(-87) can nevertheless
    // cause O(1) output errors because the weights multiply large V entries.
    const attn::Shape shape{2, 1, 1, 1, 1, 1};
    const attn::Queries q{{0}, {1.0f}};
    for (float gap : {87.0f, 88.0f, 100.0f, 104.0f, 200.0f}) {
        const attn::KVBlock kv{{0, 1}, {0.0f, -gap}, {0.0f, 1e38f}};
        const auto expected = attn::reference64(q, kv, shape, {});
        for (auto kernel : {attn::Kernel::basic, attn::Kernel::tuned})
            for (int bk : {1, 2}) {
                attn::OnlineState state(1, shape);
                attn::process_block(q, kv, shape, {}, {1, bk, 1, 0, {}, kernel}, state);
                verify(attn::finalize(state, shape), expected,
                       "extreme weight tail gap=" + std::to_string(gap) +
                       (kernel == attn::Kernel::basic ? " basic" : " tuned"));
            }
    }
    // Also exercise vector-width ranges and partial tails in the tuned path.
    const attn::Shape wide{19, 1, 1, 1, 1, 1};
    attn::KVBlock kv{attn::positions(19, 1, 0, attn::Partition::contiguous),
                     attn::Floats(19, -100.0f), attn::Floats(19, 1e38f)};
    kv.k[0] = 0.0f;
    kv.v[0] = 0.0f;
    attn::OnlineState state(1, wide);
    attn::process_block(q, kv, wide, {}, {1, 32, 1, 0, {}, attn::Kernel::tuned}, state);
    verify(attn::finalize(state, wide), attn::reference64(q, kv, wide, {}), "vector-width extreme weight tails");
}
} // namespace

int main() {
    try { kernels(); boundaries(); gqa(); extreme_weight_tails(); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    return 0;
}
