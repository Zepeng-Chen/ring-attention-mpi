#include "attn/core.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#if defined(__GNUC__) || defined(__clang__)
#define ATTN_RESTRICT __restrict__
#else
#define ATTN_RESTRICT
#endif

namespace attn {
namespace {

// ---------------------------------------------------------------------------
// Tuned kernel building blocks. Every linear-algebra loop is hand-written.
// Ordinary softmax weights use the vectorisable exponential below; online-state
// rescaling and ranges containing extreme tails use std::exp.
// The loops are written so that a compiler can vectorise them without being
// allowed to reassociate floating-point sums: independent partial sums are
// spelled out explicitly instead of relying on -ffast-math.
// ---------------------------------------------------------------------------

struct Range { std::size_t lo, hi; };

// Allowed keys of one query inside a block of ascending key positions, as at
// most two disjoint index ranges [lo, hi). Returns the number of ranges.
inline int allowed_ranges(const Mask& mask, std::int64_t qp, const std::int64_t* kp,
                          std::size_t n, Range out[2]) {
    if (mask.kind == MaskKind::none) { out[0] = {0, n}; return n ? 1 : 0; }
    const auto hi = static_cast<std::size_t>(std::upper_bound(kp, kp + n, qp) - kp);
    if (mask.kind == MaskKind::causal) { out[0] = {0, hi}; return hi ? 1 : 0; }
    // Window keys satisfy qp - k < window, i.e. k > qp - window.
    const auto lo = static_cast<std::size_t>(std::upper_bound(kp, kp + hi, qp - mask.window) - kp);
    if (mask.kind == MaskKind::window) { out[0] = {lo, hi}; return lo < hi ? 1 : 0; }
    // Sink keys k < sink are also allowed (still subject to k <= qp).
    const auto sink = static_cast<std::size_t>(std::lower_bound(kp, kp + hi, mask.sink) - kp);
    if (sink >= lo) { out[0] = {0, hi}; return hi ? 1 : 0; }
    int count = 0;
    if (sink > 0) out[count++] = {0, sink};
    if (lo < hi) out[count++] = {lo, hi};
    return count;
}

// out[j] = scale * sum_a q[a] * kt[a*bk + j] for j in [0, n). kt is one packed
// key tile stored dimension-major, so each of the eight streams below is a
// contiguous row and the j loop is a plain multiply-add over keys. Eight
// query components are folded per pass so that out[] is loaded and stored once
// per eight multiply-adds; this keeps the loop limited by multiply-add
// throughput and not by store-to-load forwarding.
inline void scores_row(float* ATTN_RESTRICT out, const float* ATTN_RESTRICT q,
                       const float* ATTN_RESTRICT kt, std::size_t bk, std::size_t n,
                       int d, float scale) {
    for (std::size_t j = 0; j < n; ++j) out[j] = 0.0f;
    int a = 0;
    for (; a + 8 <= d; a += 8) {
        const float q0 = q[a], q1 = q[a + 1], q2 = q[a + 2], q3 = q[a + 3];
        const float q4 = q[a + 4], q5 = q[a + 5], q6 = q[a + 6], q7 = q[a + 7];
        const float* ATTN_RESTRICT k0 = kt + static_cast<std::size_t>(a) * bk;
        const float* ATTN_RESTRICT k1 = k0 + bk;
        const float* ATTN_RESTRICT k2 = k1 + bk;
        const float* ATTN_RESTRICT k3 = k2 + bk;
        const float* ATTN_RESTRICT k4 = k3 + bk;
        const float* ATTN_RESTRICT k5 = k4 + bk;
        const float* ATTN_RESTRICT k6 = k5 + bk;
        const float* ATTN_RESTRICT k7 = k6 + bk;
        for (std::size_t j = 0; j < n; ++j)
            out[j] = (((((((out[j] + q0 * k0[j]) + q1 * k1[j]) + q2 * k2[j]) + q3 * k3[j])
                        + q4 * k4[j]) + q5 * k5[j]) + q6 * k6[j]) + q7 * k7[j];
    }
    for (; a < d; ++a) {
        const float qa = q[a];
        const float* ATTN_RESTRICT ka = kt + static_cast<std::size_t>(a) * bk;
        for (std::size_t j = 0; j < n; ++j) out[j] += qa * ka[j];
    }
    for (std::size_t j = 0; j < n; ++j) out[j] *= scale;
}

// Maximum of x[0..n) with sixteen explicit lanes (an elementwise select
// vectorises; a single running maximum would need relaxed NaN semantics).
inline float max_lanes(const float* ATTN_RESTRICT x, std::size_t n) {
    float lane[16];
    for (int t = 0; t < 16; ++t) lane[t] = -std::numeric_limits<float>::infinity();
    std::size_t j = 0;
    for (; j + 16 <= n; j += 16)
        for (int t = 0; t < 16; ++t) lane[t] = x[j + t] > lane[t] ? x[j + t] : lane[t];
    float result = -std::numeric_limits<float>::infinity();
    for (; j < n; ++j) result = x[j] > result ? x[j] : result;
    for (int t = 0; t < 16; ++t) result = lane[t] > result ? lane[t] : result;
    return result;
}

inline float sum_lanes(const float* ATTN_RESTRICT x, std::size_t n) {
    float lane[16] = {0};
    std::size_t j = 0;
    for (; j + 16 <= n; j += 16)
        for (int t = 0; t < 16; ++t) lane[t] += x[j + t];
    float tail = 0;
    for (; j < n; ++j) tail += x[j];
    for (int t = 0; t < 8; ++t) lane[t] += lane[t + 8];
    for (int t = 0; t < 4; ++t) lane[t] += lane[t + 4];
    return ((lane[0] + lane[2]) + (lane[1] + lane[3])) + tail;
}

// x[j] <- exp(x[j] - shift) for arguments that are <= 0 after the shift (the
// shift is the running row maximum). Range reduction x = k ln2 + r with
// |r| <= ln2/2, a degree-6 polynomial for exp(r) (Cephes expf coefficients) and
// an exact 2^k built from the exponent bits. Relative error about 1e-7 on
// [-87, 0]. A range containing a smaller argument takes the scalar std::exp
// path, preserving subnormal/underflow behaviour instead of raising its weight
// to exp(-87). Such tiny weights can still matter when V has a large magnitude.
// True when every x[j] - shift is at least -87 (false for NaN/infinite values).
// Sixteen explicit integer lanes keep this range test vectorisable; an
// early-exit loop would not be.
inline bool poly_range(const float* ATTN_RESTRICT x, std::size_t n, float shift) {
    unsigned lane[16];
    for (int t = 0; t < 16; ++t) lane[t] = 1u;
    std::size_t j = 0;
    for (; j + 16 <= n; j += 16)
        for (int t = 0; t < 16; ++t) lane[t] &= static_cast<unsigned>((x[j + t] - shift) >= -87.0f);
    unsigned ok = 1u;
    for (; j < n; ++j) ok &= static_cast<unsigned>((x[j] - shift) >= -87.0f);
    for (int t = 0; t < 16; ++t) ok &= lane[t];
    return ok != 0u;
}

inline void exp_shifted(float* ATTN_RESTRICT x, std::size_t n, float shift) {
    // Keep the ordinary polynomial loop separate, so its vectorisation does
    // not depend on whether the compiler can vectorise the libm fallback.
    if (!poly_range(x, n, shift)) {
        for (std::size_t k = 0; k < n; ++k) x[k] = std::exp(x[k] - shift);
        return;
    }
    for (std::size_t j = 0; j < n; ++j) {
        float v = x[j] - shift;
        v = v > 0.0f ? 0.0f : v;
        // v <= 0, so truncation of (t - 0.5) rounds t to the nearest integer.
        const std::int32_t k = static_cast<std::int32_t>(v * 1.44269504088896341f - 0.5f);
        const float kf = static_cast<float>(k);
        float r = v - kf * 0.693359375f;
        r = r - kf * -2.12194440e-4f;
        float p = 1.9875691500e-4f;
        p = p * r + 1.3981999507e-3f;
        p = p * r + 8.3334519073e-3f;
        p = p * r + 4.1665795894e-2f;
        p = p * r + 1.6666665459e-1f;
        p = p * r + 5.0000001201e-1f;
        p = p * (r * r) + r + 1.0f;
        const std::uint32_t bits = static_cast<std::uint32_t>(k + 127) << 23;
        float power;
        std::memcpy(&power, &bits, sizeof power);
        x[j] = p * power;
    }
}

// acc[a] += sum_j w[j] * v[j*stride + a], eight keys per pass (see scores_row).
inline void weighted_sum(float* ATTN_RESTRICT acc, const float* ATTN_RESTRICT w,
                         const float* ATTN_RESTRICT v, std::size_t stride,
                         std::size_t n, int dv) {
    std::size_t j = 0;
    for (; j + 8 <= n; j += 8) {
        const float w0 = w[j], w1 = w[j + 1], w2 = w[j + 2], w3 = w[j + 3];
        const float w4 = w[j + 4], w5 = w[j + 5], w6 = w[j + 6], w7 = w[j + 7];
        const float* ATTN_RESTRICT v0 = v + j * stride;
        const float* ATTN_RESTRICT v1 = v0 + stride;
        const float* ATTN_RESTRICT v2 = v1 + stride;
        const float* ATTN_RESTRICT v3 = v2 + stride;
        const float* ATTN_RESTRICT v4 = v3 + stride;
        const float* ATTN_RESTRICT v5 = v4 + stride;
        const float* ATTN_RESTRICT v6 = v5 + stride;
        const float* ATTN_RESTRICT v7 = v6 + stride;
        for (int a = 0; a < dv; ++a)
            acc[a] = (((((((acc[a] + w0 * v0[a]) + w1 * v1[a]) + w2 * v2[a]) + w3 * v3[a])
                        + w4 * v4[a]) + w5 * v5[a]) + w6 * v6[a]) + w7 * v7[a];
    }
    for (; j < n; ++j) {
        const float wj = w[j];
        const float* ATTN_RESTRICT vj = v + j * stride;
        for (int a = 0; a < dv; ++a) acc[a] += wj * vj[a];
    }
}

std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

std::size_t elements(std::size_t rows, int heads, int dim) {
    return checked_size(checked_size(rows, static_cast<std::size_t>(heads)),
                        static_cast<std::size_t>(dim));
}

void check(const Queries& q, const KVBlock& kv, const Shape& s) {
    s.validate();
    if (q.q.size() != elements(q.pos.size(), s.heads, s.d) ||
        kv.k.size() != elements(kv.pos.size(), s.kv_heads, s.d) ||
        kv.v.size() != elements(kv.pos.size(), s.kv_heads, s.dv))
        throw std::invalid_argument("tensor dimensions disagree with shape/positions");
}

std::vector<float> read_rows(const std::string& path,
                            const std::vector<std::int64_t>& pos,
                            std::int64_t total, int heads, int dim) {
    const std::uint32_t endian = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian) != 1)
        throw std::runtime_error("raw QKV reader requires a little-endian host");
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + path);
    const auto stride = elements(1, heads, dim);
    const auto bytes = checked_size(elements(static_cast<std::size_t>(total), heads, dim), sizeof(float));
    if (in.tellg() < 0 || static_cast<std::uint64_t>(in.tellg()) != bytes)
        throw std::runtime_error("wrong tensor file length: " + path);
    std::vector<float> out(checked_size(pos.size(), stride));
    for (std::size_t i = 0; i < pos.size();) {
        if (pos[i] < 0 || pos[i] >= total) throw std::invalid_argument("file row out of bounds");
        std::size_t end = i + 1;
        while (end < pos.size() && pos[end] == pos[end - 1] + 1) ++end;
        if (pos[end - 1] >= total) throw std::invalid_argument("file row out of bounds");
        in.seekg(static_cast<std::streamoff>(checked_size(static_cast<std::size_t>(pos[i]), stride * sizeof(float))));
        in.read(reinterpret_cast<char*>(out.data() + i * stride),
                static_cast<std::streamsize>((end - i) * stride * sizeof(float)));
        if (!in) throw std::runtime_error("short read: " + path);
        i = end;
    }
    for (float x : out)
        if (!std::isfinite(x)) throw std::runtime_error("non-finite input: " + path);
    return out;
}

template<class Real>
std::vector<Real> reference(const Queries& q, const KVBlock& kv,
                            const Shape& s, const Mask& mask) {
    check(q, kv, s);
    mask.validate();
    std::vector<Real> output(elements(q.pos.size(), s.heads, s.dv), Real(0));
    std::vector<Real> scores(kv.pos.size());
    const Real scale = Real(1) / std::sqrt(static_cast<Real>(s.d));
    for (std::size_t i = 0; i < q.pos.size(); ++i) {
        for (int h = 0; h < s.heads; ++h) {
            const int kh = h / (s.heads / s.kv_heads);
            const float* qr = q.q.data() + (i * s.heads + h) * s.d;
            Real maximum = -std::numeric_limits<Real>::infinity();
            for (std::size_t j = 0; j < kv.pos.size(); ++j) {
                scores[j] = -std::numeric_limits<Real>::infinity();
                if (!mask.allows(q.pos[i], kv.pos[j])) continue;
                const float* kr = kv.k.data() + (j * s.kv_heads + kh) * s.d;
                Real dot = 0;
                for (int a = 0; a < s.d; ++a) dot += static_cast<Real>(qr[a]) * static_cast<Real>(kr[a]);
                scores[j] = dot * scale;
                maximum = std::max(maximum, scores[j]);
            }
            if (maximum == -std::numeric_limits<Real>::infinity()) continue;
            Real denom = 0;
            Real* result = output.data() + (i * s.heads + h) * s.dv;
            for (std::size_t j = 0; j < kv.pos.size(); ++j) {
                if (!mask.allows(q.pos[i], kv.pos[j])) continue;
                const Real weight = std::exp(scores[j] - maximum);
                denom += weight;
                const float* vr = kv.v.data() + (j * s.kv_heads + kh) * s.dv;
                for (int a = 0; a < s.dv; ++a) result[a] += weight * static_cast<Real>(vr[a]);
            }
            for (int a = 0; a < s.dv; ++a) result[a] /= denom;
        }
    }
    return output;
}

} // namespace

std::size_t checked_size(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("tensor size overflow");
    return a * b;
}

void Shape::validate() const {
    if (n <= 0 || nq <= 0 || d <= 0 || dv <= 0 || heads <= 0 ||
        kv_heads <= 0 || heads % kv_heads != 0)
        throw std::invalid_argument("require positive sizes and H divisible by Hkv");
}

void Mask::validate() const {
    if (window <= 0 || sink < 0)
        throw std::invalid_argument("window must be positive and sink nonnegative");
}

bool Mask::allows(std::int64_t q, std::int64_t k) const {
    if (kind == MaskKind::none) return true;
    if (k > q) return false;
    if (kind == MaskKind::causal) return true;
    return q - k < window || (kind == MaskKind::sink_window && k < sink);
}

std::vector<std::int64_t> positions(std::int64_t n, int ranks, int rank,
                                    Partition partition) {
    if (n < 0 || ranks <= 0 || rank < 0 || rank >= ranks || ranks > std::numeric_limits<int>::max() / 2)
        throw std::invalid_argument("invalid partition arguments");
    std::vector<std::int64_t> result;
    const int chunks = partition == Partition::zigzag ? 2 * ranks : ranks;
    auto append = [&](int chunk) {
        const auto base = n / chunks;
        const auto remainder = n % chunks;
        const auto start = base * chunk + std::min<std::int64_t>(chunk, remainder);
        const auto count = base + (chunk < remainder ? 1 : 0);
        for (std::int64_t i = 0; i < count; ++i) result.push_back(start + i);
    };
    append(rank);
    if (partition == Partition::zigzag) append(chunks - 1 - rank);
    return result;
}

float synthetic_value(std::uint64_t seed, int tensor, int head,
                      std::int64_t row, int column) {
    auto x = mix(seed ^ mix(static_cast<std::uint64_t>(tensor)));
    x = mix(x ^ mix(static_cast<std::uint64_t>(head)));
    x = mix(x ^ mix(static_cast<std::uint64_t>(row)));
    x = mix(x ^ mix(static_cast<std::uint64_t>(column)));
    return static_cast<float>(x >> 40) * (2.0f / 16777216.0f) - 1.0f;
}

Queries make_queries(const Shape& s, std::vector<std::int64_t> pos,
                     std::uint64_t seed, float alpha) {
    s.validate();
    if (!std::isfinite(alpha) || alpha <= 0) throw std::invalid_argument("alpha must be finite and positive");
    Queries q{std::move(pos), {}};
    q.q.resize(elements(q.pos.size(), s.heads, s.d));
    for (std::size_t i = 0; i < q.pos.size(); ++i)
        for (int h = 0; h < s.heads; ++h)
            for (int a = 0; a < s.d; ++a)
                q.q[(i * s.heads + h) * s.d + a] = alpha * synthetic_value(seed, 0, h, q.pos[i], a);
    return q;
}

KVBlock make_kv(const Shape& s, std::vector<std::int64_t> pos,
                std::uint64_t seed, bool replicate_kv) {
    s.validate();
    KVBlock kv{std::move(pos), {}, {}};
    kv.k.resize(elements(kv.pos.size(), s.kv_heads, s.d));
    kv.v.resize(elements(kv.pos.size(), s.kv_heads, s.dv));
    for (std::size_t i = 0; i < kv.pos.size(); ++i)
        for (int h = 0; h < s.kv_heads; ++h) {
            // A one-KV-head matched MHA control, independent of rank count.
            const int source_head = replicate_kv ? 0 : h;
            for (int a = 0; a < s.d; ++a)
                kv.k[(i * s.kv_heads + h) * s.d + a] = synthetic_value(seed, 1, source_head, kv.pos[i], a);
            for (int a = 0; a < s.dv; ++a)
                kv.v[(i * s.kv_heads + h) * s.dv + a] = synthetic_value(seed, 2, source_head, kv.pos[i], a);
        }
    return kv;
}

Queries read_queries(const Shape& s, std::vector<std::int64_t> pos, const std::string& dir) {
    s.validate();
    Queries q{std::move(pos), {}};
    const auto rows = read_rows(dir + "/q.bin", q.pos, s.nq, s.heads, s.d);
    q.q.assign(rows.begin(), rows.end());
    return q;
}

KVBlock read_kv(const Shape& s, std::vector<std::int64_t> pos, const std::string& dir) {
    s.validate();
    KVBlock kv{std::move(pos), {}, {}};
    const auto keys = read_rows(dir + "/k.bin", kv.pos, s.n, s.kv_heads, s.d);
    const auto values = read_rows(dir + "/v.bin", kv.pos, s.n, s.kv_heads, s.dv);
    kv.k.assign(keys.begin(), keys.end());
    kv.v.assign(values.begin(), values.end());
    return kv;
}

std::vector<float> read_tensor(const std::string& path, const std::vector<std::int64_t>& pos,
                               std::int64_t total, int heads, int dim) {
    if (total <= 0 || heads <= 0 || dim <= 0) throw std::invalid_argument("invalid tensor file shape");
    return read_rows(path, pos, total, heads, dim);
}

KVBlock replicate_kv_heads(const KVBlock& block, const Shape& s) {
    s.validate();
    const int group = s.heads / s.kv_heads;
    const std::size_t rows = block.pos.size();
    if (block.k.size() != elements(rows, s.kv_heads, s.d) || block.v.size() != elements(rows, s.kv_heads, s.dv))
        throw std::invalid_argument("K/V block disagrees with the grouped shape");
    KVBlock out{block.pos, {}, {}};
    out.k.resize(elements(rows, s.heads, s.d));
    out.v.resize(elements(rows, s.heads, s.dv));
    for (std::size_t i = 0; i < rows; ++i)
        for (int h = 0; h < s.heads; ++h) {
            const std::size_t from = i * s.kv_heads + static_cast<std::size_t>(h / group);
            const std::size_t to = i * s.heads + static_cast<std::size_t>(h);
            std::copy_n(block.k.begin() + static_cast<std::ptrdiff_t>(from * s.d), s.d,
                        out.k.begin() + static_cast<std::ptrdiff_t>(to * s.d));
            std::copy_n(block.v.begin() + static_cast<std::ptrdiff_t>(from * s.dv), s.dv,
                        out.v.begin() + static_cast<std::ptrdiff_t>(to * s.dv));
        }
    return out;
}

OnlineState::OnlineState(std::size_t rows, const Shape& s)
{
    s.validate();
    m.resize(elements(rows, s.heads, 1));
    l.resize(m.size());
    acc.resize(elements(rows, s.heads, s.dv));
    reset();
}

void OnlineState::reset() {
    std::fill(m.begin(), m.end(), -std::numeric_limits<float>::infinity());
    std::fill(l.begin(), l.end(), 0.0f);
    std::fill(acc.begin(), acc.end(), 0.0f);
}

namespace {

// Per-thread scratch, kept across calls so the hot path never allocates.
thread_local Floats tl_scores;
thread_local std::vector<Range> tl_ranges;
thread_local std::vector<int> tl_range_count;
// Owned by the calling (MPI main) thread; worker threads only read it.
thread_local Floats tl_packed;

// First blocked version: one query row at a time, the mask is tested for every
// pair and std::exp is called per pair. Kept as the "basic" rung of the serial
// optimisation ladder and as the fallback for unsorted key positions.
std::uint64_t basic_job(const Queries& q, const KVBlock& kv, const Shape& s, const Mask& mask,
                        const KernelOptions& opt, OnlineState& st, std::int64_t job,
                        std::int64_t tiles, float scale) {
    const int h = static_cast<int>(job / tiles);
    const int kh = h / (s.heads / s.kv_heads);
    const auto ib = static_cast<std::size_t>(job % tiles) * opt.block_q;
    const auto ie = std::min(ib + opt.block_q, q.pos.size());
    auto& scores = tl_scores;
    const auto need = checked_size(static_cast<std::size_t>(opt.block_q), static_cast<std::size_t>(opt.block_k));
    if (scores.size() < need) scores.resize(need);
    std::uint64_t pairs = 0;
    for (std::size_t jb = 0; jb < kv.pos.size(); jb += opt.block_k) {
        const auto je = std::min(jb + opt.block_k, kv.pos.size());
        for (std::size_t i = ib; i < ie; ++i) {
            float maximum = -std::numeric_limits<float>::infinity();
            float* row = scores.data() + (i - ib) * opt.block_k;
            const float* qr = q.q.data() + (i * s.heads + h) * s.d;
            for (std::size_t j = jb; j < je; ++j) {
                row[j - jb] = -std::numeric_limits<float>::infinity();
                if (!mask.allows(q.pos[i], kv.pos[j])) continue;
                const float* kr = kv.k.data() + (j * s.kv_heads + kh) * s.d;
                float dot = 0;
                for (int a = 0; a < s.d; ++a) dot += qr[a] * kr[a];
                row[j - jb] = dot * scale;
                maximum = std::max(maximum, row[j - jb]);
                ++pairs;
            }
            if (maximum == -std::numeric_limits<float>::infinity()) continue;
            const auto index = i * s.heads + h;
            maximum = std::max(maximum, st.m[index]);
            const float rescale = st.l[index] == 0 ? 0.0f : std::exp(st.m[index] - maximum);
            st.l[index] *= rescale;
            float* acc = st.acc.data() + index * s.dv;
            for (int a = 0; a < s.dv; ++a) acc[a] *= rescale;
            for (std::size_t j = jb; j < je; ++j) {
                if (!mask.allows(q.pos[i], kv.pos[j])) continue;
                const float weight = std::exp(row[j - jb] - maximum);
                st.l[index] += weight;
                const float* vr = kv.v.data() + (j * s.kv_heads + kh) * s.dv;
                for (int a = 0; a < s.dv; ++a) acc[a] += weight * vr[a];
            }
            st.m[index] = maximum;
        }
    }
    return pairs;
}

// Tuned version. For one (head, query tile) and each key tile:
//   1. scores for every row of the query tile (reads only the packed K tile),
//   2. per row: maximum, rescale of the running state if the maximum moved,
//      vectorised exponentials, and the weighted sum of V rows.
// Masks are resolved into index ranges per row, so no pair is tested
// individually and wholly masked tiles cost two binary searches per row.
std::uint64_t tuned_job(const Queries& q, const KVBlock& kv, const Shape& s, const Mask& mask,
                        const KernelOptions& opt, OnlineState& st, std::int64_t job,
                        std::int64_t tiles, float scale, const float* packed,
                        std::size_t blocks) {
    const auto bq = static_cast<std::size_t>(opt.block_q), bk = static_cast<std::size_t>(opt.block_k);
    const int h = static_cast<int>(job / tiles);
    const int kh = h / (s.heads / s.kv_heads);
    const auto ib = static_cast<std::size_t>(job % tiles) * bq;
    const auto ie = std::min(ib + bq, q.pos.size());
    const auto rows = ie - ib;
    auto& scores = tl_scores;
    auto& ranges = tl_ranges;
    auto& range_count = tl_range_count;
    if (scores.size() < checked_size(bq, bk)) scores.resize(checked_size(bq, bk));
    if (ranges.size() < 2 * bq) ranges.resize(2 * bq);
    if (range_count.size() < bq) range_count.resize(bq);
    const std::int64_t qmax = *std::max_element(q.pos.begin() + static_cast<std::ptrdiff_t>(ib),
                                                q.pos.begin() + static_cast<std::ptrdiff_t>(ie));
    const std::size_t keys = kv.pos.size();
    const std::size_t tile = static_cast<std::size_t>(s.d) * bk;
    const std::size_t vstride = static_cast<std::size_t>(s.kv_heads) * static_cast<std::size_t>(s.dv);
    const float* vhead = kv.v.data() + static_cast<std::size_t>(kh) * static_cast<std::size_t>(s.dv);
    std::uint64_t pairs = 0;
    for (std::size_t b = 0; b < blocks; ++b) {
        const std::size_t jb = b * bk, nk = std::min(bk, keys - jb);
        const std::int64_t* kp = kv.pos.data() + jb;
        // Every mask other than "none" is causal and keys ascend, so once a key
        // tile starts after the latest query of this tile nothing later is valid.
        if (mask.kind != MaskKind::none && kp[0] > qmax) break;
        const float* kt = packed + (static_cast<std::size_t>(kh) * blocks + b) * tile;
        bool any = false;
        for (std::size_t r = 0; r < rows; ++r) {
            const std::size_t i = ib + r;
            Range* rg = ranges.data() + 2 * r;
            const int count = allowed_ranges(mask, q.pos[i], kp, nk, rg);
            range_count[r] = count;
            const float* qr = q.q.data() + (i * s.heads + h) * s.d;
            for (int c = 0; c < count; ++c)
                scores_row(scores.data() + r * bk + rg[c].lo, qr, kt + rg[c].lo, bk,
                           rg[c].hi - rg[c].lo, s.d, scale);
            any = any || count;
        }
        if (!any) continue;
        for (std::size_t r = 0; r < rows; ++r) {
            const int count = range_count[r];
            if (!count) continue;
            const Range* rg = ranges.data() + 2 * r;
            float* row = scores.data() + r * bk;
            float block_max = -std::numeric_limits<float>::infinity();
            for (int c = 0; c < count; ++c) {
                const float m = max_lanes(row + rg[c].lo, rg[c].hi - rg[c].lo);
                block_max = m > block_max ? m : block_max;
            }
            const auto index = (ib + r) * s.heads + h;
            const float old_max = st.m[index];
            const float new_max = block_max > old_max ? block_max : old_max;
            float* acc = st.acc.data() + index * s.dv;
            float denominator = st.l[index];
            if (new_max != old_max && denominator != 0) {
                const float rescale = std::exp(old_max - new_max);
                denominator *= rescale;
                for (int a = 0; a < s.dv; ++a) acc[a] *= rescale;
            }
            for (int c = 0; c < count; ++c) {
                const std::size_t lo = rg[c].lo, n = rg[c].hi - lo;
                exp_shifted(row + lo, n, new_max);
                denominator += sum_lanes(row + lo, n);
                weighted_sum(acc, row + lo, vhead + (jb + lo) * vstride, vstride, n, s.dv);
                pairs += n;
            }
            st.l[index] = denominator;
            st.m[index] = new_max;
        }
    }
    return pairs;
}

// Transpose K into dimension-major tiles: tile (kh, b) holds d rows of bk keys.
void pack_keys(const KVBlock& kv, const Shape& s, const KernelOptions& opt,
               std::size_t blocks, Floats& packed) {
    const auto bk = static_cast<std::size_t>(opt.block_k);
    const std::size_t tile = static_cast<std::size_t>(s.d) * bk;
    const std::size_t need = checked_size(checked_size(static_cast<std::size_t>(s.kv_heads), blocks), tile);
    if (packed.size() < need) packed.resize(need);
    const std::size_t keys = kv.pos.size();
    const auto jobs = static_cast<std::int64_t>(static_cast<std::size_t>(s.kv_heads) * blocks);
    float* out = packed.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(opt.threads)
#endif
    for (std::int64_t job = 0; job < jobs; ++job) {
        const auto kh = static_cast<std::size_t>(job) / blocks, b = static_cast<std::size_t>(job) % blocks;
        float* ATTN_RESTRICT dst = out + static_cast<std::size_t>(job) * tile;
        const std::size_t jb = b * bk, nk = std::min(bk, keys - jb);
        for (std::size_t j = 0; j < nk; ++j) {
            const float* ATTN_RESTRICT kr = kv.k.data() + ((jb + j) * s.kv_heads + kh) * s.d;
            for (int a = 0; a < s.d; ++a) dst[static_cast<std::size_t>(a) * bk + j] = kr[a];
        }
    }
}

} // namespace

std::uint64_t count_pairs(const std::vector<std::int64_t>& query_pos,
                          const std::vector<std::int64_t>& key_pos,
                          const Shape& s, const Mask& mask) {
    s.validate();
    mask.validate();
    std::uint64_t pairs = 0;
    if (std::is_sorted(key_pos.begin(), key_pos.end())) {
        Range rg[2];
        for (auto qp : query_pos) {
            const int count = allowed_ranges(mask, qp, key_pos.data(), key_pos.size(), rg);
            for (int c = 0; c < count; ++c) pairs += rg[c].hi - rg[c].lo;
        }
    } else {
        for (auto qp : query_pos)
            for (auto kp : key_pos)
                if (mask.allows(qp, kp)) ++pairs;
    }
    return pairs * static_cast<std::uint64_t>(s.heads);
}

std::uint64_t process_block(const Queries& q, const KVBlock& kv, const Shape& s,
                            const Mask& mask, const KernelOptions& requested, OnlineState& st) {
    check(q, kv, s);
    mask.validate();
    if (requested.block_q <= 0 || requested.block_k <= 0 || requested.threads <= 0 || requested.poll_tiles < 0)
        throw std::invalid_argument("invalid kernel options");
    // With several threads a shard must be cut into enough query tiles to keep
    // every thread busy: use at least four tiles per thread when the rows allow it.
    KernelOptions opt = requested;
    if (opt.threads > 1 && !q.pos.empty()) {
        const std::size_t wanted = 4 * static_cast<std::size_t>(opt.threads);
        const std::size_t rows_per_tile = std::max<std::size_t>(1, (q.pos.size() + wanted - 1) / wanted);
        opt.block_q = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(opt.block_q), rows_per_tile));
    }
#ifndef _OPENMP
    if (opt.threads != 1) throw std::invalid_argument("this build has no OpenMP support");
#endif
    if (st.m.size() != elements(q.pos.size(), s.heads, 1) || st.l.size() != st.m.size() ||
        st.acc.size() != elements(q.pos.size(), s.heads, s.dv))
        throw std::invalid_argument("invalid online state dimensions");
    const auto tiles = static_cast<std::int64_t>((q.pos.size() + opt.block_q - 1) / opt.block_q);
    const auto jobs = tiles * s.heads;
    if (!jobs || kv.pos.empty()) return 0;
    const auto batch = opt.poll_tiles ? static_cast<std::int64_t>(opt.poll_tiles) : jobs;
    const float scale = 1.0f / std::sqrt(static_cast<float>(s.d));
    // The range-based kernel needs ascending key positions inside the block;
    // every partition used here provides them. Anything else takes the basic path.
    const bool tuned = opt.kernel == Kernel::tuned && std::is_sorted(kv.pos.begin(), kv.pos.end());
    const std::size_t blocks = (kv.pos.size() + static_cast<std::size_t>(opt.block_k) - 1) /
                               static_cast<std::size_t>(opt.block_k);
    const float* packed = nullptr;
    if (tuned) {
        pack_keys(kv, s, opt, blocks, tl_packed);
        packed = tl_packed.data();
    }
    std::uint64_t pairs = 0;
    for (std::int64_t first = 0; first < jobs; first += batch) {
        const auto last = std::min(first + batch, jobs);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(opt.threads) reduction(+:pairs)
#endif
        for (std::int64_t job = first; job < last; ++job)
            pairs += tuned ? tuned_job(q, kv, s, mask, opt, st, job, tiles, scale, packed, blocks)
                           : basic_job(q, kv, s, mask, opt, st, job, tiles, scale);
        if (opt.progress) opt.progress();
    }
    return pairs;
}

std::vector<float> finalize(const OnlineState& st, const Shape& s) {
    std::vector<float> out(st.acc.size(), 0.0f);
    for (std::size_t i = 0; i < st.l.size(); ++i)
        if (st.l[i] != 0)
            for (int a = 0; a < s.dv; ++a) out[i * s.dv + a] = st.acc[i * s.dv + a] / st.l[i];
    return out; // Explicit policy: an entirely masked query produces zero.
}

std::vector<double> reference64(const Queries& q, const KVBlock& kv, const Shape& s, const Mask& m) {
    return reference<double>(q, kv, s, m);
}
std::vector<float> reference32(const Queries& q, const KVBlock& kv, const Shape& s, const Mask& m) {
    return reference<float>(q, kv, s, m);
}

Errors compare(const std::vector<float>& actual, const std::vector<double>& expected,
               double atol, double rtol) {
    if (actual.size() != expected.size()) throw std::invalid_argument("comparison shape mismatch");
    Errors e;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) { ++e.nonfinite; ++e.failed; continue; }
        const double diff = std::abs(static_cast<double>(actual[i]) - expected[i]);
        e.max_abs = std::max(e.max_abs, diff);
        e.max_rel = std::max(e.max_rel, diff / std::max(std::abs(expected[i]), 1e-12));
        if (diff > atol + rtol * std::abs(expected[i])) ++e.failed;
    }
    return e;
}

std::vector<std::size_t> sample_rows(std::size_t rows, std::size_t requested) {
    const auto count = std::min(rows, requested);
    std::vector<std::size_t> result(count);
    for (std::size_t i = 0; i < count; ++i)
        result[i] = count <= 1 ? 0 : i * (rows - 1) / (count - 1);
    return result;
}

} // namespace attn
