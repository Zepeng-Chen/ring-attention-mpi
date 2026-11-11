#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <new>
#include <string>
#include <vector>

namespace attn {

// Storage of every tensor and scratch buffer. The kernel's inner loops read and
// write whole vectors of up to 64 bytes; with the 16-byte alignment of malloc a
// 64-byte access usually straddles two cache lines, and whether it does depends
// on where the allocator happened to place the array. Aligning every buffer to
// ATTN_ALIGNMENT bytes (a page by default) removes the split accesses and gives
// all processes the same layout. ATTN_ALIGNMENT=16 restores plain malloc for the
// controlled comparison in the report.
#ifndef ATTN_ALIGNMENT
#define ATTN_ALIGNMENT 4096
#endif
template <class T>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() = default;
    template <class U> AlignedAllocator(const AlignedAllocator<U>&) {}
    T* allocate(std::size_t count) {
        if (count > static_cast<std::size_t>(-1) / sizeof(T)) throw std::bad_alloc();
        const std::size_t align = ATTN_ALIGNMENT;
        void* memory = nullptr;
        if (align <= 16) {
            memory = std::malloc(count * sizeof(T) ? count * sizeof(T) : 1);
        } else {
            const std::size_t bytes = (count * sizeof(T) + align - 1) / align * align;
            memory = std::aligned_alloc(align, bytes ? bytes : align);
        }
        if (!memory) throw std::bad_alloc();
        return static_cast<T*>(memory);
    }
    void deallocate(T* pointer, std::size_t) { std::free(pointer); }
    template <class U> bool operator==(const AlignedAllocator<U>&) const { return true; }
    template <class U> bool operator!=(const AlignedAllocator<U>&) const { return false; }
};
using Floats = std::vector<float, AlignedAllocator<float>>;

struct Shape {
    std::int64_t n = 128, nq = 128;
    int d = 64, dv = 64, heads = 1, kv_heads = 1;
    void validate() const;
};

enum class MaskKind { none, causal, window, sink_window };
struct Mask {
    MaskKind kind = MaskKind::none;
    std::int64_t window = 128, sink = 4;
    bool allows(std::int64_t query, std::int64_t key) const;
    void validate() const;
};

enum class Partition { contiguous, zigzag };
std::vector<std::int64_t> positions(std::int64_t n, int ranks, int rank,
                                    Partition partition);
std::size_t checked_size(std::size_t a, std::size_t b);

// Token-major: Q [rows,H,d], K [rows,Hkv,d], V [rows,Hkv,dv].
struct Queries {
    std::vector<std::int64_t> pos;
    Floats q;
};
struct KVBlock {
    std::vector<std::int64_t> pos;
    Floats k, v;
};
float synthetic_value(std::uint64_t seed, int tensor, int head,
                      std::int64_t row, int column);
Queries make_queries(const Shape& shape, std::vector<std::int64_t> pos,
                     std::uint64_t seed, float alpha);
KVBlock make_kv(const Shape& shape, std::vector<std::int64_t> pos,
                std::uint64_t seed, bool replicate_kv = false);
Queries read_queries(const Shape& shape, std::vector<std::int64_t> pos,
                     const std::string& directory);
KVBlock read_kv(const Shape& shape, std::vector<std::int64_t> pos,
                const std::string& directory);
// Rows `pos` of a raw little-endian float32 file laid out [total][heads][dim].
std::vector<float> read_tensor(const std::string& path, const std::vector<std::int64_t>& pos,
                               std::int64_t total, int heads, int dim);
// Matched multi-head control for grouped-query attention: every query head gets
// its own copy of the K/V head it shares. Same mathematics, G times the K/V bytes.
KVBlock replicate_kv_heads(const KVBlock& block, const Shape& shape);

struct OnlineState {
    Floats m, l, acc;
    OnlineState(std::size_t rows, const Shape& shape);
    void reset();
};
// basic: scalar tile loop that tests the mask for every pair (first blocked version).
// tuned: packed (transposed) K tiles, branch-free allowed ranges, unrolled
//        multiply-add streams and a hand-written vectorisable exponential.
enum class Kernel { basic, tuned };
struct KernelOptions {
    // Tile sizes: the measured optimum for d = 64 on the Xeon used in the report.
    int block_q = 128, block_k = 96, threads = 1;
    // Called on the MPI main thread between query-tile batches only.
    int poll_tiles = 0;
    std::function<void()> progress;
    Kernel kernel = Kernel::tuned;
};
// Number of allowed (query, key) pairs, times the number of query heads.
std::uint64_t count_pairs(const std::vector<std::int64_t>& query_pos,
                          const std::vector<std::int64_t>& key_pos,
                          const Shape& shape, const Mask& mask);
// Returns useful query-key pairs, including all query heads.
std::uint64_t process_block(const Queries& queries, const KVBlock& block,
                            const Shape& shape, const Mask& mask,
                            const KernelOptions& options, OnlineState& state);
std::vector<float> finalize(const OnlineState& state, const Shape& shape);

// Independent two-pass row references; never reuse online updates.
std::vector<double> reference64(const Queries& queries, const KVBlock& keys,
                                const Shape& shape, const Mask& mask);
std::vector<float> reference32(const Queries& queries, const KVBlock& keys,
                               const Shape& shape, const Mask& mask);
struct Errors {
    double max_abs = 0, max_rel = 0;
    std::uint64_t nonfinite = 0, failed = 0;
};
Errors compare(const std::vector<float>& actual, const std::vector<double>& expected,
               double atol = 1e-5, double rtol = 0);
std::vector<std::size_t> sample_rows(std::size_t rows, std::size_t requested);

} // namespace attn
