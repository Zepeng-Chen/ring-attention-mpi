// Hand-written CUDA kernels for blockwise online-softmax attention.
// No cuBLAS/cuDNN: every multiply-add loop below is explicit.
#include "attn/gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace attn {
namespace gpu {
namespace {

constexpr int T = 32;              // tile edge = warp size
constexpr int D = max_head_dim;    // largest d / dv of the tiled kernels

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

struct Dims {
    int nq, nk, heads, kv_heads, d, dv;
    float scale;
};
struct MaskParams {
    int kind;                      // 0 none, 1 causal, 2 window, 3 sink+window
    long long window, sink;
};

__device__ __forceinline__ bool allowed(const MaskParams& mask, long long q, long long k) {
    if (mask.kind == 0) return true;
    if (k > q) return false;
    if (mask.kind == 1) return true;
    return (q - k < mask.window) || (mask.kind == 3 && k < mask.sink);
}
__device__ __forceinline__ float warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

// ---------------------------------------------------------------------------
// K0: the serial two-pass row algorithm, one thread per (query, head).
// Q, K, V and the accumulator are all read and written in global memory.
// ---------------------------------------------------------------------------
__global__ void k0_kernel(Dims s, MaskParams mask, const float* Q, const long long* qpos,
                          const float* K, const float* V, const long long* kpos,
                          float* m, float* l, float* acc) {
    const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<long long>(s.nq) * s.heads) return;
    const int i = static_cast<int>(index / s.heads), h = static_cast<int>(index % s.heads);
    const int kh = h / (s.heads / s.kv_heads);
    const float* q = Q + index * s.d;
    const long long qp = qpos[i];
    float block_max = -INFINITY;
    for (int j = 0; j < s.nk; ++j) {
        if (!allowed(mask, qp, kpos[j])) continue;
        const float* k = K + (static_cast<long long>(j) * s.kv_heads + kh) * s.d;
        float dot = 0.0f;
        for (int a = 0; a < s.d; ++a) dot += q[a] * k[a];
        block_max = fmaxf(block_max, dot * s.scale);
    }
    if (block_max == -INFINITY) return;
    const float old_max = m[index], new_max = fmaxf(old_max, block_max);
    float denominator = l[index];
    const float rescale = denominator == 0.0f ? 0.0f : expf(old_max - new_max);
    denominator *= rescale;
    float* out = acc + index * s.dv;
    for (int a = 0; a < s.dv; ++a) out[a] *= rescale;
    for (int j = 0; j < s.nk; ++j) {
        if (!allowed(mask, qp, kpos[j])) continue;
        const float* k = K + (static_cast<long long>(j) * s.kv_heads + kh) * s.d;
        float dot = 0.0f;
        for (int a = 0; a < s.d; ++a) dot += q[a] * k[a];
        const float weight = expf(dot * s.scale - new_max);
        denominator += weight;
        const float* v = V + (static_cast<long long>(j) * s.kv_heads + kh) * s.dv;
        for (int a = 0; a < s.dv; ++a) out[a] += weight * v[a];
    }
    m[index] = new_max;
    l[index] = denominator;
}

// ---------------------------------------------------------------------------
// K1: tiled with shared memory, scores materialised in global memory.
// Thread (y, x) of a 32x32 block is (query row in tile, key in tile).
// The key tile is stored transposed in shared memory ([dim][key]) so that the
// 32 lanes of a warp read 32 different banks.
// ---------------------------------------------------------------------------
__global__ void k1_scores(Dims s, MaskParams mask, const float* __restrict__ Q, const long long* __restrict__ qpos,
                          const float* __restrict__ K, const long long* __restrict__ kpos, float* __restrict__ S) {
    __shared__ float Qs[T][D];
    __shared__ float Kt[D][T];
    const int t = threadIdx.x, r = threadIdx.y;
    const int tiles_q = (s.nq + T - 1) / T;
    const int h = blockIdx.y / tiles_q, tile = blockIdx.y % tiles_q;
    const int kh = h / (s.heads / s.kv_heads);
    const int jb = blockIdx.x * T, keys = min(T, s.nk - jb);
    const int i = tile * T + r;
    const bool row_ok = i < s.nq;
    const long long index = static_cast<long long>(i) * s.heads + h;
    if (row_ok) for (int a = t; a < s.d; a += T) Qs[r][a] = Q[index * s.d + a];
    if (r < keys) {
        const long long row = static_cast<long long>(jb + r) * s.kv_heads + kh;
        for (int a = t; a < s.d; a += T) Kt[a][r] = K[row * s.d + a];
    }
    __syncthreads();
    if (row_ok && t < keys) {
        float score = -INFINITY;
        if (allowed(mask, qpos[i], kpos[jb + t])) {
            float dot = 0.0f;
            for (int a = 0; a < s.d; ++a) dot += Qs[r][a] * Kt[a][t];
            score = dot * s.scale;
        }
        S[index * s.nk + jb + t] = score;
    }
}

// One warp per (query, head): running maximum, rescale, weights in place.
__global__ void k1_rows(Dims s, float* __restrict__ S, float* __restrict__ m, float* __restrict__ l,
                        float* __restrict__ acc) {
    const long long index = static_cast<long long>(blockIdx.y) * gridDim.x + blockIdx.x;
    if (index >= static_cast<long long>(s.nq) * s.heads) return;
    const int lane = threadIdx.x;
    float* row = S + index * s.nk;
    float block_max = -INFINITY;
    for (int j = lane; j < s.nk; j += T) block_max = fmaxf(block_max, row[j]);
    block_max = warp_max(block_max);
    if (block_max == -INFINITY) {          // nothing allowed: weights are zero
        for (int j = lane; j < s.nk; j += T) row[j] = 0.0f;
        return;
    }
    const float old_max = m[index], new_max = fmaxf(old_max, block_max);
    const float old_l = l[index];
    const float rescale = old_l == 0.0f ? 0.0f : expf(old_max - new_max);
    float sum = 0.0f;
    for (int j = lane; j < s.nk; j += T) {
        const float weight = expf(row[j] - new_max);     // exp(-inf) = 0 for masked pairs
        row[j] = weight;
        sum += weight;
    }
    sum = warp_sum(sum);
    for (int a = lane; a < s.dv; a += T) acc[index * s.dv + a] *= rescale;
    if (lane == 0) {
        m[index] = new_max;
        l[index] = old_l * rescale + sum;
    }
}

// Thread (y, x) is (query row in tile, output dimension x and x + 32).
__global__ void k1_weighted(Dims s, const float* __restrict__ S, const float* __restrict__ V,
                            float* __restrict__ acc) {
    __shared__ float Ws[T][T];
    __shared__ float Vs[T][D];
    const int t = threadIdx.x, r = threadIdx.y;
    const int tiles_q = (s.nq + T - 1) / T;
    const int h = blockIdx.x / tiles_q, tile = blockIdx.x % tiles_q;
    const int kh = h / (s.heads / s.kv_heads);
    const int i = tile * T + r;
    const bool row_ok = i < s.nq;
    const long long index = static_cast<long long>(i) * s.heads + h;
    float a0 = 0.0f, a1 = 0.0f;
    for (int jb = 0; jb < s.nk; jb += T) {
        const int keys = min(T, s.nk - jb);
        if (row_ok && t < keys) Ws[r][t] = S[index * s.nk + jb + t];
        if (r < keys) {
            const long long row = static_cast<long long>(jb + r) * s.kv_heads + kh;
            for (int a = t; a < s.dv; a += T) Vs[r][a] = V[row * s.dv + a];
        }
        __syncthreads();
        if (row_ok) {
            if (t < s.dv) for (int j = 0; j < keys; ++j) a0 += Ws[r][j] * Vs[j][t];
            if (t + T < s.dv) for (int j = 0; j < keys; ++j) a1 += Ws[r][j] * Vs[j][t + T];
        }
        __syncthreads();
    }
    if (row_ok) {
        if (t < s.dv) acc[index * s.dv + t] += a0;
        if (t + T < s.dv) acc[index * s.dv + t + T] += a1;
    }
}

// ---------------------------------------------------------------------------
// K2: fused. One 32x32 block owns a tile of 32 query rows of one head and
// streams every key tile of the block through shared memory. Each warp is one
// query row: its maximum and denominator are warp reductions, the weights stay
// in shared memory, and each thread keeps two accumulator components in
// registers until the whole K/V block has been folded in.
// ---------------------------------------------------------------------------
__global__ void k2_kernel(Dims s, MaskParams mask, const float* __restrict__ Q, const long long* __restrict__ qpos,
                          const float* __restrict__ K, const float* __restrict__ V,
                          const long long* __restrict__ kpos, float* __restrict__ m, float* __restrict__ l,
                          float* __restrict__ acc) {
    __shared__ float Qs[T][D];        // [row][dim]
    __shared__ float Kt[D][T];        // [dim][key]
    __shared__ float Vs[T][D];        // [key][dim]
    __shared__ float Ws[T][T];        // [row][key]
    __shared__ long long kps[T];
    const int t = threadIdx.x, r = threadIdx.y;
    const int tiles_q = (s.nq + T - 1) / T;
    const int h = blockIdx.x / tiles_q, tile = blockIdx.x % tiles_q;
    const int kh = h / (s.heads / s.kv_heads);
    const int i = tile * T + r;
    const bool row_ok = i < s.nq;
    const long long index = static_cast<long long>(i) * s.heads + h;
    float row_max = -INFINITY, row_l = 0.0f, acc0 = 0.0f, acc1 = 0.0f;
    long long qp = 0;
    if (row_ok) {
        for (int a = t; a < s.d; a += T) Qs[r][a] = Q[index * s.d + a];
        row_max = m[index];
        row_l = l[index];
        qp = qpos[i];
        if (t < s.dv) acc0 = acc[index * s.dv + t];
        if (t + T < s.dv) acc1 = acc[index * s.dv + t + T];
    }
    // Query positions ascend, so the last row of the tile is the latest query.
    const long long tile_qmax = qpos[min(tile * T + T, s.nq) - 1];
    __syncthreads();
    for (int jb = 0; jb < s.nk; jb += T) {
        // Keys ascend too: with a causal-type mask nothing later can be allowed.
        if (mask.kind != 0 && kpos[jb] > tile_qmax) break;
        const int keys = min(T, s.nk - jb);
        if (r < keys) {
            const long long row = static_cast<long long>(jb + r) * s.kv_heads + kh;
            for (int a = t; a < s.d; a += T) Kt[a][r] = K[row * s.d + a];
            for (int a = t; a < s.dv; a += T) Vs[r][a] = V[row * s.dv + a];
            if (t == 0) kps[r] = kpos[jb + r];
        }
        __syncthreads();
        if (row_ok) {
            float score = -INFINITY;
            if (t < keys && allowed(mask, qp, kps[t])) {
                float dot = 0.0f;
                for (int a = 0; a < s.d; ++a) dot += Qs[r][a] * Kt[a][t];
                score = dot * s.scale;
            }
            const float block_max = warp_max(score);
            if (block_max != -INFINITY) {                  // uniform across the warp
                const float new_max = fmaxf(row_max, block_max);
                const float rescale = row_l == 0.0f ? 0.0f : expf(row_max - new_max);
                const float weight = score == -INFINITY ? 0.0f : expf(score - new_max);
                row_l = row_l * rescale + warp_sum(weight);
                row_max = new_max;
                Ws[r][t] = weight;
                __syncwarp();
                float a0 = acc0 * rescale, a1 = acc1 * rescale;
                if (t < s.dv) for (int j = 0; j < keys; ++j) a0 += Ws[r][j] * Vs[j][t];
                if (t + T < s.dv) for (int j = 0; j < keys; ++j) a1 += Ws[r][j] * Vs[j][t + T];
                acc0 = a0;
                acc1 = a1;
            }
        }
        __syncthreads();
    }
    if (row_ok) {
        if (t == 0) { m[index] = row_max; l[index] = row_l; }
        if (t < s.dv) acc[index * s.dv + t] = acc0;
        if (t + T < s.dv) acc[index * s.dv + t + T] = acc1;
    }
}

__global__ void reset_kernel(long long rows, int dv, float* m, float* l, float* acc) {
    const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < rows) { m[index] = -INFINITY; l[index] = 0.0f; }
    if (index < rows * dv) acc[index] = 0.0f;
}
__global__ void finalize_kernel(long long rows, int dv, const float* l, const float* acc, float* out) {
    const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * dv) return;
    const float denominator = l[index / dv];
    out[index] = denominator != 0.0f ? acc[index] / denominator : 0.0f;
}

unsigned blocks_for(long long items, int per_block) {
    return static_cast<unsigned>((items + per_block - 1) / per_block);
}

} // namespace

struct Device {
    int id = 0;
    Dims dims{};
    MaskParams mask{};
    Variant variant = Variant::k2;
    std::size_t max_keys = 0, rows = 0, score_capacity = 0;
    cudaStream_t compute = nullptr, copy = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    float *q = nullptr, *m = nullptr, *l = nullptr, *acc = nullptr, *out = nullptr, *scores = nullptr;
    long long* qpos = nullptr;
    float *k[2] = {nullptr, nullptr}, *v[2] = {nullptr, nullptr};
    long long* kpos[2] = {nullptr, nullptr};
    float *host_k[2] = {nullptr, nullptr}, *host_v[2] = {nullptr, nullptr};
    long long* host_pos[2] = {nullptr, nullptr};
    std::size_t keys[2] = {0, 0};
    std::string label;
};

Variant parse_variant(const std::string& text) {
    if (text == "k0" || text == "K0") return Variant::k0;
    if (text == "k1" || text == "K1") return Variant::k1;
    if (text == "k2" || text == "K2") return Variant::k2;
    throw std::invalid_argument("unknown GPU kernel: " + text);
}

int device_count() {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    return count;
}

Device* create(int id, const Shape& shape, const Mask& mask, Variant variant,
               const Queries& queries, std::size_t max_keys, std::size_t k1_budget_bytes) {
    shape.validate();
    mask.validate();
    if (variant != Variant::k0 && (shape.d > D || shape.dv > D))
        throw std::invalid_argument("tiled GPU kernels support head dimensions up to 64");
    const std::size_t nq = queries.pos.size();
    if (nq > 0x7fffffffu || max_keys > 0x7fffffffu) throw std::overflow_error("GPU shard too large");
    check(cudaSetDevice(id), "cudaSetDevice");
    auto* device = new Device;
    device->id = id;
    device->variant = variant;
    device->max_keys = std::max<std::size_t>(max_keys, 1);
    device->rows = nq * static_cast<std::size_t>(shape.heads);
    device->dims = {static_cast<int>(nq), 0, shape.heads, shape.kv_heads, shape.d, shape.dv,
                    1.0f / std::sqrt(static_cast<float>(shape.d))};
    device->mask = {static_cast<int>(mask.kind), static_cast<long long>(mask.window),
                    static_cast<long long>(mask.sink)};
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, id), "cudaGetDeviceProperties");
    device->label = properties.name;
    check(cudaStreamCreateWithFlags(&device->compute, cudaStreamNonBlocking), "compute stream");
    check(cudaStreamCreateWithFlags(&device->copy, cudaStreamNonBlocking), "copy stream");
    check(cudaEventCreate(&device->begin), "event");
    check(cudaEventCreate(&device->end), "event");
    const std::size_t rows = std::max<std::size_t>(device->rows, 1);
    auto bytes = [](std::size_t count) { return std::max<std::size_t>(count, 1) * sizeof(float); };
    check(cudaMalloc(&device->q, bytes(rows * shape.d)), "cudaMalloc Q");
    check(cudaMalloc(&device->qpos, std::max<std::size_t>(nq, 1) * sizeof(long long)), "cudaMalloc qpos");
    check(cudaMalloc(&device->m, bytes(rows)), "cudaMalloc m");
    check(cudaMalloc(&device->l, bytes(rows)), "cudaMalloc l");
    check(cudaMalloc(&device->acc, bytes(rows * shape.dv)), "cudaMalloc acc");
    check(cudaMalloc(&device->out, bytes(rows * shape.dv)), "cudaMalloc out");
    if (nq) {
        check(cudaMemcpy(device->q, queries.q.data(), queries.q.size() * sizeof(float), cudaMemcpyHostToDevice), "upload Q");
        check(cudaMemcpy(device->qpos, queries.pos.data(), nq * sizeof(long long), cudaMemcpyHostToDevice), "upload qpos");
    }
    const std::size_t kcount = device->max_keys * shape.kv_heads * shape.d;
    const std::size_t vcount = device->max_keys * shape.kv_heads * shape.dv;
    for (int slot = 0; slot < 2; ++slot) {
        check(cudaMalloc(&device->k[slot], bytes(kcount)), "cudaMalloc K");
        check(cudaMalloc(&device->v[slot], bytes(vcount)), "cudaMalloc V");
        check(cudaMalloc(&device->kpos[slot], device->max_keys * sizeof(long long)), "cudaMalloc kpos");
        check(cudaMallocHost(&device->host_k[slot], bytes(kcount)), "cudaMallocHost K");
        check(cudaMallocHost(&device->host_v[slot], bytes(vcount)), "cudaMallocHost V");
        check(cudaMallocHost(&device->host_pos[slot], device->max_keys * sizeof(long long)), "cudaMallocHost kpos");
    }
    if (variant == Variant::k1) {
        // Scores for a chunk of keys: rows x chunk floats, bounded by the budget.
        std::size_t chunk = k1_budget_bytes / (rows * sizeof(float));
        chunk = std::max<std::size_t>(T, std::min(device->max_keys + T, chunk) / T * T);
        device->score_capacity = chunk;
        check(cudaMalloc(&device->scores, bytes(rows * chunk)), "cudaMalloc scores");
    }
    reset(device);
    return device;
}

void destroy(Device* device) {
    if (!device) return;
    cudaSetDevice(device->id);
    cudaStreamSynchronize(device->compute);
    cudaStreamSynchronize(device->copy);
    for (int slot = 0; slot < 2; ++slot) {
        cudaFree(device->k[slot]); cudaFree(device->v[slot]); cudaFree(device->kpos[slot]);
        cudaFreeHost(device->host_k[slot]); cudaFreeHost(device->host_v[slot]); cudaFreeHost(device->host_pos[slot]);
    }
    cudaFree(device->q); cudaFree(device->qpos); cudaFree(device->m); cudaFree(device->l);
    cudaFree(device->acc); cudaFree(device->out); cudaFree(device->scores);
    cudaEventDestroy(device->begin); cudaEventDestroy(device->end);
    cudaStreamDestroy(device->compute); cudaStreamDestroy(device->copy);
    delete device;
}

std::string name(const Device* device) { return device->label; }

void reset(Device* device) {
    const long long rows = static_cast<long long>(device->rows);
    if (!rows) return;
    reset_kernel<<<blocks_for(rows * device->dims.dv, 256), 256, 0, device->compute>>>(
        rows, device->dims.dv, device->m, device->l, device->acc);
    check(cudaGetLastError(), "reset kernel");
    check(cudaStreamSynchronize(device->compute), "reset sync");
}

float* k_device(Device* device, int slot) { return device->k[slot]; }
float* v_device(Device* device, int slot) { return device->v[slot]; }
float* k_host(Device* device, int slot) { return device->host_k[slot]; }
float* v_host(Device* device, int slot) { return device->host_v[slot]; }

void set_block(Device* device, int slot, const std::vector<std::int64_t>& positions) {
    if (positions.size() > device->max_keys) throw std::invalid_argument("K/V block exceeds the GPU slot");
    device->keys[slot] = positions.size();
    if (positions.empty()) return;
    std::memcpy(device->host_pos[slot], positions.data(), positions.size() * sizeof(long long));
    check(cudaMemcpyAsync(device->kpos[slot], device->host_pos[slot], positions.size() * sizeof(long long),
                          cudaMemcpyHostToDevice, device->copy), "upload key positions");
}

void upload(Device* device, int slot) {
    const std::size_t keys = device->keys[slot];
    if (!keys) return;
    const Dims& s = device->dims;
    check(cudaMemcpyAsync(device->k[slot], device->host_k[slot], keys * s.kv_heads * s.d * sizeof(float),
                          cudaMemcpyHostToDevice, device->copy), "upload K");
    check(cudaMemcpyAsync(device->v[slot], device->host_v[slot], keys * s.kv_heads * s.dv * sizeof(float),
                          cudaMemcpyHostToDevice, device->copy), "upload V");
}

void sync_copy(Device* device) { check(cudaStreamSynchronize(device->copy), "copy stream sync"); }

void launch(Device* device, int slot) {
    check(cudaEventRecord(device->begin, device->compute), "event record");
    const std::size_t keys = device->keys[slot];
    Dims s = device->dims;
    if (keys && s.nq) {
        const int tiles_q = (s.nq + T - 1) / T;
        const float* K = device->k[slot];
        const float* V = device->v[slot];
        const long long* kpos = device->kpos[slot];
        if (device->variant == Variant::k0) {
            s.nk = static_cast<int>(keys);
            k0_kernel<<<blocks_for(static_cast<long long>(device->rows), 128), 128, 0, device->compute>>>(
                s, device->mask, device->q, device->qpos, K, V, kpos, device->m, device->l, device->acc);
        } else if (device->variant == Variant::k2) {
            s.nk = static_cast<int>(keys);
            k2_kernel<<<static_cast<unsigned>(tiles_q) * s.heads, dim3(T, T), 0, device->compute>>>(
                s, device->mask, device->q, device->qpos, K, V, kpos, device->m, device->l, device->acc);
        } else {
            for (std::size_t offset = 0; offset < keys; offset += device->score_capacity) {
                s.nk = static_cast<int>(std::min(device->score_capacity, keys - offset));
                const float* Kc = K + offset * s.kv_heads * s.d;
                const float* Vc = V + offset * s.kv_heads * s.dv;
                const dim3 score_grid(blocks_for(s.nk, T), static_cast<unsigned>(tiles_q) * s.heads);
                k1_scores<<<score_grid, dim3(T, T), 0, device->compute>>>(
                    s, device->mask, device->q, device->qpos, Kc, kpos + offset, device->scores);
                const long long rows = static_cast<long long>(device->rows);
                const unsigned gx = static_cast<unsigned>(std::min<long long>(rows, 32768));
                const dim3 row_grid(gx, blocks_for(rows, static_cast<int>(gx)));
                k1_rows<<<row_grid, T, 0, device->compute>>>(s, device->scores, device->m, device->l, device->acc);
                k1_weighted<<<static_cast<unsigned>(tiles_q) * s.heads, dim3(T, T), 0, device->compute>>>(
                    s, device->scores, Vc, device->acc);
            }
        }
        check(cudaGetLastError(), "kernel launch");
    }
    check(cudaEventRecord(device->end, device->compute), "event record");
}

double sync_compute(Device* device) {
    check(cudaStreamSynchronize(device->compute), "compute stream sync");
    float milliseconds = 0.0f;
    check(cudaEventElapsedTime(&milliseconds, device->begin, device->end), "event time");
    return 1e-3 * static_cast<double>(milliseconds);
}

void finalize(Device* device, float* output) {
    const long long rows = static_cast<long long>(device->rows);
    if (!rows) return;
    const long long count = rows * device->dims.dv;
    finalize_kernel<<<blocks_for(count, 256), 256, 0, device->compute>>>(
        rows, device->dims.dv, device->l, device->acc, device->out);
    check(cudaGetLastError(), "finalize kernel");
    check(cudaMemcpyAsync(output, device->out, static_cast<std::size_t>(count) * sizeof(float),
                          cudaMemcpyDeviceToHost, device->compute), "download output");
    check(cudaStreamSynchronize(device->compute), "finalize sync");
}

} // namespace gpu
} // namespace attn
