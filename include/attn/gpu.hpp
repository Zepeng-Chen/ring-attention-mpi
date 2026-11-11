#pragma once
// CUDA backend: online-softmax attention of the local queries against one K/V
// block at a time, with the running state (m, l, acc) kept on the device.
// The header exposes no CUDA types, so the MPI driver is ordinary C++.

#include "attn/core.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace attn {
namespace gpu {

// k0: one thread per (query, head); every operand in global memory; two passes.
// k1: tiled, shared-memory kernels, but the score matrix is materialised in
//     global memory (scores -> row update -> weighted sum, three kernels).
// k2: one fused tile kernel; scores never leave shared memory and the
//     accumulators stay in registers for the whole K/V block.
enum class Variant { k0, k1, k2 };
Variant parse_variant(const std::string& name);

constexpr int max_head_dim = 64;   // d and dv supported by the tiled kernels

int device_count();

struct Device;   // opaque

// Creates the context on `device`, uploads the local queries and allocates two
// K/V block slots of `max_keys` rows each (device memory + pinned host staging).
Device* create(int device, const Shape& shape, const Mask& mask, Variant variant,
               const Queries& queries, std::size_t max_keys, std::size_t k1_budget_bytes);
void destroy(Device* device);
std::string name(const Device* device);

void reset(Device* device);                        // state <- (-inf, 0, 0)

// Block slots. Device pointers are for CUDA-aware MPI; host pointers are
// page-locked staging buffers for the host-staged path.
float* k_device(Device* device, int slot);
float* v_device(Device* device, int slot);
float* k_host(Device* device, int slot);
float* v_host(Device* device, int slot);
// Declares which keys the slot will hold (positions are uploaded asynchronously).
void set_block(Device* device, int slot, const std::vector<std::int64_t>& positions);
// Asynchronous host -> device copy of the slot's staging buffers (copy stream).
void upload(Device* device, int slot);
void sync_copy(Device* device);
// Asynchronous kernel launch on the compute stream.
void launch(Device* device, int slot);
// Blocks until the launched kernels finish; returns their device time in seconds.
double sync_compute(Device* device);
// O = acc / l (zero for entirely masked rows), copied to the host.
void finalize(Device* device, float* output);

} // namespace gpu
} // namespace attn
