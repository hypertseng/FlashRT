// ================================================================
// FlashRT — Patch embedding kernels
//
// 1. im2col: (nv, 224, 224, 3) → (nv*256, 588) strided copy
// 2. bias_pos: output[i,j] += bias[j] + pos_emb[i % S_per_view, j]
// ================================================================

#include "patch_embed.cuh"

__device__ __constant__ uint16_t kU8NormFp16Bits[256] = {
    0xbc00, 0xbbf0, 0xbbe0, 0xbbd0, 0xbbc0, 0xbbb0, 0xbba0, 0xbb90,
    0xbb7f, 0xbb6f, 0xbb5f, 0xbb4f, 0xbb3f, 0xbb2f, 0xbb1f, 0xbb0f,
    0xbaff, 0xbaef, 0xbadf, 0xbacf, 0xbabf, 0xbaaf, 0xba9f, 0xba8f,
    0xba7e, 0xba6e, 0xba5e, 0xba4e, 0xba3e, 0xba2e, 0xba1e, 0xba0e,
    0xb9fe, 0xb9ee, 0xb9de, 0xb9ce, 0xb9be, 0xb9ae, 0xb99e, 0xb98e,
    0xb97d, 0xb96d, 0xb95d, 0xb94d, 0xb93d, 0xb92d, 0xb91d, 0xb90d,
    0xb8fd, 0xb8ed, 0xb8dd, 0xb8cd, 0xb8bd, 0xb8ad, 0xb89d, 0xb88d,
    0xb87c, 0xb86c, 0xb85c, 0xb84c, 0xb83c, 0xb82c, 0xb81c, 0xb80c,
    0xb7f8, 0xb7d8, 0xb7b8, 0xb798, 0xb777, 0xb757, 0xb737, 0xb717,
    0xb6f7, 0xb6d7, 0xb6b7, 0xb697, 0xb676, 0xb656, 0xb636, 0xb616,
    0xb5f6, 0xb5d6, 0xb5b6, 0xb596, 0xb575, 0xb555, 0xb535, 0xb515,
    0xb4f5, 0xb4d5, 0xb4b5, 0xb495, 0xb474, 0xb454, 0xb434, 0xb414,
    0xb3e8, 0xb3a8, 0xb367, 0xb327, 0xb2e7, 0xb2a7, 0xb266, 0xb226,
    0xb1e6, 0xb1a6, 0xb165, 0xb125, 0xb0e5, 0xb0a5, 0xb064, 0xb024,
    0xafc8, 0xaf47, 0xaec7, 0xae46, 0xadc6, 0xad45, 0xacc5, 0xac44,
    0xab88, 0xaa87, 0xa986, 0xa885, 0xa707, 0xa505, 0xa206, 0x9c04,
    0x1c04, 0x2206, 0x2505, 0x2707, 0x2885, 0x2986, 0x2a87, 0x2b88,
    0x2c44, 0x2cc5, 0x2d45, 0x2dc6, 0x2e46, 0x2ec7, 0x2f47, 0x2fc8,
    0x3024, 0x3064, 0x30a5, 0x30e5, 0x3125, 0x3165, 0x31a6, 0x31e6,
    0x3226, 0x3266, 0x32a7, 0x32e7, 0x3327, 0x3367, 0x33a8, 0x33e8,
    0x3414, 0x3434, 0x3454, 0x3474, 0x3495, 0x34b5, 0x34d5, 0x34f5,
    0x3515, 0x3535, 0x3555, 0x3575, 0x3596, 0x35b6, 0x35d6, 0x35f6,
    0x3616, 0x3636, 0x3656, 0x3676, 0x3697, 0x36b7, 0x36d7, 0x36f7,
    0x3717, 0x3737, 0x3757, 0x3777, 0x3798, 0x37b8, 0x37d8, 0x37f8,
    0x380c, 0x381c, 0x382c, 0x383c, 0x384c, 0x385c, 0x386c, 0x387c,
    0x388d, 0x389d, 0x38ad, 0x38bd, 0x38cd, 0x38dd, 0x38ed, 0x38fd,
    0x390d, 0x391d, 0x392d, 0x393d, 0x394d, 0x395d, 0x396d, 0x397d,
    0x398e, 0x399e, 0x39ae, 0x39be, 0x39ce, 0x39de, 0x39ee, 0x39fe,
    0x3a0e, 0x3a1e, 0x3a2e, 0x3a3e, 0x3a4e, 0x3a5e, 0x3a6e, 0x3a7e,
    0x3a8f, 0x3a9f, 0x3aaf, 0x3abf, 0x3acf, 0x3adf, 0x3aef, 0x3aff,
    0x3b0f, 0x3b1f, 0x3b2f, 0x3b3f, 0x3b4f, 0x3b5f, 0x3b6f, 0x3b7f,
    0x3b90, 0x3ba0, 0x3bb0, 0x3bc0, 0x3bd0, 0x3be0, 0x3bf0, 0x3c00
};

__device__ __forceinline__ uint16_t normalize_u8_to_half_bits(uint8_t x_u8)
{
    return kU8NormFp16Bits[x_u8];
}

__global__ void normalize_uint8_to_fp16_kernel(
    const uint8_t* __restrict__ input,
    half* __restrict__ output,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    reinterpret_cast<uint16_t*>(output)[idx] =
        normalize_u8_to_half_bits(input[idx]);
}

void normalize_uint8_to_fp16(const uint8_t* input, half* output, int numel,
                             cudaStream_t stream)
{
    if (numel <= 0) return;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    normalize_uint8_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
        input, output, numel);
}

__global__ void normalize_uint8_to_patches_fp16_kernel(
    const uint8_t* __restrict__ input,
    half* __restrict__ output,
    int nv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nv * 256 * 588;
    if (idx >= total) return;

    int patch_idx = idx / 588;
    int feat_idx = idx % 588;

    int batch = patch_idx / 256;
    int local_patch = patch_idx % 256;
    int ph = local_patch / 16;
    int pw = local_patch % 16;

    int pxh = feat_idx / 42;
    int pxw = (feat_idx % 42) / 3;
    int c = feat_idx % 3;

    int row = ph * 14 + pxh;
    int col = pw * 14 + pxw;
    int src = batch * (224 * 224 * 3) + row * (224 * 3) + col * 3 + c;

    reinterpret_cast<uint16_t*>(output)[idx] =
        normalize_u8_to_half_bits(input[src]);
}

void normalize_uint8_to_patches_fp16(const uint8_t* input, half* output, int nv,
                                     cudaStream_t stream)
{
    int total = nv * 256 * 588;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    normalize_uint8_to_patches_fp16_kernel<<<blocks, threads, 0, stream>>>(
        input, output, nv);
}

// ── GPU im2col for SigLIP patch embedding ──
// Input:  (nv, 224, 224, 3) FP16, row-major NHWC
// Output: (nv*256, 588) FP16, each row = one 14×14×3 patch flattened
//
// Equivalent to:
//   img.reshape(nv, 16, 14, 16, 14, 3)
//      .transpose(0, 1, 3, 2, 4, 5)
//      .reshape(nv*256, 588)
__global__ void patch_im2col_kernel(
    const half* __restrict__ input,   // (nv, 224, 224, 3)
    half* __restrict__ output,        // (nv*256, 588)
    int nv)
{
    // Total output elements = nv * 256 * 588
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nv * 256 * 588;
    if (idx >= total) return;

    // Decode output index
    int patch_idx = idx / 588;         // which patch [0, nv*256)
    int feat_idx  = idx % 588;         // which feature [0, 588)

    int batch = patch_idx / 256;       // which view
    int local_patch = patch_idx % 256; // patch within view
    int ph = local_patch / 16;         // patch row [0, 16)
    int pw = local_patch % 16;         // patch col [0, 16)

    int pxh = feat_idx / 42;           // pixel row within patch [0, 14), 42=14*3
    int pxw = (feat_idx % 42) / 3;    // pixel col within patch [0, 14)
    int c   = feat_idx % 3;            // channel [0, 3)

    // Source index in (nv, 224, 224, 3) row-major
    int row = ph * 14 + pxh;
    int col = pw * 14 + pxw;
    int src = batch * (224 * 224 * 3) + row * (224 * 3) + col * 3 + c;

    output[idx] = input[src];
}

void patch_im2col(const half* input, half* output, int nv, cudaStream_t stream)
{
    int total = nv * 256 * 588;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    patch_im2col_kernel<<<blocks, threads, 0, stream>>>(input, output, nv);
}

__global__ void patch_im2col_uint8_kernel(
    const uint8_t* __restrict__ input,
    const half* __restrict__ lut,
    half* __restrict__ output,
    int nv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nv * 256 * 588;
    if (idx >= total) return;

    int patch_idx = idx / 588;
    int feat_idx = idx % 588;
    int batch = patch_idx / 256;
    int local_patch = patch_idx % 256;
    int ph = local_patch / 16;
    int pw = local_patch % 16;
    int pxh = feat_idx / 42;
    int pxw = (feat_idx % 42) / 3;
    int c = feat_idx % 3;
    int row = ph * 14 + pxh;
    int col = pw * 14 + pxw;
    int src = batch * (224 * 224 * 3) + row * (224 * 3) + col * 3 + c;

    output[idx] = lut[input[src]];
}

void patch_im2col_uint8(const uint8_t* input, const half* lut, half* output,
                        int nv, cudaStream_t stream)
{
    int total = nv * 256 * 588;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    patch_im2col_uint8_kernel<<<blocks, threads, 0, stream>>>(
        input, lut, output, nv);
}

// ── Bias + positional embedding ──

__global__ void patch_embed_bias_pos_kernel(
    half* __restrict__ output,
    const half* __restrict__ bias,
    const half* __restrict__ pos_emb,
    int S, int D, int S_per_view)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * D;
    if (idx >= total) return;

    int i = idx / D;
    int j = idx % D;
    int pos_i = i % S_per_view;

    float v = __half2float(output[idx])
            + __half2float(bias[j])
            + __half2float(pos_emb[pos_i * D + j]);
    output[idx] = __float2half(v);
}

void patch_embed_bias_pos(half* output, const half* bias, const half* pos_emb,
                          int S, int D, int S_per_view, cudaStream_t stream)
{
    int total = S * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    patch_embed_bias_pos_kernel<<<blocks, threads, 0, stream>>>(
        output, bias, pos_emb, S, D, S_per_view);
}
