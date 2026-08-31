// ================================================================
// FlashRT — Patch embedding kernel declarations
// GPU im2col + fused bias + positional embedding
// ================================================================
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Exact uint8 image normalization:
//   ((float32)x / 127.5f - 1.0f) cast to fp16
// Matches the original numpy preprocessing path byte-for-byte.
void normalize_uint8_to_fp16(const uint8_t* input, half* output, int numel,
                             cudaStream_t stream = 0);

// Exact uint8 normalization fused with the patch im2col layout transform.
// Input:  (nv, 224, 224, 3) uint8, row-major NHWC
// Output: (nv*256, 588) fp16, matching normalize_uint8_to_fp16 + patch_im2col.
void normalize_uint8_to_patches_fp16(const uint8_t* input, half* output, int nv,
                                     cudaStream_t stream = 0);

// GPU im2col: (nv, 224, 224, 3) → (nv*256, 588)
// Pure strided copy, bit-exact, no computation.
void patch_im2col(const half* input, half* output, int nv,
                  cudaStream_t stream = 0);

// GPU im2col with exact uint8 -> FP16 normalization through a 256-entry LUT.
void patch_im2col_uint8(const uint8_t* input, const half* lut, half* output,
                        int nv, cudaStream_t stream = 0);

// Add bias + positional embedding to patch GEMM output (FP16)
// output[i,j] += bias[j] + pos_emb[i % S_per_view, j]
void patch_embed_bias_pos(half* output, const half* bias, const half* pos_emb,
                          int S, int D, int S_per_view,
                          cudaStream_t stream = 0);
