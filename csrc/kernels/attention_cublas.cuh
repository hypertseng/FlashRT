// ================================================================
// FlashRT — cuBLAS decomposed attention (GQA-compatible)
// QK^T + softmax + PV, matching pi05 engine exactly.
// Stateless: receives cuBLAS handle from caller (FvkContext).
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// Full attention: Q @ K^T → softmax → @ V → out
// Supports GQA: K/V have 1 head, Q has NH heads.
void attention_qkv_fp16(
    cublasHandle_t handle,   // caller's cuBLAS handle (from FvkContext)
    const __half* Q,         // (S, NH*HD) = (S*NH, HD) contiguous
    const __half* K,         // (S_kv, HD) single KV head
    const __half* V,         // (S_kv, HD)
    __half* logits,          // (S*NH, S_kv) scratch buffer
    __half* out,             // (S*NH, HD) = (S, NH, HD) output
    int S, int S_kv, int NH, int HD,
    float attn_scale,        // 1/sqrt(HD)
    cudaStream_t stream = 0);

// Decomposed QK^T GEMM only (no softmax, no PV).
// logits_ldc is the leading dimension of the column-major logits layout.
void attention_qk_gemm_fp16(
    cublasHandle_t handle,
    const __half* Q, const __half* K, __half* logits,
    int S, int S_kv, int NH, int HD,
    int logits_ldc, float attn_scale,
    cudaStream_t stream = 0);

// Batched QK^T GEMM for independent samples with identical shapes.
// Strides are in fp16 elements, not bytes.
void attention_qk_gemm_fp16_strided_batched(
    cublasHandle_t handle,
    const __half* Q, const __half* K, __half* logits,
    int S, int S_kv, int NH, int HD,
    int logits_ldc,
    long long q_batch_stride,
    long long k_batch_stride,
    long long logits_batch_stride,
    int batch_count,
    float attn_scale,
    cudaStream_t stream = 0);

// Decomposed PV GEMM only (reads pre-computed + softmaxed logits).
void attention_pv_gemm_fp16(
    cublasHandle_t handle,
    const __half* V, const __half* logits, __half* out,
    int S, int S_kv, int NH, int HD,
    int logits_ldc,
    cudaStream_t stream = 0);

// Batched PV GEMM for independent samples with identical shapes.
// Strides are in fp16 elements, not bytes.
void attention_pv_gemm_fp16_strided_batched(
    cublasHandle_t handle,
    const __half* V, const __half* logits, __half* out,
    int S, int S_kv, int NH, int HD,
    int logits_ldc,
    long long v_batch_stride,
    long long logits_batch_stride,
    long long out_batch_stride,
    int batch_count,
    cudaStream_t stream = 0);

// Fixed-shape attention with a device-side valid K/V length.
// QK and PV keep the graph-captured S_kv_max shape; logits rows
// [seqused_k[0], S_kv_max) are masked before softmax.
void attention_qkv_fp16_seqused(
    cublasHandle_t handle,
    const __half* Q,         // (S*NH, HD)
    const __half* K,         // (S_kv_max, HD)
    const __half* V,         // (S_kv_max, HD)
    __half* logits,          // scratch: (S*NH, S_kv_max)
    __half* out,             // (S*NH, HD)
    int S, int S_kv_max, int NH, int HD,
    const int* seqused_k,    // device int32[1], valid K/V rows
    float attn_scale,
    cudaStream_t stream = 0);

// Same as attention_qkv_fp16 but supports ODD S_kv.
// Internally pads logits leading dimension to even for __half2 alignment.
// logits buffer must have room for S*NH * (S_kv+1) elements when S_kv is odd.
void attention_qkv_fp16_padded(
    cublasHandle_t handle,
    const __half* Q,         // (S*NH, HD)
    const __half* K,         // (S_kv, HD)
    const __half* V,         // (S_kv, HD)
    __half* logits,          // scratch: (S*NH, S_kv_padded) where padded = S_kv rounded up to even
    __half* out,             // (S*NH, HD)
    int S, int S_kv, int NH, int HD,
    float attn_scale,
    cudaStream_t stream = 0);

// Single-call attention with state token masking for Pi0.
// State token (first 1 query) can only attend to the first `state_nk` keys.
// Remaining keys are masked with -inf before softmax.
// This replaces the split attention (2 calls) with 1 call + 1 mask kernel.
// Handles odd S_kv via padded lda (same as attention_qkv_fp16_padded).
void attention_qkv_fp16_state_masked(
    cublasHandle_t handle,
    const __half* Q,         // (S*NH, HD)
    const __half* K,         // (S_kv, HD)
    const __half* V,         // (S_kv, HD)
    __half* logits,          // scratch: (S*NH, S_kv_padded)
    __half* out,             // (S*NH, HD)
    int S, int S_kv, int NH, int HD,
    int state_nk,            // number of keys visible to state token (typically enc_seq+1)
    float attn_scale,
    cudaStream_t stream = 0);
