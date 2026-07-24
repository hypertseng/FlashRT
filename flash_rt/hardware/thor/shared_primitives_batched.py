"""FlashRT — Thor SM110 model-agnostic primitives, B>=1 batched variants.

Companion to :mod:`flash_rt.hardware.thor.shared_primitives` which
holds the B=1 hot path (the production single-sample inference).
This module isolates the B>=1 batched kernel orchestrations so the
B=1 file stays small and easy to reason about — the non-batched
single-sample inference is the main-line product, the batched path
is opt-in (used by the fused-CFG B=2 pipeline and future RL-rollout
B>2 paths).

Mirrors the model-layer split between
:mod:`flash_rt.models.pi05.pipeline_thor` (B=1) and
:mod:`flash_rt.models.pi05.pipeline_thor_batched` (B>=1), and the
RTX layout's cfg / cfg_batched split.

Functions:
    encoder_forward_b2     — Paligemma encoder forward at B>=1
    siglip_forward_batched — SigLIP forward at B>=1 (GEMMs see M=B*S)
    postln_project_batched — PostLN + projection + per-sample lang concat
"""

import ctypes
import math
import os

_crt = ctypes.CDLL('libcudart.so')


def _select_encoder_down_gemm(fvk, B: int):
    """Choose the encoder FFN down-projection tactic for batched Thor runs."""
    tactic = os.environ.get(
        "FLASHRT_THOR_BATCH_ENCODER_DOWN_TACTIC", "auto").strip().lower()
    if tactic in ("", "auto"):
        tactic = "t2" if B >= 4 and hasattr(fvk, "cutlass_fp8_t2") else "wide"

    table = {
        "wide": fvk.cutlass_fp8_wide,
        "sq": fvk.cutlass_fp8_sq,
        "t1": fvk.cutlass_fp8_t1,
    }
    optional = {
        "t2": "cutlass_fp8_t2",
        "plain": "cutlass_fp8_plain",
    }
    if tactic in optional:
        fn = getattr(fvk, optional[tactic], None)
        if fn is not None:
            return fn
        tactic = "wide"
    return table.get(tactic, fvk.cutlass_fp8_wide)


def encoder_forward_b2(gemm, fvk, bufs, weights, dims, stream=0, *,
                       attn=None, B=2):
    """Batched encoder forward for ``B`` independent samples.

    Stage 2 of the Thor batched-CFG port. The kernel inventory split is:

      * **Flat-elementwise** kernels (``rms_norm_fp8_noweight_fp16``,
        ``residual_add_rms_norm_fp8_noweight_fp16``, ``quantize_fp8_static_fp16``,
        ``gate_geglu_merged_fp8_fp16``) auto-scale to ``M = B*Se``
        — they consume / produce a flat ``[B*Se, D]`` buffer.
      * **GEMMs** (``cutlass_fp8_sq``, ``cutlass_fp8_t1``,
        ``cutlass_fp8_wide``) likewise scale via their first dim arg
        (``M = B*Se``).
      * **Per-token-indexed ops** use batch-aware kernels when present:
        ``qkv_split_rope_kvcache_fp16_batched`` maps the sample axis to
        ``grid.y`` while preserving the original per-token RoPE math.
        Attention GEMMs are submitted through cuBLAS strided-batched
        wrappers when available. Fallbacks keep the exact same
        per-sample calls.

    Buffer contract (``B`` is the leading axis or fold):

      bufs: same key set as
        :func:`flash_rt.hardware.thor.shared_primitives.encoder_forward`;
        shapes are flat ``B*Se`` along the row dim. The frontend
        allocates a fresh set of ``_b2``-suffixed buffers and hands
        them in here.

      weights: same as ``encoder_forward`` plus ``Kc_b2`` / ``Vc_b2``
        — lists of length ``B``, each entry a device pointer to that
        sample's ``[La * total_keys * HD]`` flat KV slab. ``Kc`` /
        ``Vc`` (the B=1 keys) are ignored.

      dims: same as ``encoder_forward``. ``Se`` is per-sample
        sequence length (NOT B*Se).

    Args:
        attn: Optional :class:`flash_rt.hardware.thor.attn_backend.ThorFlashAttnBackend`.
            **Not used in Stage 2** — the backend's encoder slot is
            single-batch; Stage 2 calls ``fvk.attention_qkv_fp16``
            directly per-sample. A future Stage extends the backend
            to handle B>1.
    """
    Se = dims['Se']
    D = dims['D']
    H = dims['H']
    NH = dims['NH']
    HD = dims['HD']
    L = dims['L']
    total_keys = dims['total_keys']
    Q_dim = NH * HD
    K_dim = HD
    attn_scale = 1.0 / math.sqrt(float(HD))
    BSe = B * Se

    x = bufs['x']
    x_fp8 = bufs['x_fp8']
    qkv = bufs['qkv']
    logits = bufs['logits']
    attn_out = bufs['attn_out']
    o_fp8 = bufs['o_fp8']
    gate = bufs['gate']
    hid_fp8 = bufs['hid_fp8']
    fg = bufs['fg']

    act_scales = weights['act_scales']
    alpha_host = weights['alpha_host']

    # Per-sample KV slab device pointers (one per b, one ptr per slab).
    Kc_b2 = weights['Kc_b2']
    Vc_b2 = weights['Vc_b2']
    if len(Kc_b2) != B or len(Vc_b2) != B:
        raise ValueError(
            f"Kc_b2/Vc_b2 must each have B={B} entries; "
            f"got {len(Kc_b2)} / {len(Vc_b2)}")

    # Byte strides for the inline per-sample loop. fp16 = 2 bytes.
    qkv_stride_bytes = Se * 2560 * 2
    attn_q_stride_bytes = Se * Q_dim * 2
    qkv_stride_elems = Se * 2560
    q_stride_elems = Se * Q_dim
    kv_batch_stride_elems = ((Kc_b2[1] - Kc_b2[0]) // 2
                             if B > 1 else L * total_keys * HD)
    vc_batch_stride_elems = ((Vc_b2[1] - Vc_b2[0]) // 2
                             if B > 1 else L * total_keys * HD)
    qkv_split_batched = getattr(
        fvk, 'qkv_split_rope_kvcache_fp16_batched', None)
    qk_batched = getattr(
        fvk, 'attention_qk_gemm_fp16_strided_batched', None)
    pv_batched = getattr(
        fvk, 'attention_pv_gemm_fp16_strided_batched', None)
    gateup_gemm = fvk.cutlass_fp8_sq if B >= 2 else fvk.cutlass_fp8_t1
    down_gemm = _select_encoder_down_gemm(fvk, B)

    # ── Layer 0, step 1: RMSNorm → FP8 (first layer, no preceding residual add) ──
    as_qkv_0 = act_scales + (0 * 4 + 0) * 4
    fvk.rms_norm_fp8_noweight_fp16(x, x_fp8, BSe, D, as_qkv_0, stream)

    for l in range(L):
        last = (l == L - 1)

        as_o   = act_scales + (l * 4 + 1) * 4
        as_gu  = act_scales + (l * 4 + 2) * 4
        as_d   = act_scales + (l * 4 + 3) * 4

        # ── 1. RMSNorm → FP8 already done: ──
        #   Layer 0: done above (no preceding residual add)
        #   Layers 1..L-1: fused with step 11 of previous layer

        # ── 2. QKV GEMM (M = B*Se, output (B*Se, 2560)) ──
        fvk.cutlass_fp8_sq(x_fp8, weights['qkv_w'][l], qkv,
                           BSe, 2560, D, alpha_host[l * 4 + 0], 0.0, stream)

        # ── 3+4. QKV split + RoPE + KV cache write ──
        kv_elem_off = l * total_keys * HD
        if qkv_split_batched is not None and B > 1:
            qkv_split_batched(
                qkv, weights['rope'], attn_out, Kc_b2[0], Vc_b2[0],
                Se, Q_dim, K_dim, HD, 2560,
                kv_elem_off, HD,
                qkv_stride_elems, q_stride_elems,
                kv_batch_stride_elems, vc_batch_stride_elems,
                B, stream)
        else:
            for b in range(B):
                fvk.qkv_split_rope_kvcache_fp16(
                    qkv + b * qkv_stride_bytes,
                    weights['rope'],
                    attn_out + b * attn_q_stride_bytes,
                    Kc_b2[b], Vc_b2[b],
                    Se, Q_dim, K_dim, HD, 2560,
                    kv_elem_off, HD, stream)

        if not last:
            # ── 5. Batched attention (decomposed QK^T + softmax + PV) ──
            # All samples share the same Se (self-attention), so batch
            # the softmax across B*Se*NH rows in a single kernel call.
            qk_elem_off = Se * NH * Se  # element offset per sample in logits
            qk_stride_bytes = qk_elem_off * 2
            K_base = Kc_b2[0] + kv_elem_off * 2
            V_base = Vc_b2[0] + kv_elem_off * 2
            if qk_batched is not None and pv_batched is not None and B > 1:
                qk_batched(
                    bufs['ctx'],
                    attn_out,
                    K_base,
                    logits,
                    Se, Se, NH, HD,
                    Se,
                    q_stride_elems,
                    kv_batch_stride_elems,
                    qk_elem_off,
                    B,
                    attn_scale,
                    stream)
                fvk.softmax_fp16(logits, B * Se * NH, Se, stream)
                pv_batched(
                    bufs['ctx'],
                    V_base,
                    logits,
                    attn_out,
                    Se, Se, NH, HD,
                    Se,
                    kv_batch_stride_elems,
                    qk_elem_off,
                    q_stride_elems,
                    B,
                    stream)
            else:
                # Current Thor SM110 builds may omit the legacy decomposed
                # QK/PV helper symbols. Fall back to the fused per-sample
                # attention kernel that is already used by the B=1 path.
                for b in range(B):
                    K_ptr = Kc_b2[b] + kv_elem_off * 2
                    V_ptr = Vc_b2[b] + kv_elem_off * 2
                    fvk.attention_qkv_fp16(
                        bufs['ctx'],
                        attn_out + b * attn_q_stride_bytes,
                        K_ptr,
                        V_ptr,
                        logits + b * qk_stride_bytes,
                        attn_out + b * attn_q_stride_bytes,
                        Se, Se, NH, HD,
                        attn_scale, stream)

            # ── 6. Quantize attn → FP8 + O proj GEMM with residual fusion ──
            # Writes directly to x: x = alpha * (o_fp8 @ o_w) + x.
            fvk.quantize_fp8_static_fp16(attn_out, o_fp8, as_o, BSe * D, stream)
            fvk.cutlass_fp8_sq(o_fp8, weights['o_w'][l], x,
                               BSe, D, D, alpha_host[l * 4 + 1], 1.0, stream)

            # ── 7. RMSNorm → FP8 (residual already fused into O GEMM) ──
            fvk.rms_norm_fp8_noweight_fp16(x, x_fp8, BSe, D, as_gu, stream)

            # ── 8. Gate+Up merged GEMM (M=B*Se) ──
            gateup_gemm(x_fp8, weights['gate_w'][l], gate,
                        BSe, H * 2, D, alpha_host[l * 4 + 2], 0.0, stream)

            # ── 9. GELU(gate) × up → FP8 (flat, M=B*Se*H) ──
            fvk.gate_geglu_merged_fp8_fp16(gate, hid_fp8, BSe, H,
                                               as_d, stream)

            # ── 10. Down GEMM with residual fusion (beta=1.0, M=B*Se) ──
            # Writes directly to x: x = alpha * (hid_fp8 @ down_w) + x.
            # CUTLASS epilogue aliases C=D, so beta=1.0 folds the FFN
            # residual add into the GEMM, eliminating a separate
            # residual_add_fp16 kernel + one B*Se*D fp16 memory pass.
            down_gemm(hid_fp8, weights['down_w'][l], x,
                      BSe, D, H, alpha_host[l * 4 + 3], 1.0, stream)

            # ── 11. RMSNorm → FP8 (residual already fused into Down GEMM) ──
            as_next = act_scales + ((l + 1) * 4 + 0) * 4
            fvk.rms_norm_fp8_noweight_fp16(x, x_fp8, BSe, D, as_next, stream)

    # x[B*Se, D] now contains the encoder output for both samples,
    # contiguous: rows [0:Se] = sample 0, rows [Se:2*Se] = sample 1.


def siglip_forward_batched(gemm, fvk, bufs, weights, dims, stream=0, *,
                           attn=None, use_fp8=True):
    """Batched SigLIP forward: B samples processed as one big sequence.

    Identical to siglip_forward but treats the input as B independent
    samples, each with S_sig tokens.  GEMMs see M = B * S_sig.
    FMHA batch = B * num_views (each view attends independently).

    Args:
        bufs: same layout as siglip_forward but sized for B*S tokens
        dims: must include 'B' (batch size) in addition to standard keys
    """
    S_sig = dims['S']
    D = dims['D']
    H = dims['H']
    NH = dims['NH']
    HD = dims['HD']
    L = dims['L']
    nv = dims['num_views']
    spv = dims['seq_per_view']
    B = dims.get('B', 1)

    S = B * S_sig  # total tokens for all GEMMs

    x = bufs['x']
    x_fp8 = bufs['x_fp8']
    qkv = bufs['qkv']
    attn_out = bufs['attn_out']
    hidden = bufs['hidden']
    hid_fp8 = bufs['hid_fp8']

    alpha = weights['alpha']

    for l in range(L):
        a_qkv = alpha[l * 4 + 0]
        a_o = alpha[l * 4 + 1]
        a_up = alpha[l * 4 + 2]
        a_down = alpha[l * 4 + 3]

        # LayerNorm → FP8 (on B*S tokens)
        fvk.layer_norm_fp8(x, x_fp8, weights['ln_attn_w'][l],
                           weights['ln_attn_b'][l], S, D, 1e-6, stream)

        # QKV GEMM: [B*S, D] @ [D, 3D] → [B*S, 3D]
        gemm.fp8_nn_bias(x_fp8, weights['qkv_w'][l], qkv,
                         weights['qkv_b'][l], S, 3 * D, D, a_qkv, stream)

        # FMHA: B*nv independent views, each spv tokens
        # NOTE: attn.run() hardcodes nv = site_spec.batch_axis (num_views),
        # but we need B * nv.  Call fvk directly with the correct batch.
        stride = 3 * D
        fvk.fmha_strided_full(qkv, qkv + D * 2, qkv + 2 * D * 2,
                              attn_out, B * nv, spv, spv, NH, NH, HD,
                              stride, stride, stream)

        # Cast attn output → FP8
        fvk.quantize_fp8_static_fp16(attn_out, x_fp8, weights['unit_scale'],
                                     S * D, stream)

        # O projection + residual: [B*S, D]
        gemm.fp8_nn_bias_res(x_fp8, weights['o_w'][l], x, weights['o_b'][l],
                             S, D, D, a_o, stream)

        # FFN LayerNorm → FP8
        fvk.layer_norm_fp8(x, x_fp8, weights['ln_ffn_w'][l],
                           weights['ln_ffn_b'][l], S, D, 1e-6, stream)

        # Up GEMM + GELU: [B*S, D] @ [D, H] → [B*S, H]
        gemm.fp8_nn_gelu_bias(x_fp8, weights['up_w'][l], hidden,
                              weights['up_b'][l], S, H, D, a_up, stream)

        # Cast hidden → FP8
        fvk.quantize_fp8_static_fp16(hidden, hid_fp8, weights['unit_scale'],
                                     S * H, stream)

        # Down GEMM + residual: [B*S, H] @ [H, D] → [B*S, D]
        gemm.fp8_nn_bias_res(hid_fp8, weights['down_w'][l], x,
                             weights['down_b'][l], S, D, H, a_down, stream)


def postln_project_batched(gemm, fvk, bufs, weights, dims, stream=0):
    """Batched PostLN + projection + per-sample language concat.

    Processes B*S_sig vision tokens, projects each sample's tokens to
    D_enc, then copies per-sample language embeddings into the
    corresponding slot in enc_x_b2.

    Args:
        bufs: x_sig (B*S_sig, D_sig), enc_x_b2 (B*Se, D_enc),
              scratch (B*S_sig, D_sig)
        weights: ln_w, ln_b, proj_w, proj_b,
                 lang_emb_list: list of B lang_emb pointers
        dims: S_sig, D_sig, D_enc, S_lang, B
    """
    S_sig = dims['S_sig']
    D_sig = dims['D_sig']
    D_enc = dims['D_enc']
    S_lang = dims['S_lang']
    B = dims['B']
    Se = S_sig + S_lang  # per-sample encoder sequence length

    x_sig = bufs['x_sig']
    enc_x_b2 = bufs['enc_x_b2']
    scratch = bufs['scratch']
    proj = bufs.get('proj', 0)

    # LayerNorm on all B*S_sig vision tokens at once
    fvk.layer_norm_fp16(x_sig, weights['ln_w'], weights['ln_b'], scratch,
                        B * S_sig, D_sig, 1e-6, stream)

    if proj:
        # Projection: [B*S_sig, D_sig] -> [B*S_sig, D_enc] in one GEMM.
        gemm.fp16_nn(scratch, weights['proj_w'], proj,
                     B * S_sig, D_enc, D_sig, stream)
        fvk.add_bias_fp16(proj, weights['proj_b'], B * S_sig, D_enc, stream)
        for b in range(B):
            src = proj + b * S_sig * D_enc * 2
            dst = enc_x_b2 + b * Se * D_enc * 2
            fvk.gpu_copy(dst, src, S_sig * D_enc * 2, stream)
    else:
        # Compatibility path: write each sample's projected tokens directly.
        for b in range(B):
            src = scratch + b * S_sig * D_sig * 2  # byte offset (fp16)
            dst = enc_x_b2 + b * Se * D_enc * 2
            gemm.fp16_nn(src, weights['proj_w'], dst,
                         S_sig, D_enc, D_sig, stream)
            fvk.add_bias_fp16(dst, weights['proj_b'], S_sig, D_enc, stream)

    # Copy per-sample language embeddings
    lang_emb_list = weights['lang_emb_list']
    for b in range(B):
        nbytes = S_lang * D_enc * 2
        dst = enc_x_b2 + (b * Se + S_sig) * D_enc * 2
        _crt.cudaMemcpyAsync(ctypes.c_void_p(dst),
                             ctypes.c_void_p(lang_emb_list[b]),
                             ctypes.c_size_t(nbytes), 3,
                             ctypes.c_void_p(stream))
