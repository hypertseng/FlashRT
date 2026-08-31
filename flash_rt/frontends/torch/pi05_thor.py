"""FlashRT -- Pi05TorchFrontendThor: complete inference using ONLY flash_rt_kernels.so.

All computation goes through flash_rt_kernels (pybind11 + CUTLASS + cuBLASLt).

Usage:
    pipe = Pi05TorchFrontendThor("/path/to/checkpoint", num_views=2)
    pipe.set_prompt("pick up the red cup")
    result = pipe.infer({"image": img1, "wrist_image": img2})
    actions = result["actions"]  # (10, 7) numpy
"""

import json
import math
import logging
import os
import pathlib
import time
from typing import Optional, Union

from flash_rt.hardware.thor.shared_primitives import (
    siglip_forward,
    postln_project,
    encoder_forward,
    encoder_forward_calibrate,
)
from flash_rt.models.pi05.pipeline_thor import (
    decoder_forward,
    decoder_forward_calibrate,
)
from flash_rt.hardware.thor.attn_backend import (
    ThorFlashAttnBackend,
    make_pi05_attention_spec,
)

import numpy as np
import torch
import torch.nn.functional as F

import flash_rt.flash_rt_kernels as fvk
from flash_rt.core.cuda_buffer import CudaBuffer
from flash_rt.core.utils.actions import unnormalize_actions, LIBERO_ACTION_DIM
from flash_rt.core.utils.pi05_prompt import PI05_STATE_PROMPT_MAX_LEN
from flash_rt.core.quant.calibrator import load_calibration, save_calibration

logger = logging.getLogger(__name__)

fp16 = torch.float16
fp8 = torch.float8_e4m3fn

_CALIBRATION_CACHE_METADATA = {
    "pi05_thor_action_update": "fp32_bias_unscaled_v1",
}


def _tensor_debug_stats(name, tensor):
    arr = tensor.detach().float().reshape(-1).cpu().numpy()
    finite = np.isfinite(arr)
    return {
        f"{name}_size": int(arr.size),
        f"{name}_finite": int(finite.all()) if arr.size else 1,
        f"{name}_nan": int(np.isnan(arr).sum()) if arr.size else 0,
        f"{name}_inf": int(np.isinf(arr).sum()) if arr.size else 0,
        f"{name}_zero": int((arr == 0).sum()) if arr.size else 0,
        f"{name}_min": float(np.nanmin(arr)) if arr.size else 0.0,
        f"{name}_max": float(np.nanmax(arr)) if arr.size else 0.0,
        f"{name}_mean": float(np.nanmean(arr)) if arr.size else 0.0,
    }


def _array_debug_stats(name, arr):
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    finite = np.isfinite(arr)
    return {
        f"{name}_size": int(arr.size),
        f"{name}_finite": int(finite.all()) if arr.size else 1,
        f"{name}_nan": int(np.isnan(arr).sum()) if arr.size else 0,
        f"{name}_inf": int(np.isinf(arr).sum()) if arr.size else 0,
        f"{name}_min": float(np.nanmin(arr)) if arr.size else 0.0,
        f"{name}_max": float(np.nanmax(arr)) if arr.size else 0.0,
        f"{name}_mean": float(np.nanmean(arr)) if arr.size else 0.0,
        f"{name}_first8": [float(x) for x in arr[:8]],
    }


def _log_tensor_debug_stats(label, tensor):
    try:
        logger.info("FlashRT %s stats: %s", label, _tensor_debug_stats(label, tensor))
    except Exception as exc:
        logger.warning("Failed to collect FlashRT %s stats: %s", label, exc)


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float env %s=%r; using %.3f", name, value, default)
        return float(default)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from flash_rt.core.thor_frontend_utils import embed_prompt_torch as embed_prompt  # noqa: E402


# ---------------------------------------------------------------------------
# Calibration functions (replaces old C encoder_full_scaled / ae_forward_scaled)
# ---------------------------------------------------------------------------

# ===========================================================================
# ThorPipeline
# ===========================================================================

class Pi05TorchFrontendThor:
    """Complete Pi0.5 inference pipeline using only flash_rt_kernels.so.

    Interface compatible with FlashRTModel.predict():
        set_prompt(prompt_text)
        infer(observation) -> {"actions": np.ndarray}
    """

    _BATCH_GRAPH_BANK_ATTRS = (
        "_Kc_b2", "_Vc_b2",
        "_sig_x_b2", "_sig_x_fp8_b2", "_sig_qkv_b2", "_sig_attn_b2",
        "_sig_hidden_b2", "_sig_hid_fp8_b2", "_postln_scratch_b2",
        "_postln_proj_b2",
        "_img_buf_b2_list", "_img_buf_b2_all", "_patches_buf_b2",
        "_img_staging_torch",
        "_img_stack_host_b2", "_img_stack_host_torch_b2",
        "_img_stack_fp16_host_b2", "_img_stack_fp16_host_torch_b2",
        "_lang_emb_b2",
        "_enc_x_b2", "_enc_x_fp8_b2", "_enc_qkv_buf_b2",
        "_enc_logits_b2", "_enc_attn_b2", "_enc_o_fp8_b2",
        "_enc_gate_b2", "_enc_hid_fp8_b2", "_enc_fg_b2",
        "_ae_x_b2", "_ae_xn_b2", "_ae_gate_b2", "_ae_qkv_b2",
        "_ae_logits_b2", "_ae_attn_b2", "_ae_fg_b2",
        "_ae_action_f32_b2", "_ae_xn_fp8_b2", "_ae_hid_fp8_b2",
        "_ae_ctx_fp8_b2",
        "_g_noise_b2", "_v_b2", "_v_b2_f32",
        "_sa_all_b2", "_sf_all_b2", "_fs_all_b2",
        "_siglip_batched_graph", "_siglip_batched_B",
        "_enc_ae_graph_b2", "_cfg_b2_outer_graph",
        "_stream_attn_b2",
    )

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(self, checkpoint_dir: str, num_views: int = 2,
                 use_cuda_graph: bool = True, autotune: int = 3,
                 use_fp8: bool = True, state_prompt_mode: str = "exact",
                 state_prompt_fixed_max_len: Optional[int] = None,
                 use_fa4: bool = False, rtc_prefix_len: int = 0):
        """
        Args:
            autotune: CUDA Graph autotune trials per set_prompt().
                0 = off, 3 = default (fast), 5+ = thorough.
                Torch usually finds fast graph on trial 0-1.
        """
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.num_views = num_views
        self.use_cuda_graph = use_cuda_graph
        self.use_fp8 = bool(use_fp8)
        self.use_fa4 = bool(use_fa4)
        self._rtc_prefix_len = max(0, int(rtc_prefix_len))
        if self._rtc_prefix_len > 10:
            raise ValueError("rtc_prefix_len must be <= action horizon 10")
        self._rtc_prev_action_chunk = None
        self.autotune = int(autotune) if autotune is not True else 3
        if autotune is False:
            self.autotune = 0
        self.batched_graph_autotune = (
            os.environ.get("FLASHRT_THOR_BATCH_GRAPH_AUTOTUNE", "0") == "1"
        )
        self.batched_siglip_graph_autotune = (
            os.environ.get("FLASHRT_THOR_BATCH_SIGLIP_GRAPH_AUTOTUNE", "0") == "1"
        )
        self.batched_postln_proj_fusion = (
            os.environ.get("FLASHRT_THOR_BATCH_POSTLN_PROJ_FUSION", "0") == "1"
        )
        self.batched_skip_kv_zero = (
            os.environ.get("FLASHRT_THOR_BATCH_SKIP_KV_ZERO", "1") != "0"
        )
        self.batched_sync_after_graph = (
            os.environ.get("FLASHRT_THOR_BATCH_SYNC_AFTER_GRAPH", "1") != "0"
        )
        self.batch_graph_bank_enabled = (
            os.environ.get("FLASHRT_THOR_BATCH_GRAPH_BANK", "0") == "1"
        )
        self._batch_graph_bank = {}
        self._batch_graph_bank_active_key = None
        self.prompt_bank_enabled = (
            os.environ.get("FLASHRT_THOR_PROMPT_BANK", "0") == "1"
        )
        self._prompt_bank = {}
        self._prompt_bank_hits = 0
        self._prompt_bank_misses = 0
        self.noise_bank_enabled = (
            os.environ.get("FLASHRT_THOR_NOISE_BANK", "0") == "1"
        )
        self._noise_bank = {}
        self._noise_bank_hits = 0
        self._noise_bank_misses = 0
        self.latency_records = []
        self.calibrated = False
        self.graph_captured = False
        self._real_data_calibrated = False

        # ---- RL CFG state (set via set_rl_mode) ----
        # When ``_rl_config`` is non-None, ``set_prompt`` builds an
        # advantage-conditioned (cond) + raw (uncond) prompt pair, and
        # ``infer`` runs the encoder + decoder twice (once per branch)
        # then combines the two action chunks via
        # ``cfg_combine_into_residual_fp16``. Mirrors the RTX
        # ``Pi05CFGPipeline`` contract per-chunk (vs RTX per-step) — the
        # math collapses to cond-only at ``cfg_beta=1.0`` either way.
        self._rl_config: Optional[dict] = None
        self._lang_emb_cond: Optional[torch.Tensor] = None
        self._lang_emb_uncond: Optional[torch.Tensor] = None
        # Lazy-allocated per-set_prompt (sized by Sa once known).
        self._noise_R_snapshot: Optional[torch.Tensor] = None
        self._v_cond_buf: Optional[torch.Tensor] = None
        self._v_uncond_buf: Optional[torch.Tensor] = None
        self._rl_current_prompt_text: Optional[str] = None

        # ---- FvkContext + GemmRunner ----
        self._ctx = fvk.FvkContext()
        self._gemm = fvk.GemmRunner()

        # ---- Load CUTLASS FMHA (compiled from csrc/attention/) ----
        # Search order: next to the checkpoint, then the installed
        # ``flash_rt/`` package dir (pip + editable installs land here via
        # the ``package-data = ["*.so"]`` glob in pyproject.toml), then the
        # uncopied ``build/`` output of a fresh cmake run, then the docker
        # container convention ``/workspace/``.
        fmha_paths = [
            str(checkpoint_dir.parent / "libfmha_fp16_strided.so"),
            str(pathlib.Path(__file__).parent.parent.parent / "libfmha_fp16_strided.so"),
            str(pathlib.Path(__file__).parent.parent.parent.parent / "build" / "libfmha_fp16_strided.so"),
            "/workspace/libfmha_fp16_strided.so",
        ]
        fmha_loaded = False
        for p in fmha_paths:
            if pathlib.Path(p).exists():
                ret = fvk.load_fmha_strided_library(p)
                if ret == 0:
                    fmha_loaded = True
                    logger.info("CUTLASS FMHA loaded from %s", p)
                    break
        if not fmha_loaded:
            logger.warning("CUTLASS strided FMHA not found — SigLIP will use cuBLAS attention fallback")

        # ---- Norm stats ----
        self._load_norm_stats(checkpoint_dir)

        # ---- Weights (safetensors only — JAX/Orbax uses ThorPipelineJax) ----
        safetensors_path = checkpoint_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(
                f"safetensors not found at {safetensors_path}. "
                "For Orbax/JAX checkpoints, use ThorPipelineJax.")
        self._checkpoint_path = str(safetensors_path)
        self._load_weights(safetensors_path)
        logger.info("Pi05TorchFrontendThor initialised (num_views=%d)", num_views)

    # -----------------------------------------------------------------------
    # norm_stats
    # -----------------------------------------------------------------------

    def _load_norm_stats(self, checkpoint_dir):
        from flash_rt.core.utils.norm_stats import (
            load_norm_stats, lerobot_candidates,
        )
        candidates = [
            checkpoint_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
            checkpoint_dir.parent / "pi05_libero" / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
            checkpoint_dir / "norm_stats.json",
            pathlib.Path("/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero/"
                         "assets/physical-intelligence/libero/norm_stats.json"),
            *lerobot_candidates(checkpoint_dir),
        ]
        self.norm_stats = load_norm_stats(
            candidates, checkpoint_dir=checkpoint_dir)

    # -----------------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------------

    def _load_weights(self, safetensors_path):
        from safetensors import safe_open

        from flash_rt.executors.torch_weights import (
            SafetensorsSource, WeightLoader,
        )
        from flash_rt.frontends.torch._pi05_thor_spec import build_spec

        from flash_rt.executors.torch_weights import _autodetect_strip_prefix
        sf = safe_open(str(safetensors_path), framework='pt', device='cuda')
        # The lerobot HF policy releases (e.g. lerobot/pi05_libero
        # _finetuned_v044) wrap every weight key under an extra
        # ``model.`` namespace. Auto-detect and prepend it on the raw
        # safetensors lookup so the rest of this loader can stay
        # written against the openpi-converted bare keys.
        _strip = _autodetect_strip_prefix(set(sf.keys()))
        def g_raw(k): return sf.get_tensor((_strip + k) if _strip else k)
        def g(k): return g_raw(k).to(fp16)

        # Declarative weight-loader pass (stage 7.3). Populates:
        #   self._sig_{ln_attn,ln_ffn,qkv,o,up,down}_{w,b}  (27-layer lists)
        #   self._sig_alpha                                  (108 fp32 scales)
        #   self._enc_{qkv,o,gu,d}_w                          (18-layer lists)
        #   self._enc_w_scales                                (72 fp32 scales)
        #   self._dec_{qkv,o,gu,d}_flat                       (flat cat tensors)
        #   self._ae_w_scales                                 (72 fp32 scales)
        #   self._{attn,ffn}_mod_{w,b}                        (18-layer lists)
        _src = SafetensorsSource(str(safetensors_path), device='cuda')
        WeightLoader(source=_src, target=self,
                     spec=build_spec(use_fp8=self.use_fp8)).run()
        # FP16 path: spec drops Quant() / scale_into, so _sig_alpha,
        # _enc_w_scales, _ae_w_scales never get populated. Provide
        # empty placeholders so downstream pointer-dict construction
        # (which still reads these attrs) stays happy — the FP16
        # forward branches ignore them.
        if not self.use_fp8:
            if not hasattr(self, '_sig_alpha'):
                self._sig_alpha = []
            if not hasattr(self, '_enc_w_scales'):
                self._enc_w_scales = []
            if not hasattr(self, '_ae_w_scales'):
                self._ae_w_scales = []

        self.embedding_weight = g('paligemma_with_expert.paligemma.lm_head.weight')

        # ===============================================================
        # SigLIP  (27 layers)
        # ===============================================================
        vp = 'paligemma_with_expert.paligemma.model.vision_tower.vision_model'
        nv = self.num_views
        S_sig = nv * 256
        D_sig, H_sig, NH_sig, HD_sig, L_sig = 1152, 4304, 16, 72, 27
        self.sig_S = S_sig
        self.sig_D = D_sig
        self.sig_H = H_sig
        self.sig_NH = NH_sig
        self.sig_HD = HD_sig
        self.sig_L = L_sig

        # Per-layer weights, scales, and biases loaded declaratively above.
        # Compose the pointer dict consumed by shared_primitives.siglip_forward.
        self._sig_weights = {
            'ln_attn_w': [w.data_ptr() for w in self._sig_ln_attn_w],
            'ln_attn_b': [w.data_ptr() for w in self._sig_ln_attn_b],
            'qkv_w':     [w.data_ptr() for w in self._sig_qkv_w],
            'qkv_b':     [w.data_ptr() for w in self._sig_qkv_b],
            'o_w':       [w.data_ptr() for w in self._sig_o_w],
            'o_b':       [w.data_ptr() for w in self._sig_o_b],
            'ln_ffn_w':  [w.data_ptr() for w in self._sig_ln_ffn_w],
            'ln_ffn_b':  [w.data_ptr() for w in self._sig_ln_ffn_b],
            'up_w':      [w.data_ptr() for w in self._sig_up_w],
            'up_b':      [w.data_ptr() for w in self._sig_up_b],
            'down_w':    [w.data_ptr() for w in self._sig_down_w],
            'down_b':    [w.data_ptr() for w in self._sig_down_b],
            'alpha':     self._sig_alpha,
        }
        # Unit scale (1.0) for FP8 casts without calibration (LN output already normalized)
        self._unit_scale = torch.ones(1, dtype=torch.float32, device='cuda')
        self._sig_weights['unit_scale'] = self._unit_scale.data_ptr()
        # SiLU scale: 1/SILU_SCALE = 5.0 (kernel does val * 1/d_scale = val * 0.2)
        self._silu_scale = torch.tensor([5.0], dtype=torch.float32, device='cuda')

        # SigLIP buffers
        self._sig_x     = torch.zeros(S_sig, D_sig, dtype=fp16, device='cuda')
        self._sig_x_fp8 = torch.zeros(S_sig * D_sig, dtype=torch.uint8, device='cuda')
        self._sig_qkv   = torch.empty(S_sig, 3 * D_sig, dtype=fp16, device='cuda')
        self._sig_attn   = torch.empty(S_sig, D_sig, dtype=fp16, device='cuda')
        self._sig_hidden = torch.empty(S_sig, H_sig, dtype=fp16, device='cuda')
        self._sig_hid_fp8 = torch.zeros(S_sig * H_sig, dtype=torch.uint8, device='cuda')
        # FP16 path needs a dedicated O/Down GEMM output buffer (one
        # epilogue away from the residual add). The x_norm scratch
        # aliases ``_sig_attn`` since the LN output is consumed by the
        # QKV GEMM before attention overwrites the buffer.
        self._sig_fg = (torch.empty(S_sig, D_sig, dtype=fp16, device='cuda')
                        if not self.use_fp8 else None)

        self._sig_bufs = {
            'x':       self._sig_x.data_ptr(),
            'x_fp8':   self._sig_x_fp8.data_ptr(),
            'qkv':     self._sig_qkv.data_ptr(),
            'attn_out': self._sig_attn.data_ptr(),
            'hidden':  self._sig_hidden.data_ptr(),
            'hid_fp8': self._sig_hid_fp8.data_ptr(),
            'x_norm':  self._sig_attn.data_ptr(),
            'fg':      self._sig_fg.data_ptr() if self._sig_fg is not None else 0,
        }
        self._sig_dims = {
            'S': S_sig, 'D': D_sig, 'H': H_sig,
            'NH': NH_sig, 'HD': HD_sig, 'L': L_sig,
            'num_views': nv, 'seq_per_view': 256,
        }

        # Patch embedding weights (HWC im2col order)
        pe_w_2d = (g(f'{vp}.embeddings.patch_embedding.weight')
                   .reshape(D_sig, 3, 14, 14)
                   .permute(0, 2, 3, 1)
                   .reshape(D_sig, -1)
                   .T.contiguous())
        self._pe_w = CudaBuffer.from_numpy(pe_w_2d.cpu().numpy().copy())
        self._pe_b = CudaBuffer.from_numpy(
            g(f'{vp}.embeddings.patch_embedding.bias').cpu().numpy().copy())
        self._pos_emb = CudaBuffer.from_numpy(
            g(f'{vp}.embeddings.position_embedding.weight')[:256].cpu().numpy().copy())
        self._img_buf = CudaBuffer.device_empty(nv * 224 * 224 * 3, np.float16)
        self._infer_images_np = np.empty(
            (nv, 224, 224, 3), dtype=np.float16)
        self._infer_uint8_to_fp16 = (
            np.arange(256, dtype=np.float32) / 127.5 - 1.0
        ).astype(np.float16)
        self._img_u8_buf = CudaBuffer.device_empty(
            nv * 224 * 224 * 3, np.uint8)
        self._img_u8_lut = CudaBuffer.from_numpy(self._infer_uint8_to_fp16)
        self._infer_images_u8_np = np.empty(
            (nv, 224, 224, 3), dtype=np.uint8)
        self._patches_buf = CudaBuffer.device_empty(S_sig * 588, np.float16)

        # PostLN weights
        self._postln_w = g(f'{vp}.post_layernorm.weight')
        self._postln_b = g(f'{vp}.post_layernorm.bias')
        mp = 'paligemma_with_expert.paligemma.model.multi_modal_projector.linear'
        self._proj_w = g(f'{mp}.weight').T.contiguous()  # [D_sig, D_enc] for bf16_nn
        self._proj_b = g(f'{mp}.bias')
        self._postln_scratch = torch.empty(S_sig, max(D_sig, H_sig), dtype=fp16, device='cuda')

        # ===============================================================
        # Encoder  (18 layers, GQA)
        # ===============================================================
        De, He, Le = 2048, 16384, 18
        NHe, HDe = 8, 256
        Se_max = nv * 256 + 256  # max sequence length (views*256 + max prompt)
        self.De = De; self.He = He; self.Le = Le
        self.NHe = NHe; self.HDe = HDe; self.Se_max = Se_max

        # Encoder per-layer weights loaded declaratively above:
        #   self._enc_qkv_w   list of [2560, D]   fp8 (fused-norm + GQA interleave)
        #   self._enc_o_w     list of [2048, 2048] fp8
        #   self._enc_gu_w    list of [2*H, D]    fp8 (fused post-attn norm)
        #   self._enc_d_w     list of [H, D]      fp8
        #   self._enc_w_scales  host list of Le*4 floats (order: q, o, gu, d per layer)
        # When use_fp8=False the spec skips Quant() so _enc_w_scales is
        # not populated; keep an empty tensor so downstream pointer
        # arithmetic still works (the FP16 path never reads it).
        _enc_w_scales = getattr(self, '_enc_w_scales', []) or [0.0] * (Le * 4)
        self._enc_w_dev = torch.tensor(_enc_w_scales, dtype=torch.float32, device='cuda')

        # RoPE table.  The reference model stores ``inv_freq`` in bfloat16
        # (checkpoint dtype), so the cos/sin phase must be derived from a
        # bf16-quantized ``inv_freq`` to match it.  Computing it in fp32 keeps
        # a ~1e-3 per-frequency offset that is scaled by the absolute position
        # and grows into a large RoPE angle error at high prefix positions,
        # diverging the prefix K/V and the action decision on borderline frames.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2, dtype=torch.float32, device='cuda') / 256))
        inv_freq = inv_freq.to(torch.bfloat16).to(torch.float32)
        kp = inv_freq[None, :] * torch.arange(1200, device='cuda')[:, None].float()
        self._kc_t = torch.cos(kp).to(fp16)
        self._ks_t = torch.sin(kp).to(fp16)
        self._enc_rope = torch.empty(Se_max, 256, dtype=fp16, device='cuda')

        # KV cache
        Sa, Da, Ha, La = 10, 1024, 4096, 18
        self.Sa = Sa; self.Da = Da; self.Ha = Ha; self.La = La
        total_keys_max = Se_max + Sa
        self._Kc = torch.zeros(Le, total_keys_max, HDe, dtype=fp16, device='cuda')
        self._Vc = torch.zeros(Le, total_keys_max, HDe, dtype=fp16, device='cuda')

        # Encoder buffers
        self._enc_x      = torch.empty(Se_max, De, dtype=fp16, device='cuda')
        self._enc_x_fp8  = torch.zeros(Se_max * De, dtype=torch.uint8, device='cuda')
        self._enc_qkv_buf = torch.empty(Se_max, 2560, dtype=fp16, device='cuda')
        self._enc_logits = torch.empty(Se_max * NHe, total_keys_max, dtype=fp16, device='cuda')
        self._enc_attn   = torch.empty(Se_max, NHe * HDe, dtype=fp16, device='cuda')
        self._enc_o_fp8  = torch.zeros(Se_max * NHe * HDe, dtype=torch.uint8, device='cuda')
        self._enc_gate   = torch.empty(Se_max, 2 * He, dtype=fp16, device='cuda')
        self._enc_hidden = torch.empty(Se_max, He, dtype=fp16, device='cuda')
        self._enc_hid_fp8 = torch.zeros(Se_max * He, dtype=torch.uint8, device='cuda')
        self._enc_fg     = torch.empty(Se_max, De, dtype=fp16, device='cuda')
        # All-ones gamma for the FP16 noweight RMSNorm (encoder uses
        # ``rms_norm_fp8_noweight_fp16`` in the FP8 path; ``rms_norm_fp16``
        # takes a weight pointer, so we feed it ones for parity).
        self._enc_ones_fp16 = (torch.ones(De, dtype=fp16, device='cuda')
                               if not self.use_fp8 else None)

        # ===============================================================
        # Decoder / AE  (18 layers, 10 steps)
        # ===============================================================
        dp = 'paligemma_with_expert.gemma_expert'
        steps = 10
        self.steps = steps
        D3a = 3 * Da

        # Decoder/AE per-layer weights loaded declaratively above as flat cats:
        #   self._dec_qkv_flat  per-layer [q|k|v] fp8  (.t().contiguous(), interleaved Q/K)
        #   self._dec_o_flat    per-layer O   fp8
        #   self._dec_gu_flat   per-layer [gate|up] fp8
        #   self._dec_d_flat    per-layer down  fp8
        #   self._ae_w_scales   host list of La*4 floats (order: q, o, gu, d per layer)

        # Action in/out projections (singletons)
        self._ain_w = g('action_in_proj.weight').t().contiguous()
        self._ain_b = g('action_in_proj.bias')
        self._aow = g('action_out_proj.weight').t().contiguous()
        self._aob = g('action_out_proj.bias')
        self._ae_dt = -1.0 / steps
        self._aob_dt = self._aob * self._ae_dt

        _ae_w_scales = getattr(self, '_ae_w_scales', []) or [0.0] * (La * 4)
        self._ae_w_dev = torch.tensor(_ae_w_scales, dtype=torch.float32, device='cuda')

        # Time conditioning weights
        self._time_mlp_in_w  = g('time_mlp_in.weight')
        self._time_mlp_in_b  = g('time_mlp_in.bias')
        self._time_mlp_out_w = g('time_mlp_out.weight')
        self._time_mlp_out_b = g('time_mlp_out.bias')
        # Per-layer AdaRMS modulation Dense layers (self._{attn,ffn}_mod_{w,b})
        # loaded declaratively above. Final norm modulation stays inline.
        dp_full = 'paligemma_with_expert.gemma_expert'
        self._final_mod_w = g(f'{dp_full}.model.norm.dense.weight')
        self._final_mod_b = g(f'{dp_full}.model.norm.dense.bias')

        # Decoder buffers (pre-allocate at max sizes)
        self._dec_rope = torch.empty(Sa, 256, dtype=fp16, device='cuda')
        self._ae_x   = torch.empty(Sa, Da, dtype=fp16, device='cuda')
        self._ae_xn  = torch.empty(Sa, Da, dtype=fp16, device='cuda')
        self._ae_gate = torch.empty(Sa, Da, dtype=fp16, device='cuda')
        self._ae_qkv = torch.empty(Sa, 2560, dtype=fp16, device='cuda')
        self._ae_logits = torch.empty(Sa * 8, total_keys_max, dtype=fp16, device='cuda')
        self._ae_attn = torch.empty(Sa * 8, 256, dtype=fp16, device='cuda')
        self._ae_hid  = torch.empty(Sa, 2 * Ha, dtype=fp16, device='cuda')
        self._ae_fg   = torch.empty(Sa, 2 * Ha, dtype=fp16, device='cuda')  # must fit Gate+Up GEMM output [Sa, 2H]
        self._ae_action_f32 = torch.empty(Sa, 32, dtype=torch.float32, device='cuda')
        self._ae_xn_fp8  = torch.zeros(Sa * Da, dtype=torch.uint8, device='cuda')
        self._ae_hid_fp8 = torch.zeros(Sa * Ha, dtype=torch.uint8, device='cuda')
        self._ae_ctx_fp8 = torch.zeros(Sa * 8 * 256, dtype=torch.uint8, device='cuda')
        self._g_noise = torch.zeros(Sa, 32, dtype=fp16, device='cuda')
        # Previous normalized action chunk consumed by Prefix RTC.
        self._rtc_prev_action_chunk = torch.zeros(Sa, 32, dtype=fp16, device='cuda')

        # Calibration scale buffers
        self._enc_calib_scales = torch.zeros(Le * 4, dtype=torch.float32, device='cuda')
        self._ae_calib_scales  = torch.zeros(steps * La * 4, dtype=torch.float32, device='cuda')

        # ── B=N batched mode state (Stage 2 of Thor batched-CFG port) ──
        # Inactive by default — buffers + B=N graph capture are
        # lazily created when ``set_batched_mode(enable=True)`` fires.
        self._batched = False
        self.B = 1
        # Lazy-allocated b2-suffixed buffers (None → not yet created).
        # See ``_alloc_b2_buffers`` for the full inventory.
        self._Kc_b2 = None
        self._Vc_b2 = None
        self._enc_x_b2 = None
        self._enc_x_fp8_b2 = None
        self._enc_qkv_buf_b2 = None
        self._enc_logits_b2 = None
        self._enc_attn_b2 = None
        self._enc_o_fp8_b2 = None
        self._enc_gate_b2 = None
        self._enc_hid_fp8_b2 = None
        self._enc_fg_b2 = None
        self._ae_x_b2 = None
        self._ae_xn_b2 = None
        self._ae_gate_b2 = None
        self._ae_qkv_b2 = None
        self._ae_logits_b2 = None
        self._ae_attn_b2 = None
        self._ae_fg_b2 = None
        self._ae_xn_fp8_b2 = None
        self._ae_hid_fp8_b2 = None
        self._ae_ctx_fp8_b2 = None
        self._g_noise_b2 = None
        self._v_b2 = None
        self._img_stack_host_b2 = None
        self._img_stack_host_torch_b2 = None
        self._img_stack_fp16_host_b2 = None
        self._img_stack_fp16_host_torch_b2 = None
        # B-tiled style buffers — built in set_prompt when ``_batched``.
        self._sa_all_b2 = None
        self._sf_all_b2 = None
        self._fs_all_b2 = None
        # B=N captured graph (separate from the B=1 ``_enc_ae_graph``).
        self._enc_ae_graph_b2 = None
        # Outer fused-CFG graph: lang swap (×2) + SigLIP (×2) +
        # encoder_b2 + decoder_b2 (with per-step CFG inside). One
        # ``replay()`` per CFG inference. Captured by
        # ``_capture_cfg_b2_outer_graph`` and consumed via
        # ``Pi05ThorCFGBatchedPipeline.forward``. Mirrors RTX
        # ``Pi05CFGBatchedPipeline.forward`` /
        # ``self._graph.replay()`` at
        # ``pipeline_rtx_cfg_batched.py:419``.
        self._cfg_b2_outer_graph = None
        # When non-None, ``_capture_enc_ae_graph_b2`` bakes the per-step
        # CFG combine + noise mirror into the graph (paper-correct
        # per-step CFG; mirrors RTX
        # ``Pi05CFGBatchedPipeline.transformer_decoder_batched``).
        # Cleared by ``set_rl_mode(cfg_enable=False)``; set by
        # ``_build_cfg_batched_pipeline``.
        self._enc_ae_graph_b2_cfg_beta = None

        logger.info("Weights loaded for Pi05TorchFrontendThor")

    def _alloc_b2_buffers(self, B: int = 2, Se_dynamic: int = None) -> None:
        """Allocate all B=N inference buffers.

        Stage 2 of the Thor batched-CFG port. Called by
        :meth:`set_batched_mode` to lazily set up B-folded buffers
        without disturbing the B=1 hot path. The B=1 buffers stay
        allocated; the b2 set is parallel.

        Buffer layout convention: leading row dim is ``B * <per-sample>``,
        flat in memory. ``_Kc_b2`` and ``_Vc_b2`` are
        ``(B, La, total_keys_max, HD)`` so per-sample slabs can be
        addressed via ``_Kc_b2[b].view(-1).data_ptr()``.
        """
        if B < 2:
            raise ValueError(f"_alloc_b2_buffers requires B >= 2; got {B}")
        Le = self.Le; De = self.De; He = self.He
        NHe = self.NHe; HDe = self.HDe
        Sa = self.Sa; Da = self.Da; Ha = self.Ha; La = self.La
        Se_max = Se_dynamic if Se_dynamic is not None else self._enc_x.shape[0]
        total_keys_max = self._Kc.shape[1]

        # ── KV cache (B, La, total_keys_max, HD) ──
        self._Kc_b2 = torch.zeros(B, Le, total_keys_max, HDe,
                                   dtype=fp16, device='cuda')
        self._Vc_b2 = torch.zeros(B, Le, total_keys_max, HDe,
                                   dtype=fp16, device='cuda')

        # ── SigLIP batched buffers (for batched SigLIP CUDA Graph) ──
        # Contiguous [B*S_sig, D] layout so GEMMs see one big M dimension.
        S_sig = self.sig_S  # 256 * num_views
        D_sig = self.sig_D  # 1152
        H_sig = self.sig_H  # 4304
        self._sig_x_b2    = torch.zeros(B * S_sig, D_sig, dtype=fp16, device='cuda')
        self._sig_x_fp8_b2 = torch.zeros(B * S_sig * D_sig, dtype=torch.uint8, device='cuda')
        self._sig_qkv_b2  = torch.empty(B * S_sig, 3 * D_sig, dtype=fp16, device='cuda')
        self._sig_attn_b2 = torch.empty(B * S_sig, D_sig, dtype=fp16, device='cuda')
        self._sig_hidden_b2 = torch.empty(B * S_sig, H_sig, dtype=fp16, device='cuda')
        self._sig_hid_fp8_b2 = torch.zeros(B * S_sig * H_sig, dtype=torch.uint8, device='cuda')
        self._postln_scratch_b2 = torch.empty(B * S_sig, D_sig, dtype=fp16, device='cuda')
        self._postln_proj_b2 = (
            torch.empty(B * S_sig, self.De, dtype=fp16, device='cuda')
            if self.batched_postln_proj_fusion else None
        )
        # Per-sample image upload buffers
        self._img_buf_b2_list = [
            CudaBuffer.device_empty(self._img_buf.nbytes, np.float16)
            for _ in range(B)
        ]
        # Batched image buffer: contiguous (B*nv, 224, 224, 3) fp16
        self._img_buf_b2_all = CudaBuffer.device_empty(
            B * self._img_buf.nbytes, np.float16)
        # Batched patches buffer: (B*nv*256, 588) fp16
        self._patches_buf_b2 = CudaBuffer.device_empty(
            B * S_sig * 588, np.float16)
        # Pre-allocated torch tensor for GPU-side uint8→fp16 image conversion
        # Sized for B images (batched upload path)
        self._img_staging_torch = torch.empty(
            B * self._img_buf.nbytes // 2, dtype=torch.uint8, device='cuda')
        # Persistent host/device staging for the common uint8 image path.
        # Avoids per-inference numpy/torch allocation before patch embedding.
        img_elems = B * self.num_views * 224 * 224 * 3
        self._img_stack_host_torch_b2 = torch.empty(
            img_elems, dtype=torch.uint8, pin_memory=True)
        self._img_stack_host_b2 = self._img_stack_host_torch_b2.numpy().reshape(
            B, self.num_views, 224, 224, 3)
        self._img_stack_fp16_host_torch_b2 = torch.empty(
            img_elems, dtype=fp16, pin_memory=True)
        self._img_stack_fp16_host_b2 = (
            self._img_stack_fp16_host_torch_b2.numpy().reshape(
                B, self.num_views, 224, 224, 3))
        # Per-sample lang embedding buffers for each sample's prompt.
        # Multi-prompt batches use the batch max language length, which can
        # be larger than the prompt used to initialize the serial buffers.
        lang_slots = max(int(self._lang_emb.shape[0]),
                         int(Se_max - self.sig_S))
        self._lang_emb_b2 = torch.zeros(B, lang_slots, self.De,
                                        dtype=fp16, device='cuda')
        self._siglip_batched_graph = None
        self._siglip_batched_B = B

        # ── Encoder buffers (B*Se, ...) ──
        self._enc_x_b2     = torch.empty(B * Se_max, De, dtype=fp16, device='cuda')
        self._enc_x_fp8_b2 = torch.zeros(B * Se_max * De, dtype=torch.uint8, device='cuda')
        self._enc_qkv_buf_b2 = torch.empty(B * Se_max, 2560, dtype=fp16, device='cuda')
        self._enc_logits_b2  = torch.empty(B * Se_max * NHe, total_keys_max,
                                            dtype=fp16, device='cuda')  # scratch reused per sample
        self._enc_attn_b2    = torch.empty(B * Se_max, NHe * HDe, dtype=fp16, device='cuda')
        self._enc_o_fp8_b2   = torch.zeros(B * Se_max * NHe * HDe, dtype=torch.uint8, device='cuda')
        self._enc_gate_b2    = torch.empty(B * Se_max, 2 * He, dtype=fp16, device='cuda')
        self._enc_hid_fp8_b2 = torch.zeros(B * Se_max * He, dtype=torch.uint8, device='cuda')
        self._enc_fg_b2      = torch.empty(B * Se_max, De, dtype=fp16, device='cuda')

        # ── Decoder buffers (B*Sa, ...) ──
        self._ae_x_b2   = torch.empty(B * Sa, Da, dtype=fp16, device='cuda')
        self._ae_xn_b2  = torch.empty(B * Sa, Da, dtype=fp16, device='cuda')
        self._ae_gate_b2 = torch.empty(B * Sa, Da, dtype=fp16, device='cuda')
        self._ae_qkv_b2 = torch.empty(B * Sa, 2560, dtype=fp16, device='cuda')
        self._ae_logits_b2 = torch.empty(B * Sa * 8, total_keys_max,
                                          dtype=fp16, device='cuda')  # scratch for each sample
        self._ae_attn_b2 = torch.empty(B * Sa * 8, 256, dtype=fp16, device='cuda')
        self._ae_fg_b2   = torch.empty(B * Sa, 2 * Ha, dtype=fp16, device='cuda')
        self._ae_action_f32_b2 = torch.empty(B * Sa, 32, dtype=torch.float32, device='cuda')
        self._ae_xn_fp8_b2  = torch.zeros(B * Sa * Da, dtype=torch.uint8, device='cuda')
        self._ae_hid_fp8_b2 = torch.zeros(B * Sa * Ha, dtype=torch.uint8, device='cuda')
        self._ae_ctx_fp8_b2 = torch.zeros(B * Sa * 8 * 256, dtype=torch.uint8, device='cuda')
        self._g_noise_b2 = torch.zeros(B * Sa, 32, dtype=fp16, device='cuda')
        # Per-step velocity scratch for the CFG-batched graph: each step
        # writes (v_cond, v_uncond) here before the in-graph cfg_combine
        # mixes them into ``_g_noise_b2`` slot 0. Always allocated; the
        # non-CFG b2 path simply doesn't reference it.
        self._v_b2 = torch.zeros(B * Sa, 32, dtype=fp16, device='cuda')
        self._v_b2_f32 = torch.empty(B * Sa, 32, dtype=torch.float32, device='cuda')

        self.B = int(B)
        logger.info(
            "Allocated B=%d buffers (Se_max=%d, Sa=%d, total_keys_max=%d)",
            B, Se_max, Sa, total_keys_max)

    def enable_batch_graph_bank(self, enable: bool = True, *,
                                clear: bool = False) -> None:
        """Keep exact B-specific CUDA graph/buffer sets resident in VRAM.

        The bank is keyed by ``(B, Se, S_lang, cfg_beta)``. It does not reuse
        visual features or any image-conditioned hidden state; it only avoids
        reallocating and recapturing static CUDA graphs when runtime traffic
        switches among already-seen batch shapes.
        """
        if clear:
            self.clear_batch_graph_bank()
        self.batch_graph_bank_enabled = bool(enable)

    def clear_batch_graph_bank(self) -> None:
        """Drop all resident batch graph-bank entries."""
        self._batch_graph_bank.clear()
        self._batch_graph_bank_active_key = None

    def batch_graph_bank_keys(self):
        """Return the resident graph-bank keys for benchmarks/debugging."""
        return tuple(sorted(self._batch_graph_bank.keys(), key=repr))

    def _batch_graph_bank_key(self, B: int, Se: int):
        return (
            int(B),
            int(Se),
            int(getattr(self, "_S_lang", 0) or 0),
            self._enc_ae_graph_b2_cfg_beta,
        )

    def _save_active_batch_graph_bank_entry(self) -> None:
        if not self.batch_graph_bank_enabled:
            return
        if getattr(self, "_enc_x_b2", None) is None:
            return
        if int(getattr(self, "B", 1)) < 2:
            return
        key = self._batch_graph_bank_active_key
        if key is None:
            key = self._batch_graph_bank_key(self.B, self.Se)
        entry = {name: getattr(self, name, None)
                 for name in self._BATCH_GRAPH_BANK_ATTRS}
        entry.update({
            "B": int(self.B),
            "Se": int(self.Se),
            "total_keys": int(self.total_keys),
            "batched": bool(self._batched),
        })
        self._batch_graph_bank[key] = entry
        self._batch_graph_bank_active_key = key

    def _restore_batch_graph_bank_entry(self, B: int, Se: int) -> bool:
        if not self.batch_graph_bank_enabled:
            return False
        key = self._batch_graph_bank_key(B, Se)
        entry = self._batch_graph_bank.get(key)
        if entry is None:
            return False
        for name in self._BATCH_GRAPH_BANK_ATTRS:
            setattr(self, name, entry.get(name))
        self.B = int(entry["B"])
        self.Se = int(entry["Se"])
        self.total_keys = int(entry["total_keys"])
        self._batched = bool(entry["batched"])
        self._batch_graph_bank_active_key = key
        return True

    def _mark_batch_graph_bank_entry_ready(self, B: int, Se: int) -> None:
        if not self.batch_graph_bank_enabled:
            return
        if self._siglip_batched_graph is None or self._enc_ae_graph_b2 is None:
            return
        self._batch_graph_bank_active_key = self._batch_graph_bank_key(B, Se)
        self._save_active_batch_graph_bank_entry()

    def enable_prompt_bank(self, enable: bool = True, *,
                           clear: bool = False) -> None:
        """Keep exact prompt embeddings resident on device.

        This cache stores token-embedding outputs after the same tokenizer and
        embedding lookup used by the normal path, plus the existing even-length
        padding used by batched inference. It does not cache image-conditioned
        states and does not change model math.
        """
        if clear:
            self.clear_prompt_bank()
        self.prompt_bank_enabled = bool(enable)

    def clear_prompt_bank(self) -> None:
        """Drop all resident prompt-bank entries."""
        self._prompt_bank.clear()
        self._prompt_bank_hits = 0
        self._prompt_bank_misses = 0

    def prompt_bank_keys(self):
        """Return resident prompt-bank keys for benchmarks/debugging."""
        return tuple(sorted(self._prompt_bank.keys(), key=repr))

    def prompt_bank_stats(self):
        """Return prompt-bank hit/miss counters and resident entry count."""
        return {
            "enabled": bool(self.prompt_bank_enabled),
            "entries": len(self._prompt_bank),
            "hits": int(self._prompt_bank_hits),
            "misses": int(self._prompt_bank_misses),
        }

    @staticmethod
    def _prompt_bank_state_key(state):
        if state is None:
            return None
        arr = np.asarray(state)
        return (arr.shape, str(arr.dtype), arr.tobytes())

    def _prompt_bank_key(self, prompt_text, max_len: int, state=None):
        return (
            id(self.embedding_weight),
            int(max_len),
            prompt_text,
            self._prompt_bank_state_key(state),
        )

    def _get_prompt_bank_entry(self, prompt_text, *,
                               max_len: int = 48, state=None):
        """Return even-padded CUDA prompt embeddings for batch inference."""
        if not self.prompt_bank_enabled or not isinstance(prompt_text, str):
            embeds, _ = embed_prompt(
                prompt_text, self.embedding_weight, max_len=max_len,
                state=state)
            S_lang = embeds.shape[0]
            if S_lang % 2 != 0:
                embeds = torch.cat([embeds, embeds[-1:]], dim=0)
                S_lang = embeds.shape[0]
            return embeds, S_lang

        key = self._prompt_bank_key(prompt_text, max_len, state=state)
        entry = self._prompt_bank.get(key)
        if entry is not None:
            self._prompt_bank_hits += 1
            return entry["embeds"], entry["S_lang"]

        self._prompt_bank_misses += 1
        embeds, prompt_len = embed_prompt(
            prompt_text, self.embedding_weight, max_len=max_len, state=state)
        S_lang = embeds.shape[0]
        if S_lang % 2 != 0:
            embeds = torch.cat([embeds, embeds[-1:]], dim=0)
            S_lang = embeds.shape[0]
        entry = {
            "embeds": embeds.contiguous(),
            "S_lang": int(S_lang),
            "prompt_len": int(prompt_len),
        }
        self._prompt_bank[key] = entry
        return entry["embeds"], entry["S_lang"]

    def _copy_lang_embeds_batched(self, samples) -> None:
        """Copy per-sample language embeddings into the B-major buffer.

        Adjacent samples with the exact same prompt and language length share
        one broadcast copy. This is mathematically identical to per-row copies
        and only reduces tiny CUDA copy launches for common same-task batches.
        """
        b = 0
        B = len(samples)
        while b < B:
            s = samples[b]
            S_lang = int(s["S_lang"])
            prompt = s["prompt"]
            end = b + 1
            while end < B:
                nxt = samples[end]
                if nxt["prompt"] != prompt or int(nxt["S_lang"]) != S_lang:
                    break
                end += 1
            rows = end - b
            src = s["embeds"][:S_lang]
            if rows == 1:
                self._lang_emb_b2[b, :S_lang].copy_(src)
            else:
                self._lang_emb_b2[b:end, :S_lang].copy_(
                    src.unsqueeze(0).expand(rows, S_lang, -1))
            b = end

    def enable_noise_bank(self, enable: bool = True, *,
                          clear: bool = False) -> None:
        """Keep exact seeded diffusion noise tensors resident on device."""
        if clear:
            self.clear_noise_bank()
        self.noise_bank_enabled = bool(enable)

    def clear_noise_bank(self) -> None:
        """Drop all resident seeded noise tensors."""
        self._noise_bank.clear()
        self._noise_bank_hits = 0
        self._noise_bank_misses = 0

    def noise_bank_stats(self):
        """Return seeded-noise bank hit/miss counters."""
        return {
            "enabled": bool(self.noise_bank_enabled),
            "entries": len(self._noise_bank),
            "hits": int(self._noise_bank_hits),
            "misses": int(self._noise_bank_misses),
        }

    def _get_seed_noise_batched(self, seed: int, B: int):
        key = (int(seed), int(B), int(self.Sa), 32)
        if self.noise_bank_enabled:
            cached = self._noise_bank.get(key)
            if cached is not None:
                self._noise_bank_hits += 1
                return cached
            self._noise_bank_misses += 1

        rng_base = np.random.RandomState(seed)
        noise_np = rng_base.randn(B, self.Sa, 32).astype(np.float16)
        noise_tensor = torch.from_numpy(noise_np).to('cuda')
        if self.noise_bank_enabled:
            self._noise_bank[key] = noise_tensor
        return noise_tensor

    @staticmethod
    def _noise_indices_for_samples(samples, B: int):
        prompt_to_noise_idx = {}
        noise_idx_counter = 0
        noise_indices = []
        for s in samples:
            prompt = s['prompt']
            if prompt not in prompt_to_noise_idx:
                if noise_idx_counter >= B:
                    noise_idx_counter = 0
                prompt_to_noise_idx[prompt] = noise_idx_counter
                noise_idx_counter += 1
            noise_indices.append(prompt_to_noise_idx[prompt])
        return noise_indices

    def _fill_seed_noise_batched(self, samples, seed: int) -> None:
        B = len(samples)
        noise_tensor = self._get_seed_noise_batched(seed, B)
        noise_indices = self._noise_indices_for_samples(samples, B)
        noise_view = self._g_noise_b2[:B * self.Sa].view(B, self.Sa, 32)

        b = 0
        while b < B:
            idx = noise_indices[b]
            end = b + 1
            while end < B and noise_indices[end] == idx:
                end += 1
            rows = end - b
            if rows == 1:
                noise_view[b].copy_(noise_tensor[idx])
            else:
                noise_view[b:end].copy_(
                    noise_tensor[idx:idx + 1].expand(rows, self.Sa, 32))
            b = end

    def _upload_images_gpu_batched(self, samples, target_buf_all,
                                   target_patches_buf=None) -> bool:
        """Upload B*N views into the contiguous batched image buffer.

        The common LIBERO input is uint8. Convert that path on GPU using
        persistent staging buffers so repeated batch inference does not
        allocate a temporary numpy stack and CUDA fp16 tensor every call.
        Returns True when the patch matrix was produced directly.
        """
        B = len(samples)
        nv = self.num_views

        all_u8 = True
        for s in samples:
            for im in s['images'][:nv]:
                if not (isinstance(im, np.ndarray) and im.dtype == np.uint8):
                    all_u8 = False
                    break
            if not all_u8:
                break

        if all_u8:
            has_fused_patches = hasattr(fvk, 'normalize_uint8_to_patches_fp16')
            has_fused_fp16 = hasattr(fvk, 'normalize_uint8_to_fp16')
            if (target_patches_buf is not None and has_fused_patches) or (
                    target_patches_buf is None and has_fused_fp16):
                host_stack = self._img_stack_host_b2[:B, :nv]
                for b, s in enumerate(samples):
                    for v, im in enumerate(s['images'][:nv]):
                        host_stack[b, v] = im

                numel = host_stack.size
                src_u8 = self._img_stack_host_torch_b2[:numel]
                dev_u8 = self._img_staging_torch[:numel]
                dev_u8.copy_(src_u8, non_blocking=True)
                if target_patches_buf is not None:
                    fvk.normalize_uint8_to_patches_fp16(
                        dev_u8.data_ptr(), target_patches_buf.ptr.value, B * nv, 0)
                    return True
                fvk.normalize_uint8_to_fp16(
                    dev_u8.data_ptr(), target_buf_all.ptr.value, numel, 0)
                return False

            logger.warning(
                "FlashRT uint8 normalize fused kernel unavailable; "
                "falling back to CPU fp16 upload before patch_im2col"
            )

        host_fp16 = self._img_stack_fp16_host_b2[:B, :nv]
        for b, s in enumerate(samples):
            for v, im in enumerate(s['images'][:nv]):
                if isinstance(im, torch.Tensor):
                    host_fp16[b, v] = im.to(dtype=torch.float16).cpu().numpy()
                elif im.dtype == np.float16:
                    host_fp16[b, v] = im
                else:
                    host_fp16[b, v] = (
                        im.astype(np.float32) / 127.5 - 1.0).astype(np.float16)
        target_buf_all.upload(host_fp16)
        return False

    # NOTE: _load_weights_orbax removed — Orbax loading belongs to ThorPipelineJax
    # -----------------------------------------------------------------------
    # RL CFG inference (opt-in via set_rl_mode)
    # -----------------------------------------------------------------------

    def set_rl_mode(
        self,
        *,
        cfg_enable: bool = True,
        cfg_beta: float = 1.5,
        advantage_positive: bool = True,
    ) -> None:
        """Enable / configure advantage-conditioned RL inference (opt-in).

        When enabled, subsequent :meth:`set_prompt` calls build a
        conditioned prompt with an "Advantage: positive" / "negative"
        ACP tag appended; the unconditioned prompt is the raw task text.
        :meth:`infer` then runs the encoder + decoder twice (cond /
        uncond) and combines the two action chunks via
        ``fvk.cfg_combine_into_residual_fp16`` with strength ``cfg_beta``.

        Mirrors :meth:`Pi05TorchFrontendRtx.set_rl_mode` (the public API
        contract is byte-equal). The Thor fused B=2 path now does
        per-step CFG inside the captured ``_enc_ae_graph_b2`` (matches
        RTX :meth:`Pi05CFGBatchedPipeline.transformer_decoder_batched`
        and arXiv:2511.14759 Appendix E); the serial path is still
        per-chunk (cheaper, no graph recapture needed).

        Args:
            cfg_enable: ``True`` activates CFG inference; ``False`` clears
                any previous RL config and reverts the next
                :meth:`set_prompt` to the standard single-forward path.
            cfg_beta: Guidance strength. Must be ``>= 1.0`` (1.0 collapses
                to cond-only). Common deployment range ``[1.5, 2.5]``.
            advantage_positive: Whether the conditioned prompt uses the
                positive advantage tag (the standard "select for high
                advantage" use case). Set ``False`` only for debugging.
        """
        if not cfg_enable:
            self._rl_config = None
            self._lang_emb_cond = None
            self._lang_emb_uncond = None
            # Force the next set_prompt to rebuild the standard graph.
            self.graph_captured = False
            # If a CFG-batched B=2 graph was previously captured (with
            # in-graph cfg_combine), invalidate it so the next batched
            # use recaptures without the CFG schedule baked in.
            if self._enc_ae_graph_b2_cfg_beta is not None:
                self._enc_ae_graph_b2_cfg_beta = None
                self._enc_ae_graph_b2 = None
            return
        if cfg_beta < 1.0:
            raise ValueError(
                f"cfg_beta must be >= 1.0 (1.0 disables CFG); got {cfg_beta}")
        new_config = {
            "cfg_beta": float(cfg_beta),
            "advantage_positive": bool(advantage_positive),
        }
        if self._rl_config != new_config:
            self._rl_config = new_config
            # Force graph rebuild on next set_prompt so the new beta /
            # mode takes effect.
            self.graph_captured = False
        logger.info(
            "RL mode enabled: cfg_beta=%.2f, advantage_positive=%s",
            new_config["cfg_beta"], new_config["advantage_positive"])

    def _ensure_cfg_buffers(self):
        """Lazy-allocate the CFG-only intermediate buffers."""
        n = self.Sa * 32
        if self._noise_R_snapshot is None:
            self._noise_R_snapshot = torch.empty(
                n, dtype=fp16, device='cuda')
        if self._v_cond_buf is None:
            self._v_cond_buf = torch.empty(n, dtype=fp16, device='cuda')
        if self._v_uncond_buf is None:
            self._v_uncond_buf = torch.empty(n, dtype=fp16, device='cuda')

    def _set_prompt_rl(self, prompt_text):
        """Build cond + uncond embeds, drive the standard set_prompt path
        with cond_text (cond is always >= uncond in length because the
        ACP tag appends to the task), then pad uncond to cond_len and
        stash it for the runtime swap.

        The captured ``_siglip_graph`` reads from the cond tensor at
        ``self._lang_emb.data_ptr()``; :meth:`_infer_cfg` ``copy_``s
        either ``_lang_emb_cond`` or ``_lang_emb_uncond`` into that
        same buffer between the two siglip replays without touching
        graph capture.
        """
        if isinstance(prompt_text, (np.ndarray, list)):
            raise ValueError(
                "set_rl_mode requires a text prompt (the ACP tag is "
                "appended at the string level); pass a str, not token IDs")
        from flash_rt.core.rl import build_acp_tagged_task

        cfg = self._rl_config
        cond_text = build_acp_tagged_task(
            prompt_text, is_positive=cfg["advantage_positive"])
        uncond_text = prompt_text

        # Drive the standard graph-capture path with cond_text; this
        # sets self._lang_emb / self._S_lang / self.Se at cond_len.
        self.graph_captured = False
        self.set_prompt(cond_text)
        target_len = self._S_lang  # actual padded language slot length

        # Compute uncond embeds and pad to target_len (cond is >= uncond
        # because the ACP tag adds tokens).
        uncond_emb, uncond_len = embed_prompt(
            uncond_text, self.embedding_weight, max_len=48)
        if uncond_emb.shape[0] > target_len:
            raise RuntimeError(
                f"uncond_len={uncond_emb.shape[0]} > cond target_len="
                f"{target_len}; the ACP tag is supposed to make cond at "
                f"least as long as uncond")
        if uncond_emb.shape[0] < target_len:
            pad = uncond_emb[-1:].expand(target_len - uncond_emb.shape[0], -1)
            uncond_emb = torch.cat([uncond_emb, pad], dim=0)
        self._lang_emb_uncond = uncond_emb.contiguous()
        # Snapshot the cond embeds the standard set_prompt produced so
        # we can restore them after the uncond replay.
        self._lang_emb_cond = self._lang_emb.detach().clone()
        self._rl_current_prompt_text = prompt_text

        self._ensure_cfg_buffers()

        # Build the appropriate CFG pipeline based on whether batched
        # mode is active. Stage 0 (serial) when ``_batched`` is False;
        # Stage 3 (B=2 fused) when ``_batched`` is True.
        if self._batched:
            self._build_cfg_batched_pipeline(cfg["cfg_beta"])
        else:
            self._build_cfg_serial_pipeline(cfg["cfg_beta"])

        logger.info(
            "Set RL prompt: '%s' (cond_len=%d, uncond_len=%d, padded=%d, "
            "cfg_beta=%.2f, batched=%s)",
            prompt_text, target_len, uncond_len, target_len,
            cfg["cfg_beta"], self._batched)

    def _build_cfg_serial_pipeline(self, cfg_beta: float) -> None:
        """Build the Stage 0 serial CFG pipeline (per-chunk, B=1 graphs)."""
        from flash_rt.models.pi05.pipeline_thor_cfg import (
            Pi05ThorCFGPipeline)
        self._cfg_pipeline = Pi05ThorCFGPipeline(
            fvk,
            cfg_beta=cfg_beta,
            Sa=int(self.Sa),
            replay_siglip=lambda: self._siglip_graph.replay(),
            replay_enc_ae=lambda: self._enc_ae_graph.replay(),
            upload_cond_lang_emb=lambda: self._lang_emb.copy_(
                self._lang_emb_cond),
            upload_uncond_lang_emb=lambda: self._lang_emb.copy_(
                self._lang_emb_uncond),
            snapshot_noise=lambda: self._noise_R_snapshot.copy_(
                self._g_noise.view(-1)),
            restore_noise=lambda: self._g_noise.view(-1).copy_(
                self._noise_R_snapshot),
            snapshot_g_noise_to_v_cond=lambda: self._v_cond_buf.copy_(
                self._g_noise.view(-1)),
            snapshot_g_noise_to_v_uncond=lambda: self._v_uncond_buf.copy_(
                self._g_noise.view(-1)),
            zero_g_noise=lambda: self._g_noise.zero_(),
            g_noise_ptr=lambda: self._g_noise.data_ptr(),
            v_cond_ptr=lambda: self._v_cond_buf.data_ptr(),
            v_uncond_ptr=lambda: self._v_uncond_buf.data_ptr(),
            sync=torch.cuda.synchronize,
            stream_int=0,
        )

    def _build_cfg_batched_pipeline(self, cfg_beta: float) -> None:
        """Build the Stage 3 fused CFG pipeline (paper-correct per-step CFG).

        The fused-CFG pipeline runs as a SINGLE outer CUDA Graph: lang
        swap (×2) + SigLIP (×2) + encoder_b2 + decoder_b2 (with the
        per-step ``cfg_combine_into_residual_fp16`` + noise mirror
        baked into each denoise step). One ``forward()`` =
        ``outer_graph.replay()`` + final sync, matching RTX
        :meth:`Pi05CFGBatchedPipeline.forward` /
        ``self._graph.replay(...)``.

        Changing ``cfg_beta`` requires rebuilding the pipeline (the
        beta is baked into the captured cfg_combine kernel calls).
        """
        from flash_rt.models.pi05.pipeline_thor_cfg_batched import (
            Pi05ThorCFGBatchedPipeline)
        # Mark this beta on the inner enc_ae_b2 graph for backward
        # compatibility with any callers that still trigger the eager
        # ``run_pipeline`` fallback. The hot path uses the outer graph.
        self._enc_ae_graph_b2_cfg_beta = float(cfg_beta)
        self._enc_ae_graph_b2 = None
        self._capture_enc_ae_graph_b2()
        # Capture the full fused-CFG pipeline as one graph. When the
        # frontend is constructed with ``autotune > 0``, recapture N
        # times and keep the fastest — same parameterisation as the
        # B=1 path (``self.autotune``).
        self._cfg_b2_outer_graph = None
        if self.autotune > 0:
            self._autotune_cfg_b2_outer_graph(
                cfg_beta, n_trials=self.autotune, n_bench=10)
        else:
            self._capture_cfg_b2_outer_graph(cfg_beta)

        Se = self.Se
        Sa = self.Sa

        # B=1 SigLIP graph writes into ``_enc_x[:Se]`` (vision +
        # whatever is currently in ``_lang_emb``). For batched-CFG
        # we run it twice (once with cond_lang in _lang_emb, once
        # with uncond_lang) and copy each result into the respective
        # slot of ``_enc_x_b2``. This is identical to the serial
        # CFG pattern, just routed into B=2 buffers.
        def _siglip_for_cond():
            self._lang_emb.copy_(self._lang_emb_cond)
            self._siglip_graph.replay()
            torch.cuda.synchronize()
            self._enc_x_b2[0:Se].copy_(self._enc_x[:Se])

        def _siglip_for_uncond():
            self._lang_emb.copy_(self._lang_emb_uncond)
            self._siglip_graph.replay()
            torch.cuda.synchronize()
            self._enc_x_b2[Se:2 * Se].copy_(self._enc_x[:Se])

        def _seed_b2_noise_from_R():
            # Frontend-side R: a fresh normal draw of (Sa, 32) at
            # FP16. Both slots see the SAME R so the per-step CFG
            # mirror at the end of step 0 starts from identical x_t.
            #
            # numpy CPU RNG (not torch CUDA randn) so the bit pattern
            # matches the JAX frontend's noise draw at the same
            # ``np.random.seed`` — required for cross-backend cos
            # ≥ 0.999 at all β.
            R_np = np.random.randn(Sa, 32).astype(np.float16)
            R = torch.from_numpy(R_np).to('cuda', non_blocking=True)
            self._g_noise_b2.view(-1, 32)[0:Sa].copy_(R)
            self._g_noise_b2.view(-1, 32)[Sa:2 * Sa].copy_(R)

        self._cfg_pipeline = Pi05ThorCFGBatchedPipeline(
            fvk,
            cfg_beta=cfg_beta,
            Sa=int(self.Sa),
            replay_siglip_for_cond=_siglip_for_cond,
            replay_siglip_for_uncond=_siglip_for_uncond,
            replay_enc_ae_b2=lambda: self._enc_ae_graph_b2.replay(),
            seed_b2_noise_from_R=_seed_b2_noise_from_R,
            sync=torch.cuda.synchronize,
            stream_int=0,
            outer_graph_replay=lambda: self._cfg_b2_outer_graph.replay(),
        )

    def _infer_cfg(self, observation, debug=False, seed=None):
        """CFG inference dispatcher.

        Routes to either the serial (Stage 0) per-chunk path —
        2× (siglip + enc_ae) + cfg_combine into ``g_noise`` — or
        the Stage 3 fused B=2 path: 2× SigLIP-into-different-slots,
        single B=2 enc_ae graph replay, then cfg_combine into
        ``g_noise_b2[0:Sa]``. The Pi05ThorCFGPipeline /
        Pi05ThorCFGBatchedPipeline subclass instance picks the
        right code path under the hood; the frontend just hands
        in observations + reads back the result.
        """
        t0 = time.perf_counter()
        nv = self.num_views

        if 'images' in observation:
            img_list = observation['images']
        else:
            img_list = [observation['image']]
            if nv >= 2:
                img_list.append(
                    observation.get('wrist_image', observation['image']))
            if nv >= 3:
                img_list.append(
                    observation.get('wrist_image_right', img_list[-1]))

        def _to_np16(im):
            if isinstance(im, torch.Tensor):
                return im.to(dtype=torch.float16).cpu().numpy()
            if im.dtype == np.float16:
                return im
            return (im.astype(np.float32) / 127.5 - 1.0).astype(np.float16)
        images_np = np.stack([_to_np16(im) for im in img_list[:nv]])
        self._img_buf.upload(images_np)

        # SigLIP needs to run once for the lazy-recal hook below; the
        # CFG pipeline will run it again per branch. The hook only
        # fires on the first infer call after construction, so this
        # extra siglip replay is amortized across the lifetime of the
        # frontend.
        if not self._real_data_calibrated:
            self._lang_emb.copy_(self._lang_emb_cond)
            self._siglip_graph.replay()
            torch.cuda.synchronize()
            self._recalibrate_with_real_data()
            self._real_data_calibrated = True
            # When batched, recalibrate reallocates ``_enc_calib_scales``
            # and ``_ae_calib_scales`` (new tensors) — the B=2 graph
            # captured by pointer therefore reads stale memory. The
            # torch caching allocator usually keeps the old contents
            # around long enough for the math to roughly track, but
            # this is fragile; recapture B=2 against the fresh scales
            # so the result matches B=1 cond-only deterministically.
            if self._batched and self._enc_ae_graph_b2 is not None:
                self._enc_ae_graph_b2 = None
                if (self.batched_graph_autotune and self.autotune > 0
                        and self._cfg_b2_outer_graph is None):
                    self._autotune_enc_ae_graph_b2(
                        n_trials=self.autotune, n_bench=10)
                else:
                    self._capture_enc_ae_graph_b2()
                # The outer fused-CFG graph also references the freed
                # scale buffers; recapture against the new scales,
                # honouring autotune.
                if self._cfg_b2_outer_graph is not None:
                    self._cfg_b2_outer_graph = None
                    if self.autotune > 0:
                        self._autotune_cfg_b2_outer_graph(
                            self._enc_ae_graph_b2_cfg_beta,
                            n_trials=self.autotune, n_bench=10)
                    else:
                        self._capture_cfg_b2_outer_graph(
                            self._enc_ae_graph_b2_cfg_beta)

        if self._batched:
            # Stage 3 fused B=2 CFG path. The pipeline's ``forward()``
            # is one outer-graph replay (lang swap + SigLIP×2 +
            # encoder_b2 + decoder_b2 with per-step CFG) + sync; result
            # lives in ``_g_noise_b2[0:Sa]``.
            self._cfg_pipeline.forward()
            raw_actions = self._g_noise_b2[0:self.Sa].float().cpu().numpy()
        else:
            # Stage 0 serial CFG path — pre-seed noise R and let the
            # pipeline run the full dual-branch + combine into
            # ``_g_noise``.
            #
            # numpy CPU RNG so the bit pattern matches the JAX frontend
            # under the same ``np.random.seed`` (cross-backend cos
            # contract — see ``_seed_b2_noise_from_R`` for the rationale).
            if seed is not None:
                R_np = np.random.RandomState(seed).randn(self.Sa, 32).astype(np.float16)
            else:
                R_np = np.random.randn(self.Sa, 32).astype(np.float16)
            R = torch.from_numpy(R_np).to('cuda', non_blocking=True)
            self._g_noise.view(-1, 32).copy_(R)
            self._cfg_pipeline.run_pipeline()
            raw_actions = self._g_noise.float().cpu().numpy()

        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_records.append(latency_ms)

        unnorm = unnormalize_actions(raw_actions, self.norm_stats)
        robot_actions = unnorm[:, :LIBERO_ACTION_DIM]
        if debug:
            logger.info(
                "CFG raw[0,:5]: %s, latency: %.1f ms (beta=%.2f)",
                raw_actions[0, :5], latency_ms,
                self._cfg_pipeline.cfg_beta)
        return {"actions": robot_actions}

    def set_prompt(self, prompt_text, state=None):
        """Tokenize prompt, compute time conditioning, calibrate scales, capture graphs.

        When :meth:`set_rl_mode` has activated CFG inference and the
        caller supplies a text prompt, this routes to
        :meth:`_set_prompt_rl` which builds the cond + uncond pair and
        captures the graph at the padded length. The re-entry guard
        ``_in_rl_set_prompt`` prevents infinite recursion when
        ``_set_prompt_rl`` calls back into ``set_prompt`` with token IDs
        to drive the standard capture path.
        """
        if (self._rl_config is not None
                and not getattr(self, "_in_rl_set_prompt", False)
                and isinstance(prompt_text, str)):
            if state is not None:
                raise ValueError(
                    "Pi0.5 RL CFG mode does not support state-in-prompt yet")
            self._in_rl_set_prompt = True
            try:
                self._set_prompt_rl(prompt_text)
            finally:
                self._in_rl_set_prompt = False
            return

        S_sig = self.sig_S
        nv = self.num_views

        # ---- Tokenize ----
        if isinstance(prompt_text, (np.ndarray, list)):
            token_ids = np.asarray(prompt_text, dtype=np.int64)
            prompt_len = len(token_ids)
            embeds = F.embedding(
                torch.from_numpy(token_ids).long().cuda(), self.embedding_weight)
            embeds = embeds * float(embeds.shape[-1] ** 0.5)
        else:
            max_len = PI05_STATE_PROMPT_MAX_LEN if state is not None else 48
            embeds, prompt_len = embed_prompt(
                prompt_text, self.embedding_weight, max_len=max_len,
                state=state)

        # Se must be EVEN for cuBLASLt FP8
        Se = S_sig + prompt_len
        if Se % 2 != 0:
            Se += 1
        actual_lang = Se - S_sig
        if actual_lang > prompt_len:
            embeds = torch.cat([embeds, embeds[-1:]], dim=0)

        if (self.graph_captured and self._lang_emb is not None
                and self._S_lang == actual_lang and self.Se == Se):
            self._lang_emb.copy_(embeds)
            logger.info(
                "Updated Pi0.5 Thor prompt in place: '%s' "
                "(%d tokens, Se=%d, state=%s)",
                prompt_text, prompt_len, Se, state is not None)
            return

        self.Se = Se
        self.total_keys = Se + self.Sa

        # Stage 1.4 — build AttentionBackend. total_keys must be set first
        # because the encoder/decoder KV cache layer_stride is computed from
        # the runtime total_keys (the kernel treats Kc/Vc as a contiguous
        # [L, total_keys, HD] buffer, even though it was allocated with
        # total_keys_max along dim 1). Rebuilt on every set_prompt since
        # total_keys depends on prompt length.
        attn_scale = 1.0 / math.sqrt(float(self.HDe))
        layer_stride = int(self.total_keys) * int(self.HDe) * 2  # fp16 bytes
        kc_ptr = self._Kc.reshape(-1).data_ptr()
        vc_ptr = self._Vc.reshape(-1).data_ptr()
        self._attn = ThorFlashAttnBackend(
            make_pi05_attention_spec(
                num_views=self.num_views,
                enc_seq_max=self.Se,
                chunk_size=self.Sa,
            ),
            self._ctx,
            siglip_slots={
                "qkv": self._sig_qkv.data_ptr(),
                "O":   self._sig_attn.data_ptr(),
                "D":   self.sig_D,
            },
            encoder_slots={
                "Q_O":          self._enc_attn.data_ptr(),
                "Kc":           kc_ptr,
                "Vc":           vc_ptr,
                "logits":       self._enc_logits.data_ptr(),
                "layer_stride": layer_stride,
                "scale":        attn_scale,
            },
            decoder_slots={
                "Q_O":          self._ae_attn.data_ptr(),
                "Kc":           kc_ptr,
                "Vc":           vc_ptr,
                "logits":       self._ae_logits.data_ptr(),
                "layer_stride": layer_stride,
                "scale":        attn_scale,
            },
            use_fa4=self.use_fa4,
        )

        self._lang_emb = embeds
        self._S_lang = actual_lang

        # Propagate lang_emb to batched buffer so batched PostLN sees
        # the current prompt embeddings (not stale zeros).
        if self._batched and hasattr(self, '_lang_emb_b2') and self._lang_emb_b2 is not None:
            for b in range(self.B):
                self._lang_emb_b2[b, :actual_lang].copy_(embeds[:actual_lang])

        # ---- RoPE tables ----
        self._enc_rope[:Se].copy_(
            torch.cat([self._kc_t[:Se, :, None],
                       self._ks_t[:Se, :, None]], dim=2).reshape(Se, 256))
        dec_start = Se
        self._dec_rope.copy_(
            torch.cat([self._kc_t[dec_start:dec_start + self.Sa, :, None],
                       self._ks_t[dec_start:dec_start + self.Sa, :, None]], dim=2)
            .reshape(self.Sa, 256))

        # ---- Time conditioning (precompute per-step AdaRMSNorm styles) ----
        Sa, Da, La = self.Sa, self.Da, self.La
        steps = 10; D3a = 3 * Da
        sa_all = torch.zeros(steps * La * Sa, D3a, dtype=fp16, device='cuda')
        sf_all = torch.zeros(steps * La * Sa, D3a, dtype=fp16, device='cuda')
        fs_all = torch.zeros(steps * Sa, D3a, dtype=fp16, device='cuda')

        time_embeds = []
        for step in range(steps):
            t_val = 1.0 - step / steps
            t_tensor = torch.tensor([t_val], device='cuda')
            fraction = torch.linspace(0, 1, Da // 2, device='cuda', dtype=torch.float64)
            period = 4e-3 * (4.0 / 4e-3) ** fraction
            scaling = 1.0 / period * 2 * math.pi
            sin_input = scaling * t_tensor.double()
            emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1).to(fp16)
            time_embeds.append(emb)

        for step in range(steps):
            te = time_embeds[step].unsqueeze(0)
            tmp = (te @ self._time_mlp_in_w.t() + self._time_mlp_in_b.unsqueeze(0)).float()
            tmp = (tmp * torch.sigmoid(tmp)).to(fp16)
            tmp2 = (tmp @ self._time_mlp_out_w.t() + self._time_mlp_out_b.unsqueeze(0)).float()
            tmp2 = (tmp2 * torch.sigmoid(tmp2)).to(fp16)
            time_emb = tmp2.expand(Sa, -1).contiguous()
            for layer in range(La):
                idx = (step * La + layer) * Sa
                sa_all[idx:idx + Sa] = (time_emb @ self._attn_mod_w[layer].t()
                                        + self._attn_mod_b[layer].unsqueeze(0))
                sf_all[idx:idx + Sa] = (time_emb @ self._ffn_mod_w[layer].t()
                                        + self._ffn_mod_b[layer].unsqueeze(0))
            fidx = step * Sa
            fs_all[fidx:fidx + Sa] = (time_emb @ self._final_mod_w.t()
                                      + self._final_mod_b.unsqueeze(0))

        self._sa_all = sa_all
        self._sf_all = sf_all
        self._fs_all = fs_all

        # ── B-tile style buffers when batched mode is active ──
        # Stage 2 of the Thor batched-CFG port: ``decoder_forward_b2``
        # passes ``M = B*Sa`` to the AdaRMSNorm fused kernels and walks
        # ``sa[step, layer, row]`` for ``row ∈ [0, B*Sa)``. Each
        # (step, layer) slice must be tiled B times along the row dim
        # so both samples see the same per-step style. Mirrors RTX
        # Bug 6 fix at pipeline_rtx_batched.py:195–220.
        if self._batched and self.B >= 2:
            B = self.B
            self._sa_all_b2 = sa_all.view(steps, La, Sa, D3a).repeat_interleave(
                B, dim=2).reshape(steps * La * B * Sa, D3a).contiguous()
            self._sf_all_b2 = sf_all.view(steps, La, Sa, D3a).repeat_interleave(
                B, dim=2).reshape(steps * La * B * Sa, D3a).contiguous()
            self._fs_all_b2 = fs_all.view(steps, Sa, D3a).repeat_interleave(
                B, dim=1).reshape(steps * B * Sa, D3a).contiguous()

        # ---- Capture SigLIP graph first (warmup writes enc_x with real PostLN output) ----
        self._capture_siglip_graph()

        # ---- Calibrate FP8 scales (using SigLIP warmup output in enc_x) ----
        if self.use_fp8:
            self._calibrate(Se)
        else:
            # FP16 path: no calibration needed; populate dummy scales /
            # alpha so the enc/ae forward dicts stay shape-compatible
            # (the FP16 branches never read them).
            self._enc_calib_scales = torch.zeros(self.Le * 4, dtype=torch.float32, device='cuda')
            self._enc_alpha_host = [1.0] * (self.Le * 4)
            self._ae_calib_scales = torch.zeros(self.steps * self.La * 4, dtype=torch.float32, device='cuda')
            logger.info("use_fp8=False — skipping FP8 calibration (FP16 baseline)")

        # ---- Capture encoder+decoder graph ----
        if self.autotune > 0:
            self._autotune_enc_ae(n_trials=self.autotune, n_bench=10)
        else:
            self._capture_enc_ae_graph()

        self.graph_captured = True
        self.calibrated = True
        self._current_prompt = prompt_text
        logger.info("set_prompt done: '%s' (%d tokens, Se=%d)", prompt_text, prompt_len, Se)

    # -----------------------------------------------------------------------
    # Calibration
    # -----------------------------------------------------------------------

    def _calibrate(self, Se, force_recalibrate=False, recompute_enc=False,
                   compute_ae=True):
        """Calibrate encoder + decoder FP8 activation scales.

        Args:
            Se: encoder sequence length for buffer sizing.
            force_recalibrate: if True, skip cache and compute from current data.
            recompute_enc: if True, ALWAYS recompute encoder scales from current
                data (with flash attention), even if cache hits. This is needed
                for batch inference where cached scales may have been computed
                with DIFFERENT images than the current batch.
            compute_ae: if True, also compute decoder (AE) FP8 scales. If False,
                skip AE calibration (decoder scales must be provided by caller).
                Skipping AE calibration saves ~20ms per call — critical for
                per-sample batch inference where we only need fresh enc scales.
        """
        Le = self.Le; La = self.La
        ae_scale_count = self.steps * La * 4
        total_keys = self.total_keys

        # Try cache first
        cached = load_calibration(self._checkpoint_path, Se)
        if (cached is not None
                and cached.get("metadata") == _CALIBRATION_CACHE_METADATA
                and len(cached.get("enc_scales", [])) == Le * 4
                and len(cached.get("ae_scales", [])) == ae_scale_count):
            self._enc_calib_scales = torch.tensor(
                cached["enc_scales"], dtype=torch.float32, device='cuda')
            enc_ws = self._enc_w_dev.cpu().tolist()
            self._enc_alpha_host = [
                float(np.float32(self._enc_calib_scales[i].item()) * np.float32(enc_ws[i]))
                for i in range(Le * 4)]
            if compute_ae:
                self._ae_calib_scales = torch.tensor(
                    cached["ae_scales"], dtype=torch.float32, device='cuda')
            logger.info("Calibration loaded from cache (enc=%d, ae=%d scales)",
                        Le * 4, ae_scale_count)
            logger.info(
                "Calibration cache enc stats: %s",
                _tensor_debug_stats("enc_scale", self._enc_calib_scales),
            )
            if compute_ae:
                logger.info(
                    "Calibration cache ae stats: %s",
                    _tensor_debug_stats("ae_scale", self._ae_calib_scales),
                )
            return
        if cached is not None:
            logger.info("Ignoring stale calibration cache (expected enc=%d, ae=%d scales)",
                        Le * 4, ae_scale_count)

        HDe = self.HDe; NHe = self.NHe; De = self.De; He = self.He

        # Build encoder weight/buffer dicts for calibration
        enc_bufs = {
            'x':       self._enc_x.data_ptr(),
            'x_fp8':   self._enc_x_fp8.data_ptr(),
            'qkv':     self._enc_qkv_buf.data_ptr(),
            'logits':  self._enc_logits.data_ptr(),
            'attn_out': self._enc_attn.data_ptr(),
            'o_fp8':   self._enc_o_fp8.data_ptr(),
            'gate':    self._enc_gate.data_ptr(),
            'hidden':  self._enc_hidden.data_ptr(),
            'hid_fp8': self._enc_hid_fp8.data_ptr(),
            'fg':      self._enc_fg.data_ptr(),
            'ctx':     self._ctx,
        }
        enc_weights = {
            'qkv_w':   [w.data_ptr() for w in self._enc_qkv_w],
            'o_w':     [w.data_ptr() for w in self._enc_o_w],
            'gate_w':  [w.data_ptr() for w in self._enc_gu_w],
            'down_w':  [w.data_ptr() for w in self._enc_d_w],
            'rope':    self._enc_rope.data_ptr(),
            'Kc':      self._Kc.reshape(-1).data_ptr(),
            'Vc':      self._Vc.reshape(-1).data_ptr(),
            'w_scales': self._enc_w_dev.data_ptr(),
        }
        enc_dims = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys': total_keys,
        }

        # Encoder calibration — scratch buffers allocated here
        _norm_scratch = torch.empty(Se * De, dtype=fp16, device='cuda')
        _x_scratch = torch.empty(Se * De, dtype=fp16, device='cuda')
        _calib_buf = torch.zeros(Le * 4, dtype=torch.float32, device='cuda')
        _d_scale = torch.zeros(1, dtype=torch.float32, device='cuda')
        _fp8_scratch = torch.zeros(Se * max(De, He), dtype=torch.uint8, device='cuda')
        _ones = torch.ones(De, dtype=fp16, device='cuda')
        enc_bufs['norm_scratch'] = _norm_scratch.data_ptr()
        enc_bufs['x_scratch'] = _x_scratch.data_ptr()
        enc_bufs['calib_buf'] = _calib_buf.data_ptr()
        enc_bufs['d_scale'] = _d_scale.data_ptr()
        enc_bufs['fp8_scratch'] = _fp8_scratch.data_ptr()
        enc_bufs['ones'] = _ones.data_ptr()

        # Always recompute encoder scales in this path
        self._Kc.zero_(); self._Vc.zero_()
        enc_max = torch.zeros(Le * 4, dtype=torch.float32, device='cuda')
        encoder_forward_calibrate(
            self._gemm, fvk, enc_bufs, enc_weights, enc_dims,
            enc_max.data_ptr(), stream=0, attn=self._attn)

        # In-place copy: REPLACES values but keeps SAME tensor address.
        # This is critical: any CUDA graphs reading _enc_calib_scales.data_ptr()
        # will continue reading the same (now-updated) tensor.
        self._enc_calib_scales.copy_(enc_max)
        enc_ws = self._enc_w_dev.cpu().tolist()
        self._enc_alpha_host = [
            float(np.float32(self._enc_calib_scales[i].item()) * np.float32(enc_ws[i]))
            for i in range(Le * 4)]
        logger.info(
            "Real-data encoder calibration stats: %s",
            _tensor_debug_stats("enc_scale", self._enc_calib_scales),
        )
        logger.info("Encoder calibrated: %d scales", Le * 4)
        logger.info(
            "Encoder calibration stats: %s",
            _tensor_debug_stats("enc_scale", self._enc_calib_scales),
        )

        # Decoder calibration
        Sa, Da, Ha = self.Sa, self.Da, self.Ha
        ae_bufs = {
            'noise':   self._g_noise.data_ptr(),
            'x':       self._ae_x.data_ptr(),
            'xn':      self._ae_xn.data_ptr(),
            'gate':    self._ae_gate.data_ptr(),
            'qkv':     self._ae_qkv.data_ptr(),
            'logits':  self._ae_logits.data_ptr(),
            'attn_out': self._ae_attn.data_ptr(),
            'hid':     self._ae_hid.data_ptr(),
            'fg':      self._ae_fg.data_ptr(),
            'action_f32': self._ae_action_f32.data_ptr(),
            'xn_fp8':  self._ae_xn_fp8.data_ptr(),
            'hid_fp8': self._ae_hid_fp8.data_ptr(),
            'ctx_fp8': self._ae_ctx_fp8.data_ptr(),
            'rtc_prev_action_chunk': self._rtc_prev_action_chunk.data_ptr(),
        }
        ae_weights = {
            'ain_w':     self._ain_w.data_ptr(),
            'ain_b':     self._ain_b.data_ptr(),
            'sa':        self._sa_all.data_ptr(),
            'qw':        self._dec_qkv_flat.data_ptr(),
            'Kc':        self._Kc.reshape(-1).data_ptr(),
            'Vc':        self._Vc.reshape(-1).data_ptr(),
            'ow':        self._dec_o_flat.data_ptr(),
            'sf':        self._sf_all.data_ptr(),
            'gw':        self._dec_gu_flat.data_ptr(),
            'dw':        self._dec_d_flat.data_ptr(),
            'aow':       self._aow.data_ptr(),
            'aob':       self._aob.data_ptr(),
            'aob_dt':    self._aob_dt.data_ptr(),
            'dt':        self._ae_dt,
            'fs':        self._fs_all.data_ptr(),
            'rope':      self._dec_rope.data_ptr(),
            'w_scales':  self._ae_w_dev.data_ptr(),
        }
        ae_dims = {
            'S': Sa, 'D': Da, 'H': Ha, 'NH': 8, 'HD': 256,
            'steps': 10, 'layers': self.La, 'enc_seq': Se,
            'total_keys': total_keys,
        }

        # Decoder scratch buffers
        Sa, Da, Ha = self.Sa, self.Da, self.Ha
        _ae_calib_buf = torch.zeros(ae_scale_count, dtype=torch.float32, device='cuda')
        _ae_d_scale = torch.zeros(1, dtype=torch.float32, device='cuda')
        _ae_hidden_scratch = torch.empty(Sa * Ha, dtype=fp16, device='cuda')
        _ae_fp8_scratch = torch.zeros(Sa * max(Da, Ha), dtype=torch.uint8, device='cuda')
        ae_bufs['calib_buf'] = _ae_calib_buf.data_ptr()
        ae_bufs['d_scale'] = _ae_d_scale.data_ptr()
        ae_bufs['hidden_scratch'] = _ae_hidden_scratch.data_ptr()
        ae_bufs['fp8_scratch'] = _ae_fp8_scratch.data_ptr()

        # AE calibration: only if compute_ae=True.
        # When compute_ae=False, caller is responsible for providing _ae_calib_scales.
        if compute_ae:
            self._g_noise.normal_()
            ae_max = torch.zeros(ae_scale_count, dtype=torch.float32, device='cuda')
            decoder_forward_calibrate(
                self._ctx, fvk, ae_bufs, ae_weights, ae_dims,
                ae_max.data_ptr(), stream=0, attn=self._attn)
            self._ae_calib_scales = ae_max
            logger.info("Decoder calibrated: %d scales", ae_scale_count)
            logger.info(
                "Decoder calibration stats: %s",
                _tensor_debug_stats("ae_scale", self._ae_calib_scales),
            )
        try:
            save_calibration(
                checkpoint_path=self._checkpoint_path,
                Se=Se,
                enc_scales=self._enc_calib_scales.cpu().tolist(),
                enc_alpha=self._enc_alpha_host,
                ae_scales=self._ae_calib_scales.cpu().tolist(),
                enc_w_scales=enc_ws,
                metadata=_CALIBRATION_CACHE_METADATA,
            )
        except Exception as e:
            logger.warning("Failed to save calibration cache: %s", e)

    # -----------------------------------------------------------------------
    # Patch embedding (SigLIP input)
    # -----------------------------------------------------------------------

    def _patch_embed_ops(self, stream_int, *, uint8_input=False):
        """im2col -> GEMM -> bias+pos.  Output in self._sig_x."""
        S_sig, D_sig = self.sig_S, self.sig_D
        if uint8_input:
            fvk.patch_im2col_uint8(
                self._img_u8_buf.ptr.value, self._img_u8_lut.ptr.value,
                self._patches_buf.ptr.value, self.num_views, stream_int)
        else:
            fvk.patch_im2col(
                self._img_buf.ptr.value, self._patches_buf.ptr.value,
                self.num_views, stream_int)
        self._gemm.fp16_nn(self._patches_buf.ptr.value, self._pe_w.ptr.value,
                           self._sig_x.data_ptr(), S_sig, D_sig, 588, stream_int)
        fvk.patch_embed_bias_pos(self._sig_x.data_ptr(), self._pe_b.ptr.value,
                                 self._pos_emb.ptr.value, S_sig, D_sig, 256, stream_int)

    def _patch_embed_ops_batched(self, B, stream_int, *, patches_ready=False):
        """Batched patch embed: im2col -> GEMM -> bias+pos for B samples.

        Input: self._img_buf_b2_all (contiguous B*nv images)
        Output: self._sig_x_b2 (B*S_sig, D_sig)
        """
        S_sig = self.sig_S
        D_sig = self.sig_D
        nv_b = B * self.num_views
        m = B * S_sig
        if not patches_ready:
            fvk.patch_im2col(self._img_buf_b2_all.ptr.value,
                             self._patches_buf_b2.ptr.value, nv_b, stream_int)
        self._gemm.fp16_nn(self._patches_buf_b2.ptr.value, self._pe_w.ptr.value,
                           self._sig_x_b2.data_ptr(), m, D_sig, 588, stream_int)
        fvk.patch_embed_bias_pos(self._sig_x_b2.data_ptr(), self._pe_b.ptr.value,
                                 self._pos_emb.ptr.value, m, D_sig, 256, stream_int)

    # -----------------------------------------------------------------------
    # PostLN + projection
    # -----------------------------------------------------------------------

    def _postln_project_ops(self, stream_int):
        """LayerNorm + projection + lang concat.  Writes into self._enc_x."""
        S_sig = self.sig_S; D_sig = self.sig_D; De = self.De
        postln_bufs = {
            'x_sig':   self._sig_x.data_ptr(),
            'enc_x':   self._enc_x.data_ptr(),
            'scratch': self._postln_scratch.data_ptr(),
        }
        postln_weights = {
            'ln_w':    self._postln_w.data_ptr(),
            'ln_b':    self._postln_b.data_ptr(),
            'proj_w':  self._proj_w.data_ptr(),
            'proj_b':  self._proj_b.data_ptr(),
            'lang_emb': self._lang_emb.data_ptr(),
        }
        postln_dims = {
            'S_sig': S_sig, 'D_sig': D_sig,
            'D_enc': De, 'S_lang': self._S_lang,
        }
        postln_project(self._gemm, fvk, postln_bufs, postln_weights,
                       postln_dims, stream=stream_int)

    # -----------------------------------------------------------------------
    # CUDA graph capture
    # -----------------------------------------------------------------------

    def _capture_siglip_graph(self):
        """Capture patch_embed + SigLIP + PostLN as CUDA graph."""
        siglip_use_fp8 = self.use_fp8 and os.environ.get(
            "FLASHRT_FORCE_SIGLIP_FP16", "0"
        ).lower() not in ("1", "true", "yes", "on")
        logger.info("SigLIP capture mode: siglip_fp8=%s", siglip_use_fp8)
        # Warmup: zero SigLIP input to match production (g_xs = zeros).
        # patch_embed runs but its output is zeroed before SigLIP to avoid
        # inf from bias accumulation through 27 layers.
        dummy_img = np.zeros((self.num_views, 224, 224, 3), dtype=np.float16)
        self._img_buf.upload(dummy_img)
        for _ in range(3):
            self._patch_embed_ops(0)
            self._sig_x.zero_()  # match production: SigLIP sees zeros
            siglip_forward(self._gemm, fvk, self._sig_bufs, self._sig_weights,
                           self._sig_dims, stream=0, attn=self._attn,
                           use_fp8=siglip_use_fp8)
            self._postln_project_ops(0)
        torch.cuda.synchronize()

        # Capture
        stream = torch.cuda.Stream()
        self._siglip_graph = torch.cuda.CUDAGraph()
        s_int = stream.cuda_stream
        with torch.cuda.stream(stream):
            self._siglip_graph.capture_begin()
            self._patch_embed_ops(s_int)
            siglip_forward(self._gemm, fvk, self._sig_bufs, self._sig_weights,
                           self._sig_dims, stream=s_int, attn=self._attn,
                           use_fp8=siglip_use_fp8)
            self._postln_project_ops(s_int)
            self._siglip_graph.capture_end()
        torch.cuda.synchronize()

        dummy_u8 = np.full(
            (self.num_views, 224, 224, 3), 128, dtype=np.uint8)
        self._img_u8_buf.upload(dummy_u8)
        for _ in range(3):
            self._patch_embed_ops(0, uint8_input=True)
            self._sig_x.zero_()
            siglip_forward(self._gemm, fvk, self._sig_bufs, self._sig_weights,
                           self._sig_dims, stream=0, attn=self._attn,
                           use_fp8=self.use_fp8)
            self._postln_project_ops(0)
        torch.cuda.synchronize()

        uint8_stream = torch.cuda.Stream()
        self._siglip_u8_graph = torch.cuda.CUDAGraph()
        uint8_stream_int = uint8_stream.cuda_stream
        with torch.cuda.stream(uint8_stream):
            self._siglip_u8_graph.capture_begin()
            self._patch_embed_ops(uint8_stream_int, uint8_input=True)
            siglip_forward(
                self._gemm, fvk, self._sig_bufs, self._sig_weights,
                self._sig_dims, stream=uint8_stream_int, attn=self._attn,
                use_fp8=self.use_fp8)
            self._postln_project_ops(uint8_stream_int)
            self._siglip_u8_graph.capture_end()
        torch.cuda.synchronize()
        logger.info("SigLIP CUDA graph captured (S=%d)", self.sig_S)

    def _capture_siglip_batched_graph(self, B):
        """Capture batched SigLIP + PostLN graph for B samples.

        The graph processes B*S_sig tokens in one pass.  Patch embedding
        is run before graph replay; this graph consumes the already-filled
        batched SigLIP token buffer.
        """
        from flash_rt.hardware.thor.shared_primitives_batched import (
            siglip_forward_batched, postln_project_batched)

        S_sig = self.sig_S
        D_sig = self.sig_D
        D_enc = self.De
        S_lang = self._S_lang

        # Buffers (already allocated in _alloc_b2_buffers)
        sig_x = self._sig_x_b2          # [B*S_sig, D_sig]
        sig_x_fp8 = self._sig_x_fp8_b2  # [B*S_sig * D_sig]
        sig_qkv = self._sig_qkv_b2      # [B*S_sig, 3*D_sig]
        sig_attn = self._sig_attn_b2    # [B*S_sig, D_sig]
        sig_hidden = self._sig_hidden_b2 # [B*S_sig, H_sig]
        sig_hid_fp8 = self._sig_hid_fp8_b2
        postln_scratch = self._postln_scratch_b2

        bufs = {
            'x':       sig_x.data_ptr(),
            'x_fp8':   sig_x_fp8.data_ptr(),
            'qkv':     sig_qkv.data_ptr(),
            'attn_out': sig_attn.data_ptr(),
            'hidden':  sig_hidden.data_ptr(),
            'hid_fp8': sig_hid_fp8.data_ptr(),
            'x_norm':  sig_attn.data_ptr(),
            'fg':      0,
        }

        dims = {
            'S': S_sig, 'D': D_sig, 'H': self.sig_H,
            'NH': self.sig_NH, 'HD': self.sig_HD, 'L': self.sig_L,
            'num_views': self.num_views, 'seq_per_view': 256,
            'B': B,
        }

        postln_bufs = {
            'x_sig':    sig_x.data_ptr(),
            'enc_x_b2': self._enc_x_b2.data_ptr(),
            'scratch':  postln_scratch.data_ptr(),
        }
        if self._postln_proj_b2 is not None:
            postln_bufs['proj'] = self._postln_proj_b2.data_ptr()
        # Per-sample language embedding pointers
        lang_emb_ptrs = [
            self._lang_emb_b2[b].data_ptr() for b in range(B)
        ]
        postln_weights = {
            'ln_w':         self._postln_w.data_ptr(),
            'ln_b':         self._postln_b.data_ptr(),
            'proj_w':       self._proj_w.data_ptr(),
            'proj_b':       self._proj_b.data_ptr(),
            'lang_emb_list': lang_emb_ptrs,
        }
        postln_dims = {
            'S_sig': S_sig, 'D_sig': D_sig,
            'D_enc': D_enc, 'S_lang': S_lang, 'B': B,
        }

        # SigLIP updates sig_x in place through residual projections.
        # Lazy capture happens after patch embedding has filled sig_x_b2,
        # so restore that input before capture and before the first replay.
        sig_x_snapshot = sig_x.detach().clone()

        # Warmup
        for _ in range(3):
            siglip_forward_batched(self._gemm, fvk, bufs, self._sig_weights,
                                   dims, stream=0, attn=self._attn,
                                   use_fp8=self.use_fp8)
            postln_project_batched(self._gemm, fvk, postln_bufs,
                                   postln_weights, postln_dims, stream=0)
        torch.cuda.synchronize()
        sig_x.copy_(sig_x_snapshot)
        torch.cuda.synchronize()

        # Capture
        stream = torch.cuda.Stream()
        self._siglip_batched_graph = torch.cuda.CUDAGraph()
        s_int = stream.cuda_stream
        with torch.cuda.stream(stream):
            self._siglip_batched_graph.capture_begin()
            siglip_forward_batched(self._gemm, fvk, bufs, self._sig_weights,
                                   dims, stream=s_int, attn=self._attn,
                                   use_fp8=self.use_fp8)
            postln_project_batched(self._gemm, fvk, postln_bufs,
                                   postln_weights, postln_dims, stream=s_int)
            self._siglip_batched_graph.capture_end()
        torch.cuda.synchronize()
        sig_x.copy_(sig_x_snapshot)
        self._siglip_batched_B = B
        logger.info("Batched SigLIP CUDA graph captured (B=%d, S=%d)",
                    B, B * S_sig)

    def _capture_enc_ae_graph(self):
        """Capture encoder + decoder (static FP8) as CUDA graph."""
        Se = self.Se
        total_keys = self.total_keys
        Le = self.Le; La = self.La; De = self.De; He = self.He
        NHe = self.NHe; HDe = self.HDe
        Sa = self.Sa; Da = self.Da; Ha = self.Ha
        enc_use_fp8 = self.use_fp8 and os.environ.get(
            "FLASHRT_FORCE_ENCODER_FP16", "0"
        ).lower() not in ("1", "true", "yes", "on")
        dec_use_fp8 = self.use_fp8 and os.environ.get(
            "FLASHRT_FORCE_DECODER_FP16", "0"
        ).lower() not in ("1", "true", "yes", "on")
        logger.info("Enc+AE capture modes: encoder_fp8=%s decoder_fp8=%s",
                    enc_use_fp8, dec_use_fp8)

        # Build dicts for encoder_forward
        enc_bufs = {
            'x':       self._enc_x.data_ptr(),
            'x_fp8':   self._enc_x_fp8.data_ptr(),
            'qkv':     self._enc_qkv_buf.data_ptr(),
            'logits':  self._enc_logits.data_ptr(),
            'attn_out': self._enc_attn.data_ptr(),
            'o_fp8':   self._enc_o_fp8.data_ptr(),
            'gate':    self._enc_gate.data_ptr(),
            'hidden':  self._enc_hidden.data_ptr(),
            'hid_fp8': self._enc_hid_fp8.data_ptr(),
            'fg':      self._enc_fg.data_ptr(),
            'ctx':     self._ctx,
            # FP16 path: x_norm aliases _enc_attn (same size, disjoint
            # lifetime); ones is the all-ones vector for noweight RMS.
            'x_norm':  self._enc_attn.data_ptr(),
            'ones':    (self._enc_ones_fp16.data_ptr()
                        if self._enc_ones_fp16 is not None else 0),
        }
        enc_weights = {
            'qkv_w':     [w.data_ptr() for w in self._enc_qkv_w],
            'o_w':       [w.data_ptr() for w in self._enc_o_w],
            'gate_w':    [w.data_ptr() for w in self._enc_gu_w],
            'down_w':    [w.data_ptr() for w in self._enc_d_w],
            'rope':      self._enc_rope.data_ptr(),
            'Kc':        self._Kc.reshape(-1).data_ptr(),
            'Vc':        self._Vc.reshape(-1).data_ptr(),
            'act_scales':  self._enc_calib_scales.data_ptr(),  # device float ptr
            'alpha_host':  self._enc_alpha_host,              # host float list [L*4]
        }
        enc_dims = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys': total_keys,
        }

        # Build dicts for decoder_forward
        ae_bufs = {
            'noise':   self._g_noise.data_ptr(),
            'x':       self._ae_x.data_ptr(),
            'xn':      self._ae_xn.data_ptr(),
            'gate':    self._ae_gate.data_ptr(),
            'qkv':     self._ae_qkv.data_ptr(),
            'logits':  self._ae_logits.data_ptr(),
            'attn_out': self._ae_attn.data_ptr(),
            'hid':     self._ae_hid.data_ptr(),
            'fg':      self._ae_fg.data_ptr(),
            'action_f32': self._ae_action_f32.data_ptr(),
            'xn_fp8':  self._ae_xn_fp8.data_ptr(),
            'hid_fp8': self._ae_hid_fp8.data_ptr(),
            'ctx_fp8': self._ae_ctx_fp8.data_ptr(),
            'rtc_prev_action_chunk': self._rtc_prev_action_chunk.data_ptr(),
        }
        ae_weights = {
            'ain_w':      self._ain_w.data_ptr(),
            'ain_b':      self._ain_b.data_ptr(),
            'sa':         self._sa_all.data_ptr(),
            'qw':         self._dec_qkv_flat.data_ptr(),
            'Kc':         self._Kc.reshape(-1).data_ptr(),
            'Vc':         self._Vc.reshape(-1).data_ptr(),
            'ow':         self._dec_o_flat.data_ptr(),
            'sf':         self._sf_all.data_ptr(),
            'gw':         self._dec_gu_flat.data_ptr(),
            'dw':         self._dec_d_flat.data_ptr(),
            'aow':        self._aow.data_ptr(),
            'aob':        self._aob.data_ptr(),
            'aob_dt':     self._aob_dt.data_ptr(),
            'dt':         self._ae_dt,
            'fs':         self._fs_all.data_ptr(),
            'rope':       self._dec_rope.data_ptr(),
            'w_scales':   self._ae_w_dev.data_ptr(),
            'act_scales': self._ae_calib_scales.data_ptr(),
        }
        ae_dims = {
            'S': Sa, 'D': Da, 'H': Ha, 'NH': 8, 'HD': 256,
            'steps': 10, 'layers': La, 'enc_seq': Se,
            'total_keys': total_keys,
        }

        # Warmup
        for _ in range(3):
            self._Kc.zero_(); self._Vc.zero_()
            encoder_forward(self._gemm, fvk, enc_bufs, enc_weights,
                            enc_dims, stream=0, attn=self._attn,
                            use_fp8=enc_use_fp8)
            decoder_forward(self._ctx, fvk, ae_bufs, ae_weights,
                            ae_dims, stream=0, attn=self._attn,
                            use_fp8=dec_use_fp8,
                            rtc_prefix_len=self._rtc_prefix_len)
        torch.cuda.synchronize()

        # Capture
        stream = torch.cuda.Stream()
        self._enc_ae_graph = torch.cuda.CUDAGraph()
        s_int = stream.cuda_stream
        with torch.cuda.stream(stream):
            self._enc_ae_graph.capture_begin()
            self._Kc.zero_(); self._Vc.zero_()
            encoder_forward(self._gemm, fvk, enc_bufs, enc_weights,
                            enc_dims, stream=s_int, attn=self._attn,
                            use_fp8=enc_use_fp8)
            decoder_forward(self._ctx, fvk, ae_bufs, ae_weights,
                            ae_dims, stream=s_int, attn=self._attn,
                            use_fp8=dec_use_fp8,
                            rtc_prefix_len=self._rtc_prefix_len)
            self._enc_ae_graph.capture_end()
        torch.cuda.synchronize()
        logger.info("Enc+AE CUDA graph captured (Se=%d)", Se)

    def _run_enc_ae_eager(self, *, stream=0):
        """Run the B=1 Enc+AE path directly, without CUDA Graph replay."""
        Se = self.Se
        total_keys = self.total_keys
        Le = self.Le; La = self.La; De = self.De; He = self.He
        NHe = self.NHe; HDe = self.HDe
        Sa = self.Sa; Da = self.Da; Ha = self.Ha
        enc_use_fp8 = self.use_fp8 and os.environ.get(
            "FLASHRT_FORCE_ENCODER_FP16", "0"
        ).lower() not in ("1", "true", "yes", "on")
        dec_use_fp8 = self.use_fp8 and os.environ.get(
            "FLASHRT_FORCE_DECODER_FP16", "0"
        ).lower() not in ("1", "true", "yes", "on")

        enc_bufs = {
            'x':       self._enc_x.data_ptr(),
            'x_fp8':   self._enc_x_fp8.data_ptr(),
            'qkv':     self._enc_qkv_buf.data_ptr(),
            'logits':  self._enc_logits.data_ptr(),
            'attn_out': self._enc_attn.data_ptr(),
            'o_fp8':   self._enc_o_fp8.data_ptr(),
            'gate':    self._enc_gate.data_ptr(),
            'hidden':  self._enc_hidden.data_ptr(),
            'hid_fp8': self._enc_hid_fp8.data_ptr(),
            'fg':      self._enc_fg.data_ptr(),
            'ctx':     self._ctx,
            'x_norm':  self._enc_attn.data_ptr(),
            'ones':    (self._enc_ones_fp16.data_ptr()
                        if self._enc_ones_fp16 is not None else 0),
        }
        enc_weights = {
            'qkv_w':     [w.data_ptr() for w in self._enc_qkv_w],
            'o_w':       [w.data_ptr() for w in self._enc_o_w],
            'gate_w':    [w.data_ptr() for w in self._enc_gu_w],
            'down_w':    [w.data_ptr() for w in self._enc_d_w],
            'rope':      self._enc_rope.data_ptr(),
            'Kc':        self._Kc.reshape(-1).data_ptr(),
            'Vc':        self._Vc.reshape(-1).data_ptr(),
            'act_scales':  self._enc_calib_scales.data_ptr(),
            'alpha_host':  self._enc_alpha_host,
        }
        enc_dims = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys': total_keys,
        }

        ae_bufs = {
            'noise':   self._g_noise.data_ptr(),
            'x':       self._ae_x.data_ptr(),
            'xn':      self._ae_xn.data_ptr(),
            'gate':    self._ae_gate.data_ptr(),
            'qkv':     self._ae_qkv.data_ptr(),
            'logits':  self._ae_logits.data_ptr(),
            'attn_out': self._ae_attn.data_ptr(),
            'hid':     self._ae_hid.data_ptr(),
            'fg':      self._ae_fg.data_ptr(),
            'action_f32': self._ae_action_f32.data_ptr(),
            'xn_fp8':  self._ae_xn_fp8.data_ptr(),
            'hid_fp8': self._ae_hid_fp8.data_ptr(),
            'ctx_fp8': self._ae_ctx_fp8.data_ptr(),
        }
        ae_weights = {
            'ain_w':      self._ain_w.data_ptr(),
            'ain_b':      self._ain_b.data_ptr(),
            'sa':         self._sa_all.data_ptr(),
            'qw':         self._dec_qkv_flat.data_ptr(),
            'Kc':         self._Kc.reshape(-1).data_ptr(),
            'Vc':         self._Vc.reshape(-1).data_ptr(),
            'ow':         self._dec_o_flat.data_ptr(),
            'sf':         self._sf_all.data_ptr(),
            'gw':         self._dec_gu_flat.data_ptr(),
            'dw':         self._dec_d_flat.data_ptr(),
            'aow':        self._aow.data_ptr(),
            'aob':        self._aob.data_ptr(),
            'aob_dt':     self._aob_dt.data_ptr(),
            'dt':         self._ae_dt,
            'fs':         self._fs_all.data_ptr(),
            'rope':       self._dec_rope.data_ptr(),
            'w_scales':   self._ae_w_dev.data_ptr(),
            'act_scales': self._ae_calib_scales.data_ptr(),
        }
        ae_dims = {
            'S': Sa, 'D': Da, 'H': Ha, 'NH': 8, 'HD': 256,
            'steps': 10, 'layers': La, 'enc_seq': Se,
            'total_keys': total_keys,
        }

        self._Kc.zero_(); self._Vc.zero_()
        encoder_forward(self._gemm, fvk, enc_bufs, enc_weights,
                        enc_dims, stream=stream, attn=self._attn,
                        use_fp8=enc_use_fp8)
        decoder_forward(self._ctx, fvk, ae_bufs, ae_weights,
                        ae_dims, stream=stream, attn=self._attn,
                        use_fp8=dec_use_fp8,
                        rtc_prefix_len=self._rtc_prefix_len)

    def _bad_enc_ae_graph_replay(self, raw_actions, graph_ms):
        """Detect stale/no-op B=1 graph replay using conservative signals."""
        if not np.isfinite(raw_actions).all():
            return True, "raw_nonfinite"

        fast_ms = _env_float("FLASHRT_BAD_GRAPH_FAST_MS", 20.0)
        raw_absmax = float(np.nanmax(np.abs(raw_actions))) if raw_actions.size else 0.0
        raw_absmax_limit = _env_float("FLASHRT_BAD_GRAPH_RAW_ABSMAX", 2.0)
        fast = graph_ms >= 0.0 and graph_ms < fast_ms
        noisy = raw_absmax > raw_absmax_limit
        if fast and noisy:
            return True, f"fast_and_noisy(graph_ms={graph_ms:.3f}, raw_absmax={raw_absmax:.3f})"
        if _env_flag("FLASHRT_BAD_GRAPH_GUARD_FAST_ONLY", False) and fast:
            return True, f"fast(graph_ms={graph_ms:.3f})"
        return False, ""

    def _capture_enc_ae_graph_b2(self):
        """Capture B=N encoder + decoder as a CUDA graph (Stage 2).

        Mirror of :meth:`_capture_enc_ae_graph` for the batched path:
        wires up the ``_b2``-suffixed buffers, the per-sample KV-cache
        slabs, and the B-tiled style buffers, then drives a warmup +
        capture using
        :func:`flash_rt.hardware.thor.shared_primitives_batched.encoder_forward_b2`
        and :func:`flash_rt.models.pi05.pipeline_thor_batched.decoder_forward_b2`.

        Preconditions:
          * ``self._batched`` is True and ``self._alloc_b2_buffers``
            has run.
          * ``self._sa_all_b2`` / ``_sf_all_b2`` / ``_fs_all_b2`` have
            been built by ``set_prompt`` (B-tiled).
          * Encoder + decoder FP8 scales (``_enc_calib_scales``,
            ``_ae_calib_scales``) have been calibrated. Stage 2 reuses
            the B=1 scales — Pi05BatchedPipeline's parent contract is
            "B=1 calibration transfers to B=N". Stage 3 CFG-batched
            will override with a joint cond+uncond pass.
        """
        from flash_rt.hardware.thor.shared_primitives_batched import (
            encoder_forward_b2)
        from flash_rt.models.pi05.pipeline_thor_batched import (
            decoder_forward_b2)

        B = self.B
        Se = self.Se
        total_keys = self.total_keys
        Le = self.Le; La = self.La; De = self.De; He = self.He
        NHe = self.NHe; HDe = self.HDe
        Sa = self.Sa; Da = self.Da; Ha = self.Ha

        # Per-sample KV slab device pointers (one per b).
        kc_b2 = [self._Kc_b2[b].view(-1).data_ptr() for b in range(B)]
        vc_b2 = [self._Vc_b2[b].view(-1).data_ptr() for b in range(B)]

        enc_bufs_b2 = {
            'x':       self._enc_x_b2.data_ptr(),
            'x_fp8':   self._enc_x_fp8_b2.data_ptr(),
            'qkv':     self._enc_qkv_buf_b2.data_ptr(),
            'logits':  self._enc_logits_b2.data_ptr(),
            'attn_out': self._enc_attn_b2.data_ptr(),
            'o_fp8':   self._enc_o_fp8_b2.data_ptr(),
            'gate':    self._enc_gate_b2.data_ptr(),
            'hid_fp8': self._enc_hid_fp8_b2.data_ptr(),
            'fg':      self._enc_fg_b2.data_ptr(),
            'ctx':     self._ctx,
        }
        enc_weights_b2 = {
            'qkv_w':     [w.data_ptr() for w in self._enc_qkv_w],
            'o_w':       [w.data_ptr() for w in self._enc_o_w],
            'gate_w':    [w.data_ptr() for w in self._enc_gu_w],
            'down_w':    [w.data_ptr() for w in self._enc_d_w],
            'rope':      self._enc_rope.data_ptr(),
            'Kc_b2':     kc_b2,
            'Vc_b2':     vc_b2,
            'act_scales':  self._enc_calib_scales.data_ptr(),
            'alpha_host':  self._enc_alpha_host,
        }
        enc_dims_b2 = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys': total_keys,
        }

        ae_bufs_b2 = {
            'noise':   self._g_noise_b2.data_ptr(),
            'x':       self._ae_x_b2.data_ptr(),
            'xn':      self._ae_xn_b2.data_ptr(),
            'gate':    self._ae_gate_b2.data_ptr(),
            'qkv':     self._ae_qkv_b2.data_ptr(),
            'logits':  self._ae_logits_b2.data_ptr(),
            'attn_out': self._ae_attn_b2.data_ptr(),
            'fg':      self._ae_fg_b2.data_ptr(),
            'action_f32': self._ae_action_f32_b2.data_ptr(),
            'xn_fp8':  self._ae_xn_fp8_b2.data_ptr(),
            'hid_fp8': self._ae_hid_fp8_b2.data_ptr(),
            'ctx_fp8': self._ae_ctx_fp8_b2.data_ptr(),
            # Per-step velocity scratch, only consumed when cfg_beta is
            # set (CFG-batched capture path); harmless otherwise.
            'v_b2':    self._v_b2.data_ptr(),
            'v_b2_f32': self._v_b2_f32.data_ptr(),
        }
        # Per-sample total_keys and enc_seq (uniform for graph capture)
        _total_keys_b2 = [total_keys] * B
        _enc_seq_b2 = [Se] * B

        ae_weights_b2 = {
            'ain_w':      self._ain_w.data_ptr(),
            'ain_b':      self._ain_b.data_ptr(),
            'sa':         self._sa_all_b2.data_ptr(),
            'qw':         self._dec_qkv_flat.data_ptr(),
            'Kc_b2':      kc_b2,
            'Vc_b2':      vc_b2,
            'total_keys_b2': _total_keys_b2,
            'enc_seq_b2':    _enc_seq_b2,
            'ow':         self._dec_o_flat.data_ptr(),
            'sf':         self._sf_all_b2.data_ptr(),
            'gw':         self._dec_gu_flat.data_ptr(),
            'dw':         self._dec_d_flat.data_ptr(),
            'aow':        self._aow.data_ptr(),
            'aob':        self._aob.data_ptr(),
            'aob_dt':     self._aob_dt.data_ptr(),
            'dt':         self._ae_dt,
            'fs':         self._fs_all_b2.data_ptr(),
            'rope':       self._dec_rope.data_ptr(),
            'w_scales':   self._ae_w_dev.data_ptr(),
            'act_scales': self._ae_calib_scales.data_ptr(),
        }
        ae_dims_b2 = {
            'S': Sa, 'D': Da, 'H': Ha, 'NH': 8, 'HD': 256,
            'steps': 10, 'layers': La, 'enc_seq': Se,
            'total_keys': total_keys,
        }

        # When the CFG-batched pipeline is active, bake the per-step
        # CFG combine + noise mirror into the captured graph (matches
        # RTX). Non-CFG batched paths leave this None and use the
        # standard independent-slot integration.
        cfg_beta = self._enc_ae_graph_b2_cfg_beta

        def _b2_run(stream):
            if not self.batched_skip_kv_zero:
                self._Kc_b2.zero_(); self._Vc_b2.zero_()
            encoder_forward_b2(self._gemm, fvk, enc_bufs_b2, enc_weights_b2,
                                enc_dims_b2, stream=stream, B=B)
            decoder_forward_b2(self._ctx, fvk, ae_bufs_b2, ae_weights_b2,
                                ae_dims_b2, stream=stream, B=B,
                                cfg_beta=cfg_beta)

        # Graph warmup/capture runs the encoder and decoder in place.
        # infer_multi_prompt_batch stages fresh encoder inputs and noise
        # before first lazy capture, so restore them to keep the first
        # replay bit-identical to later replays with the same seed.
        enc_x_snapshot = self._enc_x_b2.detach().clone()
        noise_snapshot = self._g_noise_b2.detach().clone()

        # Warmup
        for _ in range(3):
            _b2_run(0)
        torch.cuda.synchronize()
        self._enc_x_b2.copy_(enc_x_snapshot)
        self._g_noise_b2.copy_(noise_snapshot)
        torch.cuda.synchronize()

        # Capture
        cstream = torch.cuda.Stream()
        self._enc_ae_graph_b2 = torch.cuda.CUDAGraph()
        s_int = cstream.cuda_stream
        with torch.cuda.stream(cstream):
            self._enc_ae_graph_b2.capture_begin()
            _b2_run(s_int)
            self._enc_ae_graph_b2.capture_end()
        torch.cuda.synchronize()
        self._enc_x_b2.copy_(enc_x_snapshot)
        self._g_noise_b2.copy_(noise_snapshot)
        logger.info(
            "Enc+AE CUDA graph captured at B=%d (Se=%d, total_keys=%d, "
            "skip_kv_zero=%s)",
            B, Se, total_keys, self.batched_skip_kv_zero)

    def _capture_cfg_b2_outer_graph(self, cfg_beta: float) -> None:
        """Capture the entire fused-CFG B=2 pipeline as one outer graph.

        Replaces the multi-call Python orchestration in
        :class:`Pi05ThorCFGBatchedPipeline.run_pipeline` (lang swap +
        SigLIP×2 + enc_ae_b2) with a single ``CUDAGraph.replay()``,
        matching RTX
        :meth:`Pi05CFGBatchedPipeline.forward` /
        ``self._graph.replay(...)``.

        Captures (in order, on a fresh stream):
          1. ``lang_emb := lang_emb_cond``        (D2D)
          2. ``patch_embed + siglip + postln``    (writes _enc_x with cond)
          3. ``_enc_x → _enc_x_b2[0:Se]``         (D2D snapshot)
          4. ``lang_emb := lang_emb_uncond``      (D2D)
          5. ``patch_embed + siglip + postln``    (writes _enc_x with uncond)
          6. ``_enc_x → _enc_x_b2[Se:2*Se]``      (D2D snapshot)
          7. ``encoder_forward_b2 + decoder_forward_b2(cfg_beta=...)``
             (the latter carries the 10-step decoder loop AND the
             per-step cfg_combine + noise mirror).

        Pre-capture inputs (frontend stages these per inference, before
        ``forward()``):
          * ``_img_buf``     — observation images (H2D)
          * ``_g_noise_b2``  — fresh noise R replicated into both slots
        After replay: ``_g_noise_b2[0:Sa]`` holds the guided action chunk.

        Requires ``self._lang_emb_cond`` / ``_lang_emb_uncond`` to be
        device tensors (set up by ``_set_prompt_rl``).
        """
        from flash_rt.hardware.thor.shared_primitives_batched import (
            encoder_forward_b2)
        from flash_rt.models.pi05.pipeline_thor_batched import (
            decoder_forward_b2)

        B = self.B
        Se = self.Se
        De = self.De
        total_keys = self.total_keys
        Le = self.Le; La = self.La; He = self.He
        NHe = self.NHe; HDe = self.HDe
        Sa = self.Sa; Da = self.Da; Ha = self.Ha

        kc_b2 = [self._Kc_b2[b].view(-1).data_ptr() for b in range(B)]
        vc_b2 = [self._Vc_b2[b].view(-1).data_ptr() for b in range(B)]

        enc_bufs_b2 = {
            'x':       self._enc_x_b2.data_ptr(),
            'x_fp8':   self._enc_x_fp8_b2.data_ptr(),
            'qkv':     self._enc_qkv_buf_b2.data_ptr(),
            'logits':  self._enc_logits_b2.data_ptr(),
            'attn_out': self._enc_attn_b2.data_ptr(),
            'o_fp8':   self._enc_o_fp8_b2.data_ptr(),
            'gate':    self._enc_gate_b2.data_ptr(),
            'hid_fp8': self._enc_hid_fp8_b2.data_ptr(),
            'fg':      self._enc_fg_b2.data_ptr(),
            'ctx':     self._ctx,
        }
        enc_weights_b2 = {
            'qkv_w':     [w.data_ptr() for w in self._enc_qkv_w],
            'o_w':       [w.data_ptr() for w in self._enc_o_w],
            'gate_w':    [w.data_ptr() for w in self._enc_gu_w],
            'down_w':    [w.data_ptr() for w in self._enc_d_w],
            'rope':      self._enc_rope.data_ptr(),
            'Kc_b2':     kc_b2,
            'Vc_b2':     vc_b2,
            'act_scales':  self._enc_calib_scales.data_ptr(),
            'alpha_host':  self._enc_alpha_host,
        }
        enc_dims_b2 = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys': total_keys,
        }

        ae_bufs_b2 = {
            'noise':   self._g_noise_b2.data_ptr(),
            'x':       self._ae_x_b2.data_ptr(),
            'xn':      self._ae_xn_b2.data_ptr(),
            'gate':    self._ae_gate_b2.data_ptr(),
            'qkv':     self._ae_qkv_b2.data_ptr(),
            'logits':  self._ae_logits_b2.data_ptr(),
            'attn_out': self._ae_attn_b2.data_ptr(),
            'fg':      self._ae_fg_b2.data_ptr(),
            'action_f32': self._ae_action_f32_b2.data_ptr(),
            'xn_fp8':  self._ae_xn_fp8_b2.data_ptr(),
            'hid_fp8': self._ae_hid_fp8_b2.data_ptr(),
            'ctx_fp8': self._ae_ctx_fp8_b2.data_ptr(),
            'v_b2':    self._v_b2.data_ptr(),
            'v_b2_f32': self._v_b2_f32.data_ptr(),
        }
        ae_weights_b2 = {
            'ain_w':      self._ain_w.data_ptr(),
            'ain_b':      self._ain_b.data_ptr(),
            'sa':         self._sa_all_b2.data_ptr(),
            'qw':         self._dec_qkv_flat.data_ptr(),
            'Kc_b2':      kc_b2,
            'Vc_b2':      vc_b2,
            'ow':         self._dec_o_flat.data_ptr(),
            'sf':         self._sf_all_b2.data_ptr(),
            'gw':         self._dec_gu_flat.data_ptr(),
            'dw':         self._dec_d_flat.data_ptr(),
            'aow':        self._aow.data_ptr(),
            'aob':        self._aob.data_ptr(),
            'aob_dt':     self._aob_dt.data_ptr(),
            'dt':         self._ae_dt,
            'fs':         self._fs_all_b2.data_ptr(),
            'rope':       self._dec_rope.data_ptr(),
            'w_scales':   self._ae_w_dev.data_ptr(),
            'act_scales': self._ae_calib_scales.data_ptr(),
        }
        ae_dims_b2 = {
            'S': Sa, 'D': Da, 'H': Ha, 'NH': 8, 'HD': 256,
            'steps': 10, 'layers': La, 'enc_seq': Se,
            'total_keys': total_keys,
        }

        lang_nbytes = self._S_lang * De * 2  # fp16
        enc_x_slot_bytes = Se * De * 2

        def _outer_run(stream_int):
            # Cond branch
            fvk.gpu_copy(
                self._lang_emb.data_ptr(),
                self._lang_emb_cond.data_ptr(),
                lang_nbytes, stream_int)
            self._patch_embed_ops(stream_int)
            siglip_forward(self._gemm, fvk, self._sig_bufs,
                           self._sig_weights, self._sig_dims,
                           stream=stream_int, attn=self._attn)
            self._postln_project_ops(stream_int)
            fvk.gpu_copy(
                self._enc_x_b2.data_ptr(),
                self._enc_x.data_ptr(),
                enc_x_slot_bytes, stream_int)
            # Uncond branch (overwrites _sig_x / _enc_x; we already
            # snapshot'd cond into _enc_x_b2[0]).
            fvk.gpu_copy(
                self._lang_emb.data_ptr(),
                self._lang_emb_uncond.data_ptr(),
                lang_nbytes, stream_int)
            self._patch_embed_ops(stream_int)
            siglip_forward(self._gemm, fvk, self._sig_bufs,
                           self._sig_weights, self._sig_dims,
                           stream=stream_int, attn=self._attn)
            self._postln_project_ops(stream_int)
            fvk.gpu_copy(
                self._enc_x_b2.data_ptr() + enc_x_slot_bytes,
                self._enc_x.data_ptr(),
                enc_x_slot_bytes, stream_int)
            # Encoder + Decoder at B=2 with per-step CFG.
            if not self.batched_skip_kv_zero:
                self._Kc_b2.zero_(); self._Vc_b2.zero_()
            encoder_forward_b2(
                self._gemm, fvk, enc_bufs_b2, enc_weights_b2,
                enc_dims_b2, stream=stream_int, B=B)
            decoder_forward_b2(
                self._ctx, fvk, ae_bufs_b2, ae_weights_b2,
                ae_dims_b2, stream=stream_int, B=B,
                cfg_beta=float(cfg_beta))

        # Warmup so cuBLAS / cuDNN selects tactics + workspace before
        # the capture freezes their kernel choice.
        for _ in range(3):
            _outer_run(0)
        torch.cuda.synchronize()

        cstream = torch.cuda.Stream()
        self._cfg_b2_outer_graph = torch.cuda.CUDAGraph()
        s_int = cstream.cuda_stream
        with torch.cuda.stream(cstream):
            self._cfg_b2_outer_graph.capture_begin()
            _outer_run(s_int)
            self._cfg_b2_outer_graph.capture_end()
        torch.cuda.synchronize()
        logger.info(
            "CFG-B=2 outer CUDA graph captured (Se=%d, S_lang=%d, beta=%.2f)",
            Se, self._S_lang, cfg_beta)

    def _autotune_cfg_b2_outer_graph(self, cfg_beta: float,
                                       n_trials: int = 3,
                                       n_bench: int = 10) -> None:
        """Capture the fused-CFG outer graph N times, keep the fastest.

        Same rationale as :meth:`_autotune_enc_ae` (B=1 path): cuBLASLt
        heuristic state and CUDA graph instantiation are not
        deterministic across captures on Thor, especially under
        process-state variation between backends. Recapturing N times
        and benching each lets each backend converge on its
        locally-optimal schedule instead of being stuck with whatever
        the first heuristic call returned.

        Driven by the frontend's ``self.autotune`` knob, parameterised
        identically to the B=1 path: ``autotune=N > 0`` runs N trials
        for the B=2 outer graph too; ``autotune=0`` does a single
        capture (current behaviour).
        """
        candidates = []
        for trial in range(n_trials):
            self._cfg_b2_outer_graph = None
            self._capture_cfg_b2_outer_graph(cfg_beta)
            graph = self._cfg_b2_outer_graph

            # Benchmark this capture: outer replay + sync. The outer
            # graph already contains lang swap + SigLIP×2 + enc_ae_b2,
            # so we don't need any pre-bench priming.
            latencies = []
            for _ in range(n_bench):
                t0 = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)
            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            candidates.append((p50, graph))
            logger.info("  [B2 autotune] trial %d/%d: p50=%.2f ms",
                        trial + 1, n_trials, p50)

        best_p50, best_graph = min(candidates, key=lambda x: x[0])
        self._cfg_b2_outer_graph = best_graph
        # Drop refs to losers so torch frees their CUDAGraph state.
        for p50, g in candidates:
            if g is not best_graph:
                del g
        logger.info("  [B2 autotune] kept best: p50=%.2f ms (of %d trials)",
                    best_p50, n_trials)

    def _autotune_siglip_batched_graph(self, B: int, n_trials: int = 3,
                                       n_bench: int = 10) -> None:
        """Capture the B=N SigLIP/PostLN graph N times, keep fastest."""
        sig_x_snapshot = self._sig_x_b2.detach().clone()
        candidates = []
        for trial in range(n_trials):
            self._siglip_batched_graph = None
            self._capture_siglip_batched_graph(B)
            graph = self._siglip_batched_graph

            latencies = []
            for _ in range(n_bench):
                self._sig_x_b2.copy_(sig_x_snapshot)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)
            self._sig_x_b2.copy_(sig_x_snapshot)
            torch.cuda.synchronize()

            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            candidates.append((p50, graph))
            logger.info("  [B%d siglip autotune] trial %d/%d: p50=%.2f ms",
                        B, trial + 1, n_trials, p50)

        best_p50, best_graph = min(candidates, key=lambda x: x[0])
        self._siglip_batched_graph = best_graph
        self._siglip_batched_B = B
        for p50, g in candidates:
            if g is not best_graph:
                del g
        self._sig_x_b2.copy_(sig_x_snapshot)
        torch.cuda.synchronize()
        logger.info(
            "  [B%d siglip autotune] kept best: p50=%.2f ms (of %d trials)",
            B, best_p50, n_trials)

    def _autotune_enc_ae_graph_b2(self, n_trials: int = 3,
                                  n_bench: int = 10) -> None:
        """Capture the generic B=N enc+ae graph N times, keep fastest."""
        enc_x_snapshot = self._enc_x_b2.detach().clone()
        noise_snapshot = self._g_noise_b2.detach().clone()
        candidates = []
        for trial in range(n_trials):
            self._enc_ae_graph_b2 = None
            self._capture_enc_ae_graph_b2()
            graph = self._enc_ae_graph_b2

            latencies = []
            for _ in range(n_bench):
                self._enc_x_b2.copy_(enc_x_snapshot)
                self._g_noise_b2.copy_(noise_snapshot)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)
            self._enc_x_b2.copy_(enc_x_snapshot)
            self._g_noise_b2.copy_(noise_snapshot)
            torch.cuda.synchronize()
            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            candidates.append((p50, graph))
            logger.info("  [B%d enc+ae autotune] trial %d/%d: p50=%.2f ms",
                        self.B, trial + 1, n_trials, p50)

        best_p50, best_graph = min(candidates, key=lambda x: x[0])
        self._enc_ae_graph_b2 = best_graph
        for p50, g in candidates:
            if g is not best_graph:
                del g
        self._enc_x_b2.copy_(enc_x_snapshot)
        self._g_noise_b2.copy_(noise_snapshot)
        torch.cuda.synchronize()
        logger.info(
            "  [B%d enc+ae autotune] kept best: p50=%.2f ms (of %d trials)",
            self.B, best_p50, n_trials)

    def set_batched_mode(self, *, enable: bool = True,
                          batch_size: int = 2) -> None:
        """Switch the frontend to / from B=N batched inference (Stage 2).

        When ``enable=True``, allocates the ``_b2``-suffixed buffer
        set, B-tiles the AdaRMSNorm style buffers, and (after the
        next :meth:`set_prompt`) captures a separate
        ``_enc_ae_graph_b2`` at B*Seq shapes. The B=1 hot path is
        preserved and remains available — this method only flips a
        flag; the :meth:`infer_batch` entry point uses the b2
        buffers, while :meth:`infer` continues to use the B=1 ones.

        When ``enable=False``, drops the b2 graph reference so a
        future re-enable will re-capture, but keeps the b2 buffers
        allocated (cheap to keep around; expensive to re-alloc).

        Args:
            enable: ``True`` activates batched mode; ``False`` clears
                the active b2 graph reference.
            batch_size: Number of fused samples. Stage 2 supports
                ``batch_size=2``; future stages may extend.

        Mirrors :meth:`flash_rt.frontends.torch.pi05_rtx.Pi05TorchFrontendRtx.set_batched_mode`
        (line 1140) at the API level.
        """
        if self.batch_graph_bank_enabled:
            self._save_active_batch_graph_bank_entry()

        if not enable:
            self._batched = False
            self._enc_ae_graph_b2 = None
            self._batch_graph_bank_active_key = None
            return

        if batch_size < 2:
            raise ValueError(
                f"set_batched_mode requires batch_size >= 2; "
                f"got {batch_size}. Use the standard infer() for B=1.")

        restored = self._restore_batch_graph_bank_entry(batch_size, self.Se)

        # Lazy alloc on first enable (or on B change).
        if not restored and (self._Kc_b2 is None or self.B != batch_size):
            self._alloc_b2_buffers(B=batch_size)

        # Pre-allocate attention backends for each stream.
        # This avoids the overhead of creating ThorFlashAttnBackend copies
        # on every infer_multi_prompt_batch call, which was causing significant
        # performance overhead due to __dict__.copy() and stream_attn configuration.
        #
        # INNOVATION: Persistent per-sample attention backends with pre-configured
        # KV cache slots. Each backend is initialized once and reused across all
        # inference calls, eliminating the per-call setup overhead.
        self._stream_attn_b2 = []
        for b in range(batch_size):
            stream_attn = ThorFlashAttnBackend.__new__(ThorFlashAttnBackend)
            stream_attn.__dict__.update(self._attn.__dict__)
            stream_attn._spec = self._attn._spec
            stream_attn.refresh_per_layer_kv = self._attn.refresh_per_layer_kv
            self._stream_attn_b2.append(stream_attn)

        # Record the per-call batch Se_max so _alloc_b2_buffers can be
        # called with the correct sequence length by infer_multi_prompt_batch.
        # This is consumed by the subsequent _alloc_b2_buffers(B=B,Se_dynamic=Se_max)
        # call in that method.

        self._batched = True
        self.B = int(batch_size)
        # Force re-capture: next set_prompt or explicit
        # _capture_enc_ae_graph_b2() will rebuild the b2 graph against
        # the new prompt length.
        if not restored:
            self._enc_ae_graph_b2 = None
            self._batch_graph_bank_active_key = None
        logger.info(
            "Batched mode ENABLED (B=%d) with %d persistent attention backends%s.",
            batch_size, len(self._stream_attn_b2),
            " from graph bank" if restored else "")

    def infer_batch(self, observations):
        """Run B=N batched inference on a list of observations.

        Stage 2 of the Thor batched-CFG port. Each observation must
        be a dict with the same keys as :meth:`infer`'s ``observation``
        argument (``image``, optional ``wrist_image``, optional
        ``wrist_image_right``). The returned list has one ``actions``
        ndarray per slot.

        For Stage 2 generic batched, all samples share the same
        SigLIP B=1 graph (we replay it once and copy the vision tokens
        into both ``_enc_x_b2`` slots). The two language slots receive
        each sample's prompt embeddings — but Stage 2 uses the
        currently-set prompt for both, so callers wanting two
        different prompts should use Stage 3's
        Pi05ThorCFGBatchedPipeline (which manages cond/uncond
        explicitly).
        """
        if not self._batched:
            raise RuntimeError(
                "set_batched_mode(enable=True) must be called first")
        if self._enc_ae_graph_b2 is None:
            if self.batched_graph_autotune and self.autotune > 0:
                self._autotune_enc_ae_graph_b2(
                    n_trials=self.autotune, n_bench=10)
            else:
                self._capture_enc_ae_graph_b2()

        if isinstance(observations, dict):
            observations = [observations] * self.B
        if len(observations) != self.B:
            raise ValueError(
                f"infer_batch expected {self.B} observations; "
                f"got {len(observations)}")

        nv = self.num_views
        Se = self.Se
        S_lang = self._S_lang
        S_sig = self.sig_S
        De = self.De

        t0 = time.perf_counter()

        # ── Run SigLIP per slot (B=1 graph, B times) and stage into _enc_x_b2 ──
        # The B=1 _siglip_graph writes into _enc_x[:Se]. We then
        # memcpy into _enc_x_b2[b*Se : b*Se+Se]. For homogeneous
        # observations this could be optimized to one SigLIP run +
        # broadcast; Stage 2 keeps the simple correct path.
        for b, obs in enumerate(observations):
            if 'images' in obs:
                img_list = obs['images']
            else:
                img_list = [obs['image']]
                if nv >= 2:
                    img_list.append(
                        obs.get('wrist_image', obs['image']))
                if nv >= 3:
                    img_list.append(
                        obs.get('wrist_image_right', img_list[-1]))

            def _to_np16(im):
                if isinstance(im, torch.Tensor):
                    return im.to(dtype=torch.float16).cpu().numpy()
                if im.dtype == np.float16:
                    return im
                return (im.astype(np.float32) / 127.5 - 1.0).astype(np.float16)
            images_np = np.stack([_to_np16(im) for im in img_list[:nv]])
            self._img_buf.upload(images_np)
            self._siglip_graph.replay()
            torch.cuda.synchronize()
            # _enc_x[:Se] now has vision + lang for this sample.
            self._enc_x_b2[b * Se : (b + 1) * Se].copy_(self._enc_x[:Se])

        # ── Seed noise per slot ──
        self._g_noise_b2.normal_()

        # ── Run B=N enc+ae graph ──
        self._enc_ae_graph_b2.replay()
        if self.batched_sync_after_graph:
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_records.append(latency_ms)

        # ── Unpack per-slot actions ──
        results = []
        for b in range(self.B):
            raw = self._g_noise_b2[b * self.Sa : (b + 1) * self.Sa
                                    ].float().cpu().numpy()
            unnorm = unnormalize_actions(raw, self.norm_stats)
            results.append(
                {"actions": unnorm[:, :LIBERO_ACTION_DIM]})
        return results

    def _run_serial_batch(self, obs_list, prompts):
        """Internal: Run serial inference for each item in the list.

        Used as fallback when B=1 or when batching is not beneficial.
        Optimized to skip set_prompt when prompt matches current state.
        """
        results = []
        current_prompt = getattr(self, '_current_prompt', None)
        for obs, prompt in zip(obs_list, prompts):
            if prompt != current_prompt:
                self.set_prompt(prompt)
                current_prompt = prompt
            if isinstance(obs, dict):
                if 'images' in obs:
                    obs_input = obs.copy()
                elif 'image' in obs:
                    obs_input = obs
                else:
                    obs_input = obs
            else:
                obs_input = obs
            result = self.infer(obs_input)
            results.append(result)
        return results

    def infer_multi_prompt_batch(self, batch_data, seed=None):
        """Run B=N batched inference with DIFFERENT prompts per sample.

        Uses per-sample SigLIP + encoder with per-sample KV cache writes
        (sequential, B=1 path for exact precision), then batched decoder
        (B=N via decoder_forward_b2). The encoder runs per-sample so each
        prompt length gets the correct FP8 calibration and RoPE positions,
        avoiding the prompt-padding problem that breaks the flat batched encoder.

        Args:
            batch_data: list of dicts, each with 'observation' and 'prompt'.
            seed: optional numpy seed for deterministic noise (matches serial).

        Args:
            batch_data: list of dicts, each with:
                - 'observation': dict with 'image'/'images'
                - 'prompt': string prompt text

        Returns:
            list of dicts with 'actions' key per sample
        """
        B = len(batch_data)
        nv = self.num_views
        S_sig = self.sig_S
        De = self.De

        # Step 1: Parse observations and compute lang embeddings
        samples = []
        max_S_lang = 0
        for item in batch_data:
            obs = item['observation']
            prompt_text = item['prompt']

            embeds, S_lang = self._get_prompt_bank_entry(
                prompt_text, max_len=48)

            if 'images' in obs:
                img_list = obs['images']
            else:
                img_list = [obs['image']]
                if nv >= 2:
                    img_list.append(obs.get('wrist_image', obs['image']))
                if nv >= 3:
                    img_list.append(obs.get('wrist_image_right', img_list[-1]))

            samples.append({
                'obs': obs,
                'prompt': prompt_text,
                'images': img_list,
                'embeds': embeds,
                'S_lang': S_lang,
            })
            if S_lang > max_S_lang:
                max_S_lang = S_lang

        if max_S_lang % 2 != 0:
            max_S_lang += 1
        Se_max = S_sig + max_S_lang
        batch_S_lang = Se_max - S_sig

        # Lazy initialization: if set_prompt was never called, we need to
        # call it once to create _lang_emb, _enc_x, and other buffers that
        # this function depends on.  Use the longest prompt so _lang_emb has
        # enough capacity for all samples (truncation if prompt exceeds init
        # is acceptable — callers should call set_prompt explicitly for
        # production use).
        if not hasattr(self, '_lang_emb') or self._lang_emb is None:
            longest_sample = max(samples, key=lambda s: s['S_lang'])
            self.set_prompt(longest_sample['prompt'])

        if int(getattr(self, "_S_lang", 0) or 0) != int(batch_S_lang):
            self._S_lang = int(batch_S_lang)
            self._siglip_batched_graph = None
            self._batch_graph_bank_active_key = None

        # CRITICAL: trigger lazy real-data calibration here (before the
        # per-sample loop).  Use multi-frame calibration to handle diverse
        # images across robots/samples robustly.
        #
        # OLD BEHAVIOR: only used first sample's images for calibration,
        # causing precision issues when images have different distributions.
        #
        # NEW BEHAVIOR: collect all samples' observations and use
        # _calibrate_multi_frame with percentile reduction. This finds
        # scales that work well across ALL samples, not just the first one.
        if not self._real_data_calibrated:
            # Collect observations from all samples for multi-frame calibration
            def _get_obs_images(sample):
                obs = sample['obs']  # samples stores observation under 'obs' key
                if 'images' in obs:
                    img_list = obs['images']
                else:
                    img_list = [obs.get('image', obs.get('wrist_image'))]
                return {'images': img_list[:self.num_views]}

            obs_list = [_get_obs_images(s) for s in samples]
            logger.info("Multi-frame calibration across %d samples for robust FP8 scales", B)
            # Lower percentile (99.0) to reduce sensitivity to outlier activations
            # from edge cases like pure white/black images. This preserves precision
            # while avoiding excessive dynamic range compression.
            self._calibrate_multi_frame(obs_list, percentile=99.0, verbose=False)
            self._real_data_calibrated = True

        # Step 2: Ensure batched mode with correctly-sized b2 buffers.
        # The common case is that set_batched_mode() already allocated _Kc_b2
        # and _enc_x_b2 with the model-level max Se (e.g. 768).  This batch may
        # use a smaller Se_max (e.g. 528).  We must re-allocate to the exact
        # Se_max so that _enc_x_b2[b*Se_max : b*Se_max+Se_b] aligns with the
        # encoder output layout the decoder expects.
        # For B=1, fall back to serial inference path
        if B == 1:
            if self.batch_graph_bank_enabled:
                self._save_active_batch_graph_bank_entry()
            obs_list = [item['observation'] for item in batch_data]
            prompts = [item['prompt'] for item in batch_data]
            return self._run_serial_batch(obs_list, prompts)

        restored_from_graph_bank = False
        if self.batch_graph_bank_enabled:
            self._save_active_batch_graph_bank_entry()
            restored_from_graph_bank = self._restore_batch_graph_bank_entry(
                B, Se_max)

        force_realloc = (
            self._enc_x_b2 is None or
            self._enc_x_b2.shape[0] < B * Se_max or
            self._lang_emb_b2 is None or
            self._lang_emb_b2.shape[0] < B or
            self._lang_emb_b2.shape[1] < batch_S_lang
        )
        if not restored_from_graph_bank and (not self._batched or force_realloc):
            self._batched = False
            # Null all b2 buffers so _alloc sees them as None and re-allocates.
            self._Kc_b2 = self._Vc_b2 = None
            self._enc_x_b2 = self._enc_x_fp8_b2 = self._enc_qkv_buf_b2 = None
            self._enc_logits_b2 = self._enc_attn_b2 = self._enc_o_fp8_b2 = None
            self._enc_gate_b2 = self._enc_hid_fp8_b2 = self._enc_fg_b2 = None
            self._ae_x_b2 = self._ae_xn_b2 = self._ae_gate_b2 = None
            self._ae_qkv_b2 = self._ae_logits_b2 = self._ae_attn_b2 = None
            self._ae_fg_b2 = None
            self._ae_xn_fp8_b2 = self._ae_hid_fp8_b2 = self._ae_ctx_fp8_b2 = None
            self._g_noise_b2 = self._v_b2 = None
            self._sa_all_b2 = self._sf_all_b2 = self._fs_all_b2 = None
            self._alloc_b2_buffers(B=B, Se_dynamic=Se_max)  # pass correct Se_max NOW
            self._enc_ae_graph_b2 = None  # force graph re-capture
            self._batch_graph_bank_active_key = None

        # Update Se/total_keys for graph capture (may differ from set_prompt)
        if self.Se != Se_max:
            self.Se = Se_max
            self.total_keys = Se_max + self.Sa
            self._enc_ae_graph_b2 = None  # force re-capture
            self._batch_graph_bank_active_key = None

        # _g_noise_b2 is fully overwritten by the noise generation loop below
        # (seed path: copy from noise_tensor; no-seed: normal_()). No zero needed.
        # _ae_x_b2 and _ae_xn_b2 are fully overwritten by decoder GEMMs.

        self._batched = True
        self.B = int(B)

        # Step 3: Ensure time conditioning buffers for decoder
        decoder_steps = 10
        Sa, Da, La = self.Sa, self.Da, self.La
        D3a = 3 * Da

        if self._sa_all_b2 is None or self._sa_all_b2.shape[0] < decoder_steps * self.La * B * self.Sa:
            # Reuse base _sa_all/_sf_all/_fs_all from set_prompt if available
            # (avoids recomputing 880 tiny kernel launches for trig + MLP + matmuls).
            need_full_recompute = (
                self._sa_all is None or
                self._sa_all.shape[0] != decoder_steps * La * Sa
            )
            if need_full_recompute:
                sa_all = torch.zeros(decoder_steps * La * Sa, D3a, dtype=fp16, device='cuda')
                sf_all = torch.zeros(decoder_steps * La * Sa, D3a, dtype=fp16, device='cuda')
                fs_all = torch.zeros(decoder_steps * Sa, D3a, dtype=fp16, device='cuda')

                time_embeds = []
                for step_idx in range(decoder_steps):
                    t_val = 1.0 - step_idx / decoder_steps
                    t_tensor = torch.tensor([t_val], device='cuda')
                    fraction = torch.linspace(0, 1, Da // 2, device='cuda', dtype=torch.float64)
                    period = 4e-3 * (4.0 / 4e-3) ** fraction
                    scaling = 1.0 / period * 2 * math.pi
                    sin_input = scaling * t_tensor.double()
                    emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1).to(fp16)
                    time_embeds.append(emb)

                for step_idx in range(decoder_steps):
                    te = time_embeds[step_idx].unsqueeze(0)
                    tmp = (te @ self._time_mlp_in_w.t() + self._time_mlp_in_b.unsqueeze(0)).float()
                    tmp = (tmp * torch.sigmoid(tmp)).to(fp16)
                    tmp2 = (tmp @ self._time_mlp_out_w.t() + self._time_mlp_out_b.unsqueeze(0)).float()
                    tmp2 = (tmp2 * torch.sigmoid(tmp2)).to(fp16)
                    time_emb = tmp2.expand(Sa, -1).contiguous()
                    for layer in range(La):
                        idx = (step_idx * La + layer) * Sa
                        sa_all[idx:idx + Sa] = (time_emb @ self._attn_mod_w[layer].t()
                                                + self._attn_mod_b[layer].unsqueeze(0))
                        sf_all[idx:idx + Sa] = (time_emb @ self._ffn_mod_w[layer].t()
                                                + self._ffn_mod_b[layer].unsqueeze(0))
                    fidx = step_idx * Sa
                    fs_all[fidx:fidx + Sa] = (time_emb @ self._final_mod_w.t()
                                              + self._final_mod_b.unsqueeze(0))

                self._sa_all = sa_all
                self._sf_all = sf_all
                self._fs_all = fs_all
            else:
                sa_all = self._sa_all
                sf_all = self._sf_all
                fs_all = self._fs_all

            self._sa_all_b2 = sa_all.view(decoder_steps, La, Sa, D3a).repeat_interleave(
                B, dim=2).reshape(decoder_steps * La * B * Sa, D3a).contiguous()
            self._sf_all_b2 = sf_all.view(decoder_steps, La, Sa, D3a).repeat_interleave(
                B, dim=2).reshape(decoder_steps * La * B * Sa, D3a).contiguous()
            self._fs_all_b2 = fs_all.view(decoder_steps, Sa, D3a).repeat_interleave(
                B, dim=1).reshape(decoder_steps * B * Sa, D3a).contiguous()
            self._enc_ae_graph_b2 = None  # force re-capture with new style buffers

        # Step 5: Batched encoding (SigLIP serial, encoder+decoder via CUDA Graph)
        # ════════════════════════════════════════════════════════════════════════
        # INNOVATION: Run complete encoder pipeline (SigLIP + Transformer) for
        # each sample in parallel using CUDA streams with per-stream buffers.
        #
        # Key insight: Use manual siglip_forward/encoder_forward (no CUDA graph)
        # so each stream can run independently without global state issues.
        # KV cache zero is not needed here: the encoder overwrites every
        # KV element the decoder reads for the uniform Se graph contract.
        # Ensure batched buffers exist
        # Calculate total_keys based on max_S_lang for proper buffer sizing
        _max_S_lang = max(s['S_lang'] for s in samples)
        _total_keys_max = (_max_S_lang if _max_S_lang % 2 == 0 else _max_S_lang + 1) + S_sig + self.Sa
        if not hasattr(self, '_enc_x_b2') or self._enc_x_b2.shape[0] < B * Se_max:
            self._enc_x_b2 = torch.empty(B * Se_max, De, dtype=fp16, device='cuda')
            self._Kc_b2 = torch.zeros(B, self.La * _total_keys_max * self.HDe, dtype=fp16, device='cuda')
            self._Vc_b2 = torch.zeros(B, self.La * _total_keys_max * self.HDe, dtype=fp16, device='cuda')

        S_sig = nv * 256

        # Step 5a: Batched SigLIP + encoder
        # Pad all prompts to max length so encoder can run in batch mode.
        # KV cache does not need a pre-zero for the uniform Se graph path;
        # encoder_forward_b2 fully covers the ranges decoder_forward_b2 reads.

        S_enc = Se_max  # all samples padded to max length

        # Batched image upload: common uint8 path can directly produce patches.
        patches_ready = self._upload_images_gpu_batched(
            samples, self._img_buf_b2_all, self._patches_buf_b2)
        self._patch_embed_ops_batched(B, 0, patches_ready=patches_ready)

        # Upload per-sample language embeddings. Identical adjacent prompts
        # are coalesced into one broadcast copy by the helper.
        self._copy_lang_embeds_batched(samples)

        # Batched SigLIP + PostLN (CUDA graph for reduced kernel launch overhead)
        from flash_rt.hardware.thor.shared_primitives_batched import (
            siglip_forward_batched, postln_project_batched)

        # Capture SigLIP + PostLN graph if not already done.
        if self._siglip_batched_graph is None or self._siglip_batched_B != B:
            if self.batched_siglip_graph_autotune and self.autotune > 0:
                self._autotune_siglip_batched_graph(
                    B, n_trials=self.autotune, n_bench=10)
            else:
                self._capture_siglip_batched_graph(B)

        # Batched SigLIP + PostLN (CUDA graph for reduced kernel launch overhead)
        if (self._siglip_batched_graph is not None and
                self._siglip_batched_B == B):
            self._siglip_batched_graph.replay()
        else:
            sig_attn = self._sig_attn_b2
            bufs_b2 = {
                'x':       self._sig_x_b2.data_ptr(),
                'x_fp8':   self._sig_x_fp8_b2.data_ptr(),
                'qkv':     self._sig_qkv_b2.data_ptr(),
                'attn_out': sig_attn.data_ptr(),
                'hidden':  self._sig_hidden_b2.data_ptr(),
                'hid_fp8': self._sig_hid_fp8_b2.data_ptr(),
                'x_norm':  sig_attn.data_ptr(),
                'fg':      0,
            }
            dims_b2 = {
                'S': S_sig, 'D': self.sig_D, 'H': self.sig_H,
                'NH': self.sig_NH, 'HD': self.sig_HD, 'L': self.sig_L,
                'num_views': nv, 'seq_per_view': 256, 'B': B,
            }
            siglip_forward_batched(self._gemm, fvk, bufs_b2,
                                   self._sig_weights, dims_b2,
                                   stream=0, attn=self._attn,
                                   use_fp8=self.use_fp8)
            postln_bufs_b2 = {
                'x_sig':    self._sig_x_b2.data_ptr(),
                'enc_x_b2': self._enc_x_b2.data_ptr(),
                'scratch':  self._postln_scratch_b2.data_ptr(),
            }
            if self._postln_proj_b2 is not None:
                postln_bufs_b2['proj'] = self._postln_proj_b2.data_ptr()
            lang_emb_ptrs = [self._lang_emb_b2[b].data_ptr() for b in range(B)]
            postln_weights_b2 = {
                'ln_w':         self._postln_w.data_ptr(),
                'ln_b':         self._postln_b.data_ptr(),
                'proj_w':       self._proj_w.data_ptr(),
                'proj_b':       self._proj_b.data_ptr(),
                'lang_emb_list': lang_emb_ptrs,
            }
            postln_dims_b2 = {
                'S_sig': S_sig, 'D_sig': self.sig_D,
                'D_enc': self.De, 'S_lang': self._S_lang, 'B': B,
            }
            postln_project_batched(self._gemm, fvk, postln_bufs_b2,
                                   postln_weights_b2, postln_dims_b2, stream=0)

        # Zero out language slots beyond each sample's actual prompt length
        for b, s in enumerate(samples):
            n_lang = min(s['S_lang'], S_enc - S_sig)
            if n_lang < S_enc - S_sig:
                self._enc_x_b2[b * S_enc + S_sig + n_lang:
                               (b + 1) * S_enc].zero_()

        # Step 6: Generate noise and run batched encoder+decoder via CUDA Graph
        if seed is None:
            self._g_noise_b2.normal_()
        else:
            self._fill_seed_noise_batched(samples, seed)

        # Replay encoder+decoder as a single CUDA Graph (b2 path).
        # By default the graph skips full KV zero because encoder_forward_b2
        # overwrites the KV ranges read by decoder_forward_b2.
        # Input buffers (_enc_x_b2, _g_noise_b2) are already filled above.
        if self._enc_ae_graph_b2 is None:
            if self.batched_graph_autotune and self.autotune > 0:
                self._autotune_enc_ae_graph_b2(
                    n_trials=self.autotune, n_bench=10)
            else:
                self._capture_enc_ae_graph_b2()
        self._mark_batch_graph_bank_entry_ready(B, Se_max)
        self._enc_ae_graph_b2.replay()
        if self.batched_sync_after_graph:
            torch.cuda.synchronize()

        # Step 7: Unpack results (single D2H transfer for all samples)
        raw_all = self._g_noise_b2[:B * self.Sa].float().cpu().numpy()
        results = []
        for b in range(B):
            raw = raw_all[b * self.Sa:(b + 1) * self.Sa]
            unnorm = unnormalize_actions(raw, self.norm_stats)
            results.append({"actions": unnorm[:, :LIBERO_ACTION_DIM]})

        # Step 8: Restore attention slots to serial defaults
        # Batch modified _attn._slots['encoder'] to point to _Kc_b2/_Vc_b2.
        # Restore serial pointers so subsequent serial infer() works correctly.
        self._attn._slots['encoder']['Kc'] = self._Kc.view(-1).data_ptr()
        self._attn._slots['encoder']['Vc'] = self._Vc.view(-1).data_ptr()
        self._attn._slots['encoder']['layer_stride'] = int(self.total_keys) * int(self.HDe) * 2
        self._attn.refresh_per_layer_kv(('encoder',))

        return results

    def _autotune_enc_ae(self, n_trials=5, n_bench=10):
        """Autotune Enc+AE graph: recapture until fast schedule is found.

        CUDA Graph instantiation is non-deterministic on Thor — the same kernels
        can produce different schedules with ~2ms variance. This recaptures the
        graph until a fast schedule is obtained or max trials are exhausted.
        The LAST captured graph is always used (no stale references).

        Benchmark replicates real infer() flow: SigLIP replay → noise → Enc+AE,
        because SigLIP changes L2 cache state which affects Enc+AE performance.

        Called once per set_prompt(). Adds ~1-5s to startup.
        """
        import ctypes
        _crt = ctypes.CDLL("libcudart.so")

        def _make_ev():
            e = ctypes.c_void_p()
            _crt.cudaEventCreate(ctypes.byref(e))
            return e

        def _elapsed(a, b):
            ms = ctypes.c_float()
            _crt.cudaEventElapsedTime(ctypes.byref(ms), a, b)
            return ms.value

        # Prepare image for SigLIP (replicate real infer flow)
        dummy_img = np.zeros((self.num_views, 224, 224, 3), dtype=np.float16)
        self._img_buf.upload(dummy_img)

        logger.info("Autotune: up to %d trials for best Enc+AE graph...", n_trials)

        for trial in range(n_trials):
            self._capture_enc_ae_graph()

            # Benchmark with SigLIP in front (replicates real infer flow)
            latencies = []
            for _ in range(n_bench):
                self._siglip_graph.replay()
                e0, e1 = _make_ev(), _make_ev()
                _crt.cudaEventRecord(e0, ctypes.c_void_p(0))
                self._g_noise.normal_()
                self._enc_ae_graph.replay()
                _crt.cudaEventRecord(e1, ctypes.c_void_p(0))
                torch.cuda.synchronize()
                latencies.append(_elapsed(e0, e1))

            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            logger.info("  Trial %d: %.2f ms", trial, p50)

            # Accept if fast enough (< 38.5ms = within fast regime)
            if p50 < 38.5:
                logger.info("Autotune done: Enc+AE = %.2f ms (trial %d)", p50, trial)
                return

        logger.info("Autotune done: Enc+AE = %.2f ms (best of %d)", p50, n_trials)

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def configure_rtc_prefix(self, prefix_len: int) -> None:
        """Enable fixed-length Prefix RTC for subsequent graph captures."""
        prefix = max(0, int(prefix_len))
        if prefix > int(self.Sa):
            raise ValueError(f"rtc prefix_len {prefix} exceeds action horizon {self.Sa}")
        if prefix == self._rtc_prefix_len:
            return
        self._rtc_prefix_len = prefix
        self._enc_ae_graph = None
        logger.info("Prefix RTC %s (prefix_len=%d)",
                    "enabled" if prefix else "disabled", prefix)

    def set_rtc_prev_action_chunk(self, actions: np.ndarray | None) -> None:
        """Upload previous normalized 32-D action chunk for Prefix RTC."""
        if self._rtc_prev_action_chunk is None:
            raise RuntimeError("RTC buffers are not initialized; call set_prompt first")
        self._rtc_prev_action_chunk.zero_()
        if actions is None:
            return
        arr = np.asarray(actions, dtype=np.float16)
        if arr.ndim != 2 or arr.shape[1] != 32:
            raise ValueError(f"rtc previous chunk must be (N, 32), got {arr.shape}")
        n = min(int(arr.shape[0]), int(self.Sa))
        self._rtc_prev_action_chunk[:n].copy_(
            torch.from_numpy(np.ascontiguousarray(arr[:n])).to(device="cuda"))

    def infer(self, observation, debug=False, seed=None, rtc_prev_action_chunk=None):
        """Run inference: images -> CUDA graph replay -> actions.

        Args:
            observation: dict with 'image' and 'wrist_image' (or 'images' list).
                         Each image is (224,224,3) uint8 or float16 numpy.
        Returns:
            {"actions": np.ndarray}  shape (Sa, LIBERO_ACTION_DIM)
        """
        if self._rl_config is not None:
            return self._infer_cfg(observation, debug, seed)
        if self._rtc_prefix_len:
            self.set_rtc_prev_action_chunk(rtc_prev_action_chunk)
        debug_timing = os.environ.get("FLASHRT_INFER_DEBUG_TIMING", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        stage_ms = {}
        last_mark = None

        def _mark(name):
            nonlocal last_mark
            if not debug_timing:
                return
            torch.cuda.synchronize()
            now = time.perf_counter()
            if last_mark is not None:
                stage_ms[name] = (now - last_mark) * 1000.0
            last_mark = now

        _mark("start")
        t0 = time.perf_counter()
        nv = self.num_views

        # ---- Collect and upload images ----
        if 'images' in observation:
            img_list = observation['images']
        else:
            img_list = [observation['image']]
            if nv >= 2:
                img_list.append(
                    observation.get('wrist_image', observation['image']))
            if nv >= 3:
                img_list.append(
                    observation.get('wrist_image_right', img_list[-1]))

        use_uint8_graph = all(
            (isinstance(image, np.ndarray) and image.dtype == np.uint8)
            or (isinstance(image, torch.Tensor) and image.dtype == torch.uint8)
            for image in img_list[:nv])
        if use_uint8_graph:
            for index, image in enumerate(img_list[:nv]):
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                np.copyto(self._infer_images_u8_np[index], image)
            _mark("image_prepare")
            self._img_u8_buf.upload(self._infer_images_u8_np)
        else:
            for index, image in enumerate(img_list[:nv]):
                if isinstance(image, torch.Tensor):
                    if image.dtype == torch.uint8:
                        image = image.detach().cpu().numpy()
                    else:
                        image = image.to(dtype=torch.float16).cpu().numpy()
                if image.dtype == np.uint8:
                    np.take(
                        self._infer_uint8_to_fp16, image,
                        out=self._infer_images_np[index])
                elif image.dtype == np.float16:
                    np.copyto(self._infer_images_np[index], image)
                else:
                    np.copyto(
                        self._infer_images_np[index],
                        (image.astype(np.float32) / 127.5 - 1.0).astype(np.float16))
            _mark("image_prepare")
            self._img_buf.upload(self._infer_images_np)
        _mark("image_upload")

        # ---- Graph 1: SigLIP + PostLN ----
        if use_uint8_graph:
            self._siglip_u8_graph.replay()
        else:
            self._siglip_graph.replay()
        _mark("siglip_graph")

        # ---- Lazy real-data recalibration on first call ----
        # Skip when use_fp8=False: the FP16 baseline path has no FP8
        # scales to refresh, and ``_recalibrate_with_real_data`` reads
        # FP8-specific attrs.
        recalib_executed = False
        if self.use_fp8 and not self._real_data_calibrated:
            torch.cuda.synchronize()
            recalib_mode = os.environ.get(
                "FLASHRT_REALDATA_RECALIB_MODE", "full"
            ).strip().lower()
            if recalib_mode in {"recapture_only", "recapture-only", "skip_scales"}:
                self._capture_siglip_graph()
                self._capture_enc_ae_graph()
                logger.info(
                    "Real-data recalibration skipped; graphs recaptured with cached FP8 scales"
                )
            else:
                self._recalibrate_with_real_data()
                recalib_executed = True
            self._real_data_calibrated = True
        _mark("realdata_recalib")

        # ---- Graph 2: Encoder + Decoder ----
        # numpy CPU RNG so the bit pattern matches the JAX frontend's
        # standard infer path (cross-backend determinism + lets the RL
        # CFG path's β=1.0 output collapse cleanly to this cond-only
        # baseline — both use the same numpy seed → same R).
        if seed is not None:
            R_np = np.random.RandomState(seed).randn(self.Sa, 32).astype(np.float16)
        else:
            R_np = np.random.randn(self.Sa, 32).astype(np.float16)
        R = torch.from_numpy(R_np).to('cuda', non_blocking=True)
        if os.environ.get("FLASHRT_RECAPTURE_ENC_AE_EACH_INFER", "0").lower() in (
            "1", "true", "yes", "on"
        ):
            self._capture_enc_ae_graph()
            _mark("enc_ae_recapture")
        self._g_noise.view(-1, 32).copy_(R)
        _mark("noise_upload")
        enc_ae_mode = "graph"
        enc_ae_exec_ms = -1.0
        graph_retry_reason = ""
        if _env_flag("FLASHRT_DISABLE_ENC_AE_GRAPH", False):
            enc_ae_mode = "eager"
            exec_t0 = time.perf_counter()
            self._run_enc_ae_eager(stream=0)
            torch.cuda.synchronize()
            enc_ae_exec_ms = (time.perf_counter() - exec_t0) * 1000.0
            _mark("enc_ae_eager")
        else:
            exec_t0 = time.perf_counter()
            self._enc_ae_graph.replay()
            torch.cuda.synchronize()
            enc_ae_exec_ms = (time.perf_counter() - exec_t0) * 1000.0
            _mark("enc_ae_graph")

            if _env_flag("FLASHRT_BAD_GRAPH_GUARD", False):
                raw_probe = self._g_noise.float().cpu().numpy()
                bad_graph, graph_retry_reason = self._bad_enc_ae_graph_replay(
                    raw_probe, enc_ae_exec_ms
                )
                if bad_graph:
                    logger.warning(
                        "Suspicious Enc+AE graph replay detected (%s); "
                        "recapturing and retrying once",
                        graph_retry_reason,
                    )
                    retry_t0 = time.perf_counter()
                    self._capture_enc_ae_graph()
                    retry_capture_ms = (time.perf_counter() - retry_t0) * 1000.0
                    stage_ms["bad_graph_recapture_ms"] = retry_capture_ms
                    self._g_noise.view(-1, 32).copy_(R)
                    retry_t1 = time.perf_counter()
                    self._enc_ae_graph.replay()
                    torch.cuda.synchronize()
                    enc_ae_exec_ms = (time.perf_counter() - retry_t1) * 1000.0
                    stage_ms["bad_graph_retry_ms"] = enc_ae_exec_ms
                    enc_ae_mode = "graph_retry"
                    _mark("enc_ae_graph_retry")

        if os.environ.get("FLASHRT_FP8_DEBUG_BUFFERS", "0").lower() in (
            "1", "true", "yes", "on"
        ):
            for label, tensor in (
                ("enc_w_scale", getattr(self, "_enc_w_dev", None)),
                ("ae_w_scale", getattr(self, "_ae_w_dev", None)),
                ("enc_act_scale", getattr(self, "_enc_calib_scales", None)),
                ("ae_act_scale", getattr(self, "_ae_calib_scales", None)),
                ("enc_x", getattr(self, "_enc_x", None)),
                ("enc_qkv", getattr(self, "_enc_qkv_buf", None)),
                ("Kc", getattr(self, "_Kc", None)),
                ("Vc", getattr(self, "_Vc", None)),
                ("ae_x", getattr(self, "_ae_x", None)),
                ("ae_xn", getattr(self, "_ae_xn", None)),
                ("ae_qkv", getattr(self, "_ae_qkv", None)),
                ("ae_attn", getattr(self, "_ae_attn", None)),
                ("ae_fg", getattr(self, "_ae_fg", None)),
                ("ae_hid", getattr(self, "_ae_hid", None)),
                ("ae_action_f32", getattr(self, "_ae_action_f32", None)),
                ("g_noise", getattr(self, "_g_noise", None)),
            ):
                if tensor is not None:
                    _log_tensor_debug_stats(label, tensor)

        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_records.append(latency_ms)

        # ---- Post-process ----
        raw_actions = self._g_noise.float().cpu().numpy()
        unnorm = unnormalize_actions(raw_actions, self.norm_stats)
        robot_actions = unnorm[:, :LIBERO_ACTION_DIM]
        _mark("postprocess")

        if debug_timing:
            logger.info("FlashRT raw action stats: %s", _array_debug_stats("raw", raw_actions))
            if hasattr(self, "_enc_calib_scales"):
                logger.info(
                    "FlashRT enc scale stats: %s",
                    _tensor_debug_stats("enc_scale", self._enc_calib_scales),
                )
            if hasattr(self, "_ae_calib_scales"):
                logger.info(
                    "FlashRT ae scale stats: %s",
                    _tensor_debug_stats("ae_scale", self._ae_calib_scales),
                )

        if debug:
            logger.info("Raw actions[0,:5]: %s", raw_actions[0, :5])
            logger.info("Robot actions[0]: %s", robot_actions[0])
            logger.info("Latency: %.1f ms", latency_ms)

        result = {"actions": robot_actions}
        if self._rtc_prefix_len:
            result["rtc_raw_actions"] = raw_actions.astype(np.float16, copy=True)
        if debug_timing:
            stage_ms["enc_ae_exec_ms"] = enc_ae_exec_ms
            stage_ms["enc_ae_mode"] = {
                "graph": 0.0,
                "eager": 1.0,
                "graph_retry": 2.0,
            }.get(enc_ae_mode, -1.0)
            stage_ms["bad_graph_retry"] = 1.0 if enc_ae_mode == "graph_retry" else 0.0
            stage_ms["total"] = latency_ms
            stage_ms["realdata_recalib_executed"] = 1.0 if recalib_executed else 0.0
            stage_ms["realdata_calibrated"] = (
                1.0 if getattr(self, "_real_data_calibrated", False) else 0.0
            )
            stage_ms["raw_finite"] = 1.0 if np.isfinite(raw_actions).all() else 0.0
            stage_ms["raw_nan_count"] = float(np.isnan(raw_actions).sum())
            result["frontend_stage_ms"] = stage_ms
            logger.info(
                "FlashRT infer stages: %s",
                {k: round(float(v), 3) for k, v in stage_ms.items()},
            )

        return result

    # -----------------------------------------------------------------------
    # Pipelined Encoder-Decoder Inference
    # -----------------------------------------------------------------------

    def enable_pipeline_mode(self):
        """Enable double-buffered KV cache for encoder-decoder pipelining.

        Creates alternate KV cache buffers and CUDA streams so that
        encoder(frame N+1) can overlap with decoder(frame N).
        Call once after set_prompt().
        """
        if not hasattr(self, '_Kc') or self._Kc is None:
            raise RuntimeError("set_prompt() must be called before enable_pipeline_mode()")

        # Create alternate KV cache buffers (same shape as originals)
        self._Kc_alt = torch.zeros_like(self._Kc)
        self._Vc_alt = torch.zeros_like(self._Vc)

        # CUDA streams for pipelining
        self._enc_stream = torch.cuda.Stream()
        self._dec_stream = torch.cuda.Stream()

        # State: which KV buffer is "current" (encoder writes here)
        # and which is "previous" (decoder reads from here)
        self._kv_is_alt = False  # False: enc→_Kc, dec→_Kc_alt; True: opposite
        self._has_prev_frame = False
        self._prev_noise = None  # noise tensor for the previous frame's decoder
        self._prev_actions = None

        logger.info("Pipeline mode enabled (double-buffered KV cache)")

    def _swap_kv_buffers(self):
        """Swap current/alternate KV cache buffers and update attention backend."""
        self._kv_is_alt = not self._kv_is_alt
        if self._kv_is_alt:
            enc_Kc, enc_Vc = self._Kc_alt, self._Vc_alt
            dec_Kc, dec_Vc = self._Kc, self._Vc
        else:
            enc_Kc, enc_Vc = self._Kc, self._Vc
            dec_Kc, dec_Vc = self._Kc_alt, self._Vc_alt

        # Update attention backend slots for encoder
        self._attn._slots['encoder']['Kc'] = enc_Kc.reshape(-1).data_ptr()
        self._attn._slots['encoder']['Vc'] = enc_Vc.reshape(-1).data_ptr()
        # Update attention backend slots for decoder
        self._attn._slots['decoder']['Kc'] = dec_Kc.reshape(-1).data_ptr()
        self._attn._slots['decoder']['Vc'] = dec_Vc.reshape(-1).data_ptr()
        self._attn.refresh_per_layer_kv(["encoder", "decoder"])

        return enc_Kc, enc_Vc, dec_Kc, dec_Vc

    def infer_pipelined(self, observation, seed=None):
        """Pipelined inference: encoder(frame N+1) overlaps decoder(frame N).

        On the first call, runs full SigLIP + Encoder + Decoder (no overlap).
        On subsequent calls, overlaps the encoder of the current frame with
        the decoder of the previous frame, saving ~3-6ms per frame.

        Args:
            observation: dict with 'image' and 'wrist_image'.
            seed: optional numpy seed for deterministic noise.

        Returns:
            {"actions": np.ndarray} shape (Sa, LIBERO_ACTION_DIM)
            On the first call, returns actions for frame 0.
            On subsequent calls, returns actions for the PREVIOUS frame
            (the one whose decoder just completed).
        """
        t0 = time.perf_counter()
        nv = self.num_views

        # ---- Collect and upload images ----
        if 'images' in observation:
            img_list = observation['images']
        else:
            img_list = [observation['image']]
            if nv >= 2:
                img_list.append(observation.get('wrist_image', observation['image']))
            if nv >= 3:
                img_list.append(observation.get('wrist_image_right', img_list[-1]))

        def _to_np16(im):
            if isinstance(im, torch.Tensor):
                return im.to(dtype=torch.float16).cpu().numpy()
            if im.dtype == np.float16:
                return im
            return (im.astype(np.float32) / 127.5 - 1.0).astype(np.float16)

        images_np = np.stack([_to_np16(im) for im in img_list[:nv]])

        # Generate noise for current frame
        if seed is not None:
            R_np = np.random.RandomState(seed).randn(self.Sa, 32).astype(np.float16)
        else:
            R_np = np.random.randn(self.Sa, 32).astype(np.float16)
        cur_noise = torch.from_numpy(R_np).to('cuda', non_blocking=True)

        if not self._has_prev_frame:
            # ---- First frame: full sequential path (no overlap) ----
            self._img_buf.upload(images_np)
            self._siglip_graph.replay()

            # Swap KV buffers: encoder writes to current, decoder reads from same
            enc_Kc, enc_Vc, dec_Kc, dec_Vc = self._swap_kv_buffers()
            enc_Kc.zero_(); enc_Vc.zero_()

            # Encoder + Decoder on default stream
            self._g_noise.view(-1, 32).copy_(cur_noise)
            self._enc_ae_graph.replay()
            torch.cuda.synchronize()

            # Save state for next call
            self._has_prev_frame = True
            self._prev_noise = cur_noise

            latency_ms = (time.perf_counter() - t0) * 1000
            self.latency_records.append(latency_ms)

            raw_actions = self._g_noise.float().cpu().numpy()
            unnorm = unnormalize_actions(raw_actions, self.norm_stats)
            return {"actions": unnorm[:, :LIBERO_ACTION_DIM]}

        # ---- Subsequent frames: pipelined execution ----

        # 1. Swap KV buffers: encoder writes to new buffer, decoder reads from old
        enc_Kc, enc_Vc, dec_Kc, dec_Vc = self._swap_kv_buffers()

        # 2. Upload image + SigLIP on enc_stream
        with torch.cuda.stream(self._enc_stream):
            self._img_buf.upload(images_np)
            self._siglip_graph.replay()
            # Zero the encoder's KV cache (must happen after swap)
            enc_Kc.zero_(); enc_Vc.zero_()
            # Encoder (writes to enc_Kc/enc_Vc)
            from flash_rt.hardware.thor.shared_primitives import encoder_forward
            enc_bufs = {
                'x': self._enc_x.data_ptr(), 'x_fp8': self._enc_x_fp8.data_ptr(),
                'qkv': self._enc_qkv_buf.data_ptr(), 'logits': self._enc_logits.data_ptr(),
                'attn_out': self._enc_attn.data_ptr(), 'o_fp8': self._enc_o_fp8.data_ptr(),
                'gate': self._enc_gate.data_ptr(), 'hidden': self._enc_hidden.data_ptr(),
                'hid_fp8': self._enc_hid_fp8.data_ptr(), 'fg': self._enc_fg.data_ptr(),
                'ctx': self._ctx,
                'x_norm': self._enc_attn.data_ptr(),
                'ones': self._enc_ones_fp16.data_ptr() if self._enc_ones_fp16 is not None else 0,
            }
            enc_weights = {
                'qkv_w': [w.data_ptr() for w in self._enc_qkv_w],
                'o_w': [w.data_ptr() for w in self._enc_o_w],
                'gate_w': [w.data_ptr() for w in self._enc_gu_w],
                'down_w': [w.data_ptr() for w in self._enc_d_w],
                'rope': self._enc_rope.data_ptr(),
                'Kc': enc_Kc.reshape(-1).data_ptr(),
                'Vc': enc_Vc.reshape(-1).data_ptr(),
                'act_scales': self._enc_calib_scales.data_ptr(),
                'alpha_host': self._enc_alpha_host,
            }
            enc_dims = {
                'Se': self.Se, 'D': self.De, 'H': self.He, 'NH': self.NHe,
                'HD': self.HDe, 'L': self.Le, 'total_keys': self.total_keys,
            }
            if self.use_fp8:
                enc_weights['act_scales'] = self._enc_calib_scales.data_ptr()
                enc_weights['alpha_host'] = self._enc_alpha_host
            s_enc = self._enc_stream.cuda_stream
            encoder_forward(self._gemm, fvk, enc_bufs, enc_weights,
                           enc_dims, stream=s_enc, attn=self._attn,
                           use_fp8=self.use_fp8)

        # 3. Decoder for PREVIOUS frame on dec_stream (reads from dec_Kc/dec_Vc)
        with torch.cuda.stream(self._dec_stream):
            self._g_noise.view(-1, 32).copy_(self._prev_noise)
            from flash_rt.models.pi05.pipeline_thor import decoder_forward
            ae_bufs = {
                'noise': self._g_noise.data_ptr(), 'x': self._ae_x.data_ptr(),
                'xn': self._ae_xn.data_ptr(), 'gate': self._ae_gate.data_ptr(),
                'qkv': self._ae_qkv.data_ptr(), 'logits': self._ae_logits.data_ptr(),
                'attn_out': self._ae_attn.data_ptr(), 'hid': self._ae_hid.data_ptr(),
                'fg': self._ae_fg.data_ptr(), 'xn_fp8': self._ae_xn_fp8.data_ptr(),
                'hid_fp8': self._ae_hid_fp8.data_ptr(), 'ctx_fp8': self._ae_ctx_fp8.data_ptr(),
            }
            ae_weights = {
                'ain_w': self._ain_w.data_ptr(), 'ain_b': self._ain_b.data_ptr(),
                'sa': self._sa_all.data_ptr(), 'qw': self._dec_qkv_flat.data_ptr(),
                'Kc': dec_Kc.reshape(-1).data_ptr(),
                'Vc': dec_Vc.reshape(-1).data_ptr(),
                'ow': self._dec_o_flat.data_ptr(), 'sf': self._sf_all.data_ptr(),
                'gw': self._dec_gu_flat.data_ptr(), 'dw': self._dec_d_flat.data_ptr(),
                'aow': self._aow.data_ptr(), 'aob': self._aob.data_ptr(),
                'fs': self._fs_all.data_ptr(), 'rope': self._dec_rope.data_ptr(),
                'w_scales': self._ae_w_dev.data_ptr(),
                'act_scales': self._ae_calib_scales.data_ptr(),
            }
            ae_dims = {
                'S': self.Sa, 'D': self.Da, 'H': self.Ha, 'NH': 8, 'HD': 256,
                'steps': 10, 'layers': self.La, 'enc_seq': self.Se,
                'total_keys': self.total_keys,
            }
            s_dec = self._dec_stream.cuda_stream
            decoder_forward(self._ctx, fvk, ae_bufs, ae_weights,
                           ae_dims, stream=s_dec, attn=self._attn,
                           use_fp8=self.use_fp8)

        # 4. Wait for both streams
        torch.cuda.synchronize()

        # Save current noise for next call's decoder
        self._prev_noise = cur_noise

        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_records.append(latency_ms)

        # Return actions from the PREVIOUS frame's decoder
        raw_actions = self._g_noise.float().cpu().numpy()
        unnorm = unnormalize_actions(raw_actions, self.norm_stats)
        return {"actions": unnorm[:, :LIBERO_ACTION_DIM]}

    # -----------------------------------------------------------------------
    # Real-data recalibration (called once after first real image)
    # -----------------------------------------------------------------------

    def calibrate(
        self,
        observations,
        *,
        percentile: float = 99.9,
        max_samples=None,
        verbose: bool = False,
    ) -> None:
        """Unified calibration API (Thor).

        N=1: runs the existing implicit recalibration path via ``infer``
        (bit-equal to the legacy "first call auto-calibrates" behaviour).

        N>=2: per-sample amax collection on both encoder and decoder
        calibrate passes, reduced via ``np.percentile(axis=0)``. The CUDA
        graph is recaptured once at the end so the caller can run
        ``infer`` immediately afterwards without the lazy-calibrate
        branch firing again.
        """
        if isinstance(observations, dict):
            obs_list = [observations]
        elif isinstance(observations, list):
            obs_list = observations
        else:
            obs_list = list(observations)
        if max_samples is not None:
            obs_list = obs_list[:max_samples]
        n = len(obs_list)
        if n == 0:
            raise ValueError("observations must contain at least 1 sample")
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")

        if n == 1:
            # Legacy implicit-calibrate path: one infer() flips
            # _real_data_calibrated via the lazy branch in infer().
            from flash_rt.core.calibration_api import implicit_calibrate
            implicit_calibrate(
                self, obs_list,
                percentile=percentile, max_samples=None, verbose=verbose,
            )
        else:
            self._calibrate_multi_frame(
                obs_list, percentile=percentile, verbose=verbose)

    def calibrate_with_real_data(self, sample_observations) -> None:
        """Legacy alias for :meth:`calibrate`."""
        self.calibrate(sample_observations)

    @property
    def precision_spec(self):
        """Thor frontends do not yet surface a structured PrecisionSpec.

        The scale buffers are populated in-device by the calibration
        kernels. A future release will add a snapshot path parallel to
        the RTX frontends.
        """
        return None

    def _recalibrate_with_real_data(self):
        """Recalibrate using real SigLIP output, then recapture enc+ae graph."""
        Se = self.Se; Le = self.Le; La = self.La
        ae_scale_count = self.steps * La * 4
        total_keys = self.total_keys
        De = self.De; He = self.He; NHe = self.NHe; HDe = self.HDe

        enc_bufs = {
            'x':       self._enc_x.data_ptr(),
            'x_fp8':   self._enc_x_fp8.data_ptr(),
            'qkv':     self._enc_qkv_buf.data_ptr(),
            'logits':  self._enc_logits.data_ptr(),
            'attn_out': self._enc_attn.data_ptr(),
            'o_fp8':   self._enc_o_fp8.data_ptr(),
            'gate':    self._enc_gate.data_ptr(),
            'hidden':  self._enc_hidden.data_ptr(),
            'hid_fp8': self._enc_hid_fp8.data_ptr(),
            'fg':      self._enc_fg.data_ptr(),
            'ctx':     self._ctx,
        }
        enc_weights = {
            'qkv_w':   [w.data_ptr() for w in self._enc_qkv_w],
            'o_w':     [w.data_ptr() for w in self._enc_o_w],
            'gate_w':  [w.data_ptr() for w in self._enc_gu_w],
            'down_w':  [w.data_ptr() for w in self._enc_d_w],
            'rope':    self._enc_rope.data_ptr(),
            'Kc':      self._Kc.reshape(-1).data_ptr(),
            'Vc':      self._Vc.reshape(-1).data_ptr(),
            'w_scales': self._enc_w_dev.data_ptr(),
        }
        enc_dims = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys': total_keys,
        }

        # Scratch buffers for recalibration
        _norm_scratch = torch.empty(Se * De, dtype=fp16, device='cuda')
        _x_scratch = torch.empty(Se * De, dtype=fp16, device='cuda')
        _calib_buf = torch.zeros(Le * 4, dtype=torch.float32, device='cuda')
        _d_scale = torch.zeros(1, dtype=torch.float32, device='cuda')
        _fp8_scratch = torch.zeros(Se * max(De, He), dtype=torch.uint8, device='cuda')
        _ones = torch.ones(De, dtype=fp16, device='cuda')
        enc_bufs['norm_scratch'] = _norm_scratch.data_ptr()
        enc_bufs['x_scratch'] = _x_scratch.data_ptr()
        enc_bufs['calib_buf'] = _calib_buf.data_ptr()
        enc_bufs['d_scale'] = _d_scale.data_ptr()
        enc_bufs['fp8_scratch'] = _fp8_scratch.data_ptr()
        enc_bufs['ones'] = _ones.data_ptr()

        self._enc_calib_scales.zero_()
        self._Kc.zero_(); self._Vc.zero_()
        encoder_forward_calibrate(
            self._gemm, fvk, enc_bufs, enc_weights, enc_dims,
            self._enc_calib_scales.data_ptr(), stream=0, attn=self._attn)
        torch.cuda.synchronize()

        enc_ws = self._enc_w_dev.cpu().tolist()
        self._enc_alpha_host = [
            float(np.float32(self._enc_calib_scales[i].item()) * np.float32(enc_ws[i]))
            for i in range(Le * 4)]

        # Recalibrate decoder
        Sa, Da, Ha = self.Sa, self.Da, self.Ha
        ae_bufs = {
            'noise':   self._g_noise.data_ptr(),
            'x':       self._ae_x.data_ptr(),
            'xn':      self._ae_xn.data_ptr(),
            'gate':    self._ae_gate.data_ptr(),
            'qkv':     self._ae_qkv.data_ptr(),
            'logits':  self._ae_logits.data_ptr(),
            'attn_out': self._ae_attn.data_ptr(),
            'hid':     self._ae_hid.data_ptr(),
            'fg':      self._ae_fg.data_ptr(),
            'action_f32': self._ae_action_f32.data_ptr(),
            'xn_fp8':  self._ae_xn_fp8.data_ptr(),
            'hid_fp8': self._ae_hid_fp8.data_ptr(),
            'ctx_fp8': self._ae_ctx_fp8.data_ptr(),
        }
        ae_weights = {
            'ain_w':      self._ain_w.data_ptr(),
            'ain_b':      self._ain_b.data_ptr(),
            'sa':         self._sa_all.data_ptr(),
            'qw':         self._dec_qkv_flat.data_ptr(),
            'Kc':         self._Kc.reshape(-1).data_ptr(),
            'Vc':         self._Vc.reshape(-1).data_ptr(),
            'ow':         self._dec_o_flat.data_ptr(),
            'sf':         self._sf_all.data_ptr(),
            'gw':         self._dec_gu_flat.data_ptr(),
            'dw':         self._dec_d_flat.data_ptr(),
            'aow':        self._aow.data_ptr(),
            'aob':        self._aob.data_ptr(),
            'aob_dt':     self._aob_dt.data_ptr(),
            'dt':         self._ae_dt,
            'fs':         self._fs_all.data_ptr(),
            'rope':       self._dec_rope.data_ptr(),
            'w_scales':   self._ae_w_dev.data_ptr(),
        }
        ae_dims = {
            'S': Sa, 'D': Da, 'H': Ha, 'NH': 8, 'HD': 256,
            'steps': 10, 'layers': self.La, 'enc_seq': Se,
            'total_keys': total_keys,
        }

        _ae_calib_buf = torch.zeros(ae_scale_count, dtype=torch.float32, device='cuda')
        _ae_d_scale = torch.zeros(1, dtype=torch.float32, device='cuda')
        _ae_hidden_scratch = torch.empty(Sa * Ha, dtype=fp16, device='cuda')
        _ae_fp8_scratch = torch.zeros(Sa * max(Da, Ha), dtype=torch.uint8, device='cuda')
        ae_bufs['calib_buf'] = _ae_calib_buf.data_ptr()
        ae_bufs['d_scale'] = _ae_d_scale.data_ptr()
        ae_bufs['hidden_scratch'] = _ae_hidden_scratch.data_ptr()
        ae_bufs['fp8_scratch'] = _ae_fp8_scratch.data_ptr()

        self._ae_calib_scales.zero_()
        self._g_noise.normal_()
        decoder_forward_calibrate(
            self._ctx, fvk, ae_bufs, ae_weights, ae_dims,
            self._ae_calib_scales.data_ptr(), stream=0, attn=self._attn)
        torch.cuda.synchronize()
        logger.info(
            "Real-data decoder calibration stats: %s",
            _tensor_debug_stats("ae_scale", self._ae_calib_scales),
        )

        # Recapture graphs with updated scales.
        # Both the SigLIP graph (PostLN FP8 scales) and the enc+ae graph
        # must be refreshed — the SigLIP graph was captured at set_prompt()
        # time with pre-calibration scales.
        self._capture_siglip_graph()
        self._capture_enc_ae_graph()
        # Also recapture the batched enc+ae graph if batched mode is active.
        if self._batched and self.B >= 2:
            if self.batched_graph_autotune and self.autotune > 0:
                self._autotune_enc_ae_graph_b2(
                    n_trials=self.autotune, n_bench=10)
            else:
                self._capture_enc_ae_graph_b2()
        logger.info("Recalibrated with real data + graphs recaptured")

    # -----------------------------------------------------------------------
    # Multi-sample dataset calibration (N>=2)
    # -----------------------------------------------------------------------

    def _calibrate_multi_frame(
        self, obs_list, *, percentile: float, verbose: bool,
    ) -> None:
        """Per-sample amax across N observations, reduced via percentile.

        Reuses the existing ``encoder_forward_calibrate`` /
        ``decoder_forward_calibrate`` kernels (no new CUDA code). Each
        sample writes amax into ``_enc_calib_scales[Le*4]`` and
        ``_ae_calib_scales[steps*La*4]``; we snapshot both to CPU per sample,
        then ``accumulate_amax(percentile)`` column-wise, upload the
        reduced scales once, recompute ``_enc_alpha_host`` (the cuBLASLt
        alpha-fold list), and recapture the encoder+decoder CUDA graph.

        Mirrors :meth:`Pi05TorchFrontendRtx._calibrate_multi_frame` in
        shape so user code is portable, but the underlying scale
        buffers differ (Thor exposes single device tensors per stage
        rather than a per-tensor dict).
        """
        from flash_rt.core.calibration import (
            accumulate_amax,
            check_scale_ceiling,
            format_summary,
            summarize_amax_dispersion,
        )

        n = len(obs_list)
        logger.info(
            "Pi0.5 Thor: calibrating FP8 across %d real samples "
            "(percentile=%.2f)...", n, percentile)

        Se = self.Se; Le = self.Le; La = self.La
        ae_scale_count = self.steps * La * 4
        total_keys = self.total_keys
        De = self.De; He = self.He; NHe = self.NHe; HDe = self.HDe
        Sa, Da, Ha = self.Sa, self.Da, self.Ha
        nv = self.num_views

        # Buffer setup identical to _recalibrate_with_real_data — allocate
        # scratch once and reuse across the N forward calibrate passes.
        _norm_scratch = torch.empty(Se * De, dtype=fp16, device='cuda')
        _x_scratch = torch.empty(Se * De, dtype=fp16, device='cuda')
        _calib_buf = torch.zeros(Le * 4, dtype=torch.float32, device='cuda')
        _d_scale = torch.zeros(1, dtype=torch.float32, device='cuda')
        _fp8_scratch_enc = torch.zeros(
            Se * max(De, He), dtype=torch.uint8, device='cuda')
        _ones = torch.ones(De, dtype=fp16, device='cuda')

        _ae_calib_buf = torch.zeros(ae_scale_count, dtype=torch.float32, device='cuda')
        _ae_d_scale = torch.zeros(1, dtype=torch.float32, device='cuda')
        _ae_hidden_scratch = torch.empty(Sa * Ha, dtype=fp16, device='cuda')
        _ae_fp8_scratch = torch.zeros(
            Sa * max(Da, Ha), dtype=torch.uint8, device='cuda')

        enc_weights = {
            'qkv_w':   [w.data_ptr() for w in self._enc_qkv_w],
            'o_w':     [w.data_ptr() for w in self._enc_o_w],
            'gate_w':  [w.data_ptr() for w in self._enc_gu_w],
            'down_w':  [w.data_ptr() for w in self._enc_d_w],
            'rope':    self._enc_rope.data_ptr(),
            'Kc':      self._Kc.reshape(-1).data_ptr(),
            'Vc':      self._Vc.reshape(-1).data_ptr(),
            'w_scales': self._enc_w_dev.data_ptr(),
        }
        enc_dims = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys': total_keys,
        }
        ae_weights = {
            'ain_w':      self._ain_w.data_ptr(),
            'ain_b':      self._ain_b.data_ptr(),
            'sa':         self._sa_all.data_ptr(),
            'qw':         self._dec_qkv_flat.data_ptr(),
            'Kc':         self._Kc.reshape(-1).data_ptr(),
            'Vc':         self._Vc.reshape(-1).data_ptr(),
            'ow':         self._dec_o_flat.data_ptr(),
            'sf':         self._sf_all.data_ptr(),
            'gw':         self._dec_gu_flat.data_ptr(),
            'dw':         self._dec_d_flat.data_ptr(),
            'aow':        self._aow.data_ptr(),
            'aob':        self._aob.data_ptr(),
            'aob_dt':     self._aob_dt.data_ptr(),
            'dt':         self._ae_dt,
            'fs':         self._fs_all.data_ptr(),
            'rope':       self._dec_rope.data_ptr(),
            'w_scales':   self._ae_w_dev.data_ptr(),
        }
        ae_dims = {
            'S': Sa, 'D': Da, 'H': Ha, 'NH': 8, 'HD': 256,
            'steps': 10, 'layers': La, 'enc_seq': Se,
            'total_keys': total_keys,
        }

        per_sample_enc: list[np.ndarray] = []
        per_sample_ae:  list[np.ndarray] = []

        # Deterministic decoder noise per-sample so amax variation reflects
        # encoder input only (not the random noise tail).
        noise_gen = torch.Generator(device='cuda').manual_seed(0)

        for i, obs in enumerate(obs_list):
            # 1. Stage images through SigLIP to populate _enc_x.
            if 'images' in obs:
                img_list = obs['images']
            else:
                img_list = [obs['image']]
                if nv >= 2:
                    img_list.append(obs.get('wrist_image', obs['image']))
                if nv >= 3:
                    img_list.append(obs.get('wrist_image_right',
                                            img_list[-1]))

            def _to_np16(im):
                if isinstance(im, torch.Tensor):
                    if im.dtype == torch.uint8 or (
                            im.is_floating_point() and
                            torch.max(im).item() > 1.5):
                        im = (im.float() / 127.5 - 1.0)
                    return im.to(dtype=fp16).cpu().numpy()
                if im.dtype == np.float16:
                    return im
                return (im.astype(np.float32) / 127.5 - 1.0).astype(np.float16)

            images_np = np.stack([_to_np16(im) for im in img_list[:nv]])
            self._img_buf.upload(images_np)
            self._siglip_graph.replay()
            torch.cuda.synchronize()

            # 2. Encoder calibrate pass.
            enc_bufs = {
                'x':       self._enc_x.data_ptr(),
                'x_fp8':   self._enc_x_fp8.data_ptr(),
                'qkv':     self._enc_qkv_buf.data_ptr(),
                'logits':  self._enc_logits.data_ptr(),
                'attn_out': self._enc_attn.data_ptr(),
                'o_fp8':   self._enc_o_fp8.data_ptr(),
                'gate':    self._enc_gate.data_ptr(),
                'hidden':  self._enc_hidden.data_ptr(),
                'hid_fp8': self._enc_hid_fp8.data_ptr(),
                'fg':      self._enc_fg.data_ptr(),
                'ctx':     self._ctx,
                'norm_scratch': _norm_scratch.data_ptr(),
                'x_scratch':    _x_scratch.data_ptr(),
                'calib_buf':    _calib_buf.data_ptr(),
                'd_scale':      _d_scale.data_ptr(),
                'fp8_scratch':  _fp8_scratch_enc.data_ptr(),
                'ones':         _ones.data_ptr(),
            }
            self._enc_calib_scales.zero_()
            self._Kc.zero_(); self._Vc.zero_()
            encoder_forward_calibrate(
                self._gemm, fvk, enc_bufs, enc_weights, enc_dims,
                self._enc_calib_scales.data_ptr(), stream=0, attn=self._attn)
            torch.cuda.synchronize()
            per_sample_enc.append(self._enc_calib_scales.cpu().numpy().copy())

            # 3. Decoder calibrate pass.
            ae_bufs = {
                'noise':   self._g_noise.data_ptr(),
                'x':       self._ae_x.data_ptr(),
                'xn':      self._ae_xn.data_ptr(),
                'gate':    self._ae_gate.data_ptr(),
                'qkv':     self._ae_qkv.data_ptr(),
                'logits':  self._ae_logits.data_ptr(),
                'attn_out': self._ae_attn.data_ptr(),
                'hid':     self._ae_hid.data_ptr(),
                'fg':      self._ae_fg.data_ptr(),
                'action_f32': self._ae_action_f32.data_ptr(),
                'xn_fp8':  self._ae_xn_fp8.data_ptr(),
                'hid_fp8': self._ae_hid_fp8.data_ptr(),
                'ctx_fp8': self._ae_ctx_fp8.data_ptr(),
                'calib_buf':      _ae_calib_buf.data_ptr(),
                'd_scale':        _ae_d_scale.data_ptr(),
                'hidden_scratch': _ae_hidden_scratch.data_ptr(),
                'fp8_scratch':    _ae_fp8_scratch.data_ptr(),
            }
            self._ae_calib_scales.zero_()
            # Deterministic but varied per-sample decoder input.
            noise = torch.empty_like(self._g_noise).normal_(generator=noise_gen)
            self._g_noise.copy_(noise)
            decoder_forward_calibrate(
                self._ctx, fvk, ae_bufs, ae_weights, ae_dims,
                self._ae_calib_scales.data_ptr(), stream=0, attn=self._attn)
            torch.cuda.synchronize()
            per_sample_ae.append(self._ae_calib_scales.cpu().numpy().copy())

            if verbose and (i + 1) % max(1, n // 10) == 0:
                logger.info("  calibration sample %d/%d", i + 1, n)

        # 4. Per-point percentile reduce.
        final_enc = accumulate_amax(per_sample_enc, percentile=percentile)
        final_ae  = accumulate_amax(per_sample_ae,  percentile=percentile)

        if verbose:
            logger.info("encoder %s",
                        format_summary(summarize_amax_dispersion(
                            per_sample_enc, final_enc)))
            logger.info("decoder %s",
                        format_summary(summarize_amax_dispersion(
                            per_sample_ae, final_ae)))

        # 5. Write reduced scales back onto the device buffers.
        self._enc_calib_scales.copy_(
            torch.from_numpy(final_enc.astype(np.float32)))
        self._ae_calib_scales.copy_(
            torch.from_numpy(final_ae.astype(np.float32)))

        # 6. Recompute encoder alpha-fold list (cuBLASLt alpha path).
        enc_ws = self._enc_w_dev.cpu().tolist()
        self._enc_alpha_host = [
            float(np.float32(final_enc[i]) * np.float32(enc_ws[i]))
            for i in range(Le * 4)]

        # 7. Outlier warning (same ratio-based check as RTX path).
        check_scale_ceiling(final_enc, label=f"pi05_thor_enc_N{n}")
        check_scale_ceiling(final_ae,  label=f"pi05_thor_ae_N{n}")

        # 8. Recapture CUDA graph with the new scales.
        self._capture_enc_ae_graph()
        self._real_data_calibrated = True
        logger.info(
            "Pi0.5 Thor multi-frame calibration + graph recapture complete "
            "(N=%d, percentile=%.2f)", n, percentile)

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def get_latency_stats(self):
        """Return latency statistics."""
        if not self.latency_records:
            return {}
        lat = np.array(self.latency_records)
        return {
            "count": len(lat),
            "mean_ms": float(np.mean(lat)),
            "std_ms": float(np.std(lat)),
            "min_ms": float(np.min(lat)),
            "max_ms": float(np.max(lat)),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)),
            "hz": float(1000 / np.mean(lat)),
        }
