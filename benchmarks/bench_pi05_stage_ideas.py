"""Pi0.5 Thor stage-level experiments for RP1 optimization ideas.

This benchmark is hardware-only but robot-safe: it loads the model and runs
model-only inference/profiling on synthetic observations. It does not touch
cameras, robot transport, CAN, or action execution.

It validates two research ideas:

* D: split Enc+AE into encoder and action decoder stages, estimate the
  upper bound for asynchronous stage pipelining, and optionally run a real
  double-buffered overlap benchmark.
* E: use measured stage service curves to evaluate heterogeneous stage batch
  plans such as vision Bv followed by Enc+AE chunks of Br.
* F: probe same-Thor cross-stage overlap between Vision(batch i+1) and
  Enc+AE(batch i) with double-buffered encoder-feature slots.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import flash_rt.flash_rt_kernels as fvk


CKPT = os.environ.get(
    "PI05_LIBERO_PYTORCH_CHECKPOINT",
    "/mnt/home/zengzixuan/workspace/checkpoints/pi05_libero_pytorch",
)


def _parse_batch_sizes(value: str) -> list[int]:
    if value.strip().lower() in {"", "none", "off"}:
        return []
    sizes: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise argparse.ArgumentTypeError(f"invalid range: {part}")
            sizes.extend(range(lo, hi + 1))
        else:
            sizes.append(int(part))
    sizes = sorted(set(sizes))
    if not sizes or sizes[0] < 1:
        raise argparse.ArgumentTypeError("batch sizes must be >= 1")
    return sizes


def _make_obs(seed: int, *, same_views: bool = False) -> dict:
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    wrist = img.copy() if same_views else rng.randint(
        0, 256, (224, 224, 3), dtype=np.uint8)
    return {"image": img, "wrist_image": wrist}


def _make_batch(prompt: str, batch_size: int, *, same_inputs: bool) -> list[dict]:
    return [
        {
            "observation": _make_obs(0 if same_inputs else i),
            "prompt": prompt,
        }
        for i in range(batch_size)
    ]


def _summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "avg": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _time_wall(fn: Callable[[], object], iters: int) -> dict[str, float]:
    times: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return _summarize(times)


def _time_cuda(fn: Callable[[int], object], iters: int) -> dict[str, float]:
    stream = torch.cuda.current_stream()
    times: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        fn(stream.cuda_stream)
        end.record(stream)
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return _summarize(times)


def _measure_graph(graph, iters: int) -> dict[str, float]:
    return _time_cuda(lambda _stream: graph.replay(), iters)


def _copy_summary(src: torch.Tensor, dst: torch.Tensor) -> None:
    dst.copy_(src)


def _build_single_stage_fns(fe):
    from flash_rt.hardware.thor.shared_primitives import encoder_forward
    from flash_rt.models.pi05.pipeline_thor import decoder_forward

    enc_bufs = {
        "x": fe._enc_x.data_ptr(),
        "x_fp8": fe._enc_x_fp8.data_ptr(),
        "qkv": fe._enc_qkv_buf.data_ptr(),
        "logits": fe._enc_logits.data_ptr(),
        "attn_out": fe._enc_attn.data_ptr(),
        "o_fp8": fe._enc_o_fp8.data_ptr(),
        "gate": fe._enc_gate.data_ptr(),
        "hidden": fe._enc_hidden.data_ptr(),
        "hid_fp8": fe._enc_hid_fp8.data_ptr(),
        "fg": fe._enc_fg.data_ptr(),
        "ctx": fe._ctx,
        "x_norm": fe._enc_attn.data_ptr(),
        "ones": fe._enc_ones_fp16.data_ptr() if fe._enc_ones_fp16 is not None else 0,
    }
    enc_weights = {
        "qkv_w": [w.data_ptr() for w in fe._enc_qkv_w],
        "o_w": [w.data_ptr() for w in fe._enc_o_w],
        "gate_w": [w.data_ptr() for w in fe._enc_gu_w],
        "down_w": [w.data_ptr() for w in fe._enc_d_w],
        "rope": fe._enc_rope.data_ptr(),
        "Kc": fe._Kc.reshape(-1).data_ptr(),
        "Vc": fe._Vc.reshape(-1).data_ptr(),
        "act_scales": fe._enc_calib_scales.data_ptr(),
        "alpha_host": fe._enc_alpha_host,
    }
    enc_dims = {
        "Se": fe.Se,
        "D": fe.De,
        "H": fe.He,
        "NH": fe.NHe,
        "HD": fe.HDe,
        "L": fe.Le,
        "total_keys": fe.total_keys,
    }

    ae_bufs = {
        "noise": fe._g_noise.data_ptr(),
        "x": fe._ae_x.data_ptr(),
        "xn": fe._ae_xn.data_ptr(),
        "gate": fe._ae_gate.data_ptr(),
        "qkv": fe._ae_qkv.data_ptr(),
        "logits": fe._ae_logits.data_ptr(),
        "attn_out": fe._ae_attn.data_ptr(),
        "hid": fe._ae_hid.data_ptr(),
        "fg": fe._ae_fg.data_ptr(),
        "action_f32": fe._ae_action_f32.data_ptr(),
        "xn_fp8": fe._ae_xn_fp8.data_ptr(),
        "hid_fp8": fe._ae_hid_fp8.data_ptr(),
        "ctx_fp8": fe._ae_ctx_fp8.data_ptr(),
    }
    ae_weights = {
        "ain_w": fe._ain_w.data_ptr(),
        "ain_b": fe._ain_b.data_ptr(),
        "sa": fe._sa_all.data_ptr(),
        "qw": fe._dec_qkv_flat.data_ptr(),
        "Kc": fe._Kc.reshape(-1).data_ptr(),
        "Vc": fe._Vc.reshape(-1).data_ptr(),
        "ow": fe._dec_o_flat.data_ptr(),
        "sf": fe._sf_all.data_ptr(),
        "gw": fe._dec_gu_flat.data_ptr(),
        "dw": fe._dec_d_flat.data_ptr(),
        "aow": fe._aow.data_ptr(),
        "aob": fe._aob.data_ptr(),
        "aob_dt": fe._aob_dt.data_ptr(),
        "dt": fe._ae_dt,
        "fs": fe._fs_all.data_ptr(),
        "rope": fe._dec_rope.data_ptr(),
        "w_scales": fe._ae_w_dev.data_ptr(),
        "act_scales": fe._ae_calib_scales.data_ptr(),
    }
    ae_dims = {
        "S": fe.Sa,
        "D": fe.Da,
        "H": fe.Ha,
        "NH": 8,
        "HD": 256,
        "steps": 10,
        "layers": fe.La,
        "enc_seq": fe.Se,
        "total_keys": fe.total_keys,
    }

    def run_encoder(stream: int) -> None:
        fe._Kc.zero_()
        fe._Vc.zero_()
        encoder_forward(
            fe._gemm,
            fvk,
            enc_bufs,
            enc_weights,
            enc_dims,
            stream=stream,
            attn=fe._attn,
            use_fp8=fe.use_fp8,
        )

    def run_decoder(stream: int) -> None:
        decoder_forward(
            fe._ctx,
            fvk,
            ae_bufs,
            ae_weights,
            ae_dims,
            stream=stream,
            attn=fe._attn,
            use_fp8=fe.use_fp8,
        )

    return run_encoder, run_decoder


def _build_batched_stage_fns(
    fe,
    batch_size: int,
    *,
    kc_tensor: torch.Tensor | None = None,
    vc_tensor: torch.Tensor | None = None,
    noise_tensor: torch.Tensor | None = None,
    enc_ctx=None,
    dec_ctx=None,
    enc_gemm=None,
):
    from flash_rt.hardware.thor.shared_primitives_batched import encoder_forward_b2
    from flash_rt.models.pi05.pipeline_thor_batched import decoder_forward_b2

    B = batch_size
    kc_owner = fe._Kc_b2 if kc_tensor is None else kc_tensor
    vc_owner = fe._Vc_b2 if vc_tensor is None else vc_tensor
    noise_owner = fe._g_noise_b2 if noise_tensor is None else noise_tensor
    enc_ctx = fe._ctx if enc_ctx is None else enc_ctx
    dec_ctx = fe._ctx if dec_ctx is None else dec_ctx
    enc_gemm = fe._gemm if enc_gemm is None else enc_gemm

    kc_b2 = [kc_owner[b].view(-1).data_ptr() for b in range(B)]
    vc_b2 = [vc_owner[b].view(-1).data_ptr() for b in range(B)]
    total_keys_b2 = [fe.total_keys] * B
    enc_seq_b2 = [fe.Se] * B

    enc_bufs = {
        "x": fe._enc_x_b2.data_ptr(),
        "x_fp8": fe._enc_x_fp8_b2.data_ptr(),
        "qkv": fe._enc_qkv_buf_b2.data_ptr(),
        "logits": fe._enc_logits_b2.data_ptr(),
        "attn_out": fe._enc_attn_b2.data_ptr(),
        "o_fp8": fe._enc_o_fp8_b2.data_ptr(),
        "gate": fe._enc_gate_b2.data_ptr(),
        "hid_fp8": fe._enc_hid_fp8_b2.data_ptr(),
        "fg": fe._enc_fg_b2.data_ptr(),
        "ctx": enc_ctx,
    }
    enc_weights = {
        "qkv_w": [w.data_ptr() for w in fe._enc_qkv_w],
        "o_w": [w.data_ptr() for w in fe._enc_o_w],
        "gate_w": [w.data_ptr() for w in fe._enc_gu_w],
        "down_w": [w.data_ptr() for w in fe._enc_d_w],
        "rope": fe._enc_rope.data_ptr(),
        "Kc_b2": kc_b2,
        "Vc_b2": vc_b2,
        "act_scales": fe._enc_calib_scales.data_ptr(),
        "alpha_host": fe._enc_alpha_host,
    }
    enc_dims = {
        "Se": fe.Se,
        "D": fe.De,
        "H": fe.He,
        "NH": fe.NHe,
        "HD": fe.HDe,
        "L": fe.Le,
        "total_keys": fe.total_keys,
    }

    ae_bufs = {
        "noise": noise_owner.data_ptr(),
        "x": fe._ae_x_b2.data_ptr(),
        "xn": fe._ae_xn_b2.data_ptr(),
        "gate": fe._ae_gate_b2.data_ptr(),
        "qkv": fe._ae_qkv_b2.data_ptr(),
        "logits": fe._ae_logits_b2.data_ptr(),
        "attn_out": fe._ae_attn_b2.data_ptr(),
        "fg": fe._ae_fg_b2.data_ptr(),
        "action_f32": fe._ae_action_f32_b2.data_ptr(),
        "xn_fp8": fe._ae_xn_fp8_b2.data_ptr(),
        "hid_fp8": fe._ae_hid_fp8_b2.data_ptr(),
        "ctx_fp8": fe._ae_ctx_fp8_b2.data_ptr(),
        "v_b2": fe._v_b2.data_ptr(),
        "v_b2_f32": fe._v_b2_f32.data_ptr(),
    }
    ae_weights = {
        "ain_w": fe._ain_w.data_ptr(),
        "ain_b": fe._ain_b.data_ptr(),
        "sa": fe._sa_all_b2.data_ptr(),
        "qw": fe._dec_qkv_flat.data_ptr(),
        "Kc_b2": kc_b2,
        "Vc_b2": vc_b2,
        "total_keys_b2": total_keys_b2,
        "enc_seq_b2": enc_seq_b2,
        "ow": fe._dec_o_flat.data_ptr(),
        "sf": fe._sf_all_b2.data_ptr(),
        "gw": fe._dec_gu_flat.data_ptr(),
        "dw": fe._dec_d_flat.data_ptr(),
        "aow": fe._aow.data_ptr(),
        "aob": fe._aob.data_ptr(),
        "aob_dt": fe._aob_dt.data_ptr(),
        "dt": fe._ae_dt,
        "fs": fe._fs_all_b2.data_ptr(),
        "rope": fe._dec_rope.data_ptr(),
        "w_scales": fe._ae_w_dev.data_ptr(),
        "act_scales": fe._ae_calib_scales.data_ptr(),
    }
    ae_dims = {
        "S": fe.Sa,
        "D": fe.Da,
        "H": fe.Ha,
        "NH": 8,
        "HD": 256,
        "steps": 10,
        "layers": fe.La,
        "enc_seq": fe.Se,
        "total_keys": fe.total_keys,
    }

    cfg_beta = fe._enc_ae_graph_b2_cfg_beta

    def run_encoder(stream: int) -> None:
        if not fe.batched_skip_kv_zero:
            kc_owner.zero_()
            vc_owner.zero_()
        encoder_forward_b2(
            enc_gemm,
            fvk,
            enc_bufs,
            enc_weights,
            enc_dims,
            stream=stream,
            B=B,
        )

    def run_decoder(stream: int) -> None:
        decoder_forward_b2(
            dec_ctx,
            fvk,
            ae_bufs,
            ae_weights,
            ae_dims,
            stream=stream,
            B=B,
            cfg_beta=cfg_beta,
        )

    return run_encoder, run_decoder


def _evaluate_hetero(results: dict[int, dict], plans: list[tuple[int, int]]) -> list[dict]:
    evaluated: list[dict] = []
    for Bv, Br in plans:
        if Bv not in results or Br not in results:
            continue
        vision_ms = results[Bv]["siglip_postln_graph"]["p50"]
        full_ms = results[Bv]["enc_ae_graph"]["p50"]
        chunks = []
        remaining = Bv
        while remaining > 0:
            chunk = min(Br, remaining)
            if chunk not in results:
                break
            chunks.append(chunk)
            remaining -= chunk
        if sum(chunks) != Bv:
            continue
        chunk_ms = sum(results[chunk]["enc_ae_graph"]["p50"] for chunk in chunks)
        unified_total = vision_ms + full_ms
        hetero_total = vision_ms + chunk_ms
        urgent_completion = vision_ms + results[chunks[0]]["enc_ae_graph"]["p50"]
        evaluated.append(
            {
                "Bv": Bv,
                "Br": Br,
                "chunks": chunks,
                "unified_total_ms": round(unified_total, 3),
                "hetero_total_ms": round(hetero_total, 3),
                "hetero_overhead_vs_unified_pct": round(
                    (hetero_total / unified_total - 1.0) * 100.0, 3),
                "first_chunk_completion_ms": round(urgent_completion, 3),
                "first_chunk_latency_reduction_vs_unified_pct": round(
                    (1.0 - urgent_completion / unified_total) * 100.0, 3),
            }
        )
    return evaluated


def _event_elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _measure_batched_overlap_pipeline(fe, batch_size: int, iters: int) -> dict:
    """Measure real encoder/decoder overlap for the batched Enc+AE stage.

    The benchmark uses two KV/noise slots. At iteration i, the decoder reads
    slot prev while the encoder overwrites slot cur for the next iteration.
    It intentionally does not touch serving, camera, or robot transport.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if fe._Kc_b2 is None or fe._Vc_b2 is None or fe._g_noise_b2 is None:
        raise RuntimeError("batched buffers must be allocated before pipeline benchmark")

    B = batch_size
    kc_slots = [torch.empty_like(fe._Kc_b2), torch.empty_like(fe._Kc_b2)]
    vc_slots = [torch.empty_like(fe._Vc_b2), torch.empty_like(fe._Vc_b2)]
    noise_slots = [torch.empty_like(fe._g_noise_b2), torch.empty_like(fe._g_noise_b2)]
    enc_x_seed = fe._enc_x_b2.detach().clone()
    noise_seed = fe._g_noise_b2.detach().clone()

    # Separate contexts avoid sharing cuBLAS/FVK state across concurrent streams.
    enc_ctxs = [fvk.FvkContext(), fvk.FvkContext()]
    dec_ctxs = [fvk.FvkContext(), fvk.FvkContext()]
    enc_gemms = [fvk.GemmRunner(), fvk.GemmRunner()]

    enc_fns = []
    dec_fns = []
    for slot in range(2):
        enc_fn, dec_fn = _build_batched_stage_fns(
            fe,
            B,
            kc_tensor=kc_slots[slot],
            vc_tensor=vc_slots[slot],
            noise_tensor=noise_slots[slot],
            enc_ctx=enc_ctxs[slot],
            dec_ctx=dec_ctxs[slot],
            enc_gemm=enc_gemms[slot],
        )
        enc_fns.append(enc_fn)
        dec_fns.append(dec_fn)

    def _reset_encoder_input() -> None:
        fe._enc_x_b2.copy_(enc_x_seed)

    # Initialize both slots with valid encoder KV and initial noise.
    for slot in range(2):
        noise_slots[slot].copy_(noise_seed)
        _reset_encoder_input()
        enc_fns[slot](torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    def _run_serial_once() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream()
        start.record(stream)
        for i in range(iters):
            prev = i & 1
            cur = 1 - prev
            noise_slots[prev].copy_(noise_seed)
            dec_fns[prev](stream.cuda_stream)
            noise_slots[cur].copy_(noise_seed)
            _reset_encoder_input()
            enc_fns[cur](stream.cuda_stream)
        end.record(stream)
        return _event_elapsed_ms(start, end)

    def _run_overlap_once() -> float:
        enc_stream = torch.cuda.Stream()
        dec_stream = torch.cuda.Stream()
        default_stream = torch.cuda.current_stream()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        enc_done = [torch.cuda.Event(), torch.cuda.Event()]
        dec_done = [torch.cuda.Event(), torch.cuda.Event()]

        # Both slots are initialized and free to overwrite after this point.
        for slot in range(2):
            enc_done[slot].record(default_stream)
            dec_done[slot].record(default_stream)

        start.record(default_stream)
        for i in range(iters):
            prev = i & 1
            cur = 1 - prev

            dec_stream.wait_event(enc_done[prev])
            enc_stream.wait_event(dec_done[cur])

            with torch.cuda.stream(dec_stream):
                noise_slots[prev].copy_(noise_seed)
                dec_fns[prev](dec_stream.cuda_stream)
                dec_done[prev].record(dec_stream)

            with torch.cuda.stream(enc_stream):
                _reset_encoder_input()
                enc_fns[cur](enc_stream.cuda_stream)
                enc_done[cur].record(enc_stream)

        default_stream.wait_stream(dec_stream)
        default_stream.wait_stream(enc_stream)
        end.record(default_stream)
        return _event_elapsed_ms(start, end)

    # One dry run to pay any lazy setup cost outside the measured samples.
    _run_serial_once()
    _run_overlap_once()

    serial_totals = [_run_serial_once() for _ in range(3)]
    overlap_totals = [_run_overlap_once() for _ in range(3)]
    serial_per_iter = [v / iters for v in serial_totals]
    overlap_per_iter = [v / iters for v in overlap_totals]

    # Parity check for one slot: run sequential encoder+decoder and compare the
    # output noise with the overlap-produced decoder output for the same slot.
    noise_slots[0].copy_(noise_seed)
    _reset_encoder_input()
    enc_fns[0](torch.cuda.current_stream().cuda_stream)
    dec_fns[0](torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    serial_action = noise_slots[0].detach().clone()

    _run_overlap_once()
    overlap_action = noise_slots[(iters - 1) & 1].detach().clone()
    max_abs = float((serial_action - overlap_action).abs().max().item())
    cos = float(torch.nn.functional.cosine_similarity(
        serial_action.float().reshape(1, -1),
        overlap_action.float().reshape(1, -1),
        dim=1,
    ).item())

    serial_summary = _summarize(serial_per_iter)
    overlap_summary = _summarize(overlap_per_iter)
    speedup = (
        serial_summary["p50"] / overlap_summary["p50"]
        if overlap_summary["p50"] > 0.0
        else math.nan
    )
    improvement = (
        (1.0 - overlap_summary["p50"] / serial_summary["p50"]) * 100.0
        if serial_summary["p50"] > 0.0
        else math.nan
    )
    return {
        "iters_per_run": iters,
        "serial_total_ms": _summarize(serial_totals),
        "overlap_total_ms": _summarize(overlap_totals),
        "serial_per_iter_ms": serial_summary,
        "overlap_per_iter_ms": overlap_summary,
        "speedup": speedup,
        "improvement_pct": improvement,
        "parity_max_abs": max_abs,
        "parity_cos": cos,
        "notes": [
            "serial uses the same double-buffer stage functions without overlap",
            "overlap enqueues decoder(prev_slot) and encoder(cur_slot) on two CUDA streams",
            "measurement covers Enc+AE stage only, not SigLIP, serving, or robot I/O",
        ],
    }


def _capture_vision_encae_slot_graphs(fe, batch_size: int, slot_count: int = 2):
    if batch_size < 2:
        raise ValueError("batched slot graph probe requires B>=2")
    if fe._enc_x_b2 is None:
        raise RuntimeError("batched buffers must be allocated before slot capture")

    original_enc_x = fe._enc_x_b2
    original_siglip_graph = fe._siglip_batched_graph
    original_siglip_B = fe._siglip_batched_B
    original_enc_ae_graph = fe._enc_ae_graph_b2
    original_ctx = fe._ctx
    original_gemm = fe._gemm

    slots = [torch.empty_like(original_enc_x) for _ in range(slot_count)]
    for slot in slots:
        slot.copy_(original_enc_x)

    siglip_graphs = []
    enc_ae_graphs = []
    keepalive = []
    try:
        for slot in slots:
            fe._enc_x_b2 = slot

            siglip_ctx = fvk.FvkContext()
            siglip_gemm = fvk.GemmRunner()
            keepalive.extend([siglip_ctx, siglip_gemm])
            fe._ctx = siglip_ctx
            fe._gemm = siglip_gemm
            fe._siglip_batched_graph = None
            fe._capture_siglip_batched_graph(batch_size)
            siglip_graphs.append(fe._siglip_batched_graph)

            enc_ctx = fvk.FvkContext()
            enc_gemm = fvk.GemmRunner()
            keepalive.extend([enc_ctx, enc_gemm])
            fe._ctx = enc_ctx
            fe._gemm = enc_gemm
            fe._enc_ae_graph_b2 = None
            fe._capture_enc_ae_graph_b2()
            enc_ae_graphs.append(fe._enc_ae_graph_b2)
    finally:
        fe._enc_x_b2 = original_enc_x
        fe._siglip_batched_graph = original_siglip_graph
        fe._siglip_batched_B = original_siglip_B
        fe._enc_ae_graph_b2 = original_enc_ae_graph
        fe._ctx = original_ctx
        fe._gemm = original_gemm

    return slots, siglip_graphs, enc_ae_graphs, keepalive


def _measure_vision_encae_overlap(fe, batch_size: int, iters: int) -> dict:
    """Measure Vision(batch i+1) overlapped with Enc+AE(batch i).

    This is a benchmark-only resource-concurrency probe. It captures separate
    graph pairs against two encoder-feature slots so the vision graph writes
    slot cur while Enc+AE reads slot prev.
    """
    if batch_size < 2:
        raise ValueError("vision/Enc+AE overlap probe requires B>=2")

    _, siglip_graphs, enc_ae_graphs, keepalive = _capture_vision_encae_slot_graphs(
        fe, batch_size, slot_count=2)
    _ = keepalive
    noise_seed = fe._g_noise_b2.detach().clone()
    kc_seed = fe._Kc_b2.detach().clone() if fe._Kc_b2 is not None else None
    vc_seed = fe._Vc_b2.detach().clone() if fe._Vc_b2 is not None else None

    def _restore_encae_state() -> None:
        fe._g_noise_b2.copy_(noise_seed)
        if kc_seed is not None:
            fe._Kc_b2.copy_(kc_seed)
        if vc_seed is not None:
            fe._Vc_b2.copy_(vc_seed)

    def _run_serial_once() -> float:
        stream = torch.cuda.current_stream()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for i in range(iters):
            slot = i & 1
            siglip_graphs[slot].replay()
            _restore_encae_state()
            enc_ae_graphs[slot].replay()
        end.record(stream)
        return _event_elapsed_ms(start, end)

    def _run_overlap_once() -> float:
        vision_stream = torch.cuda.Stream()
        enc_stream = torch.cuda.Stream()
        default_stream = torch.cuda.current_stream()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        vision_done = [torch.cuda.Event(), torch.cuda.Event()]
        enc_done = [torch.cuda.Event(), torch.cuda.Event()]

        for slot in range(2):
            vision_done[slot].record(default_stream)
            enc_done[slot].record(default_stream)

        # Prime the first feature slot. The measured loop then contains one
        # Enc+AE for the current slot and one Vision graph for the next slot.
        siglip_graphs[0].replay()
        vision_done[0].record(default_stream)
        start.record(default_stream)

        for i in range(iters):
            prev = i & 1
            cur = 1 - prev

            enc_stream.wait_event(vision_done[prev])
            vision_stream.wait_event(enc_done[cur])

            with torch.cuda.stream(enc_stream):
                _restore_encae_state()
                enc_ae_graphs[prev].replay()
                enc_done[prev].record(enc_stream)

            with torch.cuda.stream(vision_stream):
                siglip_graphs[cur].replay()
                vision_done[cur].record(vision_stream)

        default_stream.wait_stream(enc_stream)
        default_stream.wait_stream(vision_stream)
        end.record(default_stream)
        return _event_elapsed_ms(start, end)

    _run_serial_once()
    _run_overlap_once()
    serial_totals = [_run_serial_once() for _ in range(3)]
    overlap_totals = [_run_overlap_once() for _ in range(3)]
    serial_per_iter = [v / iters for v in serial_totals]
    overlap_per_iter = [v / iters for v in overlap_totals]

    # Correctness probe: Enc+AE output should not change when a vision graph
    # for the other slot runs concurrently.
    _restore_encae_state()
    enc_ae_graphs[0].replay()
    torch.cuda.synchronize()
    serial_action = fe._g_noise_b2.detach().clone()

    vision_stream = torch.cuda.Stream()
    enc_stream = torch.cuda.Stream()
    with torch.cuda.stream(enc_stream):
        _restore_encae_state()
        enc_ae_graphs[0].replay()
    with torch.cuda.stream(vision_stream):
        siglip_graphs[1].replay()
    torch.cuda.current_stream().wait_stream(enc_stream)
    torch.cuda.current_stream().wait_stream(vision_stream)
    torch.cuda.synchronize()
    overlap_action = fe._g_noise_b2.detach().clone()

    max_abs = float((serial_action - overlap_action).abs().max().item())
    cos = float(torch.nn.functional.cosine_similarity(
        serial_action.float().reshape(1, -1),
        overlap_action.float().reshape(1, -1),
        dim=1,
    ).item())
    serial_summary = _summarize(serial_per_iter)
    overlap_summary = _summarize(overlap_per_iter)
    speedup = (
        serial_summary["p50"] / overlap_summary["p50"]
        if overlap_summary["p50"] > 0.0
        else math.nan
    )
    improvement = (
        (1.0 - overlap_summary["p50"] / serial_summary["p50"]) * 100.0
        if serial_summary["p50"] > 0.0
        else math.nan
    )
    return {
        "iters_per_run": iters,
        "serial_total_ms": _summarize(serial_totals),
        "overlap_total_ms": _summarize(overlap_totals),
        "serial_per_iter_ms": serial_summary,
        "overlap_per_iter_ms": overlap_summary,
        "speedup": speedup,
        "improvement_pct": improvement,
        "parity_max_abs": max_abs,
        "parity_cos": cos,
        "notes": [
            "serial runs Vision then Enc+AE for the same feature slot",
            "overlap runs Enc+AE(prev_slot) and Vision(cur_slot) on two CUDA streams",
            "measurement covers same-Thor model stages only, not serving, camera, robot I/O, or cross-device compute",
        ],
    }


def _print_tables(results: dict[int, dict], hetero: list[dict]) -> None:
    print("\nStage profile")
    print(
        "| B | E2E P50 | SigLIP/PostLN | Enc+AE graph | Encoder | Decoder | "
        "split sum | pipe upper P50 | pipe upper gain |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for B in sorted(results):
        r = results[B]
        enc = r["encoder_manual"]["p50"]
        dec = r["decoder_manual"]["p50"]
        split_sum = enc + dec
        pipe_upper = max(enc, dec)
        graph = r["enc_ae_graph"]["p50"]
        gain = (1.0 - pipe_upper / graph) * 100.0 if graph > 0 else float("nan")
        print(
            f"| {B} | {r['end_to_end']['p50']:.1f} | "
            f"{r['siglip_postln_graph']['p50']:.1f} | {graph:.1f} | "
            f"{enc:.1f} | {dec:.1f} | {split_sum:.1f} | "
            f"{pipe_upper:.1f} | {gain:.1f}% |"
        )

    if hetero:
        print("\nHeterogeneous stage-batch estimate")
        print(
            "| Bv | Br | chunks | unified total | hetero total | overhead | "
            "first chunk completion | urgent latency reduction |"
        )
        print("| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
        for item in hetero:
            print(
                f"| {item['Bv']} | {item['Br']} | {item['chunks']} | "
                f"{item['unified_total_ms']:.1f} | {item['hetero_total_ms']:.1f} | "
                f"{item['hetero_overhead_vs_unified_pct']:.1f}% | "
                f"{item['first_chunk_completion_ms']:.1f} | "
                f"{item['first_chunk_latency_reduction_vs_unified_pct']:.1f}% |"
            )

    pipeline_rows = [
        (B, r["real_overlap_pipeline"])
        for B, r in sorted(results.items())
        if "real_overlap_pipeline" in r
    ]
    if pipeline_rows:
        print("\nReal double-buffer overlap pipeline")
        print(
            "| B | serial stage/iter | overlap stage/iter | speedup | "
            "improvement | parity max_abs | parity cos |"
        )
        print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for B, p in pipeline_rows:
            print(
                f"| {B} | {p['serial_per_iter_ms']['p50']:.1f} | "
                f"{p['overlap_per_iter_ms']['p50']:.1f} | "
                f"{p['speedup']:.3f}x | {p['improvement_pct']:.1f}% | "
                f"{p['parity_max_abs']:.3g} | {p['parity_cos']:.6f} |"
            )

    vision_rows = [
        (B, r["vision_encae_overlap_pipeline"])
        for B, r in sorted(results.items())
        if "vision_encae_overlap_pipeline" in r
    ]
    if vision_rows:
        print("\nVision/Enc+AE same-Thor overlap pipeline")
        print(
            "| B | serial stage/iter | overlap stage/iter | speedup | "
            "improvement | parity max_abs | parity cos |"
        )
        print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for B, p in vision_rows:
            print(
                f"| {B} | {p['serial_per_iter_ms']['p50']:.1f} | "
                f"{p['overlap_per_iter_ms']['p50']:.1f} | "
                f"{p['speedup']:.3f}x | {p['improvement_pct']:.1f}% | "
                f"{p['parity_max_abs']:.3g} | {p['parity_cos']:.6f} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pi0.5 Thor RP1 stage-split and hetero-batch benchmark.")
    parser.add_argument("--checkpoint", default=CKPT)
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default="1-8")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--stage-iters", type=int, default=40)
    parser.add_argument(
        "--pipeline-batches",
        type=_parse_batch_sizes,
        default=[],
        help=(
            "batch sizes for real double-buffer encoder/decoder overlap "
            "benchmark; use e.g. 2,4,8 or none"
        ),
    )
    parser.add_argument("--pipeline-iters", type=int, default=20)
    parser.add_argument(
        "--vision-pipeline-batches",
        type=_parse_batch_sizes,
        default=[],
        help=(
            "batch sizes for same-Thor Vision(batch i+1) plus Enc+AE(batch i) "
            "overlap probe; use e.g. 2,4,8 or none"
        ),
    )
    parser.add_argument("--vision-pipeline-iters", type=int, default=20)
    parser.add_argument("--prompt", default="pick up the red block and place it in the tray")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--same-inputs", action="store_true")
    parser.add_argument("--no-fp8", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this Thor benchmark.")
    if not os.path.isdir(args.checkpoint):
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor

    results: dict[int, dict] = {}
    frontend = Pi05TorchFrontendThor(
        args.checkpoint,
        num_views=2,
        use_fp8=not args.no_fp8,
    )
    frontend.set_prompt(args.prompt)

    for B in args.batch_sizes:
        print(f"\nB={B}: prepare")
        batch = _make_batch(args.prompt, B, same_inputs=args.same_inputs)
        if B == 1:
            frontend.set_batched_mode(enable=False)
            infer_fn = lambda: frontend.infer(batch[0]["observation"], seed=args.seed)
        else:
            frontend.set_batched_mode(enable=True, batch_size=B)
            infer_fn = lambda: frontend.infer_multi_prompt_batch(batch, seed=args.seed)

        for _ in range(args.warmup):
            infer_fn()
        torch.cuda.synchronize()

        print(f"B={B}: end-to-end")
        end_to_end = _time_wall(infer_fn, args.iters)

        # Ensure graphs are captured and buffers contain valid staged data.
        infer_fn()
        torch.cuda.synchronize()

        if B == 1:
            siglip_graph = frontend._siglip_graph
            enc_ae_graph = frontend._enc_ae_graph
            encoder_fn, decoder_fn = _build_single_stage_fns(frontend)
        else:
            siglip_graph = frontend._siglip_batched_graph
            enc_ae_graph = frontend._enc_ae_graph_b2
            encoder_fn, decoder_fn = _build_batched_stage_fns(frontend, B)

        if siglip_graph is None or enc_ae_graph is None:
            raise RuntimeError(f"expected graphs to be captured for B={B}")

        print(f"B={B}: graph/stage profile")
        siglip_postln = _measure_graph(siglip_graph, args.stage_iters)
        enc_ae = _measure_graph(enc_ae_graph, args.stage_iters)
        encoder = _time_cuda(encoder_fn, args.stage_iters)
        # Refresh encoder outputs before decoder timing, then time decoder alone.
        encoder_fn(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        decoder = _time_cuda(decoder_fn, args.stage_iters)

        split_sum_p50 = encoder["p50"] + decoder["p50"]
        pipe_upper_p50 = max(encoder["p50"], decoder["p50"])
        results[B] = {
            "end_to_end": end_to_end,
            "siglip_postln_graph": siglip_postln,
            "enc_ae_graph": enc_ae,
            "encoder_manual": encoder,
            "decoder_manual": decoder,
            "split_sum_p50_ms": split_sum_p50,
            "pipeline_steady_upper_bound_p50_ms": pipe_upper_p50,
            "pipeline_upper_bound_gain_vs_enc_ae_graph_pct": (
                (1.0 - pipe_upper_p50 / enc_ae["p50"]) * 100.0
                if enc_ae["p50"] > 0.0
                else math.nan
            ),
        }
        if B in args.pipeline_batches:
            if B == 1:
                print("B=1: skip real overlap pipeline (batched double-buffer path requires B>=2)")
            else:
                print(f"B={B}: real double-buffer overlap pipeline")
                results[B]["real_overlap_pipeline"] = _measure_batched_overlap_pipeline(
                    frontend,
                    B,
                    args.pipeline_iters,
                )
        if B in args.vision_pipeline_batches:
            if B == 1:
                print("B=1: skip Vision/Enc+AE overlap pipeline (batched slot path requires B>=2)")
            else:
                print(f"B={B}: Vision/Enc+AE same-Thor overlap pipeline")
                results[B]["vision_encae_overlap_pipeline"] = _measure_vision_encae_overlap(
                    frontend,
                    B,
                    args.vision_pipeline_iters,
                )

    plans = [(2, 1), (4, 2), (6, 3), (8, 4), (4, 1), (8, 2), (8, 1)]
    hetero = _evaluate_hetero(results, plans)
    _print_tables(results, hetero)

    payload = {
        "checkpoint": args.checkpoint,
        "batch_sizes": args.batch_sizes,
        "warmup": args.warmup,
        "iters": args.iters,
        "stage_iters": args.stage_iters,
        "pipeline_batches": args.pipeline_batches,
        "pipeline_iters": args.pipeline_iters,
        "vision_pipeline_batches": args.vision_pipeline_batches,
        "vision_pipeline_iters": args.vision_pipeline_iters,
        "fp8": not args.no_fp8,
        "same_inputs": args.same_inputs,
        "results": results,
        "heterogeneous_stage_batch_estimates": hetero,
        "notes": [
            "pipeline_steady_upper_bound is max(encoder, decoder), not an implemented overlap result.",
            "heterogeneous estimates reuse measured graph service curves and do not include feature-buffer copy overhead.",
            "vision_encae_overlap_pipeline is a same-Thor benchmark-only probe and does not use 4090-side or cross-device compute.",
        ],
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
