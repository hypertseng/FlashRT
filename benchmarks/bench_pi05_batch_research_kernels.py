"""Microbenchmarks for Pi0.5 Thor batch inference research ideas.

This script intentionally uses synthetic tensors but production-like shapes.
It answers two questions that the end-to-end benchmark cannot isolate:

1. Is the current CUTLASS FP8 tactic always the best tactic as B changes?
2. How much time is exposed by exact producer -> FP8 materialization before
   the next GEMM, i.e. the upper bound for precision-preserving lazy FP8
   or virtual activation GEMM work?

The script does not modify model state and does not require checkpoint files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Callable

import torch

import flash_rt.flash_rt_kernels as fvk


CutlassFn = Callable[[int, int, int, int, int, int, float, float, int], int]


def _ptr(x) -> int:
    return int(x.data_ptr()) if hasattr(x, "data_ptr") else int(x)


def _parse_batch_sizes(value: str) -> list[int]:
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
    if not sizes or min(sizes) < 1:
        raise argparse.ArgumentTypeError("batch sizes must be >= 1")
    return sizes


def _stats(times_ms: list[float]) -> dict[str, float]:
    t = torch.tensor(times_ms, dtype=torch.float64)
    return {
        "avg_ms": float(t.mean().item()),
        "p50_ms": float(t.quantile(0.50).item()),
        "p95_ms": float(t.quantile(0.95).item()),
        "min_ms": float(t.min().item()),
        "max_ms": float(t.max().item()),
    }


def _time_cuda(fn: Callable[[], None], warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return _stats(times)


def _quantize_fp8(src: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out = torch.empty(src.numel(), dtype=torch.uint8, device=src.device)
    fvk.quantize_fp8_static_fp16(_ptr(src), _ptr(out), _ptr(scale), src.numel(), 0)
    return out


def _make_fp8(shape: tuple[int, ...], scale: torch.Tensor, *, std: float = 0.05) -> torch.Tensor:
    src = torch.randn(shape, dtype=torch.float16, device="cuda") * std
    out = _quantize_fp8(src, scale)
    del src
    return out


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    denom = af.norm() * bf.norm()
    if denom.item() == 0.0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float((af @ bf / denom).item())


def _byte_diff_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, int, int]:
    exact = bool(torch.equal(a, b))
    if exact:
        return True, 0, 0
    diff = (a.to(torch.int16) - b.to(torch.int16)).abs()
    return False, int((a != b).sum().item()), int(diff.max().item())


def _run_cutlass(fn: CutlassFn, A: torch.Tensor, W: torch.Tensor, D: torch.Tensor,
                 M: int, N: int, K: int, alpha: float, beta: float) -> int:
    return int(fn(_ptr(A), _ptr(W), _ptr(D), M, N, K, alpha, beta, 0))


def _run_descale(A: torch.Tensor, W: torch.Tensor, D: torch.Tensor,
                 M: int, N: int, K: int,
                 act_scale: torch.Tensor, w_scale: torch.Tensor) -> None:
    fvk.fp8_gemm_descale_fp16(
        _ptr(A), _ptr(W), _ptr(D), M, N, K, _ptr(act_scale), _ptr(w_scale), 0)


CUTLASS_CANDIDATES: dict[str, CutlassFn] = {
    "sq": fvk.cutlass_fp8_sq,
    "t1": fvk.cutlass_fp8_t1,
    "wide": fvk.cutlass_fp8_wide,
    "plain": fvk.cutlass_fp8_plain,
}
if hasattr(fvk, "cutlass_fp8_t2"):
    CUTLASS_CANDIDATES["t2"] = fvk.cutlass_fp8_t2


def _production_gateup_tactic(B: int) -> str:
    if B >= 2:
        return "sq"
    mode = os.environ.get("FLASHRT_THOR_ENCODER_B1_TACTICS", "optimized").strip().lower()
    return "wide" if mode in (
        "optimized", "opt", "both", "gateup-wide", "gateup_wide") else "t1"


def _encoder_down_tactic(B: int, requested: str = "auto") -> str:
    tactic = requested.strip().lower()
    if tactic in ("", "auto"):
        mode = os.environ.get("FLASHRT_THOR_ENCODER_B1_TACTICS", "optimized").strip().lower()
        if B == 1 and mode in (
            "optimized", "opt", "both", "down-t1", "down_t1"):
            tactic = "t1"
        elif B >= 4 and "t2" in CUTLASS_CANDIDATES:
            tactic = "t2"
        else:
            tactic = "wide"
    if tactic not in CUTLASS_CANDIDATES:
        choices = ", ".join(sorted(["auto", *CUTLASS_CANDIDATES.keys()]))
        raise ValueError(f"unsupported encoder down tactic {requested!r}; choices: {choices}")
    return tactic


def _encoder_down_tactic(B: int, requested: str = "auto") -> str:
    tactic = requested.strip().lower()
    if tactic in ("", "auto"):
        if B == 1:
            tactic = "t1"
        elif B >= 4 and "t2" in CUTLASS_CANDIDATES:
            tactic = "t2"
        else:
            tactic = "wide"
    if tactic not in CUTLASS_CANDIDATES:
        choices = ", ".join(sorted(["auto", *CUTLASS_CANDIDATES.keys()]))
        raise ValueError(f"unsupported encoder down tactic {requested!r}; choices: {choices}")
    return tactic


def _production_down_tactic(B: int) -> str:
    return _encoder_down_tactic(B, "auto")


def _encoder_tactic_shapes(B: int, se: int) -> list[dict]:
    m = B * se
    return [
        {
            "name": "encoder_qkv",
            "M": m,
            "N": 2560,
            "K": 2048,
            "production": "sq",
            "beta": 0.0,
            "calls_per_infer": 18,
        },
        {
            "name": "encoder_o",
            "M": m,
            "N": 2048,
            "K": 2048,
            "production": "sq",
            "beta": 1.0,
            "calls_per_infer": 17,
        },
        {
            "name": "encoder_gateup",
            "M": m,
            "N": 32768,
            "K": 2048,
            "production": _production_gateup_tactic(B),
            "beta": 0.0,
            "calls_per_infer": 17,
        },
        {
            "name": "encoder_down",
            "M": m,
            "N": 2048,
            "K": 16384,
            "production": _production_down_tactic(B),
            "beta": 1.0,
            "calls_per_infer": 17,
        },
    ]


def run_cutlass_tactic_sweep(batch_sizes: list[int], se: int,
                             warmup: int, iters: int,
                             max_candidates: int | None = None) -> list[dict]:
    results: list[dict] = []
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    for B in batch_sizes:
        for shape in _encoder_tactic_shapes(B, se):
            name = shape["name"]
            M, N, K = shape["M"], shape["N"], shape["K"]
            beta = float(shape["beta"])
            prod = shape["production"]
            alpha = 1.0
            print(f"\n[tactic] B={B} {name}: M={M} N={N} K={K} beta={beta}")

            A = _make_fp8((M, K), scale)
            W = _make_fp8((K, N), scale)
            D = torch.empty((M, N), dtype=torch.float16, device="cuda")
            D_base = torch.randn_like(D) * 0.01 if beta != 0.0 else torch.zeros_like(D)
            ref = torch.empty_like(D)

            prod_fn = CUTLASS_CANDIDATES[prod]
            ref.copy_(D_base)
            rc = _run_cutlass(prod_fn, A, W, ref, M, N, K, alpha, beta)
            torch.cuda.synchronize()
            if rc != 0:
                raise RuntimeError(f"production tactic {prod} failed for {name}, rc={rc}")

            candidates = list(CUTLASS_CANDIDATES.items())
            if max_candidates is not None:
                candidates = candidates[:max_candidates]

            for cand_name, cand_fn in candidates:
                D.copy_(D_base)
                try:
                    rc = _run_cutlass(cand_fn, A, W, D, M, N, K, alpha, beta)
                    torch.cuda.synchronize()
                except Exception as exc:  # pybind/CUTLASS launch diagnostics
                    print(f"  {cand_name:>5}: failed ({exc})")
                    results.append({
                        "section": "cutlass_tactic_sweep",
                        "B": B,
                        "shape": name,
                        "candidate": cand_name,
                        "production": prod,
                        "M": M,
                        "N": N,
                        "K": K,
                        "beta": beta,
                        "status": "exception",
                        "error": str(exc),
                    })
                    continue

                if rc != 0:
                    print(f"  {cand_name:>5}: skipped rc={rc}")
                    results.append({
                        "section": "cutlass_tactic_sweep",
                        "B": B,
                        "shape": name,
                        "candidate": cand_name,
                        "production": prod,
                        "M": M,
                        "N": N,
                        "K": K,
                        "beta": beta,
                        "status": "rc",
                        "rc": rc,
                    })
                    continue

                max_abs = float((D - ref).abs().max().item())
                bit_exact = bool(torch.equal(D, ref))
                cos = _cosine(D, ref)

                def one() -> None:
                    rc_inner = _run_cutlass(cand_fn, A, W, D, M, N, K, alpha, beta)
                    if rc_inner != 0:
                        raise RuntimeError(f"{cand_name} rc={rc_inner}")

                timing = _time_cuda(one, warmup, iters)
                print(
                    f"  {cand_name:>5}: {timing['avg_ms']:.3f} ms "
                    f"(bit_exact={bit_exact}, max_abs={max_abs:.3g}, cos={cos:.9f})")
                results.append({
                    "section": "cutlass_tactic_sweep",
                    "B": B,
                    "shape": name,
                    "candidate": cand_name,
                    "production": prod,
                    "M": M,
                    "N": N,
                    "K": K,
                    "beta": beta,
                    "calls_per_infer": int(shape["calls_per_infer"]),
                    "status": "ok",
                    "bit_exact_vs_production": bit_exact,
                    "max_abs_vs_production": max_abs,
                    "cos_vs_production": cos,
                    **timing,
                })

            del A, W, D, D_base, ref
            torch.cuda.empty_cache()

    return results


def run_encoder_o_chain(B: int, se: int, warmup: int, iters: int) -> dict:
    M, D_MODEL = B * se, 2048
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    src = torch.randn((M, D_MODEL), dtype=torch.float16, device="cuda") * 0.05
    A = torch.empty(src.numel(), dtype=torch.uint8, device="cuda")
    W = _make_fp8((D_MODEL, D_MODEL), scale)
    out = torch.randn((M, D_MODEL), dtype=torch.float16, device="cuda") * 0.01

    def quant() -> None:
        fvk.quantize_fp8_static_fp16(_ptr(src), _ptr(A), _ptr(scale), src.numel(), 0)

    def gemm() -> None:
        rc = _run_cutlass(fvk.cutlass_fp8_sq, A, W, out, M, D_MODEL, D_MODEL, 1.0, 1.0)
        if rc != 0:
            raise RuntimeError(f"cutlass_fp8_sq rc={rc}")

    def chain() -> None:
        quant()
        gemm()

    quant_t = _time_cuda(quant, warmup, iters)
    gemm_t = _time_cuda(gemm, warmup, iters)
    chain_t = _time_cuda(chain, warmup, iters)
    del src, A, W, out
    torch.cuda.empty_cache()
    return {
        "section": "producer_chain",
        "name": "encoder_attn_quant_to_o_gemm",
        "B": B,
        "M": M,
        "N": D_MODEL,
        "K": D_MODEL,
        "producer": "quantize_fp8_static_fp16",
        "consumer": "cutlass_fp8_sq_beta1",
        "calls_per_infer": 17,
        "producer_ms": quant_t["avg_ms"],
        "consumer_ms": gemm_t["avg_ms"],
        "chain_ms": chain_t["avg_ms"],
        "visible_materialization_ms": max(chain_t["avg_ms"] - gemm_t["avg_ms"], 0.0),
        "producer_timing": quant_t,
        "consumer_timing": gemm_t,
        "chain_timing": chain_t,
    }


def run_encoder_geglu_down_chain(B: int, se: int, warmup: int, iters: int,
                                 down_tactic_mode: str = "auto") -> dict:
    M, H, D_MODEL = B * se, 16384, 2048
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    down_tactic = _encoder_down_tactic(B, down_tactic_mode)
    down_fn = CUTLASS_CANDIDATES[down_tactic]
    merged = torch.randn((M, 2 * H), dtype=torch.float16, device="cuda") * 0.05
    hid_fp8 = torch.empty((M * H,), dtype=torch.uint8, device="cuda")
    W = _make_fp8((H, D_MODEL), scale)
    out = torch.randn((M, D_MODEL), dtype=torch.float16, device="cuda") * 0.01

    def producer() -> None:
        fvk.gate_geglu_merged_fp8_fp16(_ptr(merged), _ptr(hid_fp8), M, H, _ptr(scale), 0)

    def gemm() -> None:
        rc = _run_cutlass(down_fn, hid_fp8, W, out, M, D_MODEL, H, 1.0, 1.0)
        if rc != 0:
            raise RuntimeError(f"cutlass_fp8_{down_tactic} rc={rc}")

    def chain() -> None:
        producer()
        gemm()

    prod_t = _time_cuda(producer, warmup, iters)
    gemm_t = _time_cuda(gemm, warmup, iters)
    chain_t = _time_cuda(chain, warmup, iters)
    del merged, hid_fp8, W, out
    torch.cuda.empty_cache()
    return {
        "section": "producer_chain",
        "name": "encoder_geglu_to_down_gemm",
        "B": B,
        "M": M,
        "N": D_MODEL,
        "K": H,
        "producer": "gate_geglu_merged_fp8_fp16",
        "consumer": f"cutlass_fp8_{down_tactic}_beta1",
        "down_tactic_mode": down_tactic_mode,
        "down_tactic": down_tactic,
        "calls_per_infer": 17,
        "producer_ms": prod_t["avg_ms"],
        "consumer_ms": gemm_t["avg_ms"],
        "chain_ms": chain_t["avg_ms"],
        "visible_materialization_ms": max(chain_t["avg_ms"] - gemm_t["avg_ms"], 0.0),
        "producer_timing": prod_t,
        "consumer_timing": gemm_t,
        "chain_timing": chain_t,
    }


def run_decoder_o_chain(B: int, sa: int, warmup: int, iters: int) -> dict:
    M, Q_DIM, D_MODEL = B * sa, 2048, 1024
    act_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    w_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    src = torch.randn((M, Q_DIM), dtype=torch.float16, device="cuda") * 0.05
    A = torch.empty(src.numel(), dtype=torch.uint8, device="cuda")
    W = _make_fp8((Q_DIM, D_MODEL), act_scale)
    out = torch.empty((M, D_MODEL), dtype=torch.float16, device="cuda")

    def quant() -> None:
        fvk.quantize_fp8_static_fp16(_ptr(src), _ptr(A), _ptr(act_scale), src.numel(), 0)

    def gemm() -> None:
        _run_descale(A, W, out, M, D_MODEL, Q_DIM, act_scale, w_scale)

    def chain() -> None:
        quant()
        gemm()

    quant_t = _time_cuda(quant, warmup, iters)
    gemm_t = _time_cuda(gemm, warmup, iters)
    chain_t = _time_cuda(chain, warmup, iters)
    del src, A, W, out
    torch.cuda.empty_cache()
    return {
        "section": "producer_chain",
        "name": "decoder_attn_quant_to_o_gemm",
        "B": B,
        "M": M,
        "N": D_MODEL,
        "K": Q_DIM,
        "producer": "quantize_fp8_static_fp16",
        "consumer": "fp8_gemm_descale_fp16",
        "calls_per_infer": 10 * 18,
        "producer_ms": quant_t["avg_ms"],
        "consumer_ms": gemm_t["avg_ms"],
        "chain_ms": chain_t["avg_ms"],
        "visible_materialization_ms": max(chain_t["avg_ms"] - gemm_t["avg_ms"], 0.0),
        "producer_timing": quant_t,
        "consumer_timing": gemm_t,
        "chain_timing": chain_t,
    }


def run_decoder_geglu_down_chain(B: int, sa: int, warmup: int, iters: int) -> dict:
    M, H, D_MODEL = B * sa, 4096, 1024
    act_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    w_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    merged = torch.randn((M, 2 * H), dtype=torch.float16, device="cuda") * 0.05
    hid_fp8 = torch.empty((M * H,), dtype=torch.uint8, device="cuda")
    W = _make_fp8((H, D_MODEL), act_scale)
    out = torch.empty((M, D_MODEL), dtype=torch.float16, device="cuda")

    def producer() -> None:
        fvk.gate_geglu_merged_fp8_fp16(_ptr(merged), _ptr(hid_fp8), M, H, _ptr(act_scale), 0)

    def gemm() -> None:
        _run_descale(hid_fp8, W, out, M, D_MODEL, H, act_scale, w_scale)

    def chain() -> None:
        producer()
        gemm()

    prod_t = _time_cuda(producer, warmup, iters)
    gemm_t = _time_cuda(gemm, warmup, iters)
    chain_t = _time_cuda(chain, warmup, iters)
    del merged, hid_fp8, W, out
    torch.cuda.empty_cache()
    return {
        "section": "producer_chain",
        "name": "decoder_geglu_to_down_gemm",
        "B": B,
        "M": M,
        "N": D_MODEL,
        "K": H,
        "producer": "gate_geglu_merged_fp8_fp16",
        "consumer": "fp8_gemm_descale_fp16",
        "calls_per_infer": 10 * 18,
        "producer_ms": prod_t["avg_ms"],
        "consumer_ms": gemm_t["avg_ms"],
        "chain_ms": chain_t["avg_ms"],
        "visible_materialization_ms": max(chain_t["avg_ms"] - gemm_t["avg_ms"], 0.0),
        "producer_timing": prod_t,
        "consumer_timing": gemm_t,
        "chain_timing": chain_t,
    }


def run_producer_chains(batch_sizes: list[int], se: int, sa: int,
                        warmup: int, iters: int,
                        include_decoder: bool,
                        encoder_down_tactic: str) -> list[dict]:
    results: list[dict] = []
    for B in batch_sizes:
        print(f"\n[chain] B={B} encoder attn quant -> O GEMM")
        r = run_encoder_o_chain(B, se, warmup, iters)
        print(
            f"  producer={r['producer_ms']:.4f} ms, "
            f"consumer={r['consumer_ms']:.4f} ms, chain={r['chain_ms']:.4f} ms")
        results.append(r)

        print(f"\n[chain] B={B} encoder GEGLU -> Down GEMM")
        r = run_encoder_geglu_down_chain(
            B, se, warmup, iters, encoder_down_tactic)
        print(
            f"  producer={r['producer_ms']:.4f} ms, "
            f"consumer={r['consumer_ms']:.4f} ms, chain={r['chain_ms']:.4f} ms "
            f"(down={r['down_tactic']})")
        results.append(r)

        if include_decoder:
            print(f"\n[chain] B={B} decoder attn quant -> O GEMM")
            r = run_decoder_o_chain(B, sa, warmup, iters)
            print(
                f"  producer={r['producer_ms']:.4f} ms, "
                f"consumer={r['consumer_ms']:.4f} ms, chain={r['chain_ms']:.4f} ms")
            results.append(r)

            print(f"\n[chain] B={B} decoder GEGLU -> Down GEMM")
            r = run_decoder_geglu_down_chain(B, sa, warmup, iters)
            print(
                f"  producer={r['producer_ms']:.4f} ms, "
                f"consumer={r['consumer_ms']:.4f} ms, chain={r['chain_ms']:.4f} ms")
            results.append(r)

    return results


def run_geglu_lut_case(name: str, B: int, S: int, H: int,
                       warmup: int, iters: int) -> dict:
    M = B * S
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    merged = torch.randn((M, 2 * H), dtype=torch.float16, device="cuda") * 0.05
    ref = torch.empty((M * H,), dtype=torch.uint8, device="cuda")
    lut = torch.empty_like(ref)
    row8 = torch.empty_like(ref) if hasattr(fvk, "gate_geglu_merged_fp8_row8_fp16") else None

    def run_ref() -> None:
        fvk.gate_geglu_merged_fp8_fp16(_ptr(merged), _ptr(ref), M, H, _ptr(scale), 0)

    def run_lut() -> None:
        fvk.gate_geglu_merged_fp8_lut_fp16(_ptr(merged), _ptr(lut), M, H, _ptr(scale), 0)

    def run_row8() -> None:
        if row8 is None:
            raise RuntimeError("row8 candidate is unavailable")
        fvk.gate_geglu_merged_fp8_row8_fp16(_ptr(merged), _ptr(row8), M, H, _ptr(scale), 0)

    run_ref()
    run_lut()
    if row8 is not None:
        run_row8()
    torch.cuda.synchronize()

    lut_exact = bool(torch.equal(ref, lut))
    lut_mismatch_count = int((ref != lut).sum().item())
    if lut_mismatch_count:
        lut_max_byte_abs = int((ref.to(torch.int16) - lut.to(torch.int16)).abs().max().item())
    else:
        lut_max_byte_abs = 0

    row8_exact = None
    row8_mismatch_count = None
    row8_max_byte_abs = None
    if row8 is not None:
        row8_exact = bool(torch.equal(ref, row8))
        row8_mismatch_count = int((ref != row8).sum().item())
        if row8_mismatch_count:
            row8_max_byte_abs = int((ref.to(torch.int16) - row8.to(torch.int16)).abs().max().item())
        else:
            row8_max_byte_abs = 0

    ref_t = _time_cuda(run_ref, warmup, iters)
    lut_t = _time_cuda(run_lut, warmup, iters)
    row8_t = _time_cuda(run_row8, warmup, iters) if row8 is not None else None
    lut_speedup = ref_t["avg_ms"] / lut_t["avg_ms"] if lut_t["avg_ms"] > 0 else math.nan
    row8_speedup = (
        ref_t["avg_ms"] / row8_t["avg_ms"]
        if row8_t is not None and row8_t["avg_ms"] > 0 else math.nan)
    print(
        f"  ref={ref_t['avg_ms']:.4f} ms, lut={lut_t['avg_ms']:.4f} ms, "
        f"lut_speedup={lut_speedup:.2f}x, lut_exact={lut_exact}, "
        f"lut_mismatches={lut_mismatch_count}")
    if row8_t is not None:
        print(
            f"  row8={row8_t['avg_ms']:.4f} ms, "
            f"row8_speedup={row8_speedup:.2f}x, row8_exact={row8_exact}, "
            f"row8_mismatches={row8_mismatch_count}")

    del merged, ref, lut, row8
    torch.cuda.empty_cache()
    return {
        "section": "geglu_lut_sweep",
        "name": name,
        "B": B,
        "S": S,
        "M": M,
        "H": H,
        "reference": "gate_geglu_merged_fp8_fp16",
        "candidate": "gate_geglu_merged_fp8_lut_fp16",
        "status": "ok",
        "bit_exact_vs_reference": lut_exact,
        "mismatch_count": lut_mismatch_count,
        "max_byte_abs_vs_reference": lut_max_byte_abs,
        "reference_avg_ms": ref_t["avg_ms"],
        "candidate_avg_ms": lut_t["avg_ms"],
        "speedup": lut_speedup,
        "reference_timing": ref_t,
        "candidate_timing": lut_t,
        "row8_candidate": "gate_geglu_merged_fp8_row8_fp16" if row8_t is not None else None,
        "row8_bit_exact_vs_reference": row8_exact,
        "row8_mismatch_count": row8_mismatch_count,
        "row8_max_byte_abs_vs_reference": row8_max_byte_abs,
        "row8_avg_ms": row8_t["avg_ms"] if row8_t is not None else None,
        "row8_speedup": row8_speedup,
        "row8_timing": row8_t,
    }


def run_geglu_lut_sweep(batch_sizes: list[int], se: int, sa: int,
                        warmup: int, iters: int,
                        include_decoder: bool) -> list[dict]:
    if not hasattr(fvk, "gate_geglu_merged_fp8_lut_fp16"):
        print("\n[geglu-lut] skipped: flash_rt_kernels lacks LUT binding")
        return []

    print("\n[geglu-lut] initializing GPU half-domain GELU LUT")
    fvk.init_gelu_half_lut(0)
    torch.cuda.synchronize()

    results: list[dict] = []
    for B in batch_sizes:
        print(f"\n[geglu-lut] B={B} encoder GEGLU FP8 M={B * se} H=16384")
        results.append(run_geglu_lut_case(
            "encoder_geglu_fp8", B, se, 16384, warmup, iters))

        if include_decoder:
            print(f"\n[geglu-lut] B={B} decoder GEGLU FP8 M={B * sa} H=4096")
            results.append(run_geglu_lut_case(
                "decoder_geglu_fp8", B, sa, 4096, warmup, iters))

    return results


def run_encoder_split_gateup_case(B: int, se: int,
                                  warmup: int, iters: int) -> dict:
    M, H, D_MODEL = B * se, 16384, 2048
    n_hid = M * H
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    merged_tactic_name = "sq" if B >= 2 else "t1"
    merged_tactic = fvk.cutlass_fp8_sq if B >= 2 else fvk.cutlass_fp8_t1

    A = _make_fp8((M, D_MODEL), scale)
    W_merged = _make_fp8((D_MODEL, 2 * H), scale)
    W_merged_2d = W_merged.view(D_MODEL, 2 * H)
    W_gate = W_merged_2d[:, :H].contiguous()
    W_up = W_merged_2d[:, H:].contiguous()

    merged = torch.empty((M, 2 * H), dtype=torch.float16, device="cuda")
    gate = torch.empty((M, H), dtype=torch.float16, device="cuda")
    up = torch.empty((M, H), dtype=torch.float16, device="cuda")
    hid_ref = torch.empty((n_hid,), dtype=torch.uint8, device="cuda")
    hid_split = torch.empty_like(hid_ref)

    def merged_gateup() -> None:
        rc = _run_cutlass(merged_tactic, A, W_merged, merged,
                          M, 2 * H, D_MODEL, 1.0, 0.0)
        if rc != 0:
            raise RuntimeError(f"merged {merged_tactic_name} rc={rc}")

    def merged_geglu() -> None:
        fvk.gate_geglu_merged_fp8_fp16(
            _ptr(merged), _ptr(hid_ref), M, H, _ptr(scale), 0)

    def merged_pair() -> None:
        merged_gateup()
        merged_geglu()

    def split_gate() -> None:
        rc = _run_cutlass(merged_tactic, A, W_gate, gate,
                          M, H, D_MODEL, 1.0, 0.0)
        if rc != 0:
            raise RuntimeError(f"split gate {merged_tactic_name} rc={rc}")

    def split_up() -> None:
        rc = _run_cutlass(merged_tactic, A, W_up, up,
                          M, H, D_MODEL, 1.0, 0.0)
        if rc != 0:
            raise RuntimeError(f"split up {merged_tactic_name} rc={rc}")

    def split_geglu() -> None:
        fvk.gate_geglu_split_fp8_fp16(
            _ptr(gate), _ptr(up), _ptr(hid_split), n_hid, _ptr(scale), 0)

    def split_pair() -> None:
        split_gate()
        split_up()
        split_geglu()

    merged_pair()
    split_pair()
    torch.cuda.synchronize()
    gate_exact = bool(torch.equal(gate, merged[:, :H]))
    up_exact = bool(torch.equal(up, merged[:, H:]))
    split_exact, split_mismatches, split_max_byte_abs = _byte_diff_stats(hid_ref, hid_split)

    merged_gateup_t = _time_cuda(merged_gateup, warmup, iters)
    merged_geglu_t = _time_cuda(merged_geglu, warmup, iters)
    merged_pair_t = _time_cuda(merged_pair, warmup, iters)
    split_gate_t = _time_cuda(split_gate, warmup, iters)
    split_up_t = _time_cuda(split_up, warmup, iters)
    split_geglu_t = _time_cuda(split_geglu, warmup, iters)
    split_pair_t = _time_cuda(split_pair, warmup, iters)

    split_speedup = (
        merged_pair_t["avg_ms"] / split_pair_t["avg_ms"]
        if split_pair_t["avg_ms"] > 0 else math.nan)

    epilogue_result: dict[str, object] = {}
    if hasattr(fvk, "cutlass_fp8_gelu") and hasattr(fvk, "mul_split_fp8_fp16"):
        gate_epi = torch.empty_like(gate)
        hid_epi = torch.empty_like(hid_ref)

        def epilogue_gate() -> None:
            rc = _run_cutlass(fvk.cutlass_fp8_gelu, A, W_gate, gate_epi,
                              M, H, D_MODEL, 1.0, 0.0)
            if rc != 0:
                raise RuntimeError(f"cutlass_fp8_gelu rc={rc}")

        def epilogue_mul_quant() -> None:
            fvk.mul_split_fp8_fp16(
                _ptr(gate_epi), _ptr(up), _ptr(hid_epi), n_hid, _ptr(scale), 0)

        def epilogue_pair() -> None:
            epilogue_gate()
            split_up()
            epilogue_mul_quant()

        epilogue_pair()
        torch.cuda.synchronize()
        epi_exact, epi_mismatches, epi_max_byte_abs = _byte_diff_stats(hid_ref, hid_epi)
        epilogue_gate_t = _time_cuda(epilogue_gate, warmup, iters)
        epilogue_mul_t = _time_cuda(epilogue_mul_quant, warmup, iters)
        epilogue_pair_t = _time_cuda(epilogue_pair, warmup, iters)
        epi_speedup = (
            merged_pair_t["avg_ms"] / epilogue_pair_t["avg_ms"]
            if epilogue_pair_t["avg_ms"] > 0 else math.nan)
        epilogue_result = {
            "epilogue_status": "ok",
            "epilogue_hid_bit_exact_vs_reference": epi_exact,
            "epilogue_mismatch_count": epi_mismatches,
            "epilogue_max_byte_abs_vs_reference": epi_max_byte_abs,
            "epilogue_gate_avg_ms": epilogue_gate_t["avg_ms"],
            "epilogue_mul_quant_avg_ms": epilogue_mul_t["avg_ms"],
            "epilogue_pair_avg_ms": epilogue_pair_t["avg_ms"],
            "epilogue_pair_speedup": epi_speedup,
            "epilogue_per_infer_delta_ms": (
                (merged_pair_t["avg_ms"] - epilogue_pair_t["avg_ms"]) * 17),
            "epilogue_gate_timing": epilogue_gate_t,
            "epilogue_mul_quant_timing": epilogue_mul_t,
            "epilogue_pair_timing": epilogue_pair_t,
        }
        del gate_epi, hid_epi
    else:
        epilogue_result = {
            "epilogue_status": "unavailable",
        }

    print(
        f"  merged_pair={merged_pair_t['avg_ms']:.4f} ms, "
        f"split_pair={split_pair_t['avg_ms']:.4f} ms, "
        f"split_speedup={split_speedup:.2f}x, "
        f"split_exact={split_exact}, mismatches={split_mismatches}")
    if epilogue_result.get("epilogue_status") == "ok":
        print(
            f"  epilogue_pair={epilogue_result['epilogue_pair_avg_ms']:.4f} ms, "
            f"epilogue_speedup={epilogue_result['epilogue_pair_speedup']:.2f}x, "
            f"epilogue_exact={epilogue_result['epilogue_hid_bit_exact_vs_reference']}, "
            f"mismatches={epilogue_result['epilogue_mismatch_count']}")

    result = {
        "section": "split_gateup_geglu_sweep",
        "name": "encoder_split_gateup_geglu_fp8",
        "B": B,
        "M": M,
        "H": H,
        "D": D_MODEL,
        "calls_per_infer": 17,
        "merged_tactic": merged_tactic_name,
        "gate_output_exact_vs_merged": gate_exact,
        "up_output_exact_vs_merged": up_exact,
        "split_hid_bit_exact_vs_reference": split_exact,
        "split_mismatch_count": split_mismatches,
        "split_max_byte_abs_vs_reference": split_max_byte_abs,
        "merged_gateup_avg_ms": merged_gateup_t["avg_ms"],
        "merged_geglu_avg_ms": merged_geglu_t["avg_ms"],
        "merged_pair_avg_ms": merged_pair_t["avg_ms"],
        "split_gate_avg_ms": split_gate_t["avg_ms"],
        "split_up_avg_ms": split_up_t["avg_ms"],
        "split_geglu_avg_ms": split_geglu_t["avg_ms"],
        "split_pair_avg_ms": split_pair_t["avg_ms"],
        "split_pair_speedup": split_speedup,
        "split_per_infer_delta_ms": (
            (merged_pair_t["avg_ms"] - split_pair_t["avg_ms"]) * 17),
        "merged_gateup_timing": merged_gateup_t,
        "merged_geglu_timing": merged_geglu_t,
        "merged_pair_timing": merged_pair_t,
        "split_gate_timing": split_gate_t,
        "split_up_timing": split_up_t,
        "split_geglu_timing": split_geglu_t,
        "split_pair_timing": split_pair_t,
        **epilogue_result,
    }
    del A, W_merged, W_gate, W_up, merged, gate, up, hid_ref, hid_split
    torch.cuda.empty_cache()
    return result


def run_encoder_split_gateup_sweep(batch_sizes: list[int], se: int,
                                   warmup: int, iters: int) -> list[dict]:
    if not hasattr(fvk, "gate_geglu_split_fp8_fp16"):
        print("\n[split-gateup] skipped: flash_rt_kernels lacks split GEGLU binding")
        return []

    results: list[dict] = []
    for B in batch_sizes:
        print(
            f"\n[split-gateup] B={B} encoder Gate+Up/GEGLU "
            f"M={B * se} H=16384 D=2048")
        results.append(run_encoder_split_gateup_case(B, se, warmup, iters))
    return results


def run_virtual_mainloop_model(results: list[dict], batch_sizes: list[int],
                               se: int, down_n_tile: int) -> list[dict]:
    """Model the GEGLU->Down fusion boundary from measured chain timings.

    This is intentionally a cost model, not a CUDA benchmark. It answers a
    design question before writing a custom mainloop: if a virtual FP8 A tile is
    regenerated once per Down N tile, the GEGLU producer work is repeated
    ceil(N / down_n_tile) times. That fusion shape is only worth pursuing if
    the repeated producer cost is below the visible materialization budget.
    """
    chain_rows = {
        int(r["B"]): r for r in results
        if r.get("section") == "producer_chain"
        and r.get("name") == "encoder_geglu_to_down_gemm"
    }
    modeled: list[dict] = []
    for B in batch_sizes:
        row = chain_rows.get(B)
        if row is None:
            print(
                f"\n[virtual-mainloop-model] B={B} skipped: "
                "run producer chains first")
            continue

        M, N, H = B * se, 2048, 16384
        elems = M * H
        n_tiles = math.ceil(N / down_n_tile)
        producer_ms = float(row["producer_ms"])
        consumer_ms = float(row["consumer_ms"])
        chain_ms = float(row["chain_ms"])
        visible_ms = float(row["visible_materialization_ms"])
        calls = int(row["calls_per_infer"])

        # Materialized path:
        #   producer reads gate+up fp16 and writes hid_fp8,
        #   Down reads hid_fp8 once per N tile in the worst CTA-local model.
        gate_up_read_bytes = 2 * elems * 2
        hid_write_bytes = elems
        hid_read_lower_bytes = elems
        hid_read_worst_n_tile_bytes = elems * n_tiles
        materialized_boundary_lower_bytes = hid_write_bytes + hid_read_lower_bytes
        materialized_boundary_worst_bytes = hid_write_bytes + hid_read_worst_n_tile_bytes

        # Naive virtual path:
        #   no hid_fp8 global write/read, but regenerate GEGLU for each N tile.
        naive_gate_up_read_bytes = gate_up_read_bytes * n_tiles
        naive_extra_gate_up_read_bytes = naive_gate_up_read_bytes - gate_up_read_bytes
        naive_recompute_ms = producer_ms * n_tiles
        naive_vs_chain_delta_ms = naive_recompute_ms + consumer_ms - chain_ms

        max_recompute_factor = (
            1.0 + visible_ms / producer_ms if producer_ms > 0 else math.nan)
        min_required_reuse = (
            n_tiles / max_recompute_factor
            if max_recompute_factor > 0 else math.inf)

        print(
            f"\n[virtual-mainloop-model] B={B} encoder GEGLU->Down: "
            f"N_tiles={n_tiles}, visible={visible_ms:.4f} ms/layer, "
            f"max_recompute_factor={max_recompute_factor:.2f}x")
        print(
            f"  naive per-N-tile recompute ~= {naive_recompute_ms:.4f} ms/layer, "
            f"delta_vs_chain={naive_vs_chain_delta_ms:.4f} ms/layer "
            f"({naive_vs_chain_delta_ms * calls:.2f} ms/infer)")
        print(
            f"  boundary_traffic_lower={materialized_boundary_lower_bytes / 1e6:.1f} MB/layer, "
            f"boundary_traffic_n_tile={materialized_boundary_worst_bytes / 1e6:.1f} MB/layer, "
            f"naive_extra_gateup_read={naive_extra_gate_up_read_bytes / 1e6:.1f} MB/layer")

        modeled.append({
            "section": "virtual_mainloop_model",
            "name": "encoder_geglu_to_down_virtual_a_model",
            "B": B,
            "M": M,
            "N": N,
            "H": H,
            "down_n_tile": down_n_tile,
            "down_n_tiles": n_tiles,
            "calls_per_infer": calls,
            "producer_ms": producer_ms,
            "consumer_ms": consumer_ms,
            "chain_ms": chain_ms,
            "visible_materialization_ms": visible_ms,
            "visible_materialization_per_infer_ms": visible_ms * calls,
            "gate_up_read_bytes_per_layer": gate_up_read_bytes,
            "hid_fp8_write_bytes_per_layer": hid_write_bytes,
            "hid_fp8_read_lower_bytes_per_layer": hid_read_lower_bytes,
            "hid_fp8_read_n_tile_bytes_per_layer": hid_read_worst_n_tile_bytes,
            "materialized_boundary_lower_bytes_per_layer": materialized_boundary_lower_bytes,
            "materialized_boundary_n_tile_bytes_per_layer": materialized_boundary_worst_bytes,
            "naive_gate_up_read_bytes_per_layer": naive_gate_up_read_bytes,
            "naive_extra_gate_up_read_bytes_per_layer": naive_extra_gate_up_read_bytes,
            "naive_recompute_ms_per_layer": naive_recompute_ms,
            "naive_delta_vs_chain_ms_per_layer": naive_vs_chain_delta_ms,
            "naive_delta_vs_chain_ms_per_infer": naive_vs_chain_delta_ms * calls,
            "max_recompute_factor_for_break_even": max_recompute_factor,
            "min_required_a_tile_reuse_vs_naive": min_required_reuse,
        })
    return modeled


def _decoder_descale_shapes(B: int, sa: int) -> list[dict]:
    m = B * sa
    return [
        {
            "name": "decoder_qkv",
            "M": m,
            "N": 2560,
            "K": 1024,
            "calls_per_infer": 10 * 18,
        },
        {
            "name": "decoder_o",
            "M": m,
            "N": 1024,
            "K": 2048,
            "calls_per_infer": 10 * 18,
        },
        {
            "name": "decoder_gateup",
            "M": m,
            "N": 8192,
            "K": 1024,
            "calls_per_infer": 10 * 18,
        },
        {
            "name": "decoder_down",
            "M": m,
            "N": 1024,
            "K": 4096,
            "calls_per_infer": 10 * 18,
        },
    ]


def run_descale_autotune_case(B: int, shape: dict,
                              warmup: int, iters: int,
                              autotune_warmup: int,
                              autotune_iters: int) -> dict:
    M, N, K = int(shape["M"]), int(shape["N"]), int(shape["K"])
    act_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    w_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    A = _make_fp8((M, K), act_scale)
    W = _make_fp8((K, N), act_scale)
    baseline_out = torch.empty((M, N), dtype=torch.float16, device="cuda")
    tuned_out = torch.empty_like(baseline_out)

    def baseline() -> None:
        _run_descale(A, W, baseline_out, M, N, K, act_scale, w_scale)

    def tuned() -> None:
        _run_descale(A, W, tuned_out, M, N, K, act_scale, w_scale)

    baseline()
    torch.cuda.synchronize()
    baseline_t = _time_cuda(baseline, warmup, iters)

    autotune_best_ms = float(fvk.fp8_gemm_descale_fp16_autotune(
        M, N, K, autotune_warmup, autotune_iters, 0))
    torch.cuda.synchronize()

    tuned()
    torch.cuda.synchronize()
    bit_exact = bool(torch.equal(baseline_out, tuned_out))
    max_abs = float((baseline_out - tuned_out).abs().max().item())
    cos = _cosine(baseline_out, tuned_out)
    tuned_t = _time_cuda(tuned, warmup, iters)
    speedup = baseline_t["avg_ms"] / tuned_t["avg_ms"] if tuned_t["avg_ms"] > 0 else math.nan
    per_infer_delta = (
        (baseline_t["avg_ms"] - tuned_t["avg_ms"]) *
        int(shape["calls_per_infer"]))
    print(
        f"  heuristic={baseline_t['avg_ms']:.4f} ms, "
        f"tuned={tuned_t['avg_ms']:.4f} ms, speedup={speedup:.2f}x, "
        f"autotune_best={autotune_best_ms:.4f} ms, "
        f"bit_exact={bit_exact}, max_abs={max_abs:.3g}, cos={cos:.9f}")

    del A, W, baseline_out, tuned_out
    torch.cuda.empty_cache()
    return {
        "section": "descale_autotune_sweep",
        "B": B,
        "shape": shape["name"],
        "M": M,
        "N": N,
        "K": K,
        "calls_per_infer": int(shape["calls_per_infer"]),
        "status": "ok",
        "bit_exact_vs_heuristic": bit_exact,
        "max_abs_vs_heuristic": max_abs,
        "cos_vs_heuristic": cos,
        "heuristic_avg_ms": baseline_t["avg_ms"],
        "tuned_avg_ms": tuned_t["avg_ms"],
        "speedup": speedup,
        "per_infer_delta_ms": per_infer_delta,
        "autotune_best_ms": autotune_best_ms,
        "heuristic_timing": baseline_t,
        "tuned_timing": tuned_t,
    }


def run_descale_autotune_sweep(batch_sizes: list[int], sa: int,
                               warmup: int, iters: int,
                               autotune_warmup: int,
                               autotune_iters: int) -> list[dict]:
    if not hasattr(fvk, "fp8_gemm_descale_fp16_autotune"):
        print("\n[descale-autotune] skipped: flash_rt_kernels lacks autotune binding")
        return []

    results: list[dict] = []
    for B in batch_sizes:
        for shape in _decoder_descale_shapes(B, sa):
            print(
                f"\n[descale-autotune] B={B} {shape['name']} "
                f"M={shape['M']} N={shape['N']} K={shape['K']}")
            try:
                results.append(run_descale_autotune_case(
                    B, shape, warmup, iters, autotune_warmup, autotune_iters))
            except Exception as exc:
                print(f"  failed: {exc}")
                results.append({
                    "section": "descale_autotune_sweep",
                    "B": B,
                    "shape": shape["name"],
                    "M": int(shape["M"]),
                    "N": int(shape["N"]),
                    "K": int(shape["K"]),
                    "status": "exception",
                    "error": str(exc),
                })

    return results


def _print_best_tactics(results: list[dict]) -> None:
    rows = [r for r in results if r.get("section") == "cutlass_tactic_sweep"
            and r.get("status") == "ok"]
    if not rows:
        return
    print("\n=== Best CUTLASS FP8 tactic by shape ===")
    grouped: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        grouped.setdefault((int(r["B"]), str(r["shape"])), []).append(r)
    for (B, shape), items in sorted(grouped.items()):
        best = min(items, key=lambda r: r["avg_ms"])
        prod = next((r for r in items if r["candidate"] == r["production"]), None)
        prod_ms = prod["avg_ms"] if prod is not None else math.nan
        speed = prod_ms / best["avg_ms"] if best["avg_ms"] > 0 else math.nan
        exact = "exact" if best["bit_exact_vs_production"] else "non-exact"
        prod_label = str(best["production"]) if prod is None else str(prod["production"])
        prod_value = "not measured" if prod is None else f"{prod_ms:.3f} ms"
        print(
            f"B={B:<2} {shape:<16} prod={prod_label:<5} "
            f"{prod_value} | best={best['candidate']:<5} "
            f"{best['avg_ms']:.3f} ms ({speed:.2f}x, {exact})")


def _print_chain_summary(results: list[dict]) -> None:
    rows = [r for r in results if r.get("section") == "producer_chain"]
    if not rows:
        return
    print("\n=== Producer -> FP8 -> GEMM chain upper bounds ===")
    for r in rows:
        share = r["producer_ms"] / r["chain_ms"] * 100.0 if r["chain_ms"] > 0 else math.nan
        per_infer = r["visible_materialization_ms"] * int(r["calls_per_infer"])
        print(
            f"B={r['B']:<2} {r['name']:<32} "
            f"producer={r['producer_ms']:.4f} ms, consumer={r['consumer_ms']:.4f} ms, "
            f"chain={r['chain_ms']:.4f} ms, producer_share={share:.1f}%, "
            f"visible_x_calls={per_infer:.2f} ms")


def _print_geglu_lut_summary(results: list[dict]) -> None:
    rows = [r for r in results if r.get("section") == "geglu_lut_sweep"]
    if not rows:
        return
    print("\n=== Exact FP16 GELU LUT GEGLU->FP8 sweep ===")
    for r in rows:
        exact = "exact" if r["bit_exact_vs_reference"] else "mismatch"
        print(
            f"B={r['B']:<2} {r['name']:<20} "
            f"ref={r['reference_avg_ms']:.4f} ms, "
            f"lut={r['candidate_avg_ms']:.4f} ms, "
            f"speedup={r['speedup']:.2f}x, {exact}, "
            f"mismatches={r['mismatch_count']}")
        if r.get("row8_avg_ms") is not None:
            row8_exact = "exact" if r["row8_bit_exact_vs_reference"] else "mismatch"
            print(
                f"     row8={r['row8_avg_ms']:.4f} ms, "
                f"speedup={r['row8_speedup']:.2f}x, {row8_exact}, "
                f"mismatches={r['row8_mismatch_count']}")


def _print_split_gateup_summary(results: list[dict]) -> None:
    rows = [r for r in results if r.get("section") == "split_gateup_geglu_sweep"]
    if not rows:
        return
    print("\n=== Split Gate/Up exactness and epilogue-fusion probe ===")
    for r in rows:
        split_exact = "exact" if r["split_hid_bit_exact_vs_reference"] else "mismatch"
        print(
            f"B={r['B']:<2} merged_pair={r['merged_pair_avg_ms']:.4f} ms, "
            f"split_pair={r['split_pair_avg_ms']:.4f} ms, "
            f"speedup={r['split_pair_speedup']:.2f}x, {split_exact}, "
            f"gate_exact={r['gate_output_exact_vs_merged']}, "
            f"up_exact={r['up_output_exact_vs_merged']}, "
            f"visible_x17={r['split_per_infer_delta_ms']:.2f} ms")
        if r.get("epilogue_status") == "ok":
            epi_exact = (
                "exact" if r["epilogue_hid_bit_exact_vs_reference"]
                else "mismatch")
            print(
                f"     gelu_epilogue_pair={r['epilogue_pair_avg_ms']:.4f} ms, "
                f"speedup={r['epilogue_pair_speedup']:.2f}x, {epi_exact}, "
                f"mismatches={r['epilogue_mismatch_count']}, "
                f"visible_x17={r['epilogue_per_infer_delta_ms']:.2f} ms")


def _print_virtual_mainloop_model_summary(results: list[dict]) -> None:
    rows = [r for r in results if r.get("section") == "virtual_mainloop_model"]
    if not rows:
        return
    print("\n=== Virtual FP8 A mainloop cost model ===")
    for r in rows:
        print(
            f"B={r['B']:<2} N_tiles={r['down_n_tiles']:<2} "
            f"visible={r['visible_materialization_ms']:.4f} ms/layer "
            f"({r['visible_materialization_per_infer_ms']:.2f} ms/infer), "
            f"naive_recompute={r['naive_recompute_ms_per_layer']:.4f} ms/layer, "
            f"naive_delta={r['naive_delta_vs_chain_ms_per_infer']:.2f} ms/infer")
        print(
            f"     max_recompute_factor="
            f"{r['max_recompute_factor_for_break_even']:.2f}x, "
            f"required_A_tile_reuse>="
            f"{r['min_required_a_tile_reuse_vs_naive']:.1f}x, "
            f"boundary_lower="
            f"{r['materialized_boundary_lower_bytes_per_layer'] / 1e6:.1f} MB/layer, "
            f"boundary_n_tile="
            f"{r['materialized_boundary_n_tile_bytes_per_layer'] / 1e6:.1f} MB/layer")


def _print_descale_autotune_summary(results: list[dict]) -> None:
    rows = [r for r in results if r.get("section") == "descale_autotune_sweep"]
    if not rows:
        return
    print("\n=== Decoder fp8_gemm_descale_fp16 autotune sweep ===")
    for r in rows:
        if r.get("status") != "ok":
            print(f"B={r['B']:<2} {r['shape']:<16} failed: {r.get('error', '')}")
            continue
        exact = "exact" if r["bit_exact_vs_heuristic"] else "non-exact"
        print(
            f"B={r['B']:<2} {r['shape']:<16} "
            f"heuristic={r['heuristic_avg_ms']:.4f} ms, "
            f"tuned={r['tuned_avg_ms']:.4f} ms, "
            f"speedup={r['speedup']:.2f}x, {exact}, "
            f"visible_x_calls={r['per_infer_delta_ms']:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pi0.5 Thor batch inference kernel research microbenchmarks.")
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default="1,2,4,8")
    parser.add_argument("--se", type=int, default=526,
                        help="Encoder sequence length per sample. Default matches the repo bench prompt.")
    parser.add_argument("--sa", type=int, default=10,
                        help="Action expert sequence length per sample.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--skip-tactics", action="store_true")
    parser.add_argument("--skip-chains", action="store_true")
    parser.add_argument("--include-geglu-lut", action="store_true",
                        help="Benchmark exact FP16-domain GELU LUT GEGLU->FP8 prototype.")
    parser.add_argument("--include-split-gateup", action="store_true",
                        help="Benchmark split Gate/Up GEGLU producer variants for encoder hot shape.")
    parser.add_argument("--include-virtual-mainloop-model", action="store_true",
                        help="Model GEGLU->Down virtual FP8 A reuse/recompute costs from producer-chain timings.")
    parser.add_argument("--down-n-tile", type=int, default=128,
                        help="N tile used by the current Down GEMM model. Default matches cutlass_fp8_wide.")
    parser.add_argument("--encoder-down-tactic", default="auto",
                        choices=["auto", *sorted(CUTLASS_CANDIDATES.keys())],
                        help="Encoder Down GEMM tactic used in producer-chain and virtual-mainloop modeling.")
    parser.add_argument("--include-descale-autotune", action="store_true",
                        help="Autotune decoder fp8_gemm_descale_fp16 shapes and compare against heuristic.")
    parser.add_argument("--autotune-warmup", type=int, default=2)
    parser.add_argument("--autotune-iters", type=int, default=8)
    parser.add_argument("--include-decoder", action="store_true",
                        help="Also benchmark small-M decoder descale GEMM chains.")
    parser.add_argument("--max-candidates", type=int,
                        help="Diagnostic knob to limit the CUTLASS candidate count.")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(1234)

    all_results: list[dict] = []
    if not args.skip_tactics:
        all_results.extend(run_cutlass_tactic_sweep(
            args.batch_sizes, args.se, args.warmup, args.iters, args.max_candidates))
    if not args.skip_chains:
        all_results.extend(run_producer_chains(
            args.batch_sizes, args.se, args.sa, args.warmup, args.iters,
            args.include_decoder, args.encoder_down_tactic))
    if args.include_descale_autotune:
        all_results.extend(run_descale_autotune_sweep(
            args.batch_sizes, args.sa, args.warmup, args.iters,
            args.autotune_warmup, args.autotune_iters))
    if args.include_geglu_lut:
        all_results.extend(run_geglu_lut_sweep(
            args.batch_sizes, args.se, args.sa, args.warmup, args.iters,
            args.include_decoder))
    if args.include_split_gateup:
        all_results.extend(run_encoder_split_gateup_sweep(
            args.batch_sizes, args.se, args.warmup, args.iters))
    if args.include_virtual_mainloop_model:
        all_results.extend(run_virtual_mainloop_model(
            all_results, args.batch_sizes, args.se, args.down_n_tile))

    _print_best_tactics(all_results)
    _print_chain_summary(all_results)
    _print_descale_autotune_summary(all_results)
    _print_geglu_lut_summary(all_results)
    _print_split_gateup_summary(all_results)
    _print_virtual_mainloop_model_summary(all_results)

    if args.json_out:
        payload = {
            "batch_sizes": args.batch_sizes,
            "se": args.se,
            "sa": args.sa,
            "warmup": args.warmup,
            "iters": args.iters,
            "results": all_results,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
