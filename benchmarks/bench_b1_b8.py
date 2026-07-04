"""Batch inference benchmark for Pi0.5 Thor B=1..8.

Default run:
    CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_b1_b8.py

Faster diagnostic run:
    CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_b1_b8.py \
        --batch-sizes 1-8 --warmup 10 --iters 30 --profile

Note:
    This script benchmarks Pi05TorchFrontendThor, i.e. the Thor FP8 path by
    default or the FP16 path with --no-fp8. It does not exercise
    Pi05TorchFrontendThorFP4. As of 2026-06-20 the FP4 frontend only overrides
    the single-sample Enc+AE graph; B>=2 still needs a separate batched-FP4
    implementation before it can be reported as an FP4 service curve.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

CKPT = os.environ.get(
    "PI05_LIBERO_PYTORCH_CHECKPOINT",
    "/mnt/home/zengzixuan/workspace/checkpoints/pi05_libero_pytorch",
)


def _make_obs(seed: int = 42, *, same_views: bool = False) -> dict:
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    wrist = img.copy() if same_views else rng.randint(
        0, 256, (224, 224, 3), dtype=np.uint8)
    return {"image": img, "wrist_image": wrist}


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
    if not sizes or sizes[0] < 1:
        raise argparse.ArgumentTypeError("batch sizes must be >= 1")
    return sizes


def _summarize(times_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "avg": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _time_call(fn: Callable[[], object], iters: int) -> list[float]:
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _profile_replay(name: str, fn: Callable[[], object], iters: int) -> dict:
    torch.cuda.synchronize()
    times = _time_call(fn, iters)
    result = _summarize(times)
    result["name"] = name
    return result


def _profile_frontend(frontend, batch_size: int, iters: int) -> dict[str, dict]:
    """Measure graph-level stages after the main benchmark has warmed them.

    This intentionally profiles graph replays and GPU-side stages only. The
    end-to-end table remains the source of truth for CPU preprocessing, host
    synchronization, and D2H unpack overhead.
    """
    profile: dict[str, dict] = {}

    if batch_size == 1:
        if getattr(frontend, "_siglip_graph", None) is not None:
            profile["siglip_graph_incl_patch_postln"] = _profile_replay(
                "siglip_graph_incl_patch_postln",
                lambda: frontend._siglip_graph.replay(),
                iters,
            )
        if getattr(frontend, "_enc_ae_graph", None) is not None:
            profile["enc_ae_graph"] = _profile_replay(
                "enc_ae_graph",
                lambda: frontend._enc_ae_graph.replay(),
                iters,
            )
        return profile

    if getattr(frontend, "_siglip_batched_graph", None) is not None:
        profile["siglip_postln_graph"] = _profile_replay(
            "siglip_postln_graph",
            lambda: frontend._siglip_batched_graph.replay(),
            iters,
        )
    if getattr(frontend, "_enc_ae_graph_b2", None) is not None:
        profile["enc_ae_graph"] = _profile_replay(
            "enc_ae_graph",
            lambda: frontend._enc_ae_graph_b2.replay(),
            iters,
        )
    return profile


def _cuda_profiler_api():
    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaProfilerStart.restype = ctypes.c_int
    cudart.cudaProfilerStop.restype = ctypes.c_int
    return cudart


def _make_batch(prompt: str, batch_size: int, *, same_inputs: bool,
                same_views: bool) -> list[dict]:
    return [
        {
            "observation": _make_obs(0 if same_inputs else i,
                                     same_views=same_views),
            "prompt": prompt,
        }
        for i in range(batch_size)
    ]


def _make_infer_fn(frontend, batch_size: int, batch_data: list[dict],
                   seed: int | None):
    if batch_size == 1:
        obs = batch_data[0]["observation"]
        return lambda: frontend.infer(obs, seed=seed)
    return lambda: frontend.infer_multi_prompt_batch(batch_data, seed=seed)


def _make_context_reuse_upper_bound_fn(frontend, batch_size: int,
                                       seed: int | None,
                                       batch_data: list[dict] | None = None):
    """Replay only the action-generation graph from a resident VLM context.

    This is an experimental upper-bound probe for memory-for-compute research.
    It assumes the current visual-language context in ``_enc_x`` or
    ``_enc_x_b2`` is reusable from a cache and keeps the normal noise,
    Enc+AE graph replay, D2H transfer, and action unnormalization path.
    """
    from flash_rt.core.utils.actions import LIBERO_ACTION_DIM, unnormalize_actions

    if batch_size == 1:
        if (batch_data is not None
                and hasattr(frontend, "stage_context")
                and hasattr(frontend, "infer_from_cached_context")):
            frontend.stage_context(batch_data[0]["observation"])
            return lambda: frontend.infer_from_cached_context(seed=seed)

        if getattr(frontend, "_enc_ae_graph", None) is None:
            raise RuntimeError("serial Enc+AE graph is not captured yet")

        def run_one():
            if seed is None:
                frontend._g_noise.normal_()
            else:
                noise_np = np.random.RandomState(seed).randn(
                    frontend.Sa, 32).astype(np.float16)
                frontend._g_noise.view(-1, 32).copy_(
                    torch.from_numpy(noise_np).to("cuda", non_blocking=True))
            frontend._enc_ae_graph.replay()
            raw = frontend._g_noise.float().cpu().numpy()
            unnorm = unnormalize_actions(raw, frontend.norm_stats)
            return {"actions": unnorm[:, :LIBERO_ACTION_DIM]}

        return run_one

    raise RuntimeError(
        "B>1 context reuse is not reported here: the batched Enc+AE graph "
        "updates _enc_x_b2 in place, so correctness requires a separate "
        "batched staging snapshot API.")


def _print_summary(results: dict[int, dict]) -> None:
    print(f"\n{'=' * 96}")
    print(
        f"{'B':>3} | {'P50(ms)':>8} | {'P95(ms)':>8} | {'Avg(ms)':>8} | "
        f"{'Per-Sample':>10} | {'Throughput/s':>12} | {'Speedup':>8} | "
        f"{'Marginal':>8}"
    )
    print("-" * 96)
    b1_avg = results.get(1, {}).get("avg")
    prev_thr = None
    for B in sorted(results):
        r = results[B]
        throughput = B * 1000.0 / r["avg"]
        speedup = f"{b1_avg * B / r['avg']:.2f}x" if b1_avg else "-"
        marginal = "-"
        if prev_thr is not None:
            marginal = f"{(throughput / prev_thr - 1.0) * 100.0:.1f}%"
        prev_thr = throughput
        print(
            f"{B:>3} | {r['p50']:>8.1f} | {r['p95']:>8.1f} | "
            f"{r['avg']:>8.1f} | {r['per_sample']:>10.1f} | "
            f"{throughput:>12.1f} | {speedup:>8} | {marginal:>8}"
        )
    print("=" * 96)


def _print_context_reuse(results: dict[int, dict]) -> None:
    probed = {
        b: r["context_reuse_upper_bound"]
        for b, r in results.items()
        if r.get("context_reuse_upper_bound")
    }
    if not probed:
        return

    print(f"\n{'=' * 96}")
    print("Context Reuse Upper Bound")
    print("-" * 96)
    print(
        f"{'B':>3} | {'Full Avg':>9} | {'Reuse Avg':>9} | "
        f"{'Saved':>8} | {'Speedup':>8} | {'Reuse Thr/s':>11}"
    )
    print("-" * 96)
    for B in sorted(probed):
        full = results[B]["avg"]
        reuse = probed[B]["avg"]
        saved = full - reuse
        speedup = full / reuse if reuse > 0 else float("nan")
        thr = B * 1000.0 / reuse
        print(
            f"{B:>3} | {full:>9.1f} | {reuse:>9.1f} | "
            f"{saved:>8.1f} | {speedup:>7.2f}x | {thr:>11.1f}"
        )
    print("=" * 96)


def _print_profile(results: dict[int, dict]) -> None:
    profiled = {b: r for b, r in results.items() if r.get("profile")}
    if not profiled:
        return
    print(f"\n{'=' * 88}")
    print("Graph Replay Profile")
    print("-" * 88)
    print(
        f"{'B':>3} | {'End2End Avg':>11} | {'SigLIP/PostLN':>13} | "
        f"{'Enc+AE':>10} | {'Enc+AE Share':>12} | {'Residual':>10}"
    )
    print("-" * 88)
    for B in sorted(profiled):
        r = profiled[B]
        p = r["profile"]
        sig = p.get("siglip_postln_graph") or p.get(
            "siglip_graph_incl_patch_postln")
        enc = p.get("enc_ae_graph")
        sig_ms = sig["avg"] if sig else float("nan")
        enc_ms = enc["avg"] if enc else float("nan")
        known = sum(v for v in (sig_ms, enc_ms) if not np.isnan(v))
        residual = r["avg"] - known
        share = enc_ms / r["avg"] * 100.0 if not np.isnan(enc_ms) else float("nan")
        print(
            f"{B:>3} | {r['avg']:>11.1f} | {sig_ms:>13.1f} | "
            f"{enc_ms:>10.1f} | {share:>11.1f}% | {residual:>10.1f}"
        )
    print("=" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Pi0.5 Thor FP8/FP16 batch inference from B=1..8. "
            "This is not an FP4 batch benchmark."
        ))
    parser.add_argument("--checkpoint", default=CKPT)
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default="1-8",
                        help=("Comma list or range. Use 1-8 for scheduler "
                              "curves; sparse lists such as 1,2,4,8 are only "
                              "for diagnostics."))
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--profile", action="store_true",
                        help="Also time SigLIP/PostLN and Enc+AE graph replays.")
    parser.add_argument("--profile-iters", type=int, default=30)
    parser.add_argument("--prompt", default="pick up the red block and place it in the tray")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional deterministic noise seed for every call.")
    parser.add_argument("--same-inputs", action="store_true",
                        help="Use the same observation for every slot.")
    parser.add_argument("--same-views", action="store_true",
                        help="Use identical image and wrist_image within an observation.")
    parser.add_argument("--no-fp8", action="store_true",
                        help="Run the FP16 baseline path instead of FP8.")
    parser.add_argument("--reuse-frontend", action="store_true",
                        help="Load weights once and recapture graphs as B changes.")
    parser.add_argument("--cuda-profiler-range", action="store_true",
                        help="Wrap measured replay region in cudaProfilerStart/Stop.")
    parser.add_argument("--context-reuse-upper-bound", action="store_true",
                        help=("After full E2E timing, replay only Enc+AE from "
                              "the resident visual-language context. This is "
                              "an idealized cache/reuse upper-bound probe, not "
                              "a production correctness mode."))
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return
    if not os.path.isdir(args.checkpoint):
        print(f"SKIP: checkpoint not found: {args.checkpoint}")
        return

    from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor

    batch_sizes = args.batch_sizes
    print("=" * 96)
    print(
        "FlashRT Batch Inference Benchmark "
        f"(warmup={args.warmup}, iters={args.iters}, profile={args.profile})"
    )
    print(f"checkpoint={args.checkpoint}")
    print(f"batch_sizes={batch_sizes} fp8={not args.no_fp8} reuse_frontend={args.reuse_frontend}")
    print("=" * 96)

    results: dict[int, dict] = {}
    frontend = None

    for B in batch_sizes:
        if frontend is None or not args.reuse_frontend:
            frontend = Pi05TorchFrontendThor(
                args.checkpoint,
                num_views=2,
                use_fp8=not args.no_fp8,
            )
            frontend.set_prompt(args.prompt)

        if B == 1:
            frontend.set_batched_mode(enable=False)
        else:
            frontend.set_batched_mode(enable=True, batch_size=B)

        batch_data = _make_batch(
            args.prompt,
            B,
            same_inputs=args.same_inputs,
            same_views=args.same_views,
        )
        infer_fn = _make_infer_fn(frontend, B, batch_data, args.seed)

        print(f"\nB={B}: warmup...")
        for _ in range(args.warmup):
            infer_fn()
        torch.cuda.synchronize()

        cudart = _cuda_profiler_api() if args.cuda_profiler_range else None
        if cudart is not None:
            torch.cuda.synchronize()
            cudart.cudaProfilerStart()
        try:
            print(f"B={B}: benchmark...")
            times = _time_call(infer_fn, args.iters)
            summary = _summarize(times)
            summary["per_sample"] = summary["avg"] / B
            summary["throughput_per_s"] = B * 1000.0 / summary["avg"]

            if args.profile:
                print(f"B={B}: graph profile...")
                summary["profile"] = _profile_frontend(
                    frontend, B, args.profile_iters)

            if args.context_reuse_upper_bound:
                print(f"B={B}: context reuse upper-bound...")
                try:
                    reuse_fn = _make_context_reuse_upper_bound_fn(
                        frontend, B, args.seed, batch_data)
                except RuntimeError as exc:
                    print(f"B={B}: context reuse skipped: {exc}")
                    summary["context_reuse_upper_bound_skipped"] = str(exc)
                else:
                    for _ in range(args.warmup):
                        reuse_fn()
                    torch.cuda.synchronize()
                    reuse_times = _time_call(reuse_fn, args.iters)
                    reuse_summary = _summarize(reuse_times)
                    reuse_summary["per_sample"] = reuse_summary["avg"] / B
                    reuse_summary["throughput_per_s"] = (
                        B * 1000.0 / reuse_summary["avg"])
                    summary["context_reuse_upper_bound"] = reuse_summary
        finally:
            if cudart is not None:
                torch.cuda.synchronize()
                cudart.cudaProfilerStop()

        results[B] = summary

        if not args.reuse_frontend:
            del frontend
            frontend = None
            torch.cuda.empty_cache()

    _print_summary(results)
    _print_context_reuse(results)
    _print_profile(results)

    if args.json_out is not None:
        payload = {
            "checkpoint": args.checkpoint,
            "batch_sizes": batch_sizes,
            "warmup": args.warmup,
            "iters": args.iters,
            "profile": args.profile,
            "profile_iters": args.profile_iters,
            "fp8": not args.no_fp8,
            "reuse_frontend": args.reuse_frontend,
            "cuda_profiler_range": args.cuda_profiler_range,
            "same_inputs": args.same_inputs,
            "same_views": args.same_views,
            "context_reuse_upper_bound": args.context_reuse_upper_bound,
            "results": results,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
