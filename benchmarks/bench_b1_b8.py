"""Batch inference benchmark for Pi0.5 Thor B=1..8.

Default run:
    CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_b1_b8.py

Faster diagnostic run:
    CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_b1_b8.py \
        --batch-sizes 1,2,4,8 --warmup 10 --iters 30 --profile
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

CKPT = os.environ.get(
    "PI05_LIBERO_PYTORCH_CHECKPOINT",
    "",
)


def _command_text(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value.splitlines()[0].strip() if value else None

def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if value else None


def _git_revision(root: Path) -> str | None:
    return _command_text(["git", "-C", str(root), "rev-parse", "HEAD"])


def _collect_environment(*, exclusive_gpu: bool) -> dict[str, object]:
    flashrt_root = Path(__file__).resolve().parents[1]
    lingshu_root = Path(__file__).resolve().parents[3]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver_version": _command_text([
            "nvidia-smi", "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ]),
        "gpu_uuid": _command_text([
            "nvidia-smi", "--query-gpu=uuid",
            "--format=csv,noheader",
        ]),
        "mig_mode": _command_text([
            "nvidia-smi", "--query-gpu=mig.mode.current",
            "--format=csv,noheader",
        ]),
        "flashrt_revision": _git_revision(flashrt_root),
        "lingshu_revision": _git_revision(lingshu_root),
        "flashrt_status_short": _command_output(
            ["git", "-C", str(flashrt_root), "status", "--short"]),
        "lingshu_status_short": _command_output(
            ["git", "-C", str(lingshu_root), "status", "--short"]),
        "flashrt_dirty_diff_stat": _command_output(
            ["git", "-C", str(flashrt_root), "diff", "--stat"]),
        "lingshu_dirty_diff_stat": _command_output(
            ["git", "-C", str(lingshu_root), "diff", "--stat"]),
        "power_draw_w": _command_text([
            "nvidia-smi", "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
        ]),
        "sm_clock_mhz": _command_text([
            "nvidia-smi", "--query-gpu=clocks.sm",
            "--format=csv,noheader,nounits",
        ]),
        "exclusive_gpu": exclusive_gpu,
    }


def _make_obs(seed: int = 42, *, same_views: bool = False) -> dict:
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    wrist = img.copy() if same_views else rng.randint(
        0, 256, (224, 224, 3), dtype=np.uint8)
    return {"image": img, "wrist_image": wrist}


def _parse_num_steps(value: str) -> int:
    steps = int(value)
    if not 4 <= steps <= 10:
        raise argparse.ArgumentTypeError("num steps must be in [4, 10]")
    return steps


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


def _time_call_with_device(
    fn: Callable[[], object], iters: int
) -> tuple[list[float], list[float]]:
    """Measure host wall and CUDA-event time for steady-state calls."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    host_times: list[float] = []
    device_times: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start_event.record()
        fn()
        end_event.record()
        end_event.synchronize()
        host_times.append((time.perf_counter() - t0) * 1000.0)
        device_times.append(float(start_event.elapsed_time(end_event)))
    return host_times, device_times


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
                                       seed: int | None):
    """Replay only the action-generation graph from a resident VLM context.

    This is an experimental upper-bound probe for memory-for-compute research.
    It assumes the current visual-language context in ``_enc_x`` or
    ``_enc_x_b2`` is reusable from a cache and keeps the normal noise,
    Enc+AE graph replay, D2H transfer, and action unnormalization path.
    """
    from flash_rt.core.utils.actions import LIBERO_ACTION_DIM, unnormalize_actions

    if batch_size == 1:
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

    if getattr(frontend, "_enc_ae_graph_b2", None) is None:
        raise RuntimeError("batched Enc+AE graph is not captured yet")

    def run_batch():
        if seed is None:
            frontend._g_noise_b2.normal_()
        else:
            noise_np = np.random.RandomState(seed).randn(
                batch_size, frontend.Sa, 32).astype(np.float16)
            noise = torch.from_numpy(noise_np).to("cuda", non_blocking=True)
            for b in range(batch_size):
                frontend._g_noise_b2[
                    b * frontend.Sa:(b + 1) * frontend.Sa].copy_(noise[b])
        frontend._enc_ae_graph_b2.replay()
        raw_all = frontend._g_noise_b2[
            :batch_size * frontend.Sa].float().cpu().numpy()
        results = []
        for b in range(batch_size):
            raw = raw_all[b * frontend.Sa:(b + 1) * frontend.Sa]
            unnorm = unnormalize_actions(raw, frontend.norm_stats)
            results.append({"actions": unnorm[:, :LIBERO_ACTION_DIM]})
        return results

    return run_batch


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
        description="Benchmark Pi0.5 Thor batch inference from B=1..8.")
    parser.add_argument("--checkpoint", default=CKPT)
    parser.add_argument(
        "--num-steps",
        type=_parse_num_steps,
        default=None,
        help="Action-expert NFE in [4, 10]; overrides FLASHRT_PI05_NUM_STEPS (default: env value or 10).",
    )
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default="1-8",
                        help="Comma list or range, e.g. 1,2,4,8 or 1-8.")
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
    parser.add_argument("--use-fa4", action="store_true",
                        help="Require the Thor FA4 attention backend.")
    parser.add_argument("--reuse-frontend", action="store_true",
                        help="Load weights once and recapture graphs as B changes.")
    parser.add_argument("--cuda-profiler-range", action="store_true",
                        help="Wrap measured replay region in cudaProfilerStart/Stop.")
    parser.add_argument("--context-reuse-upper-bound", action="store_true",
                        help=("After full E2E timing, replay only Enc+AE from "
                              "the resident visual-language context. This is "
                              "an idealized cache/reuse upper-bound probe, not "
                              "a production correctness mode."))
    parser.add_argument(
        "--exclusive-gpu",
        action="store_true",
        help="Record that the benchmark was run with an explicitly reserved GPU.",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    requested_num_steps = args.num_steps
    if requested_num_steps is None:
        raw_steps = os.environ.get("FLASHRT_PI05_NUM_STEPS", "10")
        try:
            requested_num_steps = int(raw_steps)
        except ValueError:
            parser.error(
                f"FLASHRT_PI05_NUM_STEPS must be an integer in [4, 10], got {raw_steps!r}"
            )
        if not 4 <= requested_num_steps <= 10:
            parser.error(
                f"FLASHRT_PI05_NUM_STEPS must be in [4, 10], got {requested_num_steps}"
            )

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
    print(
        f"batch_sizes={batch_sizes} num_steps={requested_num_steps} "
        f"fp8={not args.no_fp8} use_fa4={args.use_fa4} "
        f"reuse_frontend={args.reuse_frontend}"
    )
    print("=" * 96)

    results: dict[int, dict] = {}
    frontend = None
    setup_by_batch: dict[int, float] = {}
    graph_capture_by_batch: dict[int, float] = {}
    cache_miss_by_batch: dict[int, bool] = {}
    observed_num_steps: int | None = None

    for B in batch_sizes:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        setup_start = time.perf_counter()
        cache_miss = False
        graph_capture_start = None
        if frontend is None or not args.reuse_frontend:
            cache_key = (
                str((Path(args.checkpoint) / "model.safetensors").resolve()),
                bool(not args.no_fp8),
            )
            cache_miss = cache_key not in getattr(
                Pi05TorchFrontendThor, "_WEIGHT_CACHE", {}
            )
            frontend = Pi05TorchFrontendThor(
                args.checkpoint,
                num_views=2,
                use_fp8=not args.no_fp8,
                use_fa4=args.use_fa4,
                num_steps=requested_num_steps,
            )
            graph_capture_start = time.perf_counter()
            frontend.set_prompt(args.prompt)

        if graph_capture_start is None:
            graph_capture_start = time.perf_counter()
        if B == 1:
            frontend.set_batched_mode(enable=False)
        else:
            frontend.set_batched_mode(enable=True, batch_size=B)

        torch.cuda.synchronize()
        setup_ms = (time.perf_counter() - setup_start) * 1000.0
        graph_capture_ms = (time.perf_counter() - graph_capture_start) * 1000.0
        setup_peak_allocated = int(torch.cuda.max_memory_allocated())
        setup_peak_reserved = int(torch.cuda.max_memory_reserved())
        current_steps = int(frontend.steps)
        if observed_num_steps is None:
            observed_num_steps = current_steps
        elif observed_num_steps != current_steps:
            raise RuntimeError(
                f"observed NFE changed within benchmark: "
                f"{observed_num_steps} -> {current_steps}")
        setup_by_batch[B] = setup_ms
        graph_capture_by_batch[B] = graph_capture_ms
        cache_miss_by_batch[B] = cache_miss

        batch_data = _make_batch(
            args.prompt,
            B,
            same_inputs=args.same_inputs,
            same_views=args.same_views,
        )
        infer_fn = _make_infer_fn(frontend, B, batch_data, args.seed)

        print(f"\nB={B}: warmup...")
        torch.cuda.reset_peak_memory_stats()
        first_use_ms = 0.0
        first_use_device_ms = 0.0
        if args.warmup > 0:
            first_host, first_device = _time_call_with_device(infer_fn, 1)
            first_use_ms = first_host[0]
            first_use_device_ms = first_device[0]
            for _ in range(max(0, args.warmup - 1)):
                infer_fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        cudart = _cuda_profiler_api() if args.cuda_profiler_range else None
        if cudart is not None:
            torch.cuda.synchronize()
            cudart.cudaProfilerStart()
        try:
            print(f"B={B}: benchmark...")
            host_times, device_times = _time_call_with_device(
                infer_fn, args.iters)
            summary = _summarize(host_times)
            device_summary = _summarize(device_times)
            summary["per_sample"] = summary["avg"] / B
            summary["throughput_per_s"] = B * 1000.0 / summary["avg"]
            summary["avg_ms"] = summary["avg"]
            summary["p50_ms"] = summary["p50"]
            summary["p95_ms"] = summary["p95"]
            summary["per_sample_ms"] = summary["per_sample"]
            summary["throughput_samples_s"] = summary["throughput_per_s"]
            summary["host_forward_avg_ms"] = summary["avg"]
            summary["device_avg_ms"] = device_summary["avg"]
            summary["device_p50_ms"] = device_summary["p50"]
            summary["device_p95_ms"] = device_summary["p95"]
            summary["setup_ms"] = setup_ms
            summary["first_warmup_ms"] = first_use_ms
            summary["first_warmup_device_ms"] = first_use_device_ms
            summary["setup_peak_memory_allocated_bytes"] = setup_peak_allocated
            summary["setup_peak_memory_reserved_bytes"] = setup_peak_reserved
            summary["peak_memory_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated())
            summary["peak_memory_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved())
            summary["observed_num_steps"] = current_steps
            summary["measured_iters"] = args.iters
            summary["warmup"] = args.warmup
            summary["cache_miss"] = cache_miss_by_batch[B]

            if args.profile:
                print(f"B={B}: graph profile...")
                summary["profile"] = _profile_frontend(
                    frontend, B, args.profile_iters)

            if args.context_reuse_upper_bound:
                print(f"B={B}: context reuse upper-bound...")
                reuse_fn = _make_context_reuse_upper_bound_fn(
                    frontend, B, args.seed)
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
            "schema": "lingshu_g1_cost_v1",
            "config": {
                "requested_num_steps": requested_num_steps,
                "observed_num_steps": observed_num_steps,
                "batch_sizes": batch_sizes,
                "checkpoint": args.checkpoint,
                "prompt": args.prompt,
                "diffusion_seed": args.seed,
                "precision": "fp16+fp8" if not args.no_fp8 else "fp16",
                "use_fa4": args.use_fa4,
                "attention_backend": "fa4" if args.use_fa4 else "default",
                "rtc_mode": "off",
                "input_mode": {
                    "same_inputs": args.same_inputs,
                    "same_views": args.same_views,
                },
            },
            "setup": {
                "setup_ms": setup_by_batch,
                "graph_capture_ms": graph_capture_by_batch,
                "cache_miss": cache_miss_by_batch,
                "graph_capture_scope": (
                    "set_prompt and set_batched_mode wall time; includes "
                    "buffer allocation, graph capture, and autotuning"
                ),
                "cache_miss_semantics": "Pi05 model-weight cache miss on frontend construction",
                "first_use_excluded_from_steady_state": args.warmup > 0,
            },
            "checkpoint": args.checkpoint,
            "num_steps": requested_num_steps,
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
            "exclusive_gpu": args.exclusive_gpu,
            "environment": _collect_environment(
                exclusive_gpu=args.exclusive_gpu),
            "results": results,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
