"""Periodic research loop for Pi0.5 Thor batch inference optimization.

The loop is intentionally hypothesis-driven.  Each cycle records:

1. End-to-end batch latency and graph-stage breakdown.
2. GEGLU -> Down materialization cost and virtual-mainloop break-even model.
3. A low-level GEGLU producer exclusion probe on the first cycle.

It writes small JSON artifacts only.  It does not run profilers and does not
touch robot or camera hardware.

The context-reuse upper bound is intentionally optional. It assumes reusable
resident VLM context and is useful as a negative-control upper-bound probe, but
it is not a realistic core optimization for multi-robot visual inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path(os.environ.get(
    "FLASHRT_RESEARCH_OUTPUT_DIR",
    str(ROOT / "docs" / "experiments"),
))


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run(cmd: list[str], env: dict[str, str], cwd: Path) -> dict:
    t0 = time.perf_counter()
    print("\n" + "=" * 96, flush=True)
    print("RUN:", " ".join(cmd), flush=True)
    print("=" * 96, flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    elapsed = time.perf_counter() - t0
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "elapsed_s": elapsed,
    }


def _write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a periodic Pi0.5 Thor batch inference research loop.")
    parser.add_argument("--duration-min", type=float, default=120.0)
    parser.add_argument("--batch-sizes", default="1-8")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--profile-iters", type=int, default=8)
    parser.add_argument("--kernel-warmup", type=int, default=8)
    parser.add_argument("--kernel-iters", type=int, default=20)
    parser.add_argument("--include-context-reuse-upper-bound",
                        action="store_true",
                        help=("Also run the idealized context-reuse probe. "
                              "Disabled by default because real robot images "
                              "are not identical across requests."))
    parser.add_argument("--same-inputs", action="store_true",
                        help=("Use identical synthetic observations for all "
                              "batch elements. Default uses distinct synthetic "
                              "observations per request."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", default=os.environ.get(
        "PI05_LIBERO_PYTORCH_CHECKPOINT",
        "/mnt/home/zengzixuan/workspace/checkpoints/pi05_libero_pytorch"))
    parser.add_argument("--cuda-visible-devices", default=os.environ.get(
        "CUDA_VISIBLE_DEVICES", "0"))
    args = parser.parse_args()

    started = time.time()
    deadline = started + args.duration_min * 60.0
    run_id = _ts()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / f"pi05_research_loop_index_{run_id}.jsonl"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("FLASHRT_THOR_BATCH_ENCODER_DOWN_TACTIC", "auto")

    cycle = 0
    while True:
        now = time.time()
        if cycle > 0 and now >= deadline:
            break
        cycle += 1
        cycle_ts = _ts()
        cycle_rows: list[dict] = []

        e2e_json = out_dir / (
            f"pi05_loop_{run_id}_cycle{cycle:03d}_"
            f"e2e_profile_{cycle_ts}.json")
        e2e_cmd = [
            sys.executable, "benchmarks/bench_b1_b8.py",
            "--checkpoint", args.checkpoint,
            "--batch-sizes", args.batch_sizes,
            "--warmup", str(args.warmup),
            "--iters", str(args.iters),
            "--profile",
            "--profile-iters", str(args.profile_iters),
            "--reuse-frontend",
            "--json-out", str(e2e_json),
        ]
        if args.same_inputs:
            e2e_cmd.insert(-2, "--same-inputs")
        if args.include_context_reuse_upper_bound:
            e2e_cmd.insert(-2, "--context-reuse-upper-bound")
        cycle_rows.append({
            "stage": ("e2e_context_reuse_upper_bound"
                      if args.include_context_reuse_upper_bound
                      else "e2e_profile"),
            "json": str(e2e_json),
            **_run(e2e_cmd, env, ROOT),
        })

        boundary_json = out_dir / (
            f"pi05_loop_{run_id}_cycle{cycle:03d}_"
            f"boundary_model_{cycle_ts}.json")
        boundary_cmd = [
            sys.executable, "benchmarks/bench_pi05_batch_research_kernels.py",
            "--batch-sizes", args.batch_sizes,
            "--warmup", str(args.kernel_warmup),
            "--iters", str(args.kernel_iters),
            "--skip-tactics",
            "--include-virtual-mainloop-model",
            "--encoder-down-tactic", "auto",
            "--json-out", str(boundary_json),
        ]
        cycle_rows.append({
            "stage": "geglu_down_boundary_model",
            "json": str(boundary_json),
            **_run(boundary_cmd, env, ROOT),
        })

        if cycle == 1:
            lut_json = out_dir / (
                f"pi05_loop_{run_id}_cycle{cycle:03d}_"
                f"geglu_lut_exclusion_{cycle_ts}.json")
            lut_cmd = [
                sys.executable, "benchmarks/bench_pi05_batch_research_kernels.py",
                "--batch-sizes", args.batch_sizes,
                "--warmup", str(args.kernel_warmup),
                "--iters", str(args.kernel_iters),
                "--skip-tactics",
                "--skip-chains",
                "--include-geglu-lut",
                "--json-out", str(lut_json),
            ]
            cycle_rows.append({
                "stage": "geglu_lut_exclusion",
                "json": str(lut_json),
                **_run(lut_cmd, env, ROOT),
            })

        row = {
            "run_id": run_id,
            "cycle": cycle,
            "cycle_ts": cycle_ts,
            "elapsed_total_s": time.time() - started,
            "duration_min": args.duration_min,
            "batch_sizes": args.batch_sizes,
            "rows": cycle_rows,
        }
        _write_jsonl(index_path, row)
        print(f"\nCycle {cycle} recorded in {index_path}", flush=True)

    summary = {
        "run_id": run_id,
        "completed_at": _ts(),
        "cycles": cycle,
        "elapsed_total_s": time.time() - started,
        "index": str(index_path),
    }
    summary_path = out_dir / f"pi05_research_loop_summary_{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResearch loop complete: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
