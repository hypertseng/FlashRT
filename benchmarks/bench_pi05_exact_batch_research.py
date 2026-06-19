"""Exact single-GPU batch inference research probes for Pi0.5 Thor.

This benchmark focuses on computation-side optimizations that do not reuse
visual features and do not change the model math:

* resident per-(B, Se) CUDA graph bank for mixed batch sizes;
* exact prompt embedding cache hit/miss cost;
* device-resident prompt bank for repeated language paths;
* device-resident seeded-noise bank for deterministic diffusion starts;
* fused multi-view uint8 upload/normalize-to-patches path.

It uses synthetic observations only and never touches robot I/O.
"""

from __future__ import annotations

import argparse
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


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError("empty integer list")
    if any(v < 1 for v in out):
        raise argparse.ArgumentTypeError("all values must be >= 1")
    return out


def _make_obs(seed: int, *, num_views: int = 2) -> dict:
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    obs = {"image": img}
    if num_views >= 2:
        obs["wrist_image"] = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    if num_views >= 3:
        obs["wrist_image_right"] = rng.randint(
            0, 256, (224, 224, 3), dtype=np.uint8)
    return obs


def _make_batch(prompt: str, B: int, *, num_views: int) -> list[dict]:
    return [
        {"observation": _make_obs(i, num_views=num_views), "prompt": prompt}
        for i in range(B)
    ]


def _summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "avg": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _time_call(fn: Callable[[], object], iters: int = 1) -> list[float]:
    times: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _run_infer(frontend, B: int, batch_data: list[dict], seed: int | None):
    if B == 1:
        frontend.set_batched_mode(enable=False)
        return frontend.infer(batch_data[0]["observation"], seed=seed)
    frontend.set_batched_mode(enable=True, batch_size=B)
    return frontend.infer_multi_prompt_batch(batch_data, seed=seed)


def _mixed_sequence_probe(frontend, args) -> dict:
    frontend.enable_batch_graph_bank(args.graph_bank, clear=True)
    frontend.enable_prompt_bank(args.prompt_bank, clear=True)
    frontend.enable_noise_bank(args.noise_bank, clear=True)
    frontend.set_prompt(args.prompt)

    batches = {
        B: _make_batch(args.prompt, B, num_views=args.num_views)
        for B in sorted(set(args.sequence))
    }

    passes = []
    for pass_idx in range(args.passes):
        call_rows = []
        for step, B in enumerate(args.sequence):
            batch_data = batches[B]
            times = _time_call(
                lambda B=B, batch_data=batch_data: _run_infer(
                    frontend, B, batch_data, args.seed),
                iters=1,
            )
            call_rows.append({
                "step": step,
                "B": B,
                "latency_ms": times[0],
                "bank_entries": len(frontend.batch_graph_bank_keys()),
            })
        passes.append({
            "pass": pass_idx,
            "calls": call_rows,
            "summary": _summarize([r["latency_ms"] for r in call_rows]),
        })
    return {
        "enabled": bool(args.graph_bank),
        "prompt_bank_enabled": bool(args.prompt_bank),
        "sequence": args.sequence,
        "passes": passes,
        "resident_keys": [list(k) for k in frontend.batch_graph_bank_keys()],
        "prompt_bank_stats": frontend.prompt_bank_stats(),
        "noise_bank_stats": frontend.noise_bank_stats(),
    }


def _prompt_cache_probe(frontend, args) -> dict:
    from flash_rt.core.thor_frontend_utils import (
        _prompt_embed_cache,
        embed_prompt_torch,
    )

    cache_key_count_before = len(_prompt_embed_cache)
    _prompt_embed_cache.clear()

    def embed_once():
        embed_prompt_torch(args.prompt, frontend.embedding_weight, max_len=48)

    miss = _time_call(embed_once, 1)[0]
    hit_times = _time_call(embed_once, args.prompt_iters)
    return {
        "cache_entries_before_clear": cache_key_count_before,
        "miss_ms": miss,
        "hit": _summarize(hit_times),
        "cache_entries_after": len(_prompt_embed_cache),
    }


def _prompt_bank_probe(frontend, args) -> dict | None:
    B = max([b for b in args.sequence if b >= 2], default=0)
    if B < 2:
        return None

    from flash_rt.core.thor_frontend_utils import _prompt_embed_cache

    prompts = [args.prompt for _ in range(B)]

    def fetch_batch_prompts():
        for prompt in prompts:
            frontend._get_prompt_bank_entry(prompt, max_len=48)

    _prompt_embed_cache.clear()
    frontend.enable_prompt_bank(False, clear=True)
    fetch_batch_prompts()
    no_bank = _time_call(fetch_batch_prompts, args.prompt_iters)

    _prompt_embed_cache.clear()
    frontend.enable_prompt_bank(True, clear=True)
    miss_then_intrabatch_hits = _time_call(fetch_batch_prompts, 1)[0]
    hit = _time_call(fetch_batch_prompts, args.prompt_iters)
    stats = frontend.prompt_bank_stats()
    frontend.enable_prompt_bank(args.prompt_bank, clear=False)
    return {
        "B": B,
        "no_bank_embed_cache_hit": _summarize(no_bank),
        "miss_then_intrabatch_hits_ms": miss_then_intrabatch_hits,
        "bank_hit": _summarize(hit),
        "stats": stats,
    }


def _noise_bank_probe(frontend, args) -> dict | None:
    if args.seed is None:
        return None
    B = max([b for b in args.sequence if b >= 2], default=0)
    if B < 2:
        return None

    batch = _make_batch(args.prompt, B, num_views=args.num_views)
    samples = []
    for item in batch:
        samples.append({"prompt": item["prompt"]})

    if getattr(frontend, "_g_noise_b2", None) is None or frontend.B != B:
        frontend.set_batched_mode(enable=True, batch_size=B)

    def fill_seed_noise():
        frontend._fill_seed_noise_batched(samples, args.seed)

    frontend.enable_noise_bank(False, clear=True)
    no_bank = _time_call(fill_seed_noise, args.prompt_iters)

    frontend.enable_noise_bank(True, clear=True)
    miss = _time_call(fill_seed_noise, 1)[0]
    hit = _time_call(fill_seed_noise, args.prompt_iters)
    stats = frontend.noise_bank_stats()
    frontend.enable_noise_bank(args.noise_bank, clear=False)
    return {
        "B": B,
        "seed": args.seed,
        "no_bank": _summarize(no_bank),
        "miss_ms": miss,
        "hit": _summarize(hit),
        "stats": stats,
    }


def _vision_preprocess_probe(frontend, args) -> dict | None:
    B = max([b for b in args.sequence if b >= 2], default=0)
    if B < 2:
        return None

    frontend.enable_batch_graph_bank(args.graph_bank, clear=False)
    frontend.enable_prompt_bank(args.prompt_bank, clear=False)
    frontend.enable_noise_bank(args.noise_bank, clear=False)
    frontend.set_batched_mode(enable=True, batch_size=B)
    batch = _make_batch(args.prompt, B, num_views=args.num_views)
    frontend.infer_multi_prompt_batch(batch, seed=args.seed)

    samples = []
    for item in batch:
        obs = item["observation"]
        images = [obs["image"]]
        if args.num_views >= 2:
            images.append(obs["wrist_image"])
        if args.num_views >= 3:
            images.append(obs["wrist_image_right"])
        samples.append({"images": images})

    samples_fp16 = []
    for s in samples:
        images_fp16 = [
            (im.astype(np.float32) / 127.5 - 1.0).astype(np.float16)
            for im in s["images"]
        ]
        samples_fp16.append({"images": images_fp16})

    def fused_uint8_to_patches():
        patches_ready = frontend._upload_images_gpu_batched(
            samples, frontend._img_buf_b2_all, frontend._patches_buf_b2)
        frontend._patch_embed_ops_batched(B, 0, patches_ready=patches_ready)

    def fp16_image_then_patches():
        patches_ready = frontend._upload_images_gpu_batched(
            samples_fp16, frontend._img_buf_b2_all, None)
        frontend._patch_embed_ops_batched(B, 0, patches_ready=patches_ready)

    fused = _time_call(fused_uint8_to_patches, args.vision_iters)
    fp16_path = _time_call(fp16_image_then_patches, args.vision_iters)
    return {
        "B": B,
        "num_views": args.num_views,
        "uint8_fused_to_patches": _summarize(fused),
        "fp16_image_then_patches": _summarize(fp16_path),
    }


def _correctness_probe(args) -> dict | None:
    if not args.verify:
        return None
    from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor

    B = max([b for b in args.sequence if b >= 2], default=0)
    if B < 2:
        return None

    batch = _make_batch(args.prompt, B, num_views=args.num_views)
    frontend = Pi05TorchFrontendThor(
        args.checkpoint,
        num_views=args.num_views,
        use_fp8=not args.no_fp8,
        autotune=args.autotune,
    )
    frontend.set_prompt(args.prompt)
    frontend.enable_batch_graph_bank(False, clear=True)
    frontend.enable_prompt_bank(False, clear=True)
    frontend.enable_noise_bank(False, clear=True)
    ref = _run_infer(frontend, B, batch, args.seed)
    frontend.enable_batch_graph_bank(args.graph_bank, clear=True)
    frontend.enable_prompt_bank(args.prompt_bank, clear=True)
    frontend.enable_noise_bank(args.noise_bank, clear=True)
    got = _run_infer(frontend, B, batch, args.seed)

    max_abs = 0.0
    cos = []
    for r, g in zip(ref, got):
        a = np.asarray(r["actions"], dtype=np.float64).reshape(-1)
        b = np.asarray(g["actions"], dtype=np.float64).reshape(-1)
        max_abs = max(max_abs, float(np.max(np.abs(a - b))))
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cos.append(float(np.dot(a, b) / denom) if denom else 1.0)
    return {
        "B": B,
        "graph_bank": bool(args.graph_bank),
        "prompt_bank": bool(args.prompt_bank),
        "noise_bank": bool(args.noise_bank),
        "max_abs": max_abs,
        "min_cosine": float(np.min(cos)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pi0.5 exact single-machine batch optimization probes.")
    parser.add_argument("--checkpoint", default=CKPT)
    parser.add_argument("--prompt",
                        default="pick up the red block and place it in the tray")
    parser.add_argument("--sequence", type=_parse_int_list,
                        default=_parse_int_list("2,3,4,2,1,3,4,2"))
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--autotune", type=int, default=0)
    parser.add_argument("--no-fp8", action="store_true")
    parser.add_argument("--graph-bank", action="store_true",
                        help="Enable resident per-(B, Se) graph bank.")
    parser.add_argument("--prompt-bank", action="store_true",
                        help="Enable resident exact prompt embedding bank.")
    parser.add_argument("--noise-bank", action="store_true",
                        help="Enable resident exact seeded-noise bank.")
    parser.add_argument("--prompt-iters", type=int, default=50)
    parser.add_argument("--vision-iters", type=int, default=50)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return
    if not os.path.isdir(args.checkpoint):
        print(f"SKIP: checkpoint not found: {args.checkpoint}")
        return

    from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor

    frontend = Pi05TorchFrontendThor(
        args.checkpoint,
        num_views=args.num_views,
        use_fp8=not args.no_fp8,
        autotune=args.autotune,
    )
    result = {
        "checkpoint": args.checkpoint,
        "prompt": args.prompt,
        "fp8": not args.no_fp8,
        "graph_bank": args.graph_bank,
        "prompt_bank": args.prompt_bank,
        "noise_bank": args.noise_bank,
        "mixed_sequence": _mixed_sequence_probe(frontend, args),
        "prompt_cache": _prompt_cache_probe(frontend, args),
        "prompt_bank_probe": _prompt_bank_probe(frontend, args),
        "noise_bank_probe": _noise_bank_probe(frontend, args),
        "vision_preprocess": _vision_preprocess_probe(frontend, args),
        "correctness": _correctness_probe(args),
    }

    print(json.dumps(result, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
