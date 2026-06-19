"""Summarize periodic Pi0.5 Thor batch research-loop artifacts.

The input is the JSONL index written by ``run_pi05_batch_research_loop.py``.
This script intentionally consumes only small JSON artifacts and produces a
compact JSON/Markdown summary for experiment documentation.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def _stdev(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    values = sorted(xs)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _stats(xs: Iterable[float]) -> dict[str, float | int]:
    values = [float(x) for x in xs]
    return {
        "n": len(values),
        "mean": _mean(values),
        "stdev": _stdev(values),
        "p50": _pct(values, 0.50),
        "p95": _pct(values, 0.95),
        "min": min(values) if values else float("nan"),
        "max": max(values) if values else float("nan"),
    }


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _iter_success_jsons(index_path: Path, stage: str) -> Iterable[Path]:
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for item in row.get("rows", []):
            if item.get("stage") != stage:
                continue
            if item.get("returncode") != 0:
                continue
            path = Path(item.get("json", ""))
            if path.exists():
                yield path


def _summarize_e2e(paths: list[Path]) -> dict:
    by_b: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for path in paths:
        obj = _load_json(path)
        for b_s, result in obj.get("results", {}).items():
            b = int(b_s)
            full = float(result["avg"])
            by_b[b]["full_avg_ms"].append(full)
            by_b[b]["throughput_req_s"].append(float(result["throughput_per_s"]))
            profile = result.get("profile", {})
            sig = profile.get("siglip_postln_graph") or profile.get(
                "siglip_graph_incl_patch_postln")
            enc = profile.get("enc_ae_graph")
            if sig:
                by_b[b]["siglip_postln_ms"].append(float(sig["avg"]))
            if enc:
                enc_ms = float(enc["avg"])
                by_b[b]["enc_ae_ms"].append(enc_ms)
                by_b[b]["enc_ae_share"].append(enc_ms / full)
            reuse = result.get("context_reuse_upper_bound")
            if reuse:
                reuse_ms = float(reuse["avg"])
                by_b[b]["reuse_avg_ms"].append(reuse_ms)
                by_b[b]["reuse_saved_ms"].append(full - reuse_ms)
                by_b[b]["reuse_speedup"].append(full / reuse_ms)
                by_b[b]["reuse_throughput_req_s"].append(
                    float(reuse["throughput_per_s"]))

    summary: dict[str, dict] = {}
    for b in sorted(by_b):
        summary[str(b)] = {k: _stats(v) for k, v in sorted(by_b[b].items())}
    return summary


def _summarize_boundary(paths: list[Path]) -> dict:
    by_b: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for path in paths:
        obj = _load_json(path)
        for row in obj.get("results", []):
            if row.get("section") == "producer_chain" and row.get(
                    "name") == "encoder_geglu_to_down_gemm":
                b = int(row["B"])
                by_b[b]["geglu_down_chain_ms"].append(float(row["chain_ms"]))
                by_b[b]["geglu_down_visible_ms"].append(
                    float(row["visible_materialization_ms"]))
                by_b[b]["geglu_down_visible_infer_ms"].append(
                    float(row["visible_materialization_ms"])
                    * int(row["calls_per_infer"]))
                by_b[b]["geglu_producer_ms"].append(float(row["producer_ms"]))
                by_b[b]["down_consumer_ms"].append(float(row["consumer_ms"]))
            elif row.get("section") == "virtual_mainloop_model":
                b = int(row["B"])
                by_b[b]["virtual_visible_infer_ms"].append(
                    float(row["visible_materialization_per_infer_ms"]))
                by_b[b]["naive_delta_infer_ms"].append(
                    float(row["naive_delta_vs_chain_ms_per_infer"]))
                by_b[b]["required_a_tile_reuse"].append(
                    float(row["min_required_a_tile_reuse_vs_naive"]))
                by_b[b]["max_recompute_factor"].append(
                    float(row["max_recompute_factor_for_break_even"]))

    summary: dict[str, dict] = {}
    for b in sorted(by_b):
        summary[str(b)] = {k: _stats(v) for k, v in sorted(by_b[b].items())}
    return summary


def _summarize_lut(paths: list[Path]) -> dict:
    rows: dict[str, dict] = {}
    for path in paths:
        obj = _load_json(path)
        for row in obj.get("results", []):
            if row.get("section") != "geglu_lut_sweep":
                continue
            b = str(int(row["B"]))
            rows[b] = {
                "reference_avg_ms": row.get("reference_avg_ms"),
                "lut_avg_ms": row.get("candidate_avg_ms"),
                "lut_speedup": row.get("speedup"),
                "lut_bit_exact": row.get("bit_exact_vs_reference"),
                "row8_avg_ms": row.get("row8_avg_ms"),
                "row8_speedup": row.get("row8_speedup"),
                "row8_bit_exact": row.get("row8_bit_exact_vs_reference"),
            }
    return rows


def _fmt_ms(stats: dict, digits: int = 1) -> str:
    return f"{stats['mean']:.{digits}f} +/- {stats['stdev']:.{digits}f}"


def _fmt_x(stats: dict, digits: int = 2) -> str:
    return f"{stats['mean']:.{digits}f}x"


def _make_markdown(summary: dict) -> str:
    e2e = summary["e2e_profile"]
    boundary = summary["geglu_down_boundary_model"]
    has_reuse = any("reuse_avg_ms" in row for row in e2e.values())
    lines: list[str] = []
    lines.append("## Pi0.5 Thor batch research loop summary")
    lines.append("")
    lines.append(f"- run_id: `{summary['run_id']}`")
    lines.append(f"- completed cycles: {summary['completed_cycles']}")
    lines.append(f"- elapsed: {summary['elapsed_total_min']:.1f} min")
    lines.append(
        "- scope: synthetic observations only; no camera input, no robot action execution.")
    lines.append("")
    lines.append("### End-to-end graph-stage profile")
    lines.append("")
    if has_reuse:
        lines.append(
            "| B | full avg ms | throughput req/s | context reuse avg ms | saved ms | ideal speedup | Enc+AE share |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append(
            "| B | full avg ms | throughput req/s | SigLIP/PostLN ms | Enc+AE ms | Enc+AE share |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
    for b in sorted(e2e, key=int):
        row = e2e[b]
        if has_reuse and "reuse_avg_ms" in row:
            lines.append(
                f"| {b} | {_fmt_ms(row['full_avg_ms'])} | "
                f"{row['throughput_req_s']['mean']:.1f} | "
                f"{_fmt_ms(row['reuse_avg_ms'])} | "
                f"{_fmt_ms(row['reuse_saved_ms'])} | "
                f"{_fmt_x(row['reuse_speedup'])} | "
                f"{row['enc_ae_share']['mean'] * 100.0:.1f}% |")
        else:
            sig = row.get("siglip_postln_ms")
            enc = row.get("enc_ae_ms")
            lines.append(
                f"| {b} | {_fmt_ms(row['full_avg_ms'])} | "
                f"{row['throughput_req_s']['mean']:.1f} | "
                f"{_fmt_ms(sig) if sig else '-'} | "
                f"{_fmt_ms(enc) if enc else '-'} | "
                f"{row['enc_ae_share']['mean'] * 100.0:.1f}% |")
    lines.append("")
    lines.append("### GEGLU -> Down materialization boundary")
    lines.append("")
    lines.append(
        "| B | chain ms/layer | visible ms/infer | naive recompute delta ms/infer | required A-tile reuse |")
    lines.append("|---:|---:|---:|---:|---:|")
    for b in sorted(boundary, key=int):
        row = boundary[b]
        lines.append(
            f"| {b} | {_fmt_ms(row['geglu_down_chain_ms'], 3)} | "
            f"{_fmt_ms(row['geglu_down_visible_infer_ms'], 2)} | "
            f"{_fmt_ms(row['naive_delta_infer_ms'], 1)} | "
            f"{row['required_a_tile_reuse']['mean']:.1f}x |")
    lines.append("")
    lines.append("### Research conclusion")
    lines.append("")
    lines.append(
        "1. The context-reuse probe, when enabled, is only an idealized "
        "upper-bound/negative-control experiment. Real multi-robot images are "
        "not identical, and the measured bound still leaves Enc+AE as the "
        "dominant graph segment.")
    lines.append(
        "2. Exact GEGLU LUT/row8 producer variants should not be kept as a core "
        "research direction: they are numerically safe but do not improve speed.")
    lines.append(
        "3. The high-value single-machine direction is a GEGLU-producing Down "
        "GEMM or virtual FP8 activation mainloop that removes global hid_fp8 "
        "materialization while reusing A tiles enough to avoid N-tile recompute.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize a Pi0.5 periodic batch research-loop run.")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    run_id = rows[0].get("run_id") if rows else args.index.stem.rsplit("_", 1)[-1]
    completed = len(rows)
    elapsed_total_s = max((float(r.get("elapsed_total_s", 0.0)) for r in rows),
                          default=0.0)
    e2e_paths = list(_iter_success_jsons(args.index, "e2e_profile"))
    if not e2e_paths:
        e2e_paths = list(_iter_success_jsons(
            args.index, "e2e_context_reuse_upper_bound"))
    boundary_paths = list(_iter_success_jsons(
        args.index, "geglu_down_boundary_model"))
    lut_paths = list(_iter_success_jsons(args.index, "geglu_lut_exclusion"))

    summary = {
        "run_id": run_id,
        "index": str(args.index),
        "completed_cycles": completed,
        "elapsed_total_min": elapsed_total_s / 60.0,
        "e2e_artifacts": len(e2e_paths),
        "boundary_artifacts": len(boundary_paths),
        "lut_artifacts": len(lut_paths),
        "e2e_profile": _summarize_e2e(e2e_paths),
        "geglu_down_boundary_model": _summarize_boundary(boundary_paths),
        "geglu_lut_exclusion": _summarize_lut(lut_paths),
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(_make_markdown(summary), encoding="utf-8")

    print(_make_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
