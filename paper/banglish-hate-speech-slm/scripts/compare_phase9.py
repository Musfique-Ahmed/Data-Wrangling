"""Phase 9 - banglish_aware vs original prompt comparison.

For each of the 4 models, compares:
- parse success rate (ok / total)
- label distribution on ok rows
- runtime / energy / CO2

Also tests the 3 hypotheses from the Phase 8 report:
H1: Gemma's "Racism" rate should drop on banglish_aware.
H2: Llama's refusal rate should drop on banglish_aware.
H3: Qwen3-4B's thinking-budget exhaustion should be roughly
    unchanged (i.e. banglish_aware prompt is similar length).

No inference calls. Reads only existing JSON/CSV outputs.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_KEYS = ["qwen3_4b", "gemma3_4b", "qwen3_1_7b", "llama3_2_3b"]
MODEL_DISPLAY = {
    "qwen3_4b": "Qwen3-4B",
    "gemma3_4b": "Gemma 3 4B",
    "qwen3_1_7b": "Qwen3-1.7B",
    "llama3_2_3b": "Llama 3.2 3B",
}
PROMPTS = ["original", "banglish_aware"]


def _read_summary(model_key: str, prompt: str) -> dict[str, Any]:
    p = ROOT / "results" / f"{model_key}_{prompt}" / "run_summary.json"
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_predictions(model_key: str, prompt: str) -> list[dict[str, str]]:
    p = ROOT / "results" / f"{model_key}_{prompt}" / "predictions.csv"
    with p.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _per_run_block(model_key: str, prompt: str) -> dict[str, Any]:
    summary = _read_summary(model_key, prompt)
    rows = _read_predictions(model_key, prompt)
    n = len(rows)
    status_counts = Counter(r["parse_status"] for r in rows)
    n_ok = status_counts.get("ok", 0)
    n_invalid = status_counts.get("invalid_label", 0)
    n_empty = status_counts.get("empty", 0)
    label_dist = Counter(
        r["predicted_label"] for r in rows if r["parse_status"] == "ok"
    )
    return {
        "prompt": prompt,
        "n_total": n,
        "n_ok": n_ok,
        "ok_pct": round(100 * n_ok / n, 2),
        "n_invalid_label": n_invalid,
        "invalid_label_pct": round(100 * n_invalid / n, 2),
        "n_empty": n_empty,
        "empty_pct": round(100 * n_empty / n, 2),
        "wall_clock_s": round(summary["latency"]["total_runtime_seconds"], 1),
        "median_latency_s": round(summary["latency"]["median_latency_s"], 3),
        "p95_latency_s": round(summary["latency"]["p95_latency_s"], 3),
        "total_tokens": summary["total_tokens"],
        "energy_Wh": round(summary["energy"]["estimated_energy_Wh"], 2),
        "co2_g": round(summary["co2"]["estimated_co2_g"], 2),
        "label_distribution_ok": dict(label_dist),
        "racism_count_ok": label_dist.get("Racism", 0),
        "racism_pct_ok": round(100 * label_dist.get("Racism", 0) / max(1, n_ok), 2),
    }


def _deltas(orig: dict, ba: dict) -> dict[str, Any]:
    """Compute key deltas: banglish_aware - original."""
    out = {}
    for k in ["ok_pct", "invalid_label_pct", "empty_pct",
              "median_latency_s", "energy_Wh", "co2_g",
              "racism_pct_ok"]:
        v_o = orig[k]
        v_b = ba[k]
        out[k] = round(v_b - v_o, 2)
    out["delta_n_ok"] = ba["n_ok"] - orig["n_ok"]
    out["delta_wall_clock_s"] = round(ba["wall_clock_s"] - orig["wall_clock_s"], 1)
    return out


def _verdict(deltas: dict, model_key: str) -> dict[str, str]:
    """Test the 3 hypotheses from Phase 8 §8."""
    out = {}
    if model_key == "gemma3_4b":
        # H1: Racism rate should drop
        out["H1_racism_drop"] = (
            "PASS" if deltas["racism_pct_ok"] < 0 else "FAIL"
        )
        out["H1_detail"] = (
            f"Racism pct: {deltas['racism_pct_ok']:+.1f} pts"
        )
    if model_key == "llama3_2_3b":
        # H2: refusal rate should drop
        out["H2_refusal_drop"] = (
            "PASS" if deltas["invalid_label_pct"] < 0 else "FAIL"
        )
        out["H2_detail"] = (
            f"invalid_label pct: {deltas['invalid_label_pct']:+.1f} pts"
        )
    if model_key == "qwen3_4b":
        # H3: thinking-budget exhaustion should be roughly unchanged
        # (we measure as empty pct, which is the most direct signal
        # of pure thinking-budget exhaustion)
        out["H3_thinking_unchanged"] = (
            "PASS" if abs(deltas["empty_pct"]) < 10 else "FAIL"
        )
        out["H3_detail"] = (
            f"empty pct: {deltas['empty_pct']:+.1f} pts (orig={deltas.get('orig_empty_pct', '?')}%)"
        )
    return out


def main() -> int:
    per_model: dict[str, dict] = {}
    for mk in MODEL_KEYS:
        runs = {p: _per_run_block(mk, p) for p in PROMPTS}
        deltas = _deltas(runs["original"], runs["banglish_aware"])
        # for the H3 verdict we need the original empty pct in the detail
        deltas["orig_empty_pct"] = runs["original"]["empty_pct"]
        deltas["ba_empty_pct"] = runs["banglish_aware"]["empty_pct"]
        verdict = _verdict(deltas, mk)
        per_model[mk] = {
            "original": runs["original"],
            "banglish_aware": runs["banglish_aware"],
            "deltas": deltas,
            "verdict": verdict,
        }

    out = {"per_model": per_model}
    out_path = ROOT / "reports" / "phase_9_comparison.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Print a compact summary
    print("Per-model summary (ok_pct orig -> ba):")
    for mk, d in per_model.items():
        o = d["original"]["ok_pct"]
        b = d["banglish_aware"]["ok_pct"]
        v = d["verdict"]
        v_str = "; ".join(f"{k}={vv}" for k, vv in v.items() if not k.endswith("_detail"))
        print(f"  {MODEL_DISPLAY[mk]:18s}  {o:5.1f}% -> {b:5.1f}%  ({v_str})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
