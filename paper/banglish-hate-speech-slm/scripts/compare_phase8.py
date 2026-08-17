"""Phase 8 model comparison.

Pulls together:
- Wall-clock / latency / throughput / energy / CO2 from each
  model's run_summary.json (Phase 5)
- Label distribution + parse_status from each model's
  predictions.csv (Phase 5, re-loaded)
- Inter-model Cohen's kappa + chi-squared from
  reports/phase_7_evaluation.json (Phase 7)

Produces a single machine-readable table at
reports/phase_8_comparison.json, plus a human-facing markdown
summary at reports/phase_8_comparison.md.

No inference calls.
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

# Canonical labels from config/experiment.labels
CANONICAL_LABELS = [
    "Not Hate Speech",
    "Religious Hate",
    "Political Hate",
    "Personal / Social Abuse",
    "Gender Abuse",
    "Racism",
    "Geopolitical Hate",
]


def _read_summary(model_key: str) -> dict[str, Any]:
    p = ROOT / "results" / f"{model_key}_original" / "run_summary.json"
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_predictions(model_key: str) -> list[dict[str, str]]:
    p = ROOT / "results" / f"{model_key}_original" / "predictions.csv"
    with p.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_phase7() -> dict[str, Any]:
    p = ROOT / "reports" / "phase_7_evaluation.json"
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _per_model_block(model_key: str) -> dict[str, Any]:
    summary = _read_summary(model_key)
    rows = _read_predictions(model_key)

    # Latency stats per-row (sanity check vs summary)
    lats = [float(r["latency_seconds"]) for r in rows]
    pts = [int(r["prompt_tokens"]) for r in rows]
    cts = [int(r["completion_tokens"]) for r in rows]
    n = len(lats)

    # Parse status counts
    parse_counts: Counter[str] = Counter(r["parse_status"] for r in rows)
    n_ok = parse_counts.get("ok", 0)
    n_invalid = parse_counts.get("invalid_label", 0)
    n_empty = parse_counts.get("empty", 0)

    # Label distribution over ok rows
    label_dist = Counter(
        r["predicted_label"]
        for r in rows
        if r["parse_status"] == "ok"
    )

    # Cost per ok row
    cost_per_ok_row = (
        summary["energy"]["estimated_energy_Wh"] / n_ok
        if n_ok
        else float("inf")
    )

    # Energy / CO2 per ok row
    co2_per_ok = (
        summary["co2"]["estimated_co2_g"] / n_ok if n_ok else float("inf")
    )

    # Per-token cost (Wh / 1k tokens)
    wh_per_1k = (
        summary["energy"]["estimated_energy_Wh"]
        / max(1, summary["total_tokens"])
        * 1000.0
    )

    block = {
        "model": MODEL_DISPLAY[model_key],
        "parameters_b": summary["parameters_b"],
        "n_total": summary["n_rows_total"],
        "n_attempted": summary["n_attempted"],
        "parse_success_n": n_ok,
        "parse_success_pct": round(100 * n_ok / n, 2),
        "invalid_label_n": n_invalid,
        "invalid_label_pct": round(100 * n_invalid / n, 2),
        "empty_n": n_empty,
        "empty_pct": round(100 * n_empty / n, 2),
        "wall_clock_seconds": round(summary["latency"]["total_runtime_seconds"], 1),
        "wall_clock_human": _human_duration(
            summary["latency"]["total_runtime_seconds"]
        ),
        "latency_seconds_median": round(summary["latency"]["median_latency_s"], 3),
        "latency_seconds_p95": round(summary["latency"]["p95_latency_s"], 3),
        "latency_seconds_min": round(summary["latency"]["min_latency_s"], 3),
        "latency_seconds_max": round(summary["latency"]["max_latency_s"], 3),
        "samples_per_minute": round(summary["latency"]["samples_per_minute"], 2),
        "tokens_per_second": round(summary["latency"]["tokens_per_second"], 1),
        "total_prompt_tokens": summary["total_prompt_tokens"],
        "total_completion_tokens": summary["total_completion_tokens"],
        "total_tokens": summary["total_tokens"],
        "peak_vram_mib": summary["gpu"]["peak_vram_mib"],
        "estimated_energy_Wh": round(summary["energy"]["estimated_energy_Wh"], 2),
        "estimated_co2_g": round(summary["co2"]["estimated_co2_g"], 2),
        "power_source": summary["power_source"],
        "energy_per_ok_row_Wh": round(cost_per_ok_row, 4),
        "co2_per_ok_row_g": round(co2_per_ok, 4),
        "Wh_per_1k_tokens": round(wh_per_1k, 4),
        "label_distribution_ok_rows": {
            lab: label_dist.get(lab, 0) for lab in CANONICAL_LABELS
        },
        "label_distribution_ok_rows_pct": {
            lab: round(100 * label_dist.get(lab, 0) / max(1, n_ok), 2)
            for lab in CANONICAL_LABELS
        },
        "api_cloud_co2_reference_g": summary["api_cloud_co2_reference_only_g"],
    }
    return block


def _human_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def main() -> int:
    phase7 = _read_phase7()

    per_model = {k: _per_model_block(k) for k in MODEL_KEYS}

    # Pairwise kappas (ok-only) reshaped for clarity
    kappas_ok = phase7["pairwise_kappa_ok_only"]
    pairwise_kappa_table = []
    keys = list(MODEL_DISPLAY.values())
    for i, k1 in enumerate(MODEL_KEYS):
        for j, k2 in enumerate(MODEL_KEYS):
            if j <= i:
                continue
            entry = kappas_ok[k1][k2]
            pairwise_kappa_table.append({
                "model_a": MODEL_DISPLAY[k1],
                "model_b": MODEL_DISPLAY[k2],
                "n_intersection": entry["n"],
                "raw_agreement_pct": round(100 * entry["agreement"], 2),
                "kappa": round(entry["kappa"], 4),
            })

    # Chi-squared summary
    chi2 = phase7["chi_squared_model_vs_label"]

    # Identify "winner" per dimension for headline
    winners = {
        "fastest_median_latency": min(
            MODEL_KEYS,
            key=lambda k: per_model[k]["latency_seconds_median"],
        ),
        "highest_parse_success": max(
            MODEL_KEYS,
            key=lambda k: per_model[k]["parse_success_n"],
        ),
        "highest_throughput": max(
            MODEL_KEYS,
            key=lambda k: per_model[k]["samples_per_minute"],
        ),
        "lowest_total_energy": min(
            MODEL_KEYS,
            key=lambda k: per_model[k]["estimated_energy_Wh"],
        ),
        "lowest_total_co2": min(
            MODEL_KEYS,
            key=lambda k: per_model[k]["estimated_co2_g"],
        ),
        "lowest_energy_per_ok_row": min(
            MODEL_KEYS,
            key=lambda k: per_model[k]["energy_per_ok_row_Wh"]
            if per_model[k]["energy_per_ok_row_Wh"] != float("inf")
            else 1e18,
        ),
    }
    winners_named = {k: MODEL_DISPLAY[v] for k, v in winners.items()}

    out = {
        "models": [MODEL_DISPLAY[k] for k in MODEL_KEYS],
        "per_model": per_model,
        "pairwise_kappa_ok_only": pairwise_kappa_table,
        "chi_squared_model_vs_label": {
            "chi2": chi2["chi2"],
            "df": chi2["df"],
            "p_value": chi2["p_value"],
        },
        "winners_per_dimension": winners_named,
    }

    out_path = ROOT / "reports" / "phase_8_comparison.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(json.dumps(winners_named, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
