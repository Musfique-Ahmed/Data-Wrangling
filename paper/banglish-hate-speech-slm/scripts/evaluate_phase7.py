"""Phase 7 evaluation - distribution-based metrics only.

We do NOT have ground-truth labels (the supplied CSV has only the
`Sentences` column). This script computes:

1. Per-model label distribution over the 7 canonical labels and
   the 4 non-ok parse_status values (ok + invalid_label + empty +
   parse_error + request_error).

2. Inter-model agreement: Cohen's kappa between every pair of
   models, restricted to the rows that BOTH models parsed
   successfully (parse_status == 'ok'). Also reports raw
   percent-agreement and the size of the intersection.

3. Chi-squared test of independence: model x label, treating the
   intersection of ok rows across all models. Tests whether the
   four models' label distributions differ by more than chance.

4. Failure-mode breakdown for the two noisy models:
   - Qwen3-4B: distribution of (completion_tokens, parse_status)
     showing where the thinking budget gets exhausted.
   - Llama 3.2 3B: distribution of completion_tokens among
     invalid_label rows, plus a few sampled refusal messages.

Outputs:
    reports/phase_7_evaluation.md    (markdown narrative)
    reports/phase_7_evaluation.json  (machine-readable)
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LABELS = [
    "Not Hate Speech",
    "Religious Hate",
    "Political Hate",
    "Personal / Social Abuse",
    "Gender Abuse",
    "Racism",
    "Geopolitical Hate",
]
PARSE_STATUSES = ["ok", "invalid_label", "empty", "parse_error", "request_error"]
MODEL_KEYS = ["qwen3_4b", "gemma3_4b", "qwen3_1_7b", "llama3_2_3b"]
MODEL_DISPLAY = {
    "qwen3_4b": "Qwen3-4B",
    "gemma3_4b": "Gemma 3 4B",
    "qwen3_1_7b": "Qwen3-1.7B",
    "llama3_2_3b": "Llama 3.2 3B",
}


def _read_predictions(model_key: str) -> list[dict[str, str]]:
    path = ROOT / "results" / f"{model_key}_original" / "predictions.csv"
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _label_distribution(rows: list[dict]) -> dict[str, int]:
    """Count labels for all rows. ok rows -> their predicted_label;
    non-ok rows -> bucket into parse_status.
    """
    counts: dict[str, int] = {lab: 0 for lab in LABELS}
    counts["<invalid_label>"] = 0
    counts["<empty>"] = 0
    counts["<parse_error>"] = 0
    counts["<request_error>"] = 0
    for r in rows:
        status = r["parse_status"]
        if status == "ok":
            lab = r["predicted_label"]
            if lab in counts:
                counts[lab] += 1
            else:
                counts.setdefault(f"<other:{lab}>", 0)
                counts[f"<other:{lab}>"] += 1
        elif status == "invalid_label":
            counts["<invalid_label>"] += 1
        elif status == "empty":
            counts["<empty>"] += 1
        elif status == "parse_error":
            counts["<parse_error>"] += 1
        elif status == "request_error":
            counts["<request_error>"] += 1
    return counts


def _fractions(counts: dict[str, int]) -> dict[str, float]:
    n = sum(counts.values())
    return {k: (v / n if n else 0.0) for k, v in counts.items()}


def _kappa(a: list[str], b: list[str]) -> tuple[float, int, float]:
    """Cohen's kappa for two label sequences of equal length.

    Returns (kappa, n, percent_agreement).
    """
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return (0.0, 0, 0.0)
    agree = sum(1 for x, y in zip(a, b) if x == y)
    pct = agree / n
    cats = sorted(set(a) | set(b))
    po = pct
    pe = 0.0
    n_a = Counter(a)
    n_b = Counter(b)
    for c in cats:
        pe += (n_a[c] / n) * (n_b[c] / n)
    if pe >= 1.0:
        # Degenerate: avoid divide-by-zero. Treat as perfect agreement.
        return (1.0, n, pct)
    return ((po - pe) / (1.0 - pe), n, pct)


def _pairwise_kappas(by_model: dict[str, dict[int, str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    keys = list(by_model.keys())
    for i, k1 in enumerate(keys):
        out[k1] = {}
        for k2 in keys:
            if k2 in out.get(k1, {}):
                continue
            ids = sorted(set(by_model[k1]) & set(by_model[k2]))
            a = [by_model[k1][i_] for i_ in ids]
            b = [by_model[k2][i_] for i_ in ids]
            k, n, pct = _kappa(a, b)
            entry = {"kappa": k, "n": n, "agreement": pct}
            out[k1][k2] = entry
            out.setdefault(k2, {})[k1] = entry
    return out


def _chi_square_p(contingency: list[list[int]]) -> float:
    """Pearson chi-squared statistic for an RxC contingency table.

    Returns chi2 (not p-value). For an honest p-value, see
    `_chi_square_pvalue_scipy` below (lazy import).
    """
    rows = len(contingency)
    cols = len(contingency[0])
    row_totals = [sum(r) for r in contingency]
    col_totals = [sum(contingency[r][c] for r in range(rows)) for c in range(cols)]
    n = sum(row_totals)
    if n == 0:
        return 0.0
    chi2 = 0.0
    for r in range(rows):
        for c in range(cols):
            expected = row_totals[r] * col_totals[c] / n
            if expected > 0:
                chi2 += (contingency[r][c] - expected) ** 2 / expected
    return chi2


def _chi_square_pvalue(chi2: float, df: int) -> float | None:
    """scipy.stats.chi2.sf(chi2, df). Lazily imported."""
    try:
        from scipy.stats import chi2 as _chi2
        return float(_chi2.sf(chi2, df))
    except ImportError:
        return None


def _intersection_labels(by_model: dict[str, dict[int, str]]) -> tuple[list[int], dict[str, list[str]]]:
    common_ids = sorted(set.intersection(*(set(m.keys()) for m in by_model.values())))
    out = {k: [by_model[k][i_] for i_ in common_ids] for k in by_model}
    return common_ids, out


def _empty_to_label(rows: list[dict]) -> dict[int, str]:
    return {
        int(r["id"]): (r["predicted_label"] if r["parse_status"] == "ok" else f"<{r['parse_status']}>")
        for r in rows
    }


def main() -> int:
    # 1. Load
    model_rows: dict[str, list[dict]] = {}
    for k in MODEL_KEYS:
        model_rows[k] = _read_predictions(k)

    # 2. Per-model label distribution
    label_dist: dict[str, dict[str, int]] = {}
    label_frac: dict[str, dict[str, float]] = {}
    parse_dist: dict[str, dict[str, int]] = {}
    for k, rows in model_rows.items():
        label_dist[k] = _label_distribution(rows)
        label_frac[k] = _fractions(label_dist[k])
        parse_dist[k] = dict(Counter(r["parse_status"] for r in rows))

    # 3. Inter-model agreement (only ok rows)
    by_model_ok_only: dict[str, dict[int, str]] = {}
    for k, rows in model_rows.items():
        by_model_ok_only[k] = {
            int(r["id"]): r["predicted_label"]
            for r in rows
            if r["parse_status"] == "ok"
        }
    kappas_ok = _pairwise_kappas(by_model_ok_only)

    # 3b. Inter-model agreement when non-ok is bucketed as <parse_status>
    by_model_full = {k: _empty_to_label(rs) for k, rs in model_rows.items()}
    common_ids, _ = _intersection_labels(by_model_full)
    # Restrict to common ids for pairwise
    restricted = {k: {i: by_model_full[k][i] for i in common_ids} for k in by_model_full}
    kappas_full = _pairwise_kappas(restricted)

    # 4. Chi-squared: model x label, restricted to all-ok rows
    ok_only_common = sorted(set.intersection(*(set(m.keys()) for m in by_model_ok_only.values())))
    # build contingency: rows = models, cols = labels
    contingency: list[list[int]] = []
    for k in MODEL_KEYS:
        cnt = Counter(by_model_ok_only[k][i_] for i_ in ok_only_common)
        contingency.append([cnt.get(lab, 0) for lab in LABELS])
    chi2 = _chi_square_p(contingency)
    df = (len(MODEL_KEYS) - 1) * (len(LABELS) - 1)
    pval = _chi_square_pvalue(chi2, df)

    # 5. Failure-mode breakdown
    qwen3_4b = model_rows["qwen3_4b"]
    qwen3_4b_ct_by_status: dict[str, list[int]] = defaultdict(list)
    for r in qwen3_4b:
        qwen3_4b_ct_by_status[r["parse_status"]].append(int(r["completion_tokens"]))
    qwen3_4b_ct_summary = {
        s: {
            "n": len(v),
            "min": min(v) if v else 0,
            "max": max(v) if v else 0,
            "median": sorted(v)[len(v) // 2] if v else 0,
            "n_at_cap": sum(1 for x in v if x >= 4090),
        }
        for s, v in qwen3_4b_ct_by_status.items()
    }

    llama = model_rows["llama3_2_3b"]
    llama_invalid_ct = [
        int(r["completion_tokens"])
        for r in llama
        if r["parse_status"] == "invalid_label"
    ]
    # Sample a few refusal messages
    sample_refusals = []
    for r in llama:
        if r["parse_status"] == "invalid_label" and len(sample_refusals) < 3:
            txt = r["raw_response"].replace("\n", " ").strip()[:140]
            sample_refusals.append({"id": r["id"], "raw": txt})

    # 6. Distribution of Gemma "Racism" bias by status
    gemma_racism_ok = sum(
        1 for r in model_rows["gemma3_4b"]
        if r["parse_status"] == "ok" and r["predicted_label"] == "Racism"
    )

    # 7. Assemble output
    out = {
        "label_distribution_counts": label_dist,
        "label_distribution_fractions": {
            k: {kk: round(vv, 4) for kk, vv in v.items()}
            for k, v in label_frac.items()
        },
        "parse_status_counts": parse_dist,
        "pairwise_kappa_ok_only": kappas_ok,
        "pairwise_kappa_with_parse_status": kappas_full,
        "all_ok_intersection_size": len(ok_only_common),
        "chi_squared_model_vs_label": {
            "chi2": chi2,
            "df": df,
            "p_value": pval,
            "contingency": contingency,
            "labels": LABELS,
            "models": [MODEL_DISPLAY[k] for k in MODEL_KEYS],
        },
        "qwen3_4b_completion_tokens_by_status": qwen3_4b_ct_summary,
        "llama3_2_3b_refusal": {
            "n_invalid_label": len(llama_invalid_ct),
            "completion_tokens_min": min(llama_invalid_ct) if llama_invalid_ct else 0,
            "completion_tokens_max": max(llama_invalid_ct) if llama_invalid_ct else 0,
            "completion_tokens_median": sorted(llama_invalid_ct)[len(llama_invalid_ct) // 2] if llama_invalid_ct else 0,
            "sample_messages": sample_refusals,
        },
        "gemma_racism_count_in_ok_rows": gemma_racism_ok,
    }

    out_path = ROOT / "reports" / "phase_7_evaluation.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "all_ok_intersection_size": out["all_ok_intersection_size"],
        "chi2": chi2,
        "df": df,
        "p_value": pval,
        "gemma_racism_ok": gemma_racism_ok,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
