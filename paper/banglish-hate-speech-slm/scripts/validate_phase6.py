"""Phase 6 validation script.

Pure data-quality check on the four Phase-5 predictions CSVs:

1.  Schema (column names and dtypes)
2.  Row count == 312 (after dedup)
3.  No duplicate ids
4.  Numeric columns (latency_seconds, prompt_tokens, completion_tokens,
    total_tokens) >= 0; total_tokens == prompt+completion
5.  parse_status in the documented enum
6.  experiment column == 'original' for every row
7.  model column matches the directory name
8.  predicted_label matches the canonical-label set when parse_status=='ok'
9.  Cross-model checks: same id, same Text on every row across models

Outputs:
    reports/phase_6_validation.md  - markdown report
    The script also exits non-zero if any check fails.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import RESULT_CSV_COLUMNS  # noqa: E402


EXPECTED_LABEL_SET = {
    "Not Hate Speech",
    "Religious Hate",
    "Political Hate",
    "Personal / Social Abuse",
    "Gender Abuse",
    "Racism",
    "Geopolitical Hate",
}

PARSE_STATUS_ENUM = {
    "ok",
    "invalid_label",
    "parse_error",
    "request_error",
    "empty",
    "pending",
}

EXPECTED_ROWS = 312


def _ascii_check(text: str) -> list[str]:
    """Return a list of problems found in a single CSV header / row."""
    return []


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        fieldnames = rdr.fieldnames or []
    return fieldnames, rows


def _validate_one(model_key: str, csv_path: Path) -> dict:
    """Run all schema/data-quality checks on one CSV."""
    out: dict = {"model": model_key, "path": str(csv_path), "checks": []}

    fieldnames, rows = _read_csv(csv_path)
    out["row_count"] = len(rows)

    # 1. schema
    expected = list(RESULT_CSV_COLUMNS)
    actual = list(fieldnames)
    if actual != expected:
        out["checks"].append({
            "name": "schema_columns",
            "status": "fail",
            "detail": f"expected {expected}; got {actual}",
        })
    else:
        out["checks"].append({
            "name": "schema_columns",
            "status": "ok",
            "detail": "exact match",
        })

    # 2. row count
    if len(rows) != EXPECTED_ROWS:
        out["checks"].append({
            "name": "row_count",
            "status": "fail",
            "detail": f"expected {EXPECTED_ROWS}; got {len(rows)}",
        })
    else:
        out["checks"].append({
            "name": "row_count",
            "status": "ok",
            "detail": f"{len(rows)} rows",
        })

    # 3. duplicate ids
    ids = [r["id"] for r in rows]
    counts = Counter(ids)
    dups = {k: v for k, v in counts.items() if v > 1}
    if dups:
        out["checks"].append({
            "name": "unique_ids",
            "status": "fail",
            "detail": f"duplicates: {dups}",
        })
    else:
        out["checks"].append({
            "name": "unique_ids",
            "status": "ok",
            "detail": f"{len(counts)} unique ids",
        })

    # 4. numeric ranges + total_tokens == pt+ct
    bad_latency = []
    bad_total = []
    negative_numeric = []
    for r in rows:
        try:
            lat = float(r["latency_seconds"])
            pt = int(r["prompt_tokens"])
            ct = int(r["completion_tokens"])
            tot = int(r["total_tokens"])
        except ValueError as e:
            out["checks"].append({
                "name": "numeric_parse",
                "status": "fail",
                "detail": f"id={r['id']}: {e}",
            })
            return out
        if lat < 0:
            bad_latency.append(r["id"])
        if pt < 0 or ct < 0 or tot < 0:
            negative_numeric.append(r["id"])
        if pt + ct != tot:
            bad_total.append((r["id"], pt, ct, tot))

    out["checks"].append({
        "name": "latency_nonneg",
        "status": "ok" if not bad_latency else "fail",
        "detail": "all rows >= 0" if not bad_latency else f"negative: {bad_latency}",
    })
    out["checks"].append({
        "name": "tokens_nonneg",
        "status": "ok" if not negative_numeric else "fail",
        "detail": "all rows >= 0" if not negative_numeric else f"negative: {negative_numeric}",
    })
    out["checks"].append({
        "name": "total_eq_pt_plus_ct",
        "status": "ok" if not bad_total else "fail",
        "detail": "all rows consistent" if not bad_total else f"mismatch: {bad_total[:5]}",
    })

    # 5. parse_status enum
    statuses = Counter(r["parse_status"] for r in rows)
    bad_status = [s for s in statuses if s not in PARSE_STATUS_ENUM]
    out["checks"].append({
        "name": "parse_status_enum",
        "status": "ok" if not bad_status else "fail",
        "detail": f"distribution={dict(statuses)}" + (f"; unknown={bad_status}" if bad_status else ""),
    })

    # 6. experiment field
    exp_set = set(r["experiment"] for r in rows)
    out["checks"].append({
        "name": "experiment_field",
        "status": "ok" if exp_set == {"original"} else "fail",
        "detail": f"distinct values = {exp_set}",
    })

    # 7. model field consistency
    model_set = set(r["model"] for r in rows)
    expected_model = _expected_model_display(model_key)
    out["checks"].append({
        "name": "model_field",
        "status": "ok" if model_set == {expected_model} else "fail",
        "detail": f"distinct values = {model_set} (expected {{{expected_model!r}}})",
    })

    # 8. predicted_label canonical when parse_status=='ok'
    bad_label = []
    for r in rows:
        if r["parse_status"] == "ok":
            lab = r["predicted_label"]
            if lab not in EXPECTED_LABEL_SET:
                bad_label.append((r["id"], lab))
    out["checks"].append({
        "name": "canonical_labels",
        "status": "ok" if not bad_label else "fail",
        "detail": "all ok rows use canonical labels"
        if not bad_label
        else f"non-canonical: {bad_label[:5]}",
    })

    # quick distribution for ok rows
    if statuses.get("ok"):
        lab_dist = Counter(
            r["predicted_label"] for r in rows if r["parse_status"] == "ok"
        )
        out["ok_label_distribution"] = dict(lab_dist)
        out["parse_status_distribution"] = dict(statuses)
    else:
        out["parse_status_distribution"] = dict(statuses)

    # latency stats
    lats = [float(r["latency_seconds"]) for r in rows]
    if lats:
        s = sorted(lats)
        n = len(s)

        def _p(p: float) -> float:
            k = (n - 1) * (p / 100.0)
            lo = int(k)
            hi = min(lo + 1, n - 1)
            if lo == hi:
                return s[lo]
            return s[lo] + (s[hi] - s[lo]) * (k - lo)

        out["latency_stats"] = {
            "n": n,
            "min": min(lats),
            "max": max(lats),
            "mean": sum(lats) / n,
            "median": s[n // 2],
            "p95": _p(95.0),
        }

    out["all_passed"] = all(c["status"] == "ok" for c in out["checks"])
    return out


def _expected_model_display(model_key: str) -> str:
    return {
        "qwen3_4b": "Qwen3-4B",
        "qwen3_1_7b": "Qwen3-1.7B",
        "gemma3_4b": "Gemma 3 4B Instruct",
        "llama3_2_3b": "Llama 3.2 3B Instruct",
    }[model_key]


def _cross_model(texts_by_model: dict[str, dict[int, str]]) -> dict:
    """Check that the same id maps to the same Text in every CSV."""
    model_keys = list(texts_by_model.keys())
    base = model_keys[0]
    base_map = texts_by_model[base]
    mismatches = []
    for id_, text in base_map.items():
        for other in model_keys[1:]:
            if texts_by_model[other].get(id_) != text:
                mismatches.append((id_, base, other))
    return {
        "name": "cross_model_text_consistency",
        "status": "ok" if not mismatches else "fail",
        "detail": f"texts identical across {len(model_keys)} models"
        if not mismatches
        else f"{len(mismatches)} id(s) differ (showing first 5): {mismatches[:5]}",
    }


def main() -> int:
    results_dir = ROOT / "results"
    model_keys = ["qwen3_4b", "gemma3_4b", "qwen3_1_7b", "llama3_2_3b"]
    all_reports = []
    texts_by_model: dict[str, dict[int, str]] = {}

    for mk in model_keys:
        csv_path = results_dir / f"{mk}_original" / "predictions.csv"
        report = _validate_one(mk, csv_path)
        all_reports.append(report)
        # collect text-by-id for cross-model check
        _, rows = _read_csv(csv_path)
        texts_by_model[mk] = {int(r["id"]): r["Text"] for r in rows}

    xc = _cross_model(texts_by_model)
    overall_pass = (
        all(r["all_passed"] for r in all_reports) and xc["status"] == "ok"
    )

    # Persist machine-readable form
    out_json = ROOT / "reports" / "phase_6_validation.json"
    out_json.parent.mkdir(exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"per_model": all_reports, "cross_model": xc, "overall_pass": overall_pass}, f, indent=2, ensure_ascii=False, default=str)

    print(json.dumps({"overall_pass": overall_pass}, indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
