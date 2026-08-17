"""Deduplicate predictions.csv by `id`, keeping the LAST row per id.

During Phase 5 we discovered a checkpoint quirk: rows with
parse_status != 'ok' (e.g. 'invalid_label') are retried on the next
run. In cases where the retry produces the same status (e.g. Llama
3.2 3B built-in safety refusal on every attempt), the row is appended
twice. This script collapses to one row per id, keeping the most
recent attempt (largest timestamp).

Usage:
    python scripts/dedup_predictions.py results/llama3_2_3b_original/predictions.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import RESULT_CSV_COLUMNS  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: dedup_predictions.py <predictions.csv>", file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Missing: {path}", file=sys.stderr)
        return 1

    with path.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    # Keep the latest by (id, timestamp) — timestamp ISO string sorts
    # correctly for UTC ISO-8601.
    by_id: dict[str, dict] = {}
    for r in rows:
        by_id[r["id"]] = r  # last one wins
    deduped = list(by_id.values())

    n_dup = len(rows) - len(deduped)
    if n_dup == 0:
        print(f"{path}: no duplicates")
        return 0

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_CSV_COLUMNS)
        w.writeheader()
        for r in deduped:
            w.writerow(r)
    print(f"{path}: removed {n_dup} duplicate(s); kept {len(deduped)} unique rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
