"""Per-model resumable predictions CSV.

We persist incrementally so a crash at row 70 leaves rows 0..69 on disk
and a fresh `run_model.py --model ...` invocation skips them.

The CSV header is exactly the column list in utils.RESULT_CSV_COLUMNS.
A row is considered *completed* iff `parse_status == 'ok'`. Rows with
status 'request_error' / 'parse_error' / 'invalid_label' are retried.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .utils import PROJECT_ROOT, RESULT_CSV_COLUMNS, ResultRow


class Checkpoint:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = PROJECT_ROOT / self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_header()

    def _write_header(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(RESULT_CSV_COLUMNS)

    def completed_ids(self) -> set[int]:
        if not self.path.exists():
            return set()
        done: set[int] = set()
        with self.path.open("r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                try:
                    if r.get("parse_status") == "ok":
                        done.add(int(r["id"]))
                except (KeyError, ValueError):
                    continue
        return done

    def append(self, row: ResultRow) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([getattr(row, c) for c in RESULT_CSV_COLUMNS])