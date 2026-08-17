"""Small shared helpers — paths, JSON I/O, formatting."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Project root: banglish-hate-speech-slm/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp, second precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file. Imported lazily so other utils don't pull in PyYAML."""
    import yaml  # local import keeps module cheap

    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: str | Path, payload: Any) -> None:
    """Atomically write JSON with indent=2 and ensure_ascii=False (Banglish safe)."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def read_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------- result row -----------------------------

@dataclass
class ResultRow:
    """One row of the per-model predictions CSV.

    `parse_status` is one of: 'ok', 'invalid_label', 'parse_error',
    'request_error'. Resume logic only skips rows with parse_status=='ok'.
    """

    id: int
    Text: str
    predicted_label: str = ""
    reasoning: str = ""
    raw_response: str = ""
    parse_status: str = "pending"
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    model: str = ""
    experiment: str = ""
    timestamp: str = ""

    def finalize(self, model: str, experiment: str) -> "ResultRow":
        self.model = model
        self.experiment = experiment
        self.timestamp = utcnow_iso()
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self


RESULT_CSV_COLUMNS = [
    "id", "Text", "predicted_label", "reasoning", "raw_response",
    "parse_status", "error", "prompt_tokens", "completion_tokens",
    "total_tokens", "latency_seconds", "model", "experiment", "timestamp",
]


# --------------------------- text normalisation ---------------------------

_LABEL_TOKEN_MAP: dict[str, str] = {
    # canonical → canonical (used for matching)
    "not hate speech": "Not Hate Speech",
    "religious hate": "Religious Hate",
    "political hate": "Political Hate",
    "personal / social abuse": "Personal / Social Abuse",
    "gender abuse": "Gender Abuse",
    "racism": "Racism",
    "geopolitical hate": "Geopolitical Hate",
}


def normalise_label(text: str) -> str | None:
    """Return canonical label if `text` mentions one of the seven labels
    (case-insensitive, whitespace/slash tolerant), else None.

    This is used for fuzzy matching on the *first* line/segment of the
    model output. The full output is still recorded in `raw_response`.
    """
    if not text:
        return None
    t = text.strip().lower()
    # Strip common leading bullets / numbering / quoting.
    t = re.sub(r"^[\s\-\*\d\.\(\)\"\'`]+", "", t)
    # Direct match.
    for key, canon in _LABEL_TOKEN_MAP.items():
        if t.startswith(key):
            return canon
        # tolerant: collapse spaces around "/"
        t_norm = re.sub(r"\s*/\s*", "/", t)
        key_norm = re.sub(r"\s*/\s*", "/", key)
        if t_norm.startswith(key_norm):
            return canon
    return None