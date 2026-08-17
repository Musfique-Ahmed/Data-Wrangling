"""Dataset loader.

Reads the CSV (default: data/data_130.csv), normalises the text
column to `Text`, and yields rows with stable integer IDs starting at 0.
The source CSV is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from .utils import PROJECT_ROOT


@dataclass
class DatasetRow:
    id: int
    text: str


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the CSV and rename the text column to `Text`.

    Raises if the expected text column is missing.
    """
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    text_col_cfg = _configured_text_column()
    if text_col_cfg not in df.columns:
        # Try common alternates.
        for alt in ("Text", "text", "sentence", "Sentence"):
            if alt in df.columns:
                df = df.rename(columns={alt: "Text"})
                break
        else:
            raise ValueError(
                f"Text column '{text_col_cfg}' not found. "
                f"Available columns: {list(df.columns)}"
            )
    elif text_col_cfg != "Text":
        df = df.rename(columns={text_col_cfg: "Text"})

    df = df.dropna(subset=["Text"]).reset_index(drop=True)
    df["Text"] = df["Text"].astype(str)
    return df


def iter_rows(df: pd.DataFrame) -> Iterator[DatasetRow]:
    for i, t in enumerate(df["Text"].tolist()):
        yield DatasetRow(id=i, text=t)


def _configured_text_column() -> str:
    """Read the configured source column name without forcing PyYAML import."""
    try:
        import yaml  # noqa: F401
        cfg_path = PROJECT_ROOT / "config" / "config.yaml"
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg["dataset"]["text_column"]
    except Exception:
        return "Sentences"


def n_rows(path: str | Path) -> int:
    return len(load_dataset(path))