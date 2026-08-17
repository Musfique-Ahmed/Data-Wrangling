"""Inspect the local environment and (optionally) pull required Ollama models.

Usage:
    python scripts/check_environment.py            # inspect only
    python scripts/check_environment.py --pull    # pull missing models
    python scripts/check_environment.py --dry-run # no Ollama calls
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows so Unicode labels print cleanly.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Allow `python scripts/check_environment.py` from the repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests                                                     # noqa: E402
import yaml                                                          # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
OLLAMA_URL = CFG["ollama"]["base_url"].rstrip("/")


def hr(t: str) -> None:
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8)


def python_info() -> None:
    import sys as _s
    print(f"Python : {_s.version.split()[0]} ({_s.executable})")


def gpu_info() -> None:
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi: NOT FOUND")
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        for line in out.stdout.strip().splitlines():
            print(f"GPU     : {line}")
    except subprocess.CalledProcessError as e:
        print(f"nvidia-smi failed: {e.stderr or e}")


def ollama_info() -> None:
    print(f"Ollama URL: {OLLAMA_URL}")
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        data = r.json()
        models = data.get("models", [])
        print(f"Ollama : up, {len(models)} model(s) installed")
        for m in models:
            print(f"  - {m['name']}  ({m.get('size', '?')})")
    except requests.RequestException as e:
        print(f"Ollama : UNREACHABLE ({e!s})")


def required_models_status() -> dict[str, str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        installed = {m["name"] for m in r.json().get("models", [])}
    except requests.RequestException:
        installed = set()
    status = {}
    for key, cfg in CFG["models"].items():
        tag = cfg["ollama_model"]
        ok = any(n == tag or n.startswith(tag + ":") for n in installed)
        status[tag] = "present" if ok else "MISSING"
    return status


def pull(tag: str) -> None:
    print(f"Pulling {tag} (this can take a few minutes)...")
    r = requests.post(
        f"{OLLAMA_URL}/api/pull",
        json={"name": tag, "stream": False},
        timeout=3600,
    )
    if r.ok:
        print(f"  OK {tag} pulled")
    else:
        print(f"  FAIL pull failed ({r.status_code}): {r.text[:200]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true", help="pull missing models")
    ap.add_argument("--dry-run", action="store_true", help="skip Ollama HTTP calls")
    args = ap.parse_args()

    hr("Python")
    python_info()

    hr("GPU")
    gpu_info()

    hr("Ollama")
    if args.dry_run:
        print("(dry-run -- skipped)")
    else:
        ollama_info()

    hr("Required models")
    if args.dry_run:
        for k, c in CFG["models"].items():
            print(f"  - {k:14s} -> {c['ollama_model']}")
    else:
        status = required_models_status()
        any_missing = False
        for k, c in CFG["models"].items():
            tag = c["ollama_model"]
            mark = status.get(tag, "?")
            if mark == "MISSING":
                any_missing = True
            print(f"  - {k:14s} -> {tag:30s} [{mark}]")
        if any_missing:
            print("\nSome models are missing.")
            if args.pull:
                for k, c in CFG["models"].items():
                    if status.get(c["ollama_model"]) == "MISSING":
                        pull(c["ollama_model"])
            else:
                print("Re-run with --pull to fetch them.")

    hr("Dataset")
    ds = CFG["dataset"]
    p = ROOT / ds["path"]
    if p.exists():
        import pandas as pd
        df = pd.read_csv(p)
        print(f"Path   : {p}")
        print(f"Rows   : {len(df)}")
        print(f"Cols   : {list(df.columns)}")
        text_col = ds["text_column"]
        if text_col in df.columns:
            sample = df[text_col].iloc[0]
            print(f"Sample : {sample[:80]!r}")
    else:
        print(f"NOT FOUND: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())