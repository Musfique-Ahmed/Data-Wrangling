"""Run one model on the full dataset (or a limited slice).

Usage:
    python scripts/run_model.py --model qwen3_4b
    python scripts/run_model.py --model gemma3_4b --limit 10
    python scripts/run_model.py --model qwen3_4b --no-resume
    python scripts/run_model.py --model qwen3_4b --prompt-version banglish_aware
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import RunOptions, run                                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["qwen3_4b", "gemma3_4b", "qwen3_1_7b", "llama3_2_3b"])
    ap.add_argument("--prompt-version", default="original",
                    choices=["original", "banglish_aware"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N rows (for testing).")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--debug", action="store_true")
    ap.set_defaults(resume=True)
    args = ap.parse_args()

    summary = run(RunOptions(
        model_key=args.model,
        prompt_version=args.prompt_version,
        limit=args.limit,
        resume=args.resume,
        output_dir=args.output_dir,
        debug=args.debug,
    ))

    print(json.dumps(summary.__dict__, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())