"""Run all four models sequentially. Used in Phase 5/9."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from src.ollama_client import OllamaClient                               # noqa: E402
from src.utils import load_yaml, write_json                              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-version", default="original",
                    choices=["original", "banglish_aware"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--models", nargs="*",
                    default=["qwen3_4b", "gemma3_4b", "qwen3_1_7b", "llama3_2_3b"])
    ap.set_defaults(resume=True)
    args = ap.parse_args()

    cfg = load_yaml("config/config.yaml")
    summaries = []

    for key in args.models:
        print(f"\n>>> Running {key} ({args.prompt_version})")
        t0 = time.perf_counter()
        try:
            summary = run(RunOptions(
                model_key=key,
                prompt_version=args.prompt_version,
                limit=args.limit,
                resume=args.resume,
                output_dir=args.output_dir,
                debug=False,
            ))
            summary.wall_seconds = time.perf_counter() - t0
            summaries.append(summary.__dict__)
        except RuntimeError as e:
            print(f"  SKIPPED {key}: {e}")
            summaries.append({"model": key, "error": str(e)})

    out_path = Path(args.output_dir) / f"all_models_{args.prompt_version}.json"
    write_json(out_path, summaries)
    print(f"\nCombined summary -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())