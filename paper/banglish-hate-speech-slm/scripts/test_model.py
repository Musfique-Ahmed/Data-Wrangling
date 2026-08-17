"""Single-sample test for one model.

Usage:
    python scripts/test_model.py --model qwen3_4b
    python scripts/test_model.py --model gemma3_4b --prompt-version banglish_aware
"""

from __future__ import annotations

import argparse
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

from src.ollama_client import OllamaClient, OllamaError               # noqa: E402
from src.parser import parse_response                                # noqa: E402
from src.monitoring import GpuSampler                                # noqa: E402
from src.utils import load_yaml                                      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["qwen3_4b", "gemma3_4b", "qwen3_1_7b", "llama3_2_3b"])
    ap.add_argument("--prompt-version", default="original",
                    choices=["original", "banglish_aware"])
    ap.add_argument("--text", default=None,
                    help="Override the single-sample text (default: the value "
                         "from the reference notebook)")
    args = ap.parse_args()

    cfg = load_yaml("config/config.yaml")
    model_cfg = cfg["models"][args.model]
    ollama_tag = model_cfg["ollama_model"]
    gen = cfg["generation"]
    ol = cfg["ollama"]

    prompt_path = ROOT / cfg["experiment"]["prompt_files"][args.prompt_version]
    prompt_text = prompt_path.read_text(encoding="utf-8")
    text = args.text or cfg["experiment"]["single_sample_text"]

    per_model = (cfg["generation"].get("per_model_options") or {}).get(args.model, {})
    think_value = per_model.get("think", None)

    client = OllamaClient(
        base_url=ol["base_url"],
        request_timeout_s=ol["request_timeout_s"],
        retries=ol["retries"],
    )
    if not client.health():
        print(f"Ollama unreachable at {ol['base_url']}. Start with: ollama serve")
        return 1
    if not client.has_model(ollama_tag):
        print(f"Model {ollama_tag!r} not pulled. "
              f"Run: python scripts/check_environment.py --pull")
        return 1

    print(f"Model          : {model_cfg['display_name']} ({ollama_tag})")
    print(f"Prompt version : {args.prompt_version}")
    print(f"Input text     : {text!r}")
    print(f"num_predict    : {gen['num_predict']}, temp: {gen['temperature']}")
    print(f"think          : {think_value}")

    sampler = GpuSampler(poll_interval_s=0.5)
    sampler.start()
    t0 = time.perf_counter()
    try:
        resp = client.chat(
            model=ollama_tag,
            messages=[
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": text},
            ],
            options={
                "temperature": gen["temperature"],
                "num_predict": gen["num_predict"],
                "num_ctx": gen["num_ctx"],
                "seed": gen["seed"],
                "top_p": gen["top_p"],
                "top_k": gen["top_k"],
            },
            keep_alive=gen["keep_alive"],
            think=think_value,
        )
        elapsed = time.perf_counter() - t0
        gpu = sampler.stop()
        parsed = parse_response(resp.content)

        print("\n--- raw response ---")
        print(repr(resp.content))
        if resp.thinking:
            print(f"--- thinking ({len(resp.thinking)} chars, first 200) ---")
            print(resp.thinking[:200])
        print("\n--- parsed ---")
        print(f"label         : {parsed.label!r}")
        print(f"reasoning     : {parsed.reasoning!r}")
        print(f"parse_status  : {parsed.parse_status}")
        print("\n--- metrics ---")
        print(f"latency_s     : {elapsed:.3f}")
        print(f"prompt_tokens : {resp.prompt_tokens}")
        print(f"comp_tokens   : {resp.completion_tokens}")
        print(f"gpu_avg_W     : {gpu.avg_power_w}")
        print(f"gpu_vram_MiB  : {gpu.peak_vram_mib}")
        print(f"total_dur_ns  : {resp.total_duration_ns}")
        return 0
    except OllamaError as e:
        print(f"ERROR: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())