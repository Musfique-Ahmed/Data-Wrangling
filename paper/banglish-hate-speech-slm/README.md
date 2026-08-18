# Banglish Hate-Speech SLM Replication

Faithful local-SLM replication of the GPT-4o zero-shot Banglish
hate-speech classification experiment in `gpt_4o.ipynb`, using four
locally running generative SLMs via **Ollama**.

| # | Config key | Ollama tag | Display |
|---|------------|-----------|---------|
| 1 | `qwen3_4b`     | `qwen3:4b`            | Qwen3-4B |
| 2 | `gemma3_4b`    | `gemma3:4b`           | Gemma 3 4B Instruct |
| 3 | `qwen3_1_7b`   | `qwen3:1.7b`          | Qwen3-1.7B |
| 4 | `llama3_2_3b`  | `llama3.2:3b`         | Llama 3.2 3B Instruct |

Hardware target: NVIDIA RTX 5060, 8 GB VRAM, 16 GB system RAM. The
pipeline loads **one** model at a time (`keep_alive=0s`).

## Quick start

```bash
cd banglish-hate-speech-slm
python scripts/check_environment.py --pull    # inspect + pull models
python scripts/test_model.py --model qwen3_4b # single-sample smoke test
python scripts/run_model.py --model qwen3_4b  # full 312-row run
python scripts/run_all_models.py              # all four models
```

Use `--limit 10` to test with a slice, `--no-resume` to force
re-processing, `--prompt-version banglish_aware` to switch prompts.

## Layout

```
banglish-hate-speech-slm/
├── config/config.yaml            # ALL knobs (models, generation, energy)
├── data/data_130.csv             # 312-row Banglish dataset (read-only)
├── prompts/                      # system prompts (original + Banglish-aware)
├── src/                          # reusable library code
├── scripts/                      # CLI entry points
├── notebooks/slm_experiment.ipynb # research-facing notebook
├── results/<model>_<prompt>/     # per-run predictions + summary
├── logs/                         # (not used; per-run logs go to results/)
└── reports/                      # Phase 10 deliverables
```

## Methodology summary

This pipeline is a **faithful** replication of `gpt_4o.ipynb`:

1. Read `Text` from the CSV.
2. Build the zero-shot moderation prompt (verbatim).
3. Send the prompt + the user's Banglish text to Ollama's
   `/api/chat` endpoint.
4. Parse `<Label> - <reasoning>` (mirrors `parse_response` in the
   reference notebook).
5. Record per-row: latency, prompt/completion tokens, GPU power/VRAM,
   parse status.
6. Save incrementally so a crash at row 70 leaves rows 0..69 on disk
   and the next run resumes from row 70.

The CO₂ factors used in the reference notebook
(`PROMPT_CO2_FACTOR=0.0001`, `COMPLETION_CO2_FACTOR=0.0005`) are
**OpenAI-cloud approximations** and are **not** appropriate for local
inference. We preserve them in `run_summary.json` under
`api_cloud_co2_reference_only_g` for comparability, but the primary
local estimates use `nvidia-smi` GPU power samples × duration, with a
configurable carbon intensity (default: IEA 2023 global average).

## Per-model notes discovered during Phase 2

| Model | `think` | num_predict | Behaviour |
|-------|---------|-------------|-----------|
| qwen3_4b | true | 4096 | emits ~1500-2500 thinking tokens, sometimes literal `<Chosen Label>` (parser recovers). Latency ~70 s/row steady-state; ~5h total for 312 rows. ~65% parseable. |
| qwen3_1_7b | true | 2048 | emits ~500 thinking tokens, sometimes literal `<Chosen Label>` (parser recovers). Latency ~3 s/row. ~96% parseable. |
| gemma3_4b | n/a | 2048 | direct answer. Latency ~2.5 s/row after ~25-row warm-up. ~98% parseable. |
| llama3_2_3b | n/a | 2048 | built-in safety refusal on the original prompt (73 % of rows). Recorded as `invalid_label`. Latency ~0.3 s/row. ~23% parseable. |

## Phases

This project is built phase by phase. After each phase, validate and
report before proceeding.

- **Phase 0** — environment inspection (complete)
- **Phase 1** — project scaffold (complete)
- **Phase 2** — Ollama connection (complete)
- **Phase 3** — single-sample test (complete)
- **Phase 4** — small batch + resume test (complete)
- **Phase 5** — full experiment per model (complete)
- **Phase 6** — validation report (complete)
- **Phase 7** — evaluation: distribution, Cohen's κ, χ² (complete)
- **Phase 8** — model comparison table (complete)
- **Phase 9** — Banglish-aware prompt experiment (complete)
- **Phase 10** — final report

## Dataset caveat

The reference notebook reads `data_130.csv`. The CSV supplied for
this experiment is `banglish hate speech dataset - Sheet3.csv`
(312 rows, single `Sentences` column). We do **not** have the
130-row file with the GPT/Gemini/Deepseek columns. See
`data/README.md` for the column-mapping decision and the implication
that Phase 7 will report **distribution-based** metrics rather than
F1/Accuracy.

## Metrics

See **[METRICS.md](METRICS.md)** for a complete description of every
column in `predictions.csv` and every field in `run_summary.json`:
how each metric is measured, its units, its limits, and what the
failure modes look like. Highlights:

- `latency_seconds` — wall-clock from request to response
- `prompt_tokens` / `completion_tokens` — **exact** from Ollama, not estimated
- `peak_vram_mib` — coarse nvidia-smi snapshot
- `gpu_avg_W` — mean board power per inference window
- `estimated_energy_Wh` and `estimated_co2_g` — derived estimates, clearly labelled

## Security note

The reference notebook `gpt_4o.ipynb` (at the repo root) contains an
OpenAI API key in cell 6 and is therefore **not** tracked in git
(`.gitignore`). It is consulted locally for Phase 0 only. The
SLM pipeline does not require any cloud API key.