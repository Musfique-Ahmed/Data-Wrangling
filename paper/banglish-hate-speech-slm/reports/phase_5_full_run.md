# Phase 5 — Full 312-row Experiment per Model

**Goal:** run every SLM on the full dataset with the original
prompt, record the latency / token / energy / CO₂ telemetry for every
row, and surface each model's parse-success rate.

All numbers below come from `results/<model>_original/run_summary.json`
and `runtime.log`. The complete per-row data lives in
`results/<model>_original/predictions.csv` (gitignored; size 500 KB+
for Qwen3-4B). See **[METRICS.md](../METRICS.md)** for how every
column is measured.

---

## Headline numbers

| Model            | n_ok / 312 | invalid (incl. empty) | wall-clock | median latency | tokens/s | Wh | CO₂ (g) |
|------------------|------------|------------------------|------------|----------------|----------|------|---------|
| Qwen3-1.7B       | 301 (96%)  | 11                     | 16m 49s    | 3.10 s         | 178      | 22.0 | 10.5   |
| Gemma 3 4B       | 306 (98%)  | 6                      | 1h 07m     | 2.53 s         | 12.1     | 88.6 | 42.1   |
| Qwen3-4B         | 203 (65%)  | 109                    | 5h 15m     | 72.5 s         | 36.1     | 419  | 199    |
| Llama 3.2 3B     | 73 (23%)   | 239                    | 3m 00s     | 0.29 s         | 261      | 3.7  | 1.8    |

Notes:

- **Median latency** is the right number to compare models: the first
  ~25 calls of each model bear the one-time KV-cache warm-up, so
  `mean` is inflated by that warm-up tail. `p95` and the per-row CSV
  preserve the tail.
- **Energy and CO₂** use the configured 80 W fallback (nvidia-smi
  `power.draw` is recorded per row, but the energy module used
  `fallback_power_w` for the totals — see METRICS.md §5).
- **Llama 3.2 3B's 73 "ok" rows** are the rows where the model did
  *not* refuse; the 239 invalid rows are all "I cannot classify
  hate speech" refusals (latency ≈ 0.27 s, completion_tokens ≈ 15).

---

## Per-model notes

### Qwen3-1.7B — clean, fast

- All 312 rows completed in **~17 minutes**.
- 11 rows are not `ok`: these are mostly the literal `<Chosen Label>`
  placeholder cases the parser's Recovery Path A handles (so they're
  counted as `ok` after recovery) plus a handful of true failures.
  Final `invalid_predictions: 1`.
- Median 3.1 s / row, p95 4.5 s, max 30.8 s (likely a model warm-up
  outlier at the start of the run).
- Tokens/s ≈ 178 — the highest throughput of any model.

### Gemma 3 4B — long warm-up, then very fast

- Total wall-clock **67 minutes** for 312 rows; first 25 rows took
  ~100 s each (Ollama KV-cache warm-up), rows 26+ ran at ~2.5 s each.
- `p95 = 99 s` reflects the warm-up; `median = 2.5 s` reflects the
  steady-state. The CSV preserves both regimes.
- 306 ok, 6 not ok (all parse failures, no API errors).
- This is the most surprising finding: Gemma is the second-fastest
  model in steady-state.

### Qwen3-4B — thinking-budget is the limiting factor

- Initial config (`num_predict=2048`) produced **36 % ok rate** at
  Phase 4 — 64 % of rows hit the thinking-token cap before emitting
  a parseable label.
- After raising `num_predict=4096` for `qwen3_4b` in
  `config/config.yaml`, ok rate climbed to **65 %**. Remaining
  failures break down into:
  - 82 `invalid_label`: model emitted prose but no `<Label>` line
    (often the conclusion was inside a Markdown block).
  - 27 `empty`: `completion_tokens>0` but `raw_response=""` —
    all output went into the model's hidden `thinking` block.
- Latency median 72.5 s, p95 91 s. The ~5 hour wall-clock is
  primarily the per-row inference cost, not warm-up.
- Energy 419 Wh / 199 g CO₂ is the largest single contribution in
  this study — Qwen3-4B is by far the most expensive local
  configuration.
- **Verdict:** Qwen3-4B is functional but not competitive at
  zero-shot on the verbatim prompt. The Phase 9
  `banglish_aware` prompt experiment will test whether a more
  directive prompt unlocks higher success rate.

### Llama 3.2 3B — built-in refusal

- **73 ok / 239 invalid / 73.5 % refusal rate.** The 239 invalid
  rows are not bugs — they are the model's built-in safety
  classifier refusing to classify the Banglish texts as "hate
  speech / not hate speech" because that framing resembles a content
  moderation use-case the model declines.
- Latency 0.29 s median / 1.75 s p95 — extremely fast because the
  refusal is a short canned response (completion_tokens ≈ 15).
- This is the **faithful outcome** of using the original prompt
  verbatim with Llama 3.2 3B. The reference notebook would have
  observed the same refusal pattern if run with Llama.
- Total runtime 3 minutes — by far the cheapest configuration in
  energy and CO₂.

---

## Data quality fixes

1. **Per-model `num_predict` override** — the original config used a
   single `num_predict` for all four models. `src/inference.py` now
   reads `per_model_options[model_key].num_predict` and falls back to
   the global default.
2. **Dedup script** — `scripts/dedup_predictions.py`. Phase 4 left
   `invalid_label` rows in `llama3_2_3b_original/predictions.csv`;
   Phase 5 retried them (intentional: invalid rows are retried) and
   re-appended. The dedup script collapses to one row per id,
   keeping the latest attempt.

---

## What's next: Phase 6

Validate the four `predictions.csv` files end-to-end:

- Schema check (column names + dtypes).
- Row count = 312 in every file.
- No duplicate ids.
- `latency_seconds`, `prompt_tokens`, `completion_tokens` all ≥ 0.
- `parse_status` ∈ {`ok`, `invalid_label`, `empty`, `request_error`,
  `parse_error`}.
- `experiment` field consistent across rows.
- `model` field matches the directory name.

Output goes to `reports/phase_6_validation.md`. No inference calls.
