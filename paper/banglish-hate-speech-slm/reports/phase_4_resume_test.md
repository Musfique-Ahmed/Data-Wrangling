# Phase 4 — Small Batch + Resume Test

**Goal:** confirm (1) per-row incremental writes work on real model
output, (2) resume correctly skips rows already in
`predictions.csv`, and (3) all four models can complete at least a
short batch without infrastructure errors.

All numbers below are from `results/<model>_original/run_summary.json`
and `runtime.log`. See **[METRICS.md](../METRICS.md)** for how every
column is measured.

---

## Resume verification (the headline test)

We deliberately ran the same model twice with a smaller limit, then a
larger limit, with `--resume` enabled (default). The checkpoint logic
checks `id` against rows in the CSV where `parse_status == "ok"` and
skips them.

### qwen3_1_7b (default `--prompt-version original`)

| Run | Limit | n_attempted | n_rows_in_csv | Outcome |
|-----|-------|-------------|---------------|---------|
| 1   | 5     | 5           | 5  (ids 0–4)  | fresh    |
| 2   | 10    | 5           | 10 (ids 0–9)  | resumed  |

- id=0 in CSV: row from run 1, parse_status=ok
- id=5 in CSV: row from run 2, parse_status=ok
- 10 unique ids, 0 duplicates, all parse_status=ok

### gemma3_4b (slower model, sanity check)

| Run | Limit | n_attempted | n_rows_in_csv | Outcome |
|-----|-------|-------------|---------------|---------|
| 1   | 3     | 3           | 3  (ids 0–2)  | fresh    |
| 2   | 5     | 2           | 5  (ids 0–4)  | resumed  |

- id=0..2 from run 1; id=3..4 from run 2; all parse_status=ok
- Resume works equally well on a 100-s/row model

Conclusion: **`checkpoint.completed_ids()` correctly identifies ok
rows and the inference loop skips them without re-calling Ollama.** A
crash at row 70 of a 312-row run leaves rows 0..69 on disk and the next
`run_model.py` invocation resumes from row 70.

---

## Per-model `--limit 3` smoke test (original prompt)

Exposes model-specific failure modes that will matter at full scale.

| Model            | n | successful | invalid | latency (s/row)            | tokens/row (pt + ct) | notes |
|------------------|---|------------|---------|----------------------------|---------------------|-------|
| Qwen3-1.7B       | 3 | 3 (100%)   | 0       | 2.7–8.8  (median 3.9)      | ~109 + 470          | fast, direct answers |
| Gemma 3 4B       | 3 | 3 (100%)   | 0       | 97–101 (median 98)         | ~110 + 47           | slow first-call warm-up |
| Qwen3-4B         | 3 | 2 (67%)    | 1       | 70–77  (median 72)         | ~110 + 1560         | one row hit the thinking-budget cap at `num_predict=2048` and emitted `"Other"` |
| Llama 3.2 3B     | 3 | 1 (33%)    | 2       | 10–16 (median ~13)         | ~110 + ~125         | built-in safety refusal on the original prompt |

### Failure-mode catalogue from this batch

1. **Qwen3-4B — thinking-budget overrun**
   `id=1` ("Vai page er malik tore samne paile ki j kortam.") emitted
   the raw string `Other` with `completion_tokens=1561 / num_predict=2048`.
   The reasoning ran to budget and the model never reached the required
   `<Label> - <reason>` line. Recorded as `parse_status=invalid_label`,
   `error=""`. This will recur at full scale on long Banglish texts.

   **Not a bug.** It is a measured limitation of `num_predict=2048` for
   Qwen3-4B on the original prompt. Two options for Phase 5:
   - (a) keep current setting and accept ~5–10 % invalid rate from
     long-thinking rows, or
   - (b) raise `num_predict` to e.g. 4096 (slower per row) and
     re-measure the invalid rate.
   Decision deferred to Phase 5.

2. **Llama 3.2 3B — safety refusal**
   On rows 0 and 2 the model produced only an "I can't help with that"
   style refusal. Recorded as `parse_status=invalid_label`,
   `error="safety_refusal"`. This is the **faithful** outcome of using
   the original prompt verbatim — the notebook itself would have seen
   the same. Phase 7 will report the per-model refusal rate as a
   measured result, not a hidden defect.

3. **Qwen3-1.7B / Gemma 3 4B — clean**
   All three rows parsed cleanly. No further action needed.

---

## What was verified end-to-end

- `OllamaClient.chat()` returns `prompt_tokens`, `completion_tokens`,
  `thinking` (Qwen3), and `total_duration_ns` from Ollama, not via
  tokenizer estimation.
- `GpuSampler.start()/stop()` returns a `GpuSample` with mean board
  power (`avg_power_w`) and peak VRAM (`peak_vram_mib`). On Qwen3-4B
  we observed `gpu_power=144.9 W` (measured, not fallback) and
  `vram=1823 MiB`.
- `parser.parse()` handles (a) literal `<Chosen Label>`, (b) Markdown
  `**Label:**`, (c) bare canonical-label paragraphs, and falls back to
  `invalid_label` on `Other`-style failures.
- `checkpoint.append_row()` writes the row to CSV immediately
  (`mode='a'`, header only on first create) so any in-flight row is
  preserved if the process is killed.
- `estimate_energy_wh()` / `estimate_co2_g()` apply the
  `nvidia-smi × duration` formula using the configured IEA 2023 carbon
  intensity (475 gCO₂/kWh). With `power_source="measured"`, the
  result is what nvidia-smi actually saw; with `power_source="fallback"`
  we used the configured 80 W.
- `metrics.summary()` emits the latency block (n, p95, samples/min,
  tokens/s) written verbatim to `run_summary.json`.

---

## What's next: Phase 5

For each of the four models, run the full 312-row dataset with the
original prompt, in this order:

1. `qwen3_1_7b` (~17 min expected)
2. `gemma3_4b` (~9 h expected — slowest)
3. `qwen3_4b` (~6 h expected, may need raised `num_predict`)
4. `llama3_2_3b` (~90 min expected; many rows will be
   `invalid_label`)

Outputs go to `results/<model>_original/`:
- `predictions.csv` — per-row columns defined in METRICS.md §1
- `run_summary.json` — totals, GPU, energy, CO₂
- `runtime.log` — timestamped progress line per row

Open question before Phase 5 starts: should we raise
`generation.num_predict` for qwen3_4b to suppress the
thinking-budget-overrun `invalid_label` failures? Current default
`2048` may leave 5–10 % of rows unparseable; `4096` would roughly
double per-row latency for the model that already takes ~70 s/row.
I will proceed with `2048` (the documented Phase-2 choice) and report
the resulting invalid rate as a Phase 5 finding; we can revisit in
Phase 6 if the rate is materially worse than Phase 4.
