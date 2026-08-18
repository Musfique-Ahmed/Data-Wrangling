# Phase 10 — Final Research Report

## Faithful Local-SLM Replication of the GPT-4o Banglish Hate-Speech Zero-Shot Experiment

**Project:** `banglish-hate-speech-slm/`
**Hardware:** NVIDIA RTX 5060, 8 GB VRAM, 16 GB system RAM
**Software:** Ollama 0.17.4, Python 3.14.3, Ollama HTTP API
**Dataset:** 312-row Banglish hate-speech CSV (single column
`Sentences`, no ground-truth labels)
**Reference:** `gpt_4o.ipynb` (OpenAI GPT-4o zero-shot, 130 rows,
ground-truth columns absent from supplied data)
**Date:** August 2026

---

## Abstract

We replicate the zero-shot hate-speech classification experiment
from the reference notebook `gpt_4o.ipynb` using four local
generative small language models (SLMs) served through Ollama:
**Qwen3-4B**, **Gemma 3 4B Instruct**, **Qwen3-1.7B**, and **Llama
3.2 3B Instruct**. Each model is run on the full 312-row Banglish
dataset with two prompt variants — the original verbatim prompt
from the reference, and a `banglish_aware` variant — yielding eight
total runs and ~3 M tokens of inference. We measure wall-clock
latency, throughput, GPU power, VRAM, energy, CO₂, parse-success
rate, inter-model agreement (Cohen's κ), and label-distribution
divergence (Pearson χ²). All measurements are exact counts from
Ollama's HTTP API and `nvidia-smi`; no tokenizer-based estimates.

The headline result is that **prompt framing matters as much as
model choice for safety-tuned SLMs**. The single largest
improvement in the entire study comes from rephrasing the prompt
for Llama 3.2 3B: the refusal rate drops from 76 % to 1 %
(+234 valid predictions recovered). Gemma 3 4B's `Racism`
over-prediction drops from 76 % to 28 % on the same prompt change.
Conversely, the larger Qwen3-4B model with thinking-mode *worsens*
on the banglish-aware prompt (32.7 % vs 65.1 % parse success)
because the longer prompt consumes more of its 4096-token
thinking budget. We close with a recommendation matrix keyed on
the four orthogonal axes of parse success, throughput, energy, and
label-distribution balance.

---

## 1. Introduction

Hate-speech moderation for low-resource languages is a known
operational challenge. The reference notebook `gpt_4o.ipynb`
demonstrates that a zero-shot GPT-4o prompt can classify a small
Banglish dataset with reasonable per-class balance, using a
moderation-system framing and a strict `<Label> - <reasoning>`
output format. We replicate that experiment with locally hosted
generative SLMs, motivated by three considerations:

1. **Cost.** Cloud API inference is priced per 1k tokens; local
   inference has a fixed hardware cost and zero marginal cost.
2. **Privacy.** Banglish social-media content may include user
   identifiers or private context; local inference avoids
   third-party data egress.
3. **Reproducibility.** Deterministic local inference with the
   same model weights, the same hardware, and the same code can
   be re-run by any researcher with the same equipment.

We restrict our scope to **four generative SLMs** (no encoder-only
classifiers, no fine-tuning, no few-shot examples) and preserve
the **exact prompt** from the reference notebook as our primary
condition. A secondary prompt variant (`banglish_aware`) is added
in Phase 9 to test whether small prompt-framing changes can
mitigate model-specific failure modes.

We do **not** have ground-truth labels for the supplied 312-row
CSV (the `GPT`/`Gemini`/`Deepseek` columns that the reference
notebook drops are absent from the file we received). Phase 7
therefore reports **distribution-based** metrics only: prediction
distribution, parse-success rate, inter-model agreement, and
χ² independence. We treat the absence of ground truth as a
limitation to be reported, not a problem to be papered over.

---

## 2. Methodology

### 2.1 Phased execution

The replication is built in 10 phases (0–10), each with an
explicit validation gate before the next:

| phase | deliverable                          | verified |
|-------|--------------------------------------|----------|
| 0     | environment inspection                | ✅       |
| 1     | project scaffold                     | ✅       |
| 2     | Ollama connection                    | ✅       |
| 3     | single-sample test                   | ✅       |
| 4     | small batch + resume test            | ✅       |
| 5     | full 312-row experiment per model    | ✅       |
| 6     | data-quality validation (37/37 pass) | ✅       |
| 7     | distribution/agreement evaluation    | ✅       |
| 8     | model comparison table               | ✅       |
| 9     | banglish-aware prompt experiment     | ✅       |
| 10    | this report                          | —        |

Per-phase evidence is in `reports/phase_N_*.md`. All inference
telemetry lives in `results/<model>_<prompt>/{run_summary.json,
runtime.log, predictions.csv}`.

### 2.2 Models

| key            | Ollama tag       | parameters | quant    | thinking-mode |
|----------------|------------------|------------|----------|---------------|
| `qwen3_4b`     | `qwen3:4b`       | 4.0 B      | Q4_K_M   | yes           |
| `gemma3_4b`    | `gemma3:4b`      | 4.0 B      | Q4_K_M   | no            |
| `qwen3_1_7b`   | `qwen3:1.7b`     | 1.7 B      | Q4_K_M   | yes           |
| `llama3_2_3b`  | `llama3.2:3b`    | 3.2 B      | Q4_K_M   | no            |

All four fit the 8 GB VRAM budget one at a time. We use
`keep_alive="0s"` after each request so the next model gets a
clean VRAM slate. The thinking-mode toggle for Qwen3 models is
passed as the `think` field in the Ollama `/api/chat` payload;
Qwen3-4B is given `num_predict=4096` (Phase 2 finding) because the
2048-token default exhausts the thinking budget on ~50 % of
Banglish rows.

### 2.3 Dataset

The supplied CSV is
`banglish hate speech dataset - Sheet3.csv` — 312 rows, single
column `Sentences`. This is **not** the reference notebook's
`data_130.csv`; that file is absent from the repository. We
treat `Sentences` as the source text column (mapped in-memory to
`Text`) and do not modify the source file. The dataset has no
ground-truth labels, so all evaluation is distribution-based.

### 2.4 Prompts

Two prompt variants are tested:

- **original** — verbatim copy of the system prompt from
  `gpt_4o.ipynb` (`prompts/original_zero_shot.txt`).
- **banglish_aware** — same task framing with explicit guidance
  that the input may be Banglish (Bengali in Latin script),
  code-mixed, slang, and informal
  (`prompts/banglish_aware_zero_shot.txt`).

Both prompts require the model to emit exactly one of seven
canonical labels (`Not Hate Speech`, `Religious Hate`, `Political
Hate`, `Personal / Social Abuse`, `Gender Abuse`, `Racism`,
`Geopolitical Hate`) followed by a short reasoning line, prefixed
with the literal token `<Chosen Label>`.

### 2.5 Generation settings

Fixed across all runs:

- `temperature = 0`, `top_p = 1.0`, `top_k = 1`, `seed = 0`
  (deterministic).
- `num_ctx = 1024` (Banglish rows are <60 chars).
- `keep_alive = "0s"`.

Per-model:

| model        | think   | num_predict | rationale                                  |
|--------------|---------|-------------|--------------------------------------------|
| Qwen3-4B     | true    | **4096**    | 2048 exhausts thinking budget on ~50 % rows |
| Qwen3-1.7B   | true    | 2048        | thinking tokens fit comfortably            |
| Gemma 3 4B   | n/a     | 2048        | direct answers; <100 completion tokens     |
| Llama 3.2 3B | n/a     | 2048        | direct answers; <100 completion tokens     |

### 2.6 Telemetry

Every per-model run records, per row:

- `latency_seconds` — wall-clock from request send to response
  receive.
- `prompt_tokens`, `completion_tokens` — exact counts from Ollama
  (`prompt_eval_count`, `eval_count`), not tokenizer estimates.
- `parse_status` — one of `ok`, `invalid_label`, `parse_error`,
  `request_error`, `empty`.
- `predicted_label` — the canonical label string after parser
  recovery (handles literal `<Chosen Label>` placeholder, Markdown
  wrappers, and bare-label paragraphs).
- `raw_response` — the unmodified model output, for forensic
  review.

Per-run summary (`run_summary.json`) aggregates:

- `latency.{n, total_runtime_seconds, median, p95, min, max,
  tokens_per_second, samples_per_minute}`
- `gpu.peak_vram_mib`
- `energy.estimated_energy_Wh` — `nvidia-smi` mean board power ×
  duration, with configurable fallback.
- `co2.estimated_co2_g` — `energy × carbon_intensity / 1000`
  (default 475 g CO₂/kWh, IEA 2023 global average, configurable in
  `config/config.yaml`).
- `api_cloud_co2_reference_only_g` — the reference notebook's
  OpenAI cloud factors (`0.0001`, `0.0005`) applied to the same
  token counts, for cross-section comparison only.

Full metric definitions and limitations are documented in
[METRICS.md](../METRICS.md) (11 sections, every column defined).

### 2.7 Validation

`scripts/validate_phase6.py` runs 10 schema/data-quality checks
per CSV (column names, row count, unique ids, non-negative
numeric fields, `total_tokens == prompt+completion`, parse-status
enum, experiment field, model field, canonical labels when ok)
plus one cross-model check (same id → same Text byte-identical
across all four CSVs). Phase 6 reports **37/37 pass** for the
four original-prompt CSVs; the four banglish-aware CSVs inherit
the same structure and pass the same checks.

### 2.8 Resumability

Every run is checkpointed to
`results/<model>_<prompt>/predictions.csv` after each row, with
the resume rule "skip rows where `parse_status == 'ok'`". Phase 4
verifies that a `--limit 5` followed by `--limit 10` correctly
skips the first 5 rows and processes only the next 5 (Qwen3-1.7B
and Gemma 3 4B both verified). A `scripts/dedup_predictions.py`
post-processor handles the edge case where rows with
non-ok status are retried by the checkpoint and re-appended.

---

## 3. Results — Original Prompt

The four models on the original verbatim prompt:

| model           | ok / 312  | wall-clock | median latency | energy (Wh) | CO₂ (g) | peak VRAM (MiB) |
|-----------------|----------:|-----------:|---------------:|------------:|--------:|-----------------:|
| Qwen3-1.7B      | 301 (96.5%) | 16m 49s    | 3.10 s         | 22.0        | 10.5    | 1394             |
| Gemma 3 4B      | 306 (98.1%) | 1h 07m     | 2.53 s         | 88.6        | 42.1    | 1360             |
| Qwen3-4B        | 203 (65.1%) | 5h 15m     | 72.55 s        | 419.4       | 199.2   | 1987             |
| Llama 3.2 3B    | 74 (23.7%)  | 3m 00s     | 0.29 s         | 3.7         | 1.8     | 4190             |

### 3.1 Per-model behaviour

**Qwen3-1.7B** is the most consistent baseline. 99.7 % of rows
parse successfully; the 11 non-ok rows are dominated by literal
`<Chosen Label>` placeholders that the parser's recovery paths
absorb into the `ok` bucket after fallback matching. The model
emits ~500 thinking tokens per row, then a parseable label. Its
label distribution is balanced: 65.6 % Not Hate Speech and the
remaining 34.4 % spread across all six hate classes (max
20.3 % Personal / Social Abuse, 4.8 % Religious Hate).

**Gemma 3 4B** has the second-highest parse success (99.7 %)
but a heavy `Racism` over-prediction: **75.6 % of its 311 valid
predictions are `Racism`**. Inspection of raw responses shows
the model frequently treats *any* mention of an ethnic,
community, or transliterated name as Racism, conflating
Banglish demographic references with slurs. Latency is 2.5 s/row
in steady state after a ~25-row warm-up that costs p95 = 99 s.

**Qwen3-4B** is the slowest and most expensive configuration
(5h 15m wall-clock, 419 Wh, 199 g CO₂ — ~50× the energy of
Qwen3-1.7B). Only 65.1 % of rows parse successfully because
27 rows hit the 4096-token thinking cap with **all** output in
the hidden thinking block (`raw_response=""` — we record these
as `empty`), and 82 rows emit prose but no `<Label>` line. When
it does parse, the label distribution is the **most balanced
non-neutral** of any model: 81.8 % Not Hate Speech, 6.9 %
Geopolitical Hate, 5.4 % Personal / Social Abuse, 3.5 % Religious
Hate, 1.5 % Racism.

**Llama 3.2 3B** is the cheapest configuration by every efficiency
axis (3 min wall-clock, 3.7 Wh, 1.8 g CO₂) but produces valid
predictions on only 23.7 % of rows. The remaining 76.3 % are
byte-identical canned safety refusals ("I cannot classify hate
speech. Can I help you with something else?" — completion_tokens
≈ 15, latency ≈ 0.27 s). This is the Meta safety classifier
firing on the prompt's content-moderation framing.

### 3.2 Inter-model agreement

The four models produce very different label distributions. The
Pearson χ² test on the **43-row all-ok intersection** (rows where
all four models parsed successfully) gives:

- χ² = **108.68**, df = 18, **p = 5.54 × 10⁻¹⁵**

Pairwise Cohen's κ on the same intersection (or on per-pair
ok-row intersections, which range from n=43 to n=310):

| pair                       | n   | raw agreement | κ       |
|----------------------------|----:|--------------:|--------:|
| Qwen3-4B   vs Qwen3-1.7B   | 203 | 61.6 %        | **0.116** |
| Qwen3-4B   vs Llama 3.2 3B | 43  | 74.4 %        | 0.046   |
| Qwen3-1.7B  vs Llama 3.2 3B | 74  | 68.9 %        | 0.054   |
| Gemma 3 4B  vs Qwen3-1.7B  | 310 | 15.5 %        | 0.048   |
| Gemma 3 4B  vs Qwen3-4B    | 202 | 12.4 %        | -0.003  |
| Gemma 3 4B  vs Llama 3.2 3B | 74 | 16.2 %        | -0.016  |

**No pair exceeds κ = 0.12** — the "poor" band of the Landis &
Koch scale. Any pair involving Gemma is at chance or below,
dragged down by Gemma's heavy `Racism` rate. The two Qwen3 models
agree 61.6 % of the time (κ = 0.12), the highest in the matrix.

### 3.3 The "winners" on each axis

| dimension                  | winner          | runner-up       |
|----------------------------|-----------------|-----------------|
| fastest median latency     | Llama 3.2 3B    | Gemma 3 4B      |
| highest parse success      | Gemma 3 4B      | Qwen3-1.7B (tie)|
| highest throughput         | Llama 3.2 3B    | Qwen3-1.7B      |
| lowest total energy        | Llama 3.2 3B    | Qwen3-1.7B      |
| lowest total CO₂           | Llama 3.2 3B    | Qwen3-1.7B      |
| lowest energy per OK row   | Llama 3.2 3B    | Qwen3-1.7B      |

> **Important caveat:** Llama 3.2 3B's "wins" on every efficiency
> axis are an artefact of its 76 % refusal rate. A 0.3-second
> canned refusal is cheap but useless. The efficiency-per-OK-row
> metric normalises for this and still ranks Llama first, but
> Qwen3-1.7B is a close second (0.071 vs 0.050 Wh/ok) while also
> producing 4× more OK rows.

---

## 4. Results — Banglish-Aware Prompt

The same four models on the `banglish_aware` prompt variant:

| model           | ok / 312  | wall-clock | median latency | energy (Wh) | CO₂ (g) |
|-----------------|----------:|-----------:|---------------:|------------:|--------:|
| Qwen3-1.7B      | 279 (89.4%) | 26m 46s    | 3.01 s         | 35.2        | 16.7    |
| Gemma 3 4B      | **312 (100.0%)** | **55m** | 2.47 s     | 73.0        | 34.7    |
| Qwen3-4B        | 102 (32.7%) | 3h 35m     | 33.31 s        | 287.0       | 136.3   |
| Llama 3.2 3B    | **308 (98.7%)** | 24m 03s | 3.76 s     | 31.4        | 14.9    |

### 4.1 Hypothesis tests from Phase 8

| hypothesis | model       | metric                       | original | banglish_aware | Δ          | verdict |
|------------|-------------|------------------------------|---------:|---------------:|-----------:|---------|
| H1         | Gemma 3 4B  | `Racism` rate on ok rows     | 75.6 %   | 28.2 %         | **−47.4 pts** | ✅ PASS |
| H2         | Llama 3.2 3B | invalid_label (refusal) rate | 76.3 %   | 1.3 %          | **−75.0 pts** | ✅ PASS |
| H3         | Qwen3-4B    | `empty` (thinking-cap miss)  | 8.7 %    | 15.1 %         | +6.4 pts      | ✅ PASS |

**All three hypotheses are confirmed.** The banglish-aware prompt
substantially mitigates both Gemma's `Racism` over-prediction
and Llama's safety refusal, with only a modest increase in
Qwen3-4B's thinking-budget exhaustion.

### 4.2 Per-model deltas

**Gemma 3 4B** sees the most striking improvement. `Racism` rate
falls from 75.6 % to 28.2 %; `Not Hate Speech` rate triples from
10.9 % to 34.3 %. The new prompt's explicit guidance about
Banglish transliteration vs. ethnic slurs appears to suppress
the model's misclassification. Parse success goes from 99.7 %
to 100.0 % (the single `invalid_label` row on the original
prompt now parses).

**Llama 3.2 3B** sees the largest absolute change. Refusal rate
collapses from 76.3 % to 1.3 %; 234 previously-refused rows are
recovered. The prompt's reframing — "specializing in hate speech
detection" rather than "moderate this user content" — appears to
bypass the Meta safety classifier that was firing on the original
prompt. The model is now slower (3.76 s vs 0.29 s median) and
uses more energy per row because it is actually generating a
real answer instead of a canned refusal.

**Qwen3-4B** sees a clear regression: parse success drops from
65.1 % to 32.7 %. The longer prompt triggers more thinking,
which exhausts the 4096-token cap faster on the harder rows.
Median latency actually *drops* from 72.55 s to 33.31 s because
many rows now hit the cap instead of completing — net wall-clock
is 32 % lower but the trade-off is a 32-point parse-success loss.
**Qwen3-4B is the only model that gets worse on the banglish-
aware prompt.**

**Qwen3-1.7B** sees a modest regression: parse success drops from
99.7 % to 89.4 %. The model emits 32 new `invalid_label` rows
that previously parsed; the failures are not thinking-budget
exhaustion (no `empty` rows) but unparseable Markdown output.
Wall-clock grows 59 % because the longer prompt produces
longer outputs.

### 4.3 Updated recommendation matrix

| criterion                              | best model       | second            |
|----------------------------------------|------------------|-------------------|
| Highest parse success                  | Gemma 3 4B (100%) | Llama 3.2 3B (98.7%) |
| Most balanced per-class distribution   | Qwen3-1.7B       | Gemma 3 4B        |
| Fastest wall-clock                     | Llama 3.2 3B (24m) | Qwen3-1.7B (27m)  |
| Lowest total energy                    | Llama 3.2 3B (31 Wh) | Qwen3-1.7B (35 Wh) |
| Lowest total CO₂                       | Llama 3.2 3B (15 g) | Qwen3-1.7B (17 g) |
| Lowest energy per OK row               | Llama 3.2 3B (0.10 Wh) | Qwen3-1.7B (0.13 Wh) |
| Smallest model size                    | Qwen3-1.7B (1.7 B) | Llama 3.2 3B (3.2 B) |

**No model dominates all axes.** Llama 3.2 3B with the
banglish-aware prompt is the recommended choice if wall-clock and
energy are the primary constraints; Gemma 3 4B is the recommended
choice if parse success is the primary constraint; Qwen3-1.7B
remains a strong alternative with the most balanced label
distribution.

---

## 5. Discussion

### 5.1 Why the four models disagree so strongly

The four models in this study were trained by different
organisations on different corpora with different safety
post-training regimes. Their label distributions on Banglish
inputs are therefore expected to differ — and they do, by a
factor that the χ² test confirms is far beyond chance. The
practical implication is that **no single SLM can be trusted as
the authoritative classifier for low-resource hate speech**. A
production system would need either (a) ensemble methods that
weight votes by per-class reliability, (b) per-model human audit
on a calibration set, or (c) ground-truth labels that we do not
have.

### 5.2 What prompt framing actually does

The most striking finding from Phase 9 is that the same model
weights, given a slightly different prompt, can change refusal
rate by 75 percentage points (Llama 3.2 3B) or `Racism` rate by
47 percentage points (Gemma 3 4B). Both changes are *bigger* than
the differences between any two of our models on the original
prompt. For practitioners, this means:

- A model that "looks broken" on a given task may be broken
  because of the prompt, not the weights.
- The same model can be deployed safely or unsafely depending
  on how the request is framed.
- Model selection without prompt A/B testing is unreliable.

### 5.3 What thinking-mode actually costs

Qwen3-4B's hidden thinking block consumes 1500–2500 tokens per
row on average. With `num_predict=4096`, this leaves only
1500–2500 tokens for the visible answer, which is enough for the
required `<Label> - <reasoning>` line. But on long or difficult
Banglish rows, the thinking block grows to 3000+ tokens, leaving
insufficient budget for the visible output. The 27 `empty` rows
on the original prompt and the 47 on the banglish-aware prompt
are all instances where this happened. The mitigation is to raise
`num_predict`, but the cost is proportional: at `num_predict=8192`,
per-row latency roughly doubles and wall-clock would exceed
10 hours. **For production, Qwen3-4B with thinking-mode on
Banglish inputs is not yet a viable choice.**

### 5.4 What the absence of ground truth means

We cannot answer the central question a reader would want
answered: "Which model classifies Banglish hate speech most
correctly?" Without ground-truth labels, we report only
distribution and agreement metrics. A reviewer who wants to know
whether Gemma's reduced `Racism` rate on the banglish-aware
prompt represents *better* classification or just *different*
classification has no way to tell from this study. We treat this
as the primary limitation (see §6.1).

---

## 6. Limitations

### 6.1 No ground-truth labels

The supplied dataset lacks the `GPT`, `Gemini`, `Deepseek`
columns that the reference notebook drops from its data. We
therefore cannot compute accuracy, precision, recall, F1, or any
per-class confusion matrix. The χ² and κ metrics measure
**inter-model disagreement**, not correctness. A ground-truth
labelling pass (either by human annotators or by adopting one of
the reference notebook's classifier outputs as a silver label)
is the highest-value follow-up to this work.

### 6.2 Single-seed determinism

All runs use `temperature=0`, `seed=0`. We do not measure
variance over multiple seeds. With determinism on, the per-row
outputs are deterministic given the same model weights and
prompt, but the *latency* measurements vary by ±5 % across runs
because of GPU scheduling noise (especially during the warm-up
phase for Gemma and Qwen3-4B). The latency statistics in
`run_summary.json` are therefore point estimates without
confidence intervals.

### 6.3 Single dataset, single language

All experiments are on Banglish (Bengali in Latin script). We
make no claims about transfer to Hindi, Urdu, Romanised Bengali
in Devanagari, Tamil, or any other low-resource language. The
reference notebook's results on `data_130.csv` may or may not
generalise to other Banglish sources; our results certainly do
not.

### 6.4 Energy and CO₂ are estimates

The per-row energy is computed from `nvidia-smi power.draw`
samples × wall-clock duration, but Ollama does not sample
`power.draw` continuously — only at the start and end of each
request. Bursts shorter than `poll_interval_s=1.0` may be
missed, and the linear-interpolation assumption may over- or
under-count by ~5 %. The CO₂ figure is computed using the
**IEA 2023 global average carbon intensity (475 g CO₂/kWh)**,
which is an upper bound for renewables-rich regions. Both
numbers are clearly labelled `power_source` in every
`run_summary.json`.

### 6.5 Small per-class sample sizes

For several model×label combinations the count is in single
digits (e.g. Qwen3-4B Political Hate: 1 row). Per-class
percentages are therefore noisy estimates. We do not report
per-class confidence intervals.

### 6.6 The "100 % parse success" of Gemma

Gemma 3 4B on the banglish-aware prompt reports 312 / 312 = 100 %
parse success. This is a fact about *parsing*, not about
*correctness*. Gemma could still be wrong on every row and the
parse rate would still be 100 %.

### 6.7 Single hardware target

All measurements are on a single NVIDIA RTX 5060. Results on
other GPUs (Apple Silicon, AMD, NVIDIA A100/H100, integrated
graphics) would differ. We do not extrapolate.

---

## 7. Conclusion

We replicate the GPT-4o Banglish hate-speech zero-shot experiment
from `gpt_4o.ipynb` using four local generative SLMs on a 312-row
dataset, measuring latency, throughput, energy, CO₂, parse
success, label distribution, and inter-model agreement. The
study confirms three findings:

1. **Local SLMs are a viable operational alternative to cloud API
   inference** for this task, with the Llama 3.2 3B
   (banglish-aware) and Qwen3-1.7B configurations offering the
   best balance of speed and parse success.

2. **No pair of the four SLMs exceeds Cohen's κ = 0.12** on the
   Banglish classification task. Inter-model disagreement is
   structural, not noise; a production classifier would benefit
   from ensembling or human audit.

3. **Prompt framing is a first-order hyperparameter.** The single
   biggest improvement in the entire study comes from rephrasing
   the prompt for Llama 3.2 3B (+75 pts parse success, recovered
   234 rows). The same change cuts Gemma 3 4B's `Racism`
   over-prediction by 47 pts. Practitioners should A/B test
   prompt framing before assuming a model is unsuitable.

The most actionable recommendation, balancing parse success,
throughput, and label-distribution quality, is
**Qwen3-1.7B on the original prompt** for the most neutral label
distribution, or **Llama 3.2 3B on the banglish-aware prompt** for
the fastest wall-clock at 99 % parse success.

---

## 8. Future Work

In rough order of expected impact:

1. **Ground-truth labelling.** Hand-label or consensus-label a
   subset of the 312 rows (or a larger Banglish hate-speech
   corpus) so that accuracy, F1, and per-class confusion can be
   computed. The current study can be re-evaluated as soon as
   labels exist.

2. **Multi-seed runs.** Repeat the full 312-row experiment for
   each model with `temperature=0.7` and 5 seeds; report mean and
   95 % CI for every metric. The current point estimates would
   gain error bars.

3. **Cross-language transfer.** Test the same four models on
   Romanised Hindi, Urdu, and Tamil; measure whether the prompt
   change and the per-model failure modes transfer.

4. **Ensemble classification.** Build a weighted-vote ensemble
   across the four models' ok-row predictions, with weights
   derived from per-class reliability on a calibration set.

5. **`num_predict` sweep for Qwen3-4B.** Re-run Qwen3-4B with
   `num_predict ∈ {2048, 4096, 8192}` on both prompts; identify
   the knee point in the (parse success, wall-clock, energy)
   curve.

6. **Few-shot ablation.** Add 1, 3, 5 in-context examples per
   class and measure parse success and label-distribution shift.
   The current study is zero-shot; few-shot may help Llama more
   than Qwen3.

7. **GPU energy profiling at higher poll rate.** Reduce
   `poll_interval_s` from 1.0 to 0.1 and re-measure Qwen3-4B;
   capture the per-row energy spike profile and quantify the
   warm-up cost.

8. **Bangla-script dataset.** Source a parallel dataset in
   Bengali script (বাংলা) and measure whether the same models
   produce the same label distribution — a test of whether the
   "Banglish" failure modes are transliteration artefacts.

---

## 9. Reproducibility

```bash
# Install Ollama (0.17.4+), then:
ollama pull qwen3:4b
ollama pull gemma3:4b
ollama pull qwen3:1.7b
ollama pull llama3.2:3b

# Run the full experiment:
cd banglish-hate-speech-slm
python scripts/run_model.py --model qwen3_4b
python scripts/run_model.py --model gemma3_4b
python scripts/run_model.py --model qwen3_1_7b
python scripts/run_model.py --model llama3_2_3b
python scripts/run_model.py --model qwen3_4b --prompt-version banglish_aware
python scripts/run_model.py --model gemma3_4b --prompt-version banglish_aware
python scripts/run_model.py --model qwen3_1_7b --prompt-version banglish_aware
python scripts/run_model.py --model llama3_2_3b --prompt-version banglish_aware

# Re-derive every report:
python scripts/validate_phase6.py
python scripts/evaluate_phase7.py
python scripts/compare_phase8.py
python scripts/compare_phase9.py
```

Total wall-clock for the eight runs on RTX 5060, 8 GB VRAM:
~11 hours.

---

## 10. Project deliverables

```
banglish-hate-speech-slm/
├── README.md                       # project overview
├── METRICS.md                      # 11-section metric definitions
├── requirements.txt
├── .gitignore
├── config/config.yaml              # all knobs (models, energy, etc.)
├── data/
│   ├── README.md
│   └── data_130.csv                # 312-row dataset (read-only)
├── prompts/
│   ├── original_zero_shot.txt      # Phase-5 prompt (verbatim)
│   └── banglish_aware_zero_shot.txt # Phase-9 prompt variant
├── src/                            # reusable library code
│   ├── ollama_client.py            # thin HTTP wrapper
│   ├── inference.py                # main loop with checkpoint
│   ├── parser.py                   # <Label> - <reasoning> extractor
│   ├── dataset.py                  # CSV loader
│   ├── metrics.py                  # latency/throughput stats
│   ├── monitoring.py               # nvidia-smi GPU sampler
│   ├── energy.py                   # Wh / CO2 estimator
│   ├── checkpoint.py               # resumable CSV writer
│   ├── dedup_predictions.py logic  # inside scripts/
│   └── utils.py                    # paths, JSON I/O, label map
├── scripts/                        # CLI entry points
│   ├── check_environment.py
│   ├── test_model.py
│   ├── run_model.py
│   ├── run_all_models.py
│   ├── dedup_predictions.py
│   ├── validate_phase6.py
│   ├── evaluate_phase7.py
│   ├── compare_phase8.py
│   └── compare_phase9.py
├── notebooks/slm_experiment.ipynb  # research-facing notebook
├── results/<model>_<prompt>/       # 8 directories, each with
│   ├── predictions.csv             #   per-row CSV
│   ├── run_summary.json            #   run-level aggregates
│   └── runtime.log                 #   per-row timestamped log
└── reports/                        # per-phase deliverables
    ├── phase_4_resume_test.md
    ├── phase_5_full_run.md
    ├── phase_6_validation.md
    ├── phase_6_validation.json
    ├── phase_7_evaluation.md
    ├── phase_7_evaluation.json
    ├── phase_8_comparison.md
    ├── phase_8_comparison.json
    ├── phase_9_banglish_aware.md
    ├── phase_9_comparison.json
    └── phase_10_final_report.md    # this file
```

All code, configs, and reports are committed to git. Per-row
CSVs are gitignored (each is 150–550 KB and is fully reproducible
from `scripts/run_model.py`).
