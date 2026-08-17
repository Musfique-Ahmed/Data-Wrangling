# METRICS — what we measure, how, and the limits

This document explains every metric that appears in `predictions.csv` and
`run_summary.json`. The goal is **research reproducibility**: anyone reading
the results should be able to reconstruct exactly how each number was
obtained, what its units are, what the failure modes look like, and what is
measured versus estimated.

All inference happens via `src/inference.py`, which calls `src/ollama_client.py`
for HTTP requests and `src/monitoring.py` for GPU sampling. The CSV column
names match `src/utils.py:RESULT_CSV_COLUMNS`.

---

## 1. `latency_seconds`

**What:** Wall-clock seconds between sending the chat request to Ollama and
receiving the response.

**How:**
```python
t0 = time.perf_counter()
resp = client.chat(...)
latency = time.perf_counter() - t0
```

**Units:** seconds, float, rounded to 4 decimal places.

**What it includes:**
- HTTP transport time (loopback, typically <1 ms)
- Ollama request queueing
- Model warm-up (tokenizer, KV cache alloc) on the first request after `keep_alive` expires
- All transformer forward passes for both prompt evaluation and completion generation
- Ollama post-processing (streaming/JSON encode)

**What it does NOT include:**
- Python startup time
- CSV append / log write time (sub-millisecond)
- Inter-request sleep (`runtime.inter_request_sleep_s`)

**Failure mode:** if `client.chat()` raises `OllamaError`, the latency for
that row is still recorded (the elapsed time up to the exception) and the
row is marked `parse_status=request_error`. `latency_seconds` for failed
rows is **included** in summary statistics; remove it with
`df[df.parse_status == 'ok'].latency_seconds` if you want "successful
requests only" stats.

**Limits:**
- Single-threaded; not a true concurrency-safe latency.
- For Qwen3 with `think=true`, latency includes both the hidden
  chain-of-thought generation and the visible answer.

---

## 2. `prompt_tokens` and `completion_tokens`

**What:** Token counts reported by the Ollama server for one chat request.

**How:** extracted from the JSON response:
```python
prompt_tokens     = data.get("prompt_eval_count", 0)
completion_tokens = data.get("eval_count", 0)
```

These come straight from Ollama's inference engine (llama.cpp for Q4_K_M
GGUF models). They count **BPE tokens** that were actually evaluated on the
GPU, not characters.

**Units:** integer token counts.

**Distinction:**
- `prompt_tokens` = tokens evaluated during the prefill phase
  (system prompt + user text + chat-template scaffolding).
- `completion_tokens` = tokens generated during the decode phase
  (the model's response, INCLUDING thinking tokens for Qwen3 when
  `think=true`).
- `total_tokens` = `prompt_tokens + completion_tokens` (computed in
  `ResultRow.finalize`).

**Accuracy:** these are **exact** numbers reported by Ollama itself, not
estimates. They are not derived from a Python tokenizer (e.g. tiktoken),
which would be approximate for non-GPT models.

**Failure mode:** on transport error (`request_error`) both fields are 0
because Ollama never returned a response.

**Caveat for Qwen3:** with `think=true`, `completion_tokens` counts the
hidden chain-of-thought **plus** the visible answer. To get the visible-only
count you would need to tokenize `resp.content` separately, which we do
**not** do — leaving the metric consistent with Ollama's reported totals.
For total cost / energy purposes this is the right number.

---

## 3. VRAM (`gpu.peak_vram_mib` in run_summary, per-row not recorded)

**What:** Maximum GPU memory used during the inference run, in MiB.

**How:** sampled via `nvidia-smi`:
```
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```

Two layers of sampling:
- **Per-request:** `GpuSampler.start()` records a baseline sample;
  `GpuSampler.stop()` records an end-of-request sample. Peak = max across
  samples (currently n=2 per request — coarse).
- **Per-run:** `sample_now()` is called once after the entire model run
  completes; this gives `gpu.peak_vram_mib` in `run_summary.json`.

**Units:** MiB (mebibytes, 2²⁰ bytes).

**What it measures:** total `memory.used` for the GPU, including Ollama's
process, any system overhead, and any other GPU-using processes. We do not
isolate per-process VRAM.

**Limits:**
- Coarse: only 2 samples per request. Bursts shorter than `poll_interval_s`
  (default 1.0 s) may be missed. `poll_interval_s` is configurable in
  `config.yaml > monitoring.poll_interval_s`.
- `memory.used` is GPU-wide. If another process (browser, screen
  recorder, etc.) uses VRAM, our peak reflects the union.
- When `nvidia-smi` is unavailable (e.g. non-NVIDIA system), VRAM is
  reported as `null` and the field is omitted from the summary.

---

## 4. GPU average power (`gpu_avg_W`)

**What:** Mean GPU power draw in watts during the inference run.

**How:** sampled via `nvidia-smi`:
```
nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits
```

`GpuSampler` takes a sample every `poll_interval_s` and reports
`avg_power_w = mean(samples)` where each sample is in watts.

**Units:** watts.

**What it measures:** the entire GPU board power (not just the CUDA cores),
including VRAM, fans, etc. On the RTX 5060 this is read from the
`power.draw` sensor (NVML).

**Failure mode:** `[N/A]` is returned when the sensor cannot read (some
laptops, some WDDM modes). `GpuSample.avg_power_w` is then `None`.

**Important limitation:** per-request power samples are **not currently
aggregated** into the run summary. The run summary's energy estimate uses
`sampler.stop()` output of the *last* request, which is incomplete. To
fix this, Phase 5+ should collect a list of `GpuSample` per request and
average them; for now, the run-level energy estimate uses
`energy.fallback_gpu_power_w` (default 80 W) as a conservative substitute.

**Per-row logging:** the per-row `runtime.log` line records
`gpu_power=X.X` for that row, so post-hoc aggregation is possible from
the log file.

---

## 5. Energy (`energy.estimated_energy_Wh`)

**What:** Estimated electrical energy consumed during the inference run, in
watt-hours.

**How:**
```
energy_Wh = power_W × duration_s / 3600
```

Two paths:
- **measured:** `power_W` comes from `GpuSample.avg_power_w` (nvidia-smi).
- **fallback:** if nvidia-smi is unavailable or returned N/A,
  `power_W = energy.fallback_gpu_power_w` (default 80 W, configurable).

The `power_source` field in `run_summary.json` records which path was used
(`measured` / `fallback` / `unavailable`).

**Units:** watt-hours.

**Limits:**
- "Duration" is currently `sum(latencies)` — wall-clock time the request
  was active. This is approximate; the GPU idle window between requests
  is excluded, but a hot model that keeps VRAM allocated does draw idle
  power that we do **not** account for.
- Power sample count per request is small (n=2). The mean is therefore
  mostly determined by the *end* of the request.
- "GPU power" is board power, not "energy to produce one inference".
  Wall-plug energy (which includes PSU inefficiency, ~85%) is higher; we
  report GPU-side only.

---

## 6. CO₂ (`co2.estimated_co2_g`)

**What:** Estimated CO₂-equivalent emissions in grams for the run.

**How:**
```
CO2_g = energy_Wh × carbon_intensity_gCO2_per_kWh / 1000
```

The carbon intensity is configurable:
- `energy.carbon_intensity_gCO2_per_kWh` (default **475**, IEA 2023 global average).
- `energy.carbon_intensity_source` (default `"IEA 2023 global average (configurable)"`).

To switch to a Bangladesh-specific factor (e.g. ~600 gCO₂/kWh for the
national grid), edit `config/config.yaml`. The change is recorded in the
run summary so the assumption is always auditable.

**Units:** grams of CO₂-equivalent.

**Limits:**
- CO₂ is a derived estimate, not a measurement.
- It only covers the GPU's electrical draw; it does not include embodied
  carbon (manufacturing, transport) or cooling overhead.
- The IEA global average is a *blended* figure (renewables + fossil). For a
  strict local-machine accounting, use the marginal intensity of your
  grid at run time.

---

## 7. Reference-only cloud CO₂ (`api_cloud_co2_reference_only_g`)

**What:** The CO₂ figure that the original `gpt_4o.ipynb` would compute if
the same token counts were billed to OpenAI's cloud.

**How:**
```python
api_cloud_co2_g = (total_p_tok / 1000) * 0.0001 +
                 (total_c_tok / 1000) * 0.0005
```

(Factors from `gpt_4o.ipynb` cell 16; units: g CO₂eq per 1000 tokens.)

**Why kept:** for comparison. These factors assume a cloud LLM and are
**not appropriate** for local-inference accounting — that's why the field
is named `..._reference_only_g` and stored separately from the local
estimate. Never use this number as the experiment's primary CO₂ value.

---

## 8. Throughput metrics

Computed in `src/metrics.py:summary` and stored in
`run_summary.json > latency`:

| Metric | Formula | Units |
|--------|---------|-------|
| `average_latency_s` | `mean(latencies)` | seconds |
| `median_latency_s` | `statistics.median(latencies)` | seconds |
| `p95_latency_s` | linear-interpolated 95th percentile | seconds |
| `min_latency_s`, `max_latency_s` | extremes | seconds |
| `tokens_per_second` | `total_tokens / total_runtime_seconds` | tokens/s |
| `samples_per_minute` | `n / total_runtime_seconds × 60` | samples/min |

Where `total_runtime_seconds` is wall-clock time of the whole run (from
before the first request to after the last). `n` is `n_attempted` (rows
that we actually sent to Ollama), **not** `n_rows_total` (which counts
rows we skipped via resume).

---

## 9. Per-row CSV schema

```
id                  row index in dataset (0-based)
Text                Banglish input (verbatim from source CSV)
predicted_label     canonical label, or "" if invalid
reasoning           free text (or first paragraph)
raw_response        model's raw message.content
parse_status        'ok' | 'invalid_label' | 'parse_error' |
                    'request_error' | 'empty' | 'pending'
error               error message if any (else "")
prompt_tokens       int
completion_tokens   int
total_tokens        int (= prompt + completion)
latency_seconds     float (rounded to 4 dp)
model               display name
experiment          'original' | 'banglish_aware'
timestamp           ISO-8601 UTC at write time
```

---

## 10. Resume semantics

A row is treated as **completed** iff `parse_status == 'ok'`. On restart:
- Rows already in `predictions.csv` with `parse_status='ok'` are skipped.
- Rows with `parse_status='invalid_label'`, `'parse_error'`,
  `'request_error'`, `'empty'`, or missing are retried.

Use `--no-resume` to force re-processing of all rows.

---

## 11. What this pipeline does **not** measure

- GPU temperature (could be added; `nvidia-smi --query-gpu=temperature.gpu`).
- CPU and RAM utilization (would require `psutil`).
- Wall-plug energy (requires a smart plug or RAPL on Intel CPUs).
- Inter-token latency distribution (would require Ollama streaming).
- Carbon cost of training or model download.
- Time spent in the Ollama queue when multiple requests are pending.

All of the above can be added incrementally without changing the schema.