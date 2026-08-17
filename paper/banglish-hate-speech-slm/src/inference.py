"""High-level inference loop with checkpointing, monitoring, and metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .checkpoint import Checkpoint
from .dataset import DatasetRow, load_dataset, iter_rows
from .monitoring import GpuSampler, sample_now
from .ollama_client import OllamaClient, OllamaError
from .parser import parse_response
from .utils import (
    PROJECT_ROOT, ResultRow, load_yaml, utcnow_iso, write_json,
)
from .energy import estimate_energy_wh, estimate_co2_g
from .metrics import summary as latency_summary


@dataclass
class RunOptions:
    model_key: str               # e.g. 'qwen3_4b'
    prompt_version: str = "original"  # 'original' | 'banglish_aware'
    limit: int | None = None
    resume: bool = True
    output_dir: str = "results"
    debug: bool = False
    save_summary: bool = True


@dataclass
class RunSummary:
    model: str
    ollama_tag: str
    display_name: str
    parameters_b: float
    experiment: str
    dataset_path: str
    n_rows_total: int
    n_attempted: int
    successful_samples: int
    failed_samples: int
    invalid_predictions: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    latency: dict
    gpu: dict
    energy: dict
    co2: dict
    carbon_intensity_gCO2_per_kWh: float
    carbon_intensity_source: str
    power_source: str
    api_cloud_co2_reference_only_g: float
    started_at: str
    finished_at: str


def _load_cfg() -> dict:
    return load_yaml("config/config.yaml")


def _prompt_text(cfg: dict, version: str) -> str:
    rel = cfg["experiment"]["prompt_files"][version]
    p = PROJECT_ROOT / rel
    return p.read_text(encoding="utf-8")


def _build_messages(prompt: str, text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]


def run(options: RunOptions) -> RunSummary:
    cfg = _load_cfg()
    model_cfg = cfg["models"][options.model_key]
    ollama_tag = model_cfg["ollama_model"]
    display = model_cfg["display_name"]
    gen = cfg["generation"]
    mon = cfg["monitoring"]
    en = cfg["energy"]
    ol = cfg["ollama"]

    client = OllamaClient(
        base_url=ol["base_url"],
        request_timeout_s=ol["request_timeout_s"],
        retries=ol["retries"],
        backoff_initial_s=ol["backoff_initial_s"],
        backoff_max_s=ol["backoff_max_s"],
    )
    if not client.health():
        raise RuntimeError(
            f"Ollama server unreachable at {ol['base_url']}. "
            "Start it with: ollama serve"
        )
    if not client.has_model(ollama_tag):
        raise RuntimeError(
            f"Model '{ollama_tag}' not pulled. Run "
            f"`python scripts/check_environment.py --pull`."
        )

    prompt_text = _prompt_text(cfg, options.prompt_version)
    per_model = (gen.get("per_model_options") or {}).get(options.model_key, {})
    options_block = {
        "temperature": gen["temperature"],
        "num_predict": per_model.get("num_predict", gen["num_predict"]),
        "num_ctx": gen["num_ctx"],
        "seed": gen["seed"],
        "top_p": gen["top_p"],
        "top_k": gen["top_k"],
    }
    if gen.get("stop"):
        options_block["stop"] = gen["stop"]
    think_value = per_model.get("think", None)

    df = load_dataset(cfg["dataset"]["path"])
    rows: list[DatasetRow] = list(iter_rows(df))
    n_total = len(rows)
    if options.limit is not None:
        rows = rows[: options.limit]

    exp_dir = (
        PROJECT_ROOT
        / options.output_dir
        / f"{options.model_key}_{options.prompt_version}"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = exp_dir / "predictions.csv"
    runtime_log = exp_dir / "runtime.log"
    summary_path = exp_dir / "run_summary.json"

    ck = Checkpoint(pred_csv)
    done_ids = ck.completed_ids() if options.resume else set()
    if options.debug:
        print(f"[resume] {len(done_ids)} rows already completed")

    sampler = GpuSampler(poll_interval_s=mon.get("poll_interval_s", 1.0))
    inter_sleep = cfg["runtime"]["inter_request_sleep_s"]

    latencies: list[float] = []
    total_p_tok = 0
    total_c_tok = 0
    n_success = 0
    n_failed = 0
    n_invalid = 0
    n_attempted = 0
    started = utcnow_iso()
    t_run0 = time.perf_counter()

    with runtime_log.open("a", encoding="utf-8") as logf:
        for row in rows:
            if row.id in done_ids:
                continue
            n_attempted += 1
            t0 = time.perf_counter()
            sampler.start()
            try:
                resp = client.chat(
                    model=ollama_tag,
                    messages=_build_messages(prompt_text, row.text),
                    options=options_block,
                    keep_alive=gen["keep_alive"],
                    think=think_value,
                )
                raw = resp.content
                p_tok = resp.prompt_tokens
                c_tok = resp.completion_tokens
                error = ""
            except OllamaError as e:
                raw = ""
                p_tok = 0
                c_tok = 0
                error = str(e)
            latency = time.perf_counter() - t0
            gpu = sampler.stop()
            latencies.append(latency)

            total_p_tok += p_tok
            total_c_tok += c_tok

            parsed = parse_response(raw) if not error else None
            if error:
                parse_status = "request_error"
                label = ""
                reason = error
                n_failed += 1
            elif parsed is None or not raw:
                parse_status = "empty"
                label = ""
                reason = ""
                n_failed += 1
            else:
                parse_status = parsed.parse_status
                label = parsed.label
                reason = parsed.reasoning
                if parse_status == "ok":
                    n_success += 1
                elif parse_status == "invalid_label":
                    n_invalid += 1
                    n_failed += 1
                else:
                    n_failed += 1

            result = ResultRow(
                id=row.id,
                Text=row.text,
                predicted_label=label,
                reasoning=reason,
                raw_response=raw,
                parse_status=parse_status,
                error=error,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                latency_seconds=round(latency, 4),
            ).finalize(display, options.prompt_version)

            ck.append(result)

            log_line = (
                f"{utcnow_iso()} model={display} row={row.id} "
                f"status={parse_status} latency={latency:.3f}s "
                f"pt={p_tok} ct={c_tok} "
                f"gpu_power={(gpu.avg_power_w or 0):.1f}W "
                f"vram={(gpu.peak_vram_mib or 0)}MiB "
                f"err={error!r}\n"
            )
            logf.write(log_line)
            logf.flush()

            if options.debug or n_attempted % 10 == 0 or parse_status != "ok":
                print(
                    f"  [{display}/{options.prompt_version}] "
                    f"{row.id}/{n_total - 1} status={parse_status} "
                    f"lat={latency:.2f}s ok={n_success} "
                    f"fail={n_failed} invalid={n_invalid}"
                )

            if inter_sleep > 0:
                time.sleep(inter_sleep)

    total_runtime = time.perf_counter() - t_run0
    finished = utcnow_iso()

    avg_power = None
    peak_vram = None
    # Coarse run-level energy: use sum-of-latencies as a proxy for GPU-active
    # time. Per-request power samples are logged in runtime.log but not
    # aggregated here; the fallback power config is used as a safe default.
    # See METRICS.md §4 and §5 for details.
    total_gpu_s = sum(latencies)
    post = sample_now()
    if post:
        peak_vram = post.get("vram_mib")
    energy = estimate_energy_wh(
        GpuSample_dummy(total_gpu_s),
        fallback_power_w=en["fallback_gpu_power_w"],
    )

    lat_summary = latency_summary(latencies, total_p_tok + total_c_tok, total_runtime)
    co2_g = estimate_co2_g(
        energy["estimated_energy_Wh"],
        en["carbon_intensity_gCO2_per_kWh"],
    )

    ref = cfg.get("api_cloud_co2_factors_for_reference_only", {})
    api_co2_g = (
        (total_p_tok / 1000.0) * ref.get("prompt_per_1k_tokens_g", 0)
        + (total_c_tok / 1000.0) * ref.get("completion_per_1k_tokens_g", 0)
    )

    summary = RunSummary(
        model=display,
        ollama_tag=ollama_tag,
        display_name=display,
        parameters_b=float(model_cfg.get("parameters_b", 0)),
        experiment=options.prompt_version,
        dataset_path=cfg["dataset"]["path"],
        n_rows_total=n_total,
        n_attempted=n_attempted,
        successful_samples=n_success,
        failed_samples=n_failed,
        invalid_predictions=n_invalid,
        total_prompt_tokens=total_p_tok,
        total_completion_tokens=total_c_tok,
        total_tokens=total_p_tok + total_c_tok,
        latency=lat_summary,
        gpu={"peak_vram_mib": peak_vram},
        energy=energy,
        co2={
            "estimated_co2_g": co2_g,
            "carbon_intensity_gCO2_per_kWh": en["carbon_intensity_gCO2_per_kWh"],
            "carbon_intensity_source": en["carbon_intensity_source"],
        },
        carbon_intensity_gCO2_per_kWh=en["carbon_intensity_gCO2_per_kWh"],
        carbon_intensity_source=en["carbon_intensity_source"],
        power_source=energy["power_source"],
        api_cloud_co2_reference_only_g=api_co2_g,
        started_at=started,
        finished_at=finished,
    )

    if options.save_summary:
        write_json(summary_path, summary.__dict__)

    return summary


class GpuSample_dummy:
    """Adapter so we can reuse estimate_energy_wh with a synthetic window."""

    def __init__(self, duration_s: float) -> None:
        self.avg_power_w = None
        self.duration_s = float(duration_s)
        self.peak_vram_mib = None
        self.util_pct_avg = None
        self.raw = {}