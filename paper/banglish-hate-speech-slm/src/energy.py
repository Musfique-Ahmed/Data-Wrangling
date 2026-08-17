"""Energy and CO₂ estimation.

Inputs:
  - gpu_sample.duration_s   (window length)
  - gpu_sample.avg_power_w  (W; or None if nvidia-smi failed)
  - config['energy']['fallback_gpu_power_w']
  - config['energy']['carbon_intensity_gCO2_per_kWh']

All values are clearly labelled in run_summary.json so reviewers know
which were measured and which are estimates.
"""

from __future__ import annotations

from .monitoring import GpuSample


def estimate_energy_wh(gpu: GpuSample, fallback_power_w: float) -> dict:
    """Estimate energy in Wh from a `GpuSample` window.

    Returns {"estimated_energy_Wh": float, "power_source": str} where
    `power_source` is one of:
      - "measured":   used the actual nvidia-smi mean power
      - "fallback":   nvidia-smi returned None; used `fallback_power_w`
      - "unavailable": duration was zero or no fallback was provided

    Formula: Wh = W × s / 3600. Limitations documented in METRICS.md §5.
    """
    if gpu.avg_power_w is not None and gpu.duration_s > 0:
        wh = gpu.avg_power_w * gpu.duration_s / 3600.0
        return {"estimated_energy_Wh": wh, "power_source": "measured"}
    if gpu.duration_s > 0:
        wh = fallback_power_w * gpu.duration_s / 3600.0
        return {"estimated_energy_Wh": wh, "power_source": "fallback"}
    return {"estimated_energy_Wh": 0.0, "power_source": "unavailable"}


def estimate_co2_g(energy_wh: float, carbon_intensity_g_per_kwh: float) -> float:
    """Estimate CO₂-eq in grams.

    CO2_g = energy_Wh × intensity_gCO2_per_kWh / 1000.

    `carbon_intensity_g_per_kwh` is configurable in config.yaml; the source
    of the default (IEA 2023 global average ≈ 475) is recorded alongside
    every run in `run_summary.json > co2.carbon_intensity_source`.
    Limitations documented in METRICS.md §6.
    """
    if energy_wh <= 0 or carbon_intensity_g_per_kwh <= 0:
        return 0.0
    return energy_wh * carbon_intensity_g_per_kwh / 1000.0


# Reference-only: the cloud API factors used in gpt_4o.ipynb. NOT applied
# to local results; kept so the report can show both numbers side by side.
def api_cloud_co2_g(prompt_tokens: int, completion_tokens: int,
                    prompt_factor: float, completion_factor: float) -> float:
    return (prompt_tokens / 1000.0) * prompt_factor + \
           (completion_tokens / 1000.0) * completion_factor