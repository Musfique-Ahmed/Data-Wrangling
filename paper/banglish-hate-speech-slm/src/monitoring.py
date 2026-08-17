"""GPU / RAM resource monitoring.

Primary path: ``nvidia-smi`` CSV output parsed once per call. We do NOT
poll in a background thread — that adds complexity for a research script.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class GpuSample:
    avg_power_w: float | None
    peak_vram_mib: int | None
    duration_s: float
    util_pct_avg: float | None
    raw: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _query() -> dict[str, Any] | None:
    """One nvidia-smi CSV sample. None on failure."""
    if not nvidia_smi_available():
        return None
    cmd = [
        "nvidia-smi",
        "--query-gpu=power.draw,utilization.gpu,memory.used,timestamp",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        first = out.stdout.strip().splitlines()[0]
        power_s, util_s, mem_s, ts_s = [c.strip() for c in first.split(",")]
        power = float(power_s) if power_s not in ("[N/A]", "") else None
        util = float(util_s) if util_s not in ("[N/A]", "") else None
        mem = int(mem_s) if mem_s not in ("[N/A]", "") else None
        return {
            "power_w": power,
            "util_pct": util,
            "vram_mib": mem,
            "timestamp": ts_s,
        }
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


class GpuSampler:
    """Coarse-grained GPU power/VRAM sampler.

    Strategy: query nvidia-smi every `poll_interval_s` seconds between
    `start()` and `stop()`. Average power draw is computed across samples;
    peak VRAM is the max sample.
    """

    def __init__(self, poll_interval_s: float = 1.0) -> None:
        self.poll_interval_s = max(0.1, poll_interval_s)
        self._samples: list[dict[str, Any]] = []
        self._t0: float | None = None
        self._stopping: bool = False

    def start(self) -> None:
        self._samples.clear()
        self._stopping = False
        s = _query()
        if s:
            self._samples.append(s)
        self._t0 = time.perf_counter()

    def stop(self) -> GpuSample:
        self._stopping = True
        s = _query()
        if s:
            self._samples.append(s)
        duration = (time.perf_counter() - (self._t0 or time.perf_counter()))
        powers = [x["power_w"] for x in self._samples if x.get("power_w") is not None]
        utils = [x["util_pct"] for x in self._samples if x.get("util_pct") is not None]
        vrams = [x["vram_mib"] for x in self._samples if x.get("vram_mib") is not None]
        return GpuSample(
            avg_power_w=(sum(powers) / len(powers)) if powers else None,
            peak_vram_mib=max(vrams) if vrams else None,
            duration_s=duration,
            util_pct_avg=(sum(utils) / len(utils)) if utils else None,
            raw={"samples": self._samples},
        )


def sample_now() -> dict[str, Any] | None:
    """One-shot snapshot — used for pre/post inference VRAM deltas."""
    return _query()