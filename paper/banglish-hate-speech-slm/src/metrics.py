"""Latency / throughput statistics."""

from __future__ import annotations

import statistics
from typing import Iterable


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile. Robust for n in [1, ~100k]."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def summary(latencies_s: Iterable[float], total_tokens: int,
            total_runtime_s: float) -> dict:
    """Return the standard latency / throughput summary block."""
    lat = list(latencies_s)
    n = len(lat)
    out = {
        "n": n,
        "total_runtime_seconds": float(total_runtime_s),
        "average_latency_s": (sum(lat) / n) if n else 0.0,
        "median_latency_s": statistics.median(lat) if n else 0.0,
        "p95_latency_s": percentile(lat, 95.0),
        "min_latency_s": min(lat) if n else 0.0,
        "max_latency_s": max(lat) if n else 0.0,
        "total_tokens": int(total_tokens),
        "tokens_per_second": (total_tokens / total_runtime_s) if total_runtime_s > 0 else 0.0,
        "samples_per_minute": (n / total_runtime_s * 60.0) if total_runtime_s > 0 else 0.0,
    }
    return out