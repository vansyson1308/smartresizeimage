"""Dependency-free Prometheus text-format metrics."""

from __future__ import annotations

import threading
from collections import defaultdict

_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)


class Metrics:
    """Thread-safe counters, gauges and histograms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._hist: dict[str, list[float]] = {}
        self._hist_sum: dict[str, float] = defaultdict(float)
        self._hist_count: dict[str, int] = defaultdict(int)
        self._help: dict[str, tuple[str, str]] = {}

    def _describe(self, name: str, kind: str, help_text: str) -> None:
        self._help.setdefault(name, (kind, help_text))

    def inc(self, name: str, help_text: str = "", value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._describe(name, "counter", help_text)
            self._counters[(name, tuple(sorted(labels.items())))] += value

    def set_gauge(self, name: str, value: float, help_text: str = "") -> None:
        with self._lock:
            self._describe(name, "gauge", help_text)
            self._gauges[name] = value

    def observe(self, name: str, seconds: float, help_text: str = "") -> None:
        with self._lock:
            self._describe(name, "histogram", help_text)
            buckets = self._hist.setdefault(name, [0.0] * len(_BUCKETS))
            for i, bound in enumerate(_BUCKETS):
                if seconds <= bound:
                    buckets[i] += 1
            self._hist_sum[name] += seconds
            self._hist_count[name] += 1

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            names = sorted(self._help)
            for name in names:
                kind, help_text = self._help[name]
                lines.append(f"# HELP {name} {help_text or name}")
                lines.append(f"# TYPE {name} {kind}")
                if kind == "counter":
                    for (cname, labels), value in sorted(self._counters.items()):
                        if cname != name:
                            continue
                        label_str = ",".join(f'{k}="{v}"' for k, v in labels)
                        suffix = f"{{{label_str}}}" if label_str else ""
                        lines.append(f"{name}{suffix} {value:g}")
                elif kind == "gauge":
                    lines.append(f"{name} {self._gauges.get(name, 0):g}")
                else:
                    buckets = self._hist.get(name, [0.0] * len(_BUCKETS))
                    for bound, count in zip(_BUCKETS, buckets, strict=True):
                        lines.append(f'{name}_bucket{{le="{bound:g}"}} {count:g}')
                    lines.append(f'{name}_bucket{{le="+Inf"}} {self._hist_count[name]}')
                    lines.append(f"{name}_sum {self._hist_sum[name]:.6f}")
                    lines.append(f"{name}_count {self._hist_count[name]}")
        return "\n".join(lines) + "\n"
