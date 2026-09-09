"""Quality contract v2: rendered-output and structural verification.

The legacy Phase 2.1 benchmark metrics (``backend.app.layout.bench_metrics``)
only inspect layout metadata. This package evaluates what the customer sees:
the rendered pixels, the visibility of every content element, and the
legibility of native text. Every check reports an explicit status; a check
that could not run is reported as ``NOT_CHECKED`` and never counts as
positive evidence.
"""

from .contract import (
    CONTRACT_VERSION,
    CheckResult,
    CheckStatus,
    QualityReport,
    Severity,
    Verdict,
)
from .evaluate import QualityConfig, evaluate_composition

__all__ = [
    "CONTRACT_VERSION",
    "CheckResult",
    "CheckStatus",
    "QualityConfig",
    "QualityReport",
    "Severity",
    "Verdict",
    "evaluate_composition",
]
