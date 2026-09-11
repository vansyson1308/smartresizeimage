"""Typed quality contract shared by benchmark, production pipeline and UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

CONTRACT_VERSION = "2.0.0"


class CheckStatus(str, Enum):
    """Outcome of a single check.

    ``NOT_CHECKED`` means the check could not run (missing dependency, missing
    input). It is never treated as a pass.
    """

    PASS = "pass"
    FAIL = "fail"
    NEEDS_REVIEW = "needs_review"
    NOT_CHECKED = "not_checked"


class Severity(str, Enum):
    """How much a failing check matters to the customer."""

    CRITICAL = "critical"  # text/price/logo/product identity errors
    MAJOR = "major"  # visibly broken composition
    MINOR = "minor"  # polish issues


class Verdict(str, Enum):
    """Overall outcome for one rendered variant."""

    ACCEPTED = "accepted"  # every implemented check passed (minor notes allowed)
    NEEDS_REVIEW = "needs_review"  # a check needs a human, or a critical check did not run
    FAILED = "failed"  # at least one critical/major check failed


@dataclass
class CheckResult:
    """Result of one check on one subject (element or whole canvas)."""

    check_id: str
    status: CheckStatus
    severity: Severity
    message: str
    subject_id: str | None = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["severity"] = self.severity.value
        return payload


@dataclass
class QualityReport:
    """All checks for one rendered output plus the derived verdict."""

    contract_version: str
    verdict: Verdict
    checks: list[CheckResult]
    config: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    @property
    def needs_review(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.NEEDS_REVIEW]

    @property
    def not_checked(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.NOT_CHECKED]

    def issues(self) -> list[CheckResult]:
        """Checks a reviewer should look at, most severe first.

        Critical checks that could not run are included: a reviewer must know
        that text legibility (for example) was never verified.
        """
        order = {Severity.CRITICAL: 0, Severity.MAJOR: 1, Severity.MINOR: 2}
        status_order = {
            CheckStatus.FAIL: 0,
            CheckStatus.NEEDS_REVIEW: 1,
            CheckStatus.NOT_CHECKED: 2,
        }
        relevant = [
            c
            for c in self.checks
            if c.status in (CheckStatus.FAIL, CheckStatus.NEEDS_REVIEW)
            or (c.status == CheckStatus.NOT_CHECKED and c.severity == Severity.CRITICAL)
        ]
        return sorted(
            relevant, key=lambda c: (status_order[c.status], order[c.severity], c.check_id)
        )

    def to_dict(self) -> dict:
        return {
            "contract_version": self.contract_version,
            "verdict": self.verdict.value,
            "checks": [c.to_dict() for c in self.checks],
            "config": dict(self.config),
            "summary": dict(self.summary),
        }


def derive_verdict(checks: list[CheckResult]) -> Verdict:
    """Combine check statuses into a verdict.

    Rules:
    - any FAIL with severity CRITICAL or MAJOR -> FAILED
    - any FAIL with severity MINOR -> NEEDS_REVIEW
    - any NEEDS_REVIEW with severity CRITICAL or MAJOR -> NEEDS_REVIEW
    - any NOT_CHECKED with severity CRITICAL -> NEEDS_REVIEW (skipped is not evidence)
    - MINOR NEEDS_REVIEW items are notes: they are listed but do not block ACCEPTED
    - otherwise ACCEPTED
    """
    verdict = Verdict.ACCEPTED
    for c in checks:
        if c.status == CheckStatus.FAIL and c.severity in (Severity.CRITICAL, Severity.MAJOR):
            return Verdict.FAILED
    for c in checks:
        review = (
            c.status == CheckStatus.FAIL
            or (c.status == CheckStatus.NEEDS_REVIEW and c.severity != Severity.MINOR)
            or (c.status == CheckStatus.NOT_CHECKED and c.severity == Severity.CRITICAL)
        )
        if review:
            verdict = Verdict.NEEDS_REVIEW
    return verdict


def summarize(checks: list[CheckResult]) -> dict:
    counts = {s.value: 0 for s in CheckStatus}
    for c in checks:
        counts[c.status.value] += 1
    return {
        "total": len(checks),
        "by_status": counts,
        "notes": sum(
            1
            for c in checks
            if c.status == CheckStatus.NEEDS_REVIEW and c.severity == Severity.MINOR
        ),
        "critical_failures": [
            c.check_id + (f":{c.subject_id}" if c.subject_id else "")
            for c in checks
            if c.status == CheckStatus.FAIL and c.severity == Severity.CRITICAL
        ],
    }
