"""Local, append-only event log for pilot measurement.

Events are written as JSON lines under ``data/events/<owner>.jsonl`` and never
leave the machine: this is the instrument for measuring first-pass acceptance,
time-to-decision and correction effort during pilots. Enable/disable with
``AUTOBANNER_EVENTS`` (default ``local``; ``off`` disables).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, root: Path, enabled: bool | None = None) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        if enabled is None:
            enabled = os.environ.get("AUTOBANNER_EVENTS", "local").strip().lower() != "off"
        self.enabled = enabled
        self._lock = threading.Lock()

    def _path(self, owner: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in owner)[:64]
        return self.root / f"{safe or 'owner'}.jsonl"

    def record(self, owner: str, kind: str, **fields: Any) -> None:
        if not self.enabled:
            return
        event = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), "kind": kind, **fields}
        line = json.dumps(event, ensure_ascii=False)
        with self._lock, self._path(owner).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read(self, owner: str, limit: int = 5000) -> list[dict]:
        p = self._path(owner)
        if not p.exists():
            return []
        lines = p.read_text(encoding="utf-8").splitlines()[-limit:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def summary(self, owner: str) -> dict:
        """Pilot metrics derived from the log.

        - first_pass_acceptance: approved variants that were never regenerated or
          edited after generation / all decided variants
        - median_seconds_to_decision: generation done -> approve/reject
        - corrections_per_project: document operations per project
        """
        events = self.read(owner)
        generated: dict[str, str] = {}
        regenerated: set[str] = set()
        decisions: list[dict] = []
        ops_per_project: dict[str, int] = {}
        rejections: dict[str, int] = {}
        for e in events:
            kind = e.get("kind")
            if kind == "variant_done":
                vid = e.get("variant_id")
                if vid in generated:
                    regenerated.add(vid)
                generated[vid] = e["ts"]
            elif kind == "approval" and e.get("approval") in ("approved", "rejected"):
                decisions.append(e)
                if e["approval"] == "rejected":
                    reason = str(e.get("reason", "")).strip().lower()[:60] or "(no reason)"
                    rejections[reason] = rejections.get(reason, 0) + 1
            elif kind == "document_ops":
                pid = e.get("project_id")
                ops_per_project[pid] = ops_per_project.get(pid, 0) + int(e.get("count", 1))
        decided = {}
        for d in decisions:
            decided[d.get("variant_id")] = d  # last decision wins
        approved_first_pass = sum(
            1
            for vid, d in decided.items()
            if d["approval"] == "approved" and vid not in regenerated
        )
        durations = []
        for vid, d in decided.items():
            gen_ts = generated.get(vid)
            if gen_ts:
                try:
                    t0 = datetime.fromisoformat(gen_ts)
                    t1 = datetime.fromisoformat(d["ts"])
                    durations.append((t1 - t0).total_seconds())
                except ValueError:
                    continue
        durations.sort()
        median = durations[len(durations) // 2] if durations else None
        return {
            "owner": owner,
            "events": len(events),
            "variants_generated": len(generated),
            "variants_decided": len(decided),
            "approved": sum(1 for d in decided.values() if d["approval"] == "approved"),
            "rejected": sum(1 for d in decided.values() if d["approval"] == "rejected"),
            "first_pass_acceptance": (
                round(approved_first_pass / len(decided), 4) if decided else None
            ),
            "median_seconds_to_decision": median,
            "corrections_per_project": ops_per_project,
            "rejection_reasons": rejections,
            "note": "Local log only; decisions by the account holder, not blinded review.",
        }
