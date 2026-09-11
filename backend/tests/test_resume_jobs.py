"""Variant jobs cut short by a restart continue on the next start (durable queue).

Every variant record stores its brief and every job record the layout families
chosen for the size set, so a restart re-queues the unfinished items in a new job
that points back at the interrupted one. ``AUTOBANNER_RESUME_JOBS=0`` restores the
old behaviour (unfinished variants marked failed).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from backend.app.api.service import LOCAL_OWNER, ProjectService
from backend.app.design.project import Project
from backend.app.design.variant import VariantBrief
from backend.tests.test_planner import _doc


def _crashed_state(tmp_path: Path) -> tuple[Path, str, list[str], str]:
    """A project with three variants of one job, as left by a process killed mid-run."""
    data = tmp_path / "data"
    doc, store = _doc(tmp_path / "fixture")
    pid = "proj_resume"
    project = Project.create(data / "projects" / pid, doc, name="Resume", owner=LOCAL_OWNER)
    project.meta["id"] = pid
    shutil.copytree(store.root, project.assets.root, dirs_exist_ok=True)
    job_id = "job_crashed"
    targets = [(600, 314, "Wide"), (300, 300, "Square"), (300, 600, "Tall")]
    ids = []
    for w, h, name in targets:
        brief = VariantBrief(w, h, name=name, text_overrides={"headline": "RESUMED"},
                             row={"id": "r1", "label": "Row 1"})
        rec = project.new_variant(name, w, h, brief.to_dict(), job_id=job_id)
        ids.append(rec.id)
    # the first variant finished before the crash, the second was running, the third pending
    project.mark_variant(ids[0], "done")
    project.variants[ids[0]].verdict = "accepted"
    project.mark_variant(ids[1], "running")
    project.save()
    jobs_dir = data / "jobs"
    jobs_dir.mkdir(parents=True)
    families = {
        "landscape": {"name": "landscape_text_left",
                      "text": {"x": 0.05, "y": 0.10, "w": 0.50, "h": 0.80},
                      "subject": {"x": 0.58, "y": 0.06, "w": 0.38, "h": 0.88},
                      "logo": {"x": 0.80, "y": 0.04, "w": 0.16, "h": 0.14},
                      "text_align": "left", "subject_first": False},
    }
    record = {
        "id": job_id, "kind": "variants", "project_id": pid, "owner": LOCAL_OWNER,
        "status": "running", "created_at": "2026-09-11T00:00:00+00:00",
        "started_at": "2026-09-11T00:00:01+00:00", "finished_at": None,
        "idempotency_key": "camp-1", "error": None, "meta": {"families": families},
        "items": [
            {"item_id": ids[0], "label": "Wide", "status": "done", "progress": 1.0,
             "stage": "done", "error": None, "result": {}},
            {"item_id": ids[1], "label": "Square", "status": "running", "progress": 0.4,
             "stage": "render", "error": None, "result": {}},
            {"item_id": ids[2], "label": "Tall", "status": "pending", "progress": 0.0,
             "stage": "", "error": None, "result": {}},
        ],
    }
    (jobs_dir / f"{job_id}.json").write_text(json.dumps(record))
    return data, pid, ids, job_id


def test_interrupted_job_is_resumed_on_start(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AUTOBANNER_RESUME_JOBS", raising=False)
    data, pid, ids, job_id = _crashed_state(tmp_path)
    service = ProjectService(data, max_workers=1)
    try:
        old = service.jobs.get(job_id)
        assert old.status == "interrupted" and old.resumed_by
        new = service.jobs.get(old.resumed_by)
        assert new is not None and new.resumed_from == job_id
        assert [i.item_id for i in new.items] == ids[1:]  # the finished one is not redone
        assert [i.label for i in new.items] == ["Square", "Tall"]
        assert new.meta.get("families") == old.meta.get("families")
        done = service.jobs.wait(new.id, timeout=300)
        assert done.status == "done", done.to_dict()
        project = service.get_project(pid, LOCAL_OWNER)
        assert project.variants[ids[0]].status == "done"  # untouched
        for vid in ids[1:]:
            rec = project.variants[vid]
            assert rec.status == "done" and rec.error is None and rec.job_id == new.id
            detail = project.variant_detail(vid)
            assert detail["plan"]["brief"]["text_overrides"] == {"headline": "RESUMED"}
            assert detail["plan"]["brief"]["row"] == {"id": "r1", "label": "Row 1"}
        # a second start finds nothing to resume and does not touch the finished variants
        on_disk = json.loads((data / "jobs" / f"{job_id}.json").read_text())
        assert on_disk["resumed_by"] == new.id
        events = (data / "events").glob("*")
        text = "".join(p.read_text() for p in events if p.is_file())
        assert "job_resumed" in text
    finally:
        service.shutdown()
    again = ProjectService(data, max_workers=1)
    try:
        assert not again.jobs.interrupted()
        project = again.get_project(pid, LOCAL_OWNER)
        assert all(r.status == "done" for r in project.variants.values())
    finally:
        again.shutdown()


def test_resume_can_be_disabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOBANNER_RESUME_JOBS", "0")
    data, pid, ids, job_id = _crashed_state(tmp_path)
    service = ProjectService(data, max_workers=1)
    try:
        assert service.jobs.get(job_id).resumed_by is None
        project = service.get_project(pid, LOCAL_OWNER)
        assert project.variants[ids[0]].status == "done"
        for vid in ids[1:]:
            assert project.variants[vid].status == "failed"
            assert project.variants[vid].error == "interrupted by restart"
    finally:
        service.shutdown()


def test_unfinished_variants_without_a_job_record_are_failed(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AUTOBANNER_RESUME_JOBS", raising=False)
    data, pid, ids, job_id = _crashed_state(tmp_path)
    (data / "jobs" / f"{job_id}.json").unlink()  # nothing to continue from
    service = ProjectService(data, max_workers=1)
    try:
        project = service.get_project(pid, LOCAL_OWNER)
        assert {project.variants[v].status for v in ids[1:]} == {"failed"}
    finally:
        service.shutdown()
