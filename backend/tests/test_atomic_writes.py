"""Atomic JSON writes survive concurrent writers on the same path.

Regression for the restart race: two savers sharing one ``<name>.tmp`` raced on
the rename and one of them died with ``FileNotFoundError``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from backend.app.design.atomic import atomic_write_json, atomic_write_text


def test_concurrent_writers_never_collide(tmp_path: Path) -> None:
    target = tmp_path / "index.json"
    errors: list[BaseException] = []
    start = threading.Barrier(8)

    def writer(n: int) -> None:
        try:
            start.wait(timeout=10)
            for i in range(60):
                atomic_write_json(target, {"writer": n, "i": i, "pad": "x" * 2000})
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["i"] == 59 and payload["writer"] in range(8)
    # no temp files linger next to the target
    assert [p.name for p in tmp_path.iterdir()] == ["index.json"]


def test_failed_write_leaves_old_content_and_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    atomic_write_json(target, {"v": 1})

    class Boom:
        pass

    with pytest.raises(TypeError):
        atomic_write_json(target, {"v": Boom()})
    assert json.loads(target.read_text()) == {"v": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["data.json"]


def test_text_writer_replaces_whole_file(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    atomic_write_text(target, "first")
    atomic_write_text(target, "second, longer")
    assert target.read_text() == "second, longer"
    assert [p.name for p in tmp_path.iterdir()] == ["note.txt"]
