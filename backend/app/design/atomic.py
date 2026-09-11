"""Crash- and race-safe JSON writes.

Every store in AutoBanner persists by writing a temporary file next to the target
and renaming it over the target, so a reader never sees a half-written file.  The
temporary file must be private to the writer: two threads saving the same path
through one shared ``<name>.tmp`` would each rename the other's file away and one
of them would fail with ``FileNotFoundError`` (seen on restart, when the recovery
pass and a resumed worker both saved a project's variant index).  ``os.replace``
is atomic on the same file system, so the last writer wins whole.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

__all__ = ["atomic_write_json", "atomic_write_text"]


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` through a uniquely named sibling temp file."""
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
) -> None:
    atomic_write_text(path, json.dumps(payload, indent=indent, ensure_ascii=ensure_ascii))
