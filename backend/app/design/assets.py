"""Content-addressed asset store for a project directory."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

from PIL import Image

from .document import AssetRef

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_FILENAME = re.compile(r"^(?!\.)[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}$")


class AssetStore:
    """Stores original assets as PNG files named by content hash.

    Assets are immutable: the same pixels always map to the same id, so a
    project can reference one logo from many elements and variants without
    duplication, and exports can prove asset identity by hash.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, image: Image.Image, original_name: str = "") -> AssetRef:
        rgba = image.convert("RGBA")
        buf = io.BytesIO()
        rgba.save(buf, format="PNG", optimize=False)
        data = buf.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        asset_id = digest[:16]
        filename = f"{asset_id}.png"
        path = self.root / filename
        if not path.exists():
            path.write_bytes(data)
        return AssetRef(
            asset_id=asset_id,
            content_hash=digest,
            mime="image/png",
            width=rgba.width,
            height=rgba.height,
            path=filename,
            original_name=_SAFE.sub("_", original_name)[:80],
        )

    def put_file(self, path: str | Path, original_name: str | None = None) -> AssetRef:
        p = Path(path)
        with Image.open(p) as img:
            return self.put(img, original_name or p.name)

    def path(self, ref: AssetRef | str) -> Path:
        """On-disk path of an asset; raises ``ValueError`` for a name that would leave the store.

        Asset references travel inside editable project archives, so ``ref.path`` is
        untrusted: only a bare ``<hash>.png`` style filename is accepted.
        """
        name = ref.path if isinstance(ref, AssetRef) else f"{ref}.png"
        if not isinstance(name, str) or not _FILENAME.match(name):
            raise ValueError(f"unsafe asset path {name!r}")
        return self.root / name

    def get(self, ref: AssetRef | str) -> Image.Image:
        p = self.path(ref)
        with Image.open(p) as img:
            return img.convert("RGBA").copy()

    def exists(self, ref: AssetRef | str) -> bool:
        try:
            return self.path(ref).is_file()
        except ValueError:
            return False

    def verify(self, ref: AssetRef) -> bool:
        """Confirm the stored bytes still match the recorded content hash."""
        p = self.path(ref)
        if not p.exists():
            return False
        return hashlib.sha256(p.read_bytes()).hexdigest() == ref.content_hash
