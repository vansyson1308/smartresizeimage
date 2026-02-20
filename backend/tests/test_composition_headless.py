"""Headless-safety tests for composition dependency handling."""

from __future__ import annotations

import builtins
import importlib
import sys


def test_composition_module_import_does_not_require_cv2(monkeypatch):
    """Importing composition modules should not crash if cv2 import fails."""
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("simulated missing cv2")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    for module_name in [
        "backend.app.composition.background",
        "backend.app.composition.content_aware_fit",
    ]:
        sys.modules.pop(module_name, None)
        module = importlib.import_module(module_name)
        assert module is not None
