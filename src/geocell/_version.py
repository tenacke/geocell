"""Runtime version helper for `geocell`.

Reads the repository `VERSION` file at package import time so the runtime
package `__version__` stays in sync with the project version.
"""

from __future__ import annotations

from pathlib import Path
import os


def _read_version() -> str:
    # Resolve project root relative to this file: src/geocell/_version.py
    root = Path(__file__).resolve().parents[2]
    version_file = root / "VERSION"
    try:
        return version_file.read_text(encoding="utf8").strip()
    except Exception:
        # Fallback if VERSION is not present (e.g., installed as wheel)
        return os.environ.get("GEOCELL_VERSION", "0.0.0")


__version__ = _read_version()
