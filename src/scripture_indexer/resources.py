"""Access the biblical text bundled with the package."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

BIBLE_RESOURCE = "data/vulgata.json"


def default_bible_file() -> Path:
    """Return the path of the bundled Sixto-Clementine Vulgate text."""
    return Path(files("scripture_indexer").joinpath(BIBLE_RESOURCE))
