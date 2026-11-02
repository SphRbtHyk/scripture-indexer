"""Bridge between the Streamlit app and the scripture_indexer pipeline.

Wraps ``scripture_indexer.run_full_pipeline`` with the hyperparameters
optimized on MS10 (see ``paper/evaluation.py``) and returns, alongside the
page index, the rendered page images so the UI can pair each index entry
with its manuscript page.
"""

from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from scripture_indexer.resources import default_bible_file
from scripture_indexer.run_full_pipeline import run_full_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = REPO_ROOT / "htr" / "models" / "catmus-medieval-1.6.0.mlmodel"
SHARED_CACHE = REPO_ROOT / "app" / "cache"

# Hyperparameters from the MS10 optimization (paper/evaluation.py).
# This is the only configuration duplicated in the app: every change to the
# pipeline itself lives in src/scripture_indexer and is picked up thanks to
# the editable install.
OPTIMIZED_PIPELINE_OPTIONS = {
    "jump_penalty": 0.04,
    "lookback": 2,
    "max_hits": 5,
    "max_verse_jump": 2,
    "ngram_size": 3,
    "noise_level": 0.15,
    "shift": 7,
    "skip_penalty": 0.01,
    "start_penalty": 0.15,
    "window_size": 15,
}


@dataclass(frozen=True)
class IndexingResult:
    """Everything the UI needs after one indexing run."""

    index: dict[str, Any]
    yaml_path: Path
    json_path: Path
    htr_dir: Path
    pdf_name: str
    source_ids: tuple[str, ...]
    calibration_path: Path | None = None


def _page_number(source_id: str) -> int:
    """Extract the trailing page number from a rendered-PDF source id."""
    match = re.search(r"(\d+)\s*$", source_id)
    return int(match.group(1)) if match else 0


def sort_source_ids(source_ids: list[str]) -> list[str]:
    """Sort source ids in natural (page) order."""
    return sorted(source_ids, key=lambda item: (_page_number(item), item))


def rendered_page_image(htr_dir: Path, source_id: str) -> Path | None:
    """Return the rendered image file for one source id, if it exists."""
    rendered_root = htr_dir / "_rendered_pdf_pages"
    if not rendered_root.is_dir():
        return None
    # The source id comes from the PDF name and may contain glob-special
    # characters (e.g. "[...]" in Gallica filenames), so it must be escaped.
    for image in rendered_root.glob(f"*/{glob.escape(source_id)}.*"):
        return image
    return None


def transcription_path(htr_dir: Path, source_id: str) -> Path:
    """Return the transcription file produced by Kraken for one page."""
    return htr_dir / source_id / "transcription.txt"


def run_indexing(
    pdf_path: str | Path,
    work_dir: str | Path,
    *,
    limit: int = 0,
    pdf_dpi: int = 300,
    device: str = "auto",
    min_reference_words: int = 40,
    missing_verse_tolerance: int = 3,
    exclude_psalms: bool = True,
    auto_hyperparam: bool = False,
) -> IndexingResult:
    """Run HTR + indexing on one PDF inside a dedicated work directory."""
    pdf_path = Path(pdf_path)
    work_dir = Path(work_dir)
    htr_dir = work_dir / "htr_output"
    yaml_path = work_dir / "index.yaml"
    json_path = htr_dir / "index.json"
    calibration_path = work_dir / "calibration.json" if auto_hyperparam else None

    SHARED_CACHE.mkdir(parents=True, exist_ok=True)
    if not MODEL.is_file():
        raise FileNotFoundError(
            f"Modele Kraken introuvable: {MODEL}. "
            "Placez catmus-medieval-1.6.0.mlmodel dans htr/models/."
        )

    index = run_full_pipeline(
        input_path=pdf_path,
        bible_file=None,
        output_file=yaml_path,
        model=MODEL,
        htr_output_dir=htr_dir,
        lookup_file=SHARED_CACHE / "bible_lookup.json",
        optimizer_name="global",
        limit=limit,
        pdf_dpi=pdf_dpi,
        device=device,
        min_reference_words=min_reference_words,
        missing_verse_tolerance=missing_verse_tolerance,
        excluded_books=("Psalms",) if exclude_psalms else (),
        # When on, the calibration pass overrides ngram_size, noise_level,
        # window_size and shift with manuscript-specific estimates; the
        # OPTIMIZED_PIPELINE_OPTIONS values then act only as fallback.
        auto_hyperparam=auto_hyperparam,
        auto_hyperparam_diagnostics=calibration_path,
        auto_hyperparam_cache=SHARED_CACHE / "auto_hyperparam_constants.json",
        **OPTIMIZED_PIPELINE_OPTIONS,
    )

    return IndexingResult(
        index=index,
        yaml_path=yaml_path,
        json_path=json_path,
        htr_dir=htr_dir,
        pdf_name=pdf_path.name,
        source_ids=source_ids_for(htr_dir, index),
        calibration_path=calibration_path,
    )


def source_ids_for(htr_dir: Path, index: dict[str, Any]) -> tuple[str, ...]:
    """Page ids of a run, from the HTR output, falling back to the index.

    Every processed page gets a transcription directory, even when no
    reading is detected on it, so the page list comes from the HTR
    output rather than from the index itself.
    """
    transcription_dirs = [
        item.name
        for item in htr_dir.iterdir()
        if item.is_dir() and item.name != "_rendered_pdf_pages"
    ]
    return tuple(
        sort_source_ids(
            transcription_dirs
            or [entry["page"] for entry in index.get("pages", [])]
        )
    )


def load_existing_result(
    work_dir: str | Path, pdf_name: str
) -> IndexingResult | None:
    """Reload a previous run's artifacts from disk, without any re-run."""
    work_dir = Path(work_dir)
    yaml_path = work_dir / "index.yaml"
    htr_dir = work_dir / "htr_output"
    if not yaml_path.is_file() or not htr_dir.is_dir():
        return None
    index = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    calibration_path = work_dir / "calibration.json"
    return IndexingResult(
        index=index,
        yaml_path=yaml_path,
        json_path=htr_dir / "index.json",
        htr_dir=htr_dir,
        pdf_name=pdf_name,
        source_ids=source_ids_for(htr_dir, index),
        calibration_path=calibration_path if calibration_path.is_file() else None,
    )


# --- Vulgate and missal text for the indexed readings ---------------------

_REFERENCE_RUN_RE = re.compile(
    r"^(?P<book>.+?)\s+(?P<chapter_start>\d+):(?P<verse_start>\d+)"
    r"(?:[–-](?:(?P<chapter_end>\d+):)?(?P<verse_end>\d+))?$"
)


@lru_cache(maxsize=1)
def _vulgate_order() -> tuple[dict[str, Any], ...]:
    """Flatten the bundled Sixto-Clementine Vulgate in canonical order."""
    biblical_text = json.loads(
        default_bible_file().read_text(encoding="utf-8")
    )
    entries: list[dict[str, Any]] = []
    for book, chapters in biblical_text.items():
        for chapter, verses in chapters.items():
            for verse, text in verses.items():
                entries.append(
                    {
                        "book": book,
                        "chapter": int(chapter),
                        "verse": int(verse),
                        "text": str(text).strip(),
                    }
                )
    return tuple(entries)


@lru_cache(maxsize=1)
def _vulgate_positions() -> dict[tuple[str, int, int], int]:
    """Canonical position of every (book, chapter, verse) key."""
    return {
        (entry["book"], entry["chapter"], entry["verse"]): position
        for position, entry in enumerate(_vulgate_order())
    }


@lru_cache(maxsize=1)
def _chapter_verses() -> dict[tuple[str, int], tuple[int, ...]]:
    """Verse numbers present in each biblical chapter."""
    grouped: dict[tuple[str, int], list[int]] = {}
    for entry in _vulgate_order():
        grouped.setdefault(
            (entry["book"], entry["chapter"]), []
        ).append(entry["verse"])
    return {
        key: tuple(sorted(numbers)) for key, numbers in grouped.items()
    }


def reference_verses(reference: str) -> tuple[tuple[str, int, int], ...]:
    """Expand a formatted index reference into its verse keys.

    Handles the compact runs produced by the index ("Book 1:4",
    "Book 1:4–9", "Book 1:4–2:9") and the "; "-separated discontinuous
    references; keys absent from the Vulgate are skipped.
    """
    positions = _vulgate_positions()
    keys: list[tuple[str, int, int]] = []
    for run in reference.split(";"):
        match = _REFERENCE_RUN_RE.match(run.strip())
        if match is None:
            continue
        book = match.group("book").strip()
        chapter_start = int(match.group("chapter_start"))
        verse_start = int(match.group("verse_start"))
        chapter_end = int(match.group("chapter_end") or chapter_start)
        verse_end = int(match.group("verse_end") or verse_start)

        start = positions.get((book, chapter_start, verse_start))
        end = positions.get((book, chapter_end, verse_end))
        if start is not None and end is not None and end >= start:
            keys.extend(
                (entry["book"], entry["chapter"], entry["verse"])
                for entry in _vulgate_order()[start : end + 1]
            )
            continue

        # Endpoint absent from the Vulgate: keep the existing keys of
        # the requested ranges instead.
        for chapter in range(chapter_start, chapter_end + 1):
            numbers = _chapter_verses().get((book, chapter), ())
            for verse in numbers:
                if (
                    (chapter, verse) >= (chapter_start, verse_start)
                    and (chapter, verse) <= (chapter_end, verse_end)
                ):
                    keys.append((book, chapter, verse))
    return tuple(keys)


def vulgata_verses(reference: str) -> list[dict[str, Any]]:
    """Vulgate text of every verse of one reference, in canonical order."""
    positions = _vulgate_positions()
    order = _vulgate_order()
    entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    for key in reference_verses(reference):
        position = positions.get(key)
        if position is None or position in seen:
            continue
        seen.add(position)
        entries.append(dict(order[position]))
    return entries


@lru_cache(maxsize=4)
def _detailed_lectiones(json_path: str) -> tuple[dict[str, Any], ...]:
    """Every lectio of one detailed pipeline output, any optimizer."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    lectiones: list[dict[str, Any]] = []
    for optimizer_results in data.get("results", {}).values():
        lectiones.extend(optimizer_results.get("lectiones", []))
    return tuple(lectiones)


def missal_lines_for_reading(
    json_path: str | Path,
    source_id: str,
    reference: str,
    transcription_lines: list[str],
) -> list[str] | None:
    """HTR lines of one page backing one index reference.

    The lines are the union of the source spans of the detailed-pipeline
    lectiones whose steps carry a verse of the reference on this page;
    None when no lectio step matches.
    """
    wanted = set(reference_verses(reference))
    if not wanted:
        return None

    spans: list[tuple[int, int]] = []
    for lectio in _detailed_lectiones(str(json_path)):
        for step in lectio.get("steps", []):
            try:
                key = (
                    str(step.get("book")),
                    int(step.get("chapter")),
                    int(step.get("verse")),
                )
            except (TypeError, ValueError):
                continue
            if key not in wanted:
                continue
            for span in step.get("source_spans", []):
                if span.get("source_id") != source_id:
                    continue
                spans.append(
                    (int(span["line_start"]), int(span["line_end"]))
                )

    if not spans:
        return None
    low = max(0, min(start for start, _ in spans))
    high = min(len(transcription_lines), max(end for _, end in spans) + 1)
    if low >= high:
        return None
    return transcription_lines[low:high]
