"""Build a page-level, human-readable index from optimized biblical paths."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripture_indexer.mapper import BiblicalReference
from scripture_indexer.optimizer import OptimizationResult
from scripture_indexer.segmenter import SourceSpan


@dataclass(frozen=True, slots=True)
class PageReading:
    """The page-level portion of one reconstructed lectio."""

    lectio_id: int
    reference: str
    page_verse_count: int
    lectio_verse_count: int
    page_reference_word_count: int
    lectio_reference_word_count: int
    tolerated_missing_verses: int
    continues_from_previous_page: bool
    continues_on_next_page: bool


@dataclass(frozen=True, slots=True)
class PageIndex:
    """All detected biblical readings on one page."""

    page: str
    readings: tuple[PageReading, ...]


@dataclass(frozen=True, slots=True)
class HumanReadableIndex:
    """Ordered collection of page-level index entries."""

    pages: tuple[PageIndex, ...]

    @property
    def reading_count(self) -> int:
        """Return the number of page-level reading entries."""
        return sum(len(page.readings) for page in self.pages)


@dataclass(frozen=True, slots=True)
class AggregatedPageReading:
    """References assigned to one source page of a reading."""

    source_id: str
    references: tuple[BiblicalReference, ...]


@dataclass(frozen=True, slots=True)
class AggregatedReading:
    """One retained index entry with page-level evidence."""

    lectio_id: int
    references: tuple[BiblicalReference, ...]
    pages: tuple[AggregatedPageReading, ...]


@dataclass(frozen=True, slots=True)
class _IndexStep:
    """Normalized step used by both object and JSON input routes."""

    reference: BiblicalReference
    source_spans: tuple[SourceSpan, ...]


@dataclass(frozen=True, slots=True)
class _IndexLectio:
    """Normalized lectio independent of its serialization format."""

    lectio_id: int
    steps: tuple[_IndexStep, ...]
    evaluation_block: str | None = None


class IndexBuilder:
    """Convert optimized paths into a page-level biblical index."""

    def __init__(
        self,
        reference_word_counts: Mapping[int, int],
        min_reference_words: int = 1,
        missing_verse_tolerance: int = 0,
        page_label_fn: Callable[[str], str] | None = None,
    ) -> None:
        """Configure pruning and page-label formatting.

        Args:
            reference_word_counts: Canonical word count for every verse ID.
            min_reference_words: Minimum number of words in the complete
                canonical biblical reference required to retain a lectio.
            missing_verse_tolerance: Maximum missing verses allowed when
                merging biblically connected fragments sharing a page.
            page_label_fn: Optional function converting a source ID to the
                displayed page label. The source basename is used by default.
        """
        if min_reference_words < 1:
            raise ValueError("min_reference_words must be positive")
        if any(count < 0 for count in reference_word_counts.values()):
            raise ValueError("reference word counts cannot be negative")
        if missing_verse_tolerance < 0:
            raise ValueError("missing_verse_tolerance cannot be negative")

        self.reference_word_counts = dict(reference_word_counts)
        self.min_reference_words = min_reference_words
        self.missing_verse_tolerance = missing_verse_tolerance
        self.page_label_fn = page_label_fn or (
            lambda source_id: Path(source_id).name
        )

    @staticmethod
    def _unique_references(
        steps: Iterable[_IndexStep],
    ) -> tuple[BiblicalReference, ...]:
        """Return references once in canonical biblical order."""
        references: dict[int, BiblicalReference] = {}
        for step in steps:
            references.setdefault(step.reference.verse_id, step.reference)
        return tuple(references[key] for key in sorted(references))

    @staticmethod
    def _split_runs(
        references: Sequence[BiblicalReference],
        missing_verse_tolerance: int = 0,
    ) -> list[list[BiblicalReference]]:
        """Split references into runs under a missing-verse tolerance."""
        if not references:
            return []

        runs = [[references[0]]]
        for reference in references[1:]:
            previous = runs[-1][-1]
            verse_jump = reference.verse_id - previous.verse_id
            if (
                reference.book == previous.book
                and verse_jump >= 0
                and max(0, verse_jump - 1) <= missing_verse_tolerance
            ):
                runs[-1].append(reference)
            else:
                runs.append([reference])

        return runs

    @staticmethod
    def _format_run(references: Sequence[BiblicalReference]) -> str:
        """Format one consecutive run as a compact biblical reference."""
        first = references[0]
        last = references[-1]

        if len(references) == 1:
            return f"{first.book} {first.chapter}:{first.verse}"
        if first.chapter == last.chapter:
            return (
                f"{first.book} {first.chapter}:"
                f"{first.verse}–{last.verse}"
            )

        return (
            f"{first.book} {first.chapter}:{first.verse}–"
            f"{last.chapter}:{last.verse}"
        )

    @classmethod
    def format_reference(
        cls,
        references: Sequence[BiblicalReference],
        missing_verse_tolerance: int = 0,
    ) -> str:
        """Format possibly discontinuous references without hiding gaps."""
        return "; ".join(
            cls._format_run(run)
            for run in cls._split_runs(
                references,
                missing_verse_tolerance=missing_verse_tolerance,
            )
        )

    @staticmethod
    def _missing_verse_count(
        references: Sequence[BiblicalReference],
    ) -> int:
        """Count inferred verses absent between selected references."""
        missing = 0

        for previous, current in zip(references, references[1:]):
            if previous.book == current.book:
                missing += max(0, current.verse_id - previous.verse_id - 1)

        return missing

    def _reference_word_count(
        self,
        references: Sequence[BiblicalReference],
    ) -> int:
        """Count canonical words in the displayed biblical reference.

        Verses bridged under ``missing_verse_tolerance`` are included because
        the formatted reference presents the run as one inclusive interval.
        Repeated optimizer steps for the same verse are counted only once.
        """
        verse_ids = {
            verse_id
            for run in self._split_runs(
                references,
                missing_verse_tolerance=self.missing_verse_tolerance,
            )
            for verse_id in range(
                run[0].verse_id,
                run[-1].verse_id + 1,
            )
        }
        missing_ids = {
            verse_id
            for verse_id in verse_ids
            if verse_id not in self.reference_word_counts
        }
        if missing_ids:
            missing = ", ".join(str(verse_id) for verse_id in sorted(missing_ids))
            raise KeyError(
                "Missing canonical word counts for verse IDs: " + missing
            )

        return sum(
            self.reference_word_counts[verse_id]
            for verse_id in verse_ids
        )

    @classmethod
    def _lectio_interval(
        cls,
        lectio: _IndexLectio,
    ) -> tuple[str, int, int] | None:
        """Return one lectio's book and inclusive canonical verse interval."""
        references = cls._unique_references(lectio.steps)
        books = {reference.book for reference in references}
        if not references or len(books) != 1:
            return None
        verse_ids = [reference.verse_id for reference in references]
        return next(iter(books)), min(verse_ids), max(verse_ids)

    @staticmethod
    def _lectio_pages(lectio: _IndexLectio) -> frozenset[str]:
        """Return the source pages touched by a lectio, ignoring geometry."""
        pages = set()
        for step in lectio.steps:
            for span in step.source_spans:
                pages.add(span.source_id)
        return frozenset(pages)

    @staticmethod
    def _source_page_number(source_id: str) -> int:
        """Extract an image page number from a source identifier.

        Identifiers ending in a folio label, such as ``54 - f. 024v``, use
        the image page before the spaced dash. Other identifiers use their
        final integer, for example ``ms103__page_0083`` becomes ``83``.
        """
        stem = Path(source_id).stem
        page_and_folio = re.split(r"\s+[-–—]\s+", stem, maxsplit=1)
        if len(page_and_folio) == 2:
            page_numbers = re.findall(r"\d+", page_and_folio[0])
            if page_numbers:
                return int(page_numbers[-1])

        numbers = re.findall(r"\d+", stem)
        if not numbers:
            raise ValueError(
                f"Cannot extract a page number from {source_id!r}"
            )
        return int(numbers[-1])

    @classmethod
    def _lectio_page_numbers(cls, lectio: _IndexLectio) -> frozenset[int]:
        """Return numeric source pages touched by a lectio."""
        return frozenset(
            cls._source_page_number(source_id)
            for source_id in cls._lectio_pages(lectio)
        )

    @staticmethod
    def _page_locations_connected(
        left_pages: frozenset[str],
        right_pages: frozenset[str],
        left_page_numbers: frozenset[int],
        right_page_numbers: frozenset[int],
        left_block: str | None,
        right_block: str | None,
    ) -> bool:
        """Allow same-page joins and safe joins across consecutive pages.

        Production outputs have no evaluation block, so ``None == None`` is
        deliberately treated as one continuous inference stream. Explicitly
        different evaluation blocks remain disconnected.
        """
        if left_block != right_block:
            return False

        if left_pages & right_pages:
            return True

        return any(
            abs(left_page - right_page) == 1
            for left_page in left_page_numbers
            for right_page in right_page_numbers
        )

    def _intervals_connected(
        self,
        left: tuple[str, int, int],
        right: tuple[str, int, int],
    ) -> bool:
        """Test overlap or biblical continuity in either traversal order."""
        left_book, left_start, left_end = left
        right_book, right_start, right_end = right
        allowance = self.missing_verse_tolerance + 1
        return (
            left_book == right_book
            and left_start <= right_end + allowance
            and right_start <= left_end + allowance
        )

    def _find_page_component(
        self,
        lectiones: Sequence[_IndexLectio],
    ) -> set[int] | None:
        """Find a connected same-page or adjacent-page component."""
        intervals = [self._lectio_interval(lectio) for lectio in lectiones]
        pages = [self._lectio_pages(lectio) for lectio in lectiones]
        page_numbers = [
            self._lectio_page_numbers(lectio) for lectio in lectiones
        ]
        blocks = [lectio.evaluation_block for lectio in lectiones]
        unvisited = set(
            index for index, interval in enumerate(intervals)
            if interval is not None
        )

        while unvisited:
            seed = min(unvisited)
            component = {seed}
            frontier = [seed]
            unvisited.remove(seed)

            while frontier:
                current = frontier.pop()
                connected = {
                    candidate
                    for candidate in unvisited
                    if self._page_locations_connected(
                        pages[current],
                        pages[candidate],
                        page_numbers[current],
                        page_numbers[candidate],
                        blocks[current],
                        blocks[candidate],
                    )
                    and self._intervals_connected(
                        intervals[current],  # type: ignore[arg-type]
                        intervals[candidate],  # type: ignore[arg-type]
                    )
                }
                component.update(connected)
                frontier.extend(connected)
                unvisited.difference_update(connected)

            if len(component) > 1:
                return component

        return None

    @staticmethod
    def _step_biblical_key(step: _IndexStep) -> int:
        """Sort fragments solely by canonical biblical position."""
        return step.reference.verse_id

    def _merge_page_fragments(
        self,
        lectiones: Iterable[_IndexLectio],
    ) -> list[_IndexLectio]:
        """Merge connected fragments within one evaluation block."""
        merged = list(lectiones)

        while True:
            component = self._find_page_component(merged)
            if component is None:
                return merged

            first = min(component)
            steps = sorted(
                (
                    step
                    for index in sorted(component)
                    for step in merged[index].steps
                ),
                key=self._step_biblical_key,
            )
            repaired = _IndexLectio(
                lectio_id=merged[first].lectio_id,
                steps=tuple(steps),
                evaluation_block=merged[first].evaluation_block,
            )
            merged = [
                repaired if index == first else lectio
                for index, lectio in enumerate(merged)
                if index == first or index not in component
            ]

    def _reconstruct_lectiones(
        self,
        lectiones: Iterable[_IndexLectio],
    ) -> list[_IndexLectio]:
        """Reconstruct readings using only page identity and references."""
        return self._merge_page_fragments(lectiones)

    @staticmethod
    def _object_lectiones(result: OptimizationResult) -> list[_IndexLectio]:
        """Normalize an in-memory optimizer result."""
        lectiones = []

        for lectio_id, lectio in enumerate(result.lectiones, start=1):
            steps = []
            for step in lectio.steps:
                spans = step.segment.source_spans
                if not spans:
                    spans = IndexBuilder._fallback_spans(
                        step.segment.source_start,
                        step.segment.line_start,
                        step.segment.source_end,
                        step.segment.line_end,
                    )

                steps.append(
                    _IndexStep(
                        reference=step.reference,
                        source_spans=spans,
                    )
                )

            lectiones.append(
                _IndexLectio(
                    lectio_id=lectio_id,
                    steps=tuple(steps),
                    evaluation_block=None,
                )
            )

        return lectiones

    @staticmethod
    def _fallback_spans(
        source_start: str,
        line_start: int,
        source_end: str,
        line_end: int,
    ) -> tuple[SourceSpan, ...]:
        """Support old single-page outputs lacking per-page spans."""
        if source_start != source_end:
            raise ValueError(
                "A multi-page lectio lacks per-page source_spans. "
                "Rerun the pipeline with the updated serializer."
            )

        return (
            SourceSpan(
                source_id=source_start,
                line_start=line_start,
                line_end=line_end,
            ),
        )

    @staticmethod
    def _parse_reference(step: Mapping[str, Any]) -> BiblicalReference:
        """Read a structured step reference, with legacy-label fallback."""
        if all(key in step for key in ("book", "chapter", "verse")):
            book = str(step["book"])
            chapter = str(step["chapter"])
            verse = str(step["verse"])
        else:
            book, chapter, verse = str(step["reference"]).rsplit("-", 2)

        return BiblicalReference(
            verse_id=int(step["verse_id"]),
            book=book,
            chapter=chapter,
            verse=verse,
        )

    @staticmethod
    def _serialized_spans(
        lectio: Mapping[str, Any],
        step: Mapping[str, Any],
    ) -> tuple[SourceSpan, ...]:
        """Load step spans or support a legacy single-page lectio."""
        serialized = step.get("source_spans", [])
        if serialized:
            return tuple(
                SourceSpan(
                    source_id=str(span["source_id"]),
                    line_start=int(span["line_start"]),
                    line_end=int(span["line_end"]),
                )
                for span in serialized
            )

        return IndexBuilder._fallback_spans(
            source_start=str(lectio["source_start"]),
            line_start=int(lectio["line_start"]),
            source_end=str(lectio["source_end"]),
            line_end=int(lectio["line_end"]),
        )

    @staticmethod
    def _serialized_lectiones(
        pipeline_output: Mapping[str, Any],
        optimizer_name: str,
    ) -> list[_IndexLectio]:
        """Normalize optimizer output previously written as JSON."""
        try:
            serialized = pipeline_output["results"][optimizer_name]["lectiones"]
        except KeyError as error:
            raise KeyError(
                f"No {optimizer_name!r} optimizer result in pipeline output"
            ) from error

        lectiones = []
        for position, lectio in enumerate(serialized, start=1):
            raw_block = lectio.get("evaluation_block")
            steps = tuple(
                _IndexStep(
                    reference=IndexBuilder._parse_reference(step),
                    source_spans=IndexBuilder._serialized_spans(lectio, step),
                )
                for step in lectio["steps"]
            )
            lectiones.append(
                _IndexLectio(
                    lectio_id=int(lectio.get("lectio_id", position)),
                    steps=steps,
                    evaluation_block=(
                        None if raw_block is None else str(raw_block)
                    ),
                )
            )

        return lectiones

    def _page_readings(
        self,
        lectio: _IndexLectio,
    ) -> list[tuple[str, PageReading]]:
        """Split one retained lectio across all pages it traverses."""
        all_references = self._unique_references(lectio.steps)
        lectio_reference_word_count = self._reference_word_count(all_references)
        if lectio_reference_word_count < self.min_reference_words:
            return []

        page_references: dict[str, dict[int, BiblicalReference]] = {}

        for step in lectio.steps:
            for span in step.source_spans:
                page_references.setdefault(span.source_id, {}).setdefault(
                    step.reference.verse_id,
                    step.reference,
                )

        page_ids = list(page_references)
        readings = []

        for page_position, page_id in enumerate(page_ids):
            references = tuple(
                page_references[page_id][verse_id]
                for verse_id in sorted(page_references[page_id])
            )
            page_reference_word_count = self._reference_word_count(references)

            readings.append(
                (
                    page_id,
                    PageReading(
                        lectio_id=lectio.lectio_id,
                        reference=self.format_reference(
                            references,
                            missing_verse_tolerance=(
                                self.missing_verse_tolerance
                            ),
                        ),
                        page_verse_count=len(references),
                        lectio_verse_count=len(all_references),
                        page_reference_word_count=page_reference_word_count,
                        lectio_reference_word_count=lectio_reference_word_count,
                        tolerated_missing_verses=(
                            self._missing_verse_count(references)
                        ),
                        continues_from_previous_page=page_position > 0,
                        continues_on_next_page=page_position < len(page_ids) - 1,
                    ),
                )
            )

        return readings

    @staticmethod
    def _aggregate_pages(
        lectio: _IndexLectio,
    ) -> tuple[AggregatedPageReading, ...]:
        """Collect distinct references for every page, ignoring geometry."""
        page_references: dict[str, dict[int, BiblicalReference]] = {}

        for step in lectio.steps:
            for span in step.source_spans:
                page_references.setdefault(span.source_id, {}).setdefault(
                    step.reference.verse_id,
                    step.reference,
                )

        return tuple(
            AggregatedPageReading(
                source_id=source_id,
                references=tuple(
                    page_references[source_id][verse_id]
                    for verse_id in sorted(page_references[source_id])
                ),
            )
            for source_id in page_references
        )

    def _build_aggregated(
        self,
        lectiones: Iterable[_IndexLectio],
    ) -> tuple[AggregatedReading, ...]:
        """Build retained whole-lectio entries using index merge rules."""
        readings = []

        for lectio in self._reconstruct_lectiones(lectiones):
            references = self._unique_references(lectio.steps)
            if (
                self._reference_word_count(references)
                < self.min_reference_words
            ):
                continue

            readings.append(
                AggregatedReading(
                    lectio_id=lectio.lectio_id,
                    references=references,
                    pages=self._aggregate_pages(lectio),
                )
            )

        return tuple(readings)

    def _build(self, lectiones: Iterable[_IndexLectio]) -> HumanReadableIndex:
        """Build and order page entries from normalized lectiones."""
        pages: defaultdict[str, list[PageReading]] = defaultdict(list)

        for lectio in self._reconstruct_lectiones(lectiones):
            for source_id, reading in self._page_readings(lectio):
                pages[source_id].append(reading)

        return HumanReadableIndex(
            pages=tuple(
                PageIndex(
                    page=self.page_label_fn(source_id),
                    readings=tuple(readings),
                )
                for source_id, readings in pages.items()
            )
        )

    def build(self, result: OptimizationResult) -> HumanReadableIndex:
        """Build an index from an in-memory optimizer result."""
        return self._build(self._object_lectiones(result))

    def build_from_pipeline_output(
        self,
        pipeline_output: Mapping[str, Any],
        optimizer_name: str = "global",
    ) -> HumanReadableIndex:
        """Build an index from the JSON-compatible pipeline output."""
        return self._build(
            self._serialized_lectiones(pipeline_output, optimizer_name)
        )

    def build_aggregated(
        self,
        result: OptimizationResult,
    ) -> tuple[AggregatedReading, ...]:
        """Build whole-lectio entries from an in-memory optimizer result."""
        return self._build_aggregated(self._object_lectiones(result))

    def build_aggregated_from_pipeline_output(
        self,
        pipeline_output: Mapping[str, Any],
        optimizer_name: str = "global",
    ) -> tuple[AggregatedReading, ...]:
        """Build whole-lectio entries from serialized pipeline output."""
        return self._build_aggregated(
            self._serialized_lectiones(pipeline_output, optimizer_name)
        )


def index_to_dict(index: HumanReadableIndex) -> dict[str, Any]:
    """Convert a page index to a JSON-compatible dictionary."""
    return {
        "pages": [
            {
                "page": page.page,
                "readings": [
                    {
                        "lectio_id": reading.lectio_id,
                        "reference": reading.reference,
                        "page_verse_count": reading.page_verse_count,
                        "lectio_verse_count": reading.lectio_verse_count,
                        "page_reference_word_count": (
                            reading.page_reference_word_count
                        ),
                        "lectio_reference_word_count": (
                            reading.lectio_reference_word_count
                        ),
                        "tolerated_missing_verses": (
                            reading.tolerated_missing_verses
                        ),
                        "continues_from_previous_page": (
                            reading.continues_from_previous_page
                        ),
                        "continues_on_next_page": (
                            reading.continues_on_next_page
                        ),
                    }
                    for reading in page.readings
                ],
            }
            for page in index.pages
        ]
    }


def format_index(index: HumanReadableIndex) -> str:
    """Render a compact, page-by-page plain-text index."""
    lines = []

    for page in index.pages:
        lines.append(page.page)
        for reading in page.readings:
            continuation = []
            if reading.continues_from_previous_page:
                continuation.append("continued")
            if reading.continues_on_next_page:
                continuation.append("continues")

            suffix = f" ({', '.join(continuation)})" if continuation else ""
            lines.append(f"  {reading.reference}{suffix}")

    return "\n".join(lines) + ("\n" if lines else "")


def load_pipeline_output(input_file: str | Path) -> dict[str, Any]:
    """Load the JSON output written by ``run_pipeline.py``."""
    with Path(input_file).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_reference_word_counts(
    bible_file: str | Path,
) -> dict[int, int]:
    """Derive canonical verse word counts from the biblical text."""
    with Path(bible_file).open("r", encoding="utf-8") as file:
        biblical_text = json.load(file)

    counts = {}
    verse_id = 0
    for chapters in biblical_text.values():
        for verses in chapters.values():
            for text in verses.values():
                normalized = re.sub(r"[^\w\s]", "", text.lower())
                normalized = re.sub(r"\s+", " ", normalized).strip()
                counts[verse_id] = len(normalized.split())
                verse_id += 1

    return counts


def resolve_bible_file(
    input_file: Path,
    pipeline_output: Mapping[str, Any],
    explicit_bible: Path | None,
) -> Path:
    """Resolve an explicit or pipeline-configured biblical text path."""
    configured = explicit_bible
    if configured is None:
        configured_value = pipeline_output.get("configuration", {}).get(
            "bible_file"
        )
        if configured_value is None:
            raise ValueError(
                "No biblical text supplied. Use --bible or include "
                "configuration.bible_file in the pipeline output."
            )
        configured = Path(configured_value)

    candidates = [configured]
    if not configured.is_absolute():
        candidates.append(input_file.parent / configured)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Biblical text not found: {configured}")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Build a page-level index from optimized biblical paths."
    )
    parser.add_argument("input", type=Path, help="Pipeline JSON output.")
    parser.add_argument("--output", type=Path, default=Path("index.txt"))
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--optimizer",
        choices=("global", "naive"),
        default="global",
    )
    parser.add_argument(
        "--bible",
        type=Path,
        default=None,
        help=(
            "Canonical biblical text used to calculate reference length. "
            "Defaults to configuration.bible_file from the pipeline output."
        ),
    )
    parser.add_argument("--min-reference-words", type=int, default=1)
    parser.add_argument("--missing-verse-tolerance", type=int, default=0)
    return parser


def main() -> None:
    """Build and write the human-readable page index."""
    args = build_parser().parse_args()
    pipeline_output = load_pipeline_output(args.input)
    bible_file = resolve_bible_file(args.input, pipeline_output, args.bible)
    builder = IndexBuilder(
        reference_word_counts=load_reference_word_counts(bible_file),
        min_reference_words=args.min_reference_words,
        missing_verse_tolerance=args.missing_verse_tolerance,
    )
    index = builder.build_from_pipeline_output(
        pipeline_output,
        optimizer_name=args.optimizer,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_index(index), encoding="utf-8")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as file:
            json.dump(index_to_dict(index), file, ensure_ascii=False, indent=2)

    print(f"Wrote {args.output}: {index.reading_count} page entries")


if __name__ == "__main__":
    main()
