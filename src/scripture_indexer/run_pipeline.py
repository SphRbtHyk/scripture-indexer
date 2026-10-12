"""Run the complete missal-to-biblical-index pipeline from the command line."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from scripture_indexer.index import (
    HumanReadableIndex,
    IndexBuilder,
    format_index,
    load_reference_word_counts,
)
from scripture_indexer.mapper import SegmentMapper, VerseMapper, load_lookup
from scripture_indexer.naive_optimizer import NaiveOptimizer
from scripture_indexer.optimizer import LectioPath, OptimizationResult, PathOptimizer
from scripture_indexer.resources import default_bible_file
from scripture_indexer.segmenter import Segmenter


def _format_elapsed(seconds: float) -> str:
    """Format an elapsed duration with millisecond precision."""
    hours, remainder = divmod(seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{remaining_seconds:06.3f}"


@contextmanager
def timed_stage(label: str):
    """Report wall-clock start/end timestamps and precise elapsed time."""
    started_at = datetime.now().astimezone()
    started = perf_counter()
    print(
        f"[{started_at.isoformat(timespec='milliseconds')}] {label} started",
        file=sys.stderr,
        flush=True,
    )
    try:
        yield
    except BaseException:
        finished_at = datetime.now().astimezone()
        elapsed = perf_counter() - started
        print(
            f"[{finished_at.isoformat(timespec='milliseconds')}] {label} "
            f"failed after {_format_elapsed(elapsed)}",
            file=sys.stderr,
            flush=True,
        )
        raise
    else:
        finished_at = datetime.now().astimezone()
        elapsed = perf_counter() - started
        print(
            f"[{finished_at.isoformat(timespec='milliseconds')}] {label} "
            f"finished in {_format_elapsed(elapsed)}",
            file=sys.stderr,
            flush=True,
        )


def load_biblical_text(bible_file: str | Path) -> dict[str, Any]:
    """Load the nested book/chapter/verse JSON file."""
    with Path(bible_file).open("r", encoding="utf-8") as file:
        return json.load(file)


def prepare_verse_mapper(
    biblical_text: dict[str, Any],
    lookup_file: str | Path,
    ngram_size: int,
) -> VerseMapper:
    """Build retrieval structures and protect an existing stable lookup."""
    mapper = VerseMapper(
        biblical_text=biblical_text,
        n=ngram_size,
        index_output=None,
    )

    lookup_path = Path(lookup_file)
    if lookup_path.exists():
        existing_lookup = load_lookup(lookup_path)
        if existing_lookup != mapper.lookup:
            raise ValueError(
                f"{lookup_path} does not match the current biblical text. "
                "Refusing to overwrite the stable lookup."
            )
    else:
        mapper.dump_lookup(lookup_path)

    return mapper


def serialize_lectio(lectio: LectioPath, lectio_id: int) -> dict[str, Any]:
    """Convert one reconstructed lectio to a JSON-compatible dictionary."""
    first = lectio.steps[0]
    last = lectio.steps[-1]

    return {
        "lectio_id": lectio_id,
        "book": first.reference.book,
        "chapter_start": first.reference.chapter,
        "verse_start": first.reference.verse,
        "chapter_end": last.reference.chapter,
        "verse_end": last.reference.verse,
        "source_start": first.segment.source_start,
        "line_start": first.segment.line_start,
        "source_end": last.segment.source_end,
        "line_end": last.segment.line_end,
        "path_score": lectio.score,
        "mean_jaccard": lectio.mean_jaccard,
        "boundary_precision": "window-level",
        "steps": [
            {
                "window_id": step.segment.window_id,
                "token_start": step.segment.token_start,
                "token_end": step.segment.token_end,
                "verse_id": step.candidate.verse_id,
                "reference": step.reference.label,
                "jaccard": step.candidate.score,
                "transition": step.transition,
                "coverage_weight": step.coverage_weight,
                "contribution": step.contribution,
                "book": step.reference.book,
                "chapter": step.reference.chapter,
                "verse": step.reference.verse,
                "source_spans": [
                    {
                        "source_id": span.source_id,
                        "line_start": span.line_start,
                        "line_end": span.line_end,
                    }
                    for span in step.segment.source_spans
                ],
            }
            for step in lectio.steps
        ],
    }


def serialize_result(result: OptimizationResult) -> dict[str, Any]:
    """Convert an optimizer result to a JSON-compatible dictionary."""
    return {
        "total_score": result.total_score,
        "lectio_count": len(result.lectiones),
        "lectiones": [
            serialize_lectio(lectio, lectio_id=index + 1)
            for index, lectio in enumerate(result.lectiones)
        ],
    }


def write_json(data: dict[str, Any], output_file: str | Path) -> None:
    """Write pipeline results as UTF-8 JSON."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def format_index_markdown(
    index: HumanReadableIndex,
    optimizer_name: str,
    min_reference_words: int,
    missing_verse_tolerance: int,
    excluded_books: Sequence[str] = (),
) -> str:
    """Render the aggregated page index as Markdown."""
    lines = [
        "# Biblical index",
        "",
        f"- Optimizer: `{optimizer_name}`",
        f"- Minimum reference words: {min_reference_words}",
        f"- Missing verse tolerance: {missing_verse_tolerance}",
    ]
    if excluded_books:
        lines.append(f"- Excluded books: {', '.join(excluded_books)}")
    lines.append("")

    if not index.pages:
        lines.append("_No retained readings._")
        return "\n".join(lines) + "\n"

    for page in index.pages:
        lines.extend((f"## {page.page}", ""))
        for reading in page.readings:
            continuation = []
            if reading.continues_from_previous_page:
                continuation.append("continued")
            if reading.continues_on_next_page:
                continuation.append("continues")

            suffix = (
                f" _({', '.join(continuation)})_" if continuation else ""
            )
            lines.append(f"- {reading.reference}{suffix}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_markdown_index(
    index: HumanReadableIndex,
    output_file: str | Path,
    optimizer_name: str,
    min_reference_words: int,
    missing_verse_tolerance: int,
    excluded_books: Sequence[str] = (),
) -> None:
    """Write the aggregated index as UTF-8 Markdown."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        format_index_markdown(
            index=index,
            optimizer_name=optimizer_name,
            min_reference_words=min_reference_words,
            missing_verse_tolerance=missing_verse_tolerance,
            excluded_books=excluded_books,
        ),
        encoding="utf-8",
    )


def build_aggregated_index(
    pipeline_output: dict[str, Any],
    optimizer_name: str,
    bible_file: str | Path,
    min_reference_words: int,
    missing_verse_tolerance: int,
    excluded_books: Sequence[str] = (),
) -> HumanReadableIndex:
    """Build the aggregated page index displayed by the CLI."""
    excluded = {
        book.strip().casefold()
        for book in excluded_books
        if book.strip()
    }
    index_input = pipeline_output
    if excluded:
        optimizer_result = pipeline_output["results"][optimizer_name]

        def is_retained(lectio: dict[str, Any]) -> bool:
            book = lectio.get("book")
            if book is None and lectio.get("steps"):
                book = lectio["steps"][0].get("book")
            return book is None or str(book).casefold() not in excluded

        retained_lectiones = [
            lectio
            for lectio in optimizer_result["lectiones"]
            if is_retained(lectio)
        ]
        index_input = {
            **pipeline_output,
            "results": {
                **pipeline_output["results"],
                optimizer_name: {
                    **optimizer_result,
                    "lectio_count": len(retained_lectiones),
                    "lectiones": retained_lectiones,
                },
            },
        }

    builder = IndexBuilder(
        reference_word_counts=load_reference_word_counts(bible_file),
        min_reference_words=min_reference_words,
        missing_verse_tolerance=missing_verse_tolerance,
    )
    return builder.build_from_pipeline_output(
        index_input,
        optimizer_name=optimizer_name,
    )


def select_index_optimizer(
    requested: str | None,
    pipeline_optimizer: str,
) -> str:
    """Choose which optimizer result should be rendered in the console."""
    if requested is not None:
        return requested
    if pipeline_optimizer == "naive":
        return "naive"
    return "global"


def print_pipeline_report(
    output_file: str | Path,
    index_output_file: str | Path,
    pipeline_output: dict[str, Any],
    index: HumanReadableIndex,
    index_optimizer: str,
    min_reference_words: int,
    missing_verse_tolerance: int,
    excluded_books: Sequence[str] = (),
) -> None:
    """Print a concise pipeline summary and its aggregated index."""
    print(f"Wrote {output_file}")
    print(f"Wrote {index_output_file}")
    for name, result in pipeline_output["results"].items():
        print(f"{name}: {result['lectio_count']} lectiones")

    print(
        f"\nAggregated index ({index_optimizer}, "
        f"min_reference_words={min_reference_words}, "
        f"missing_verse_tolerance={missing_verse_tolerance}, "
        f"excluded_books={list(excluded_books)})"
    )
    rendered = format_index(index)
    if not rendered:
        print("  (no retained readings)")
        return
    print(rendered, end="" if rendered.endswith("\n") else "\n")


def resolve_bible_file(bible_file: str | Path | None) -> Path:
    """Fall back to the bundled Vulgate when no biblical text is given."""
    return Path(bible_file) if bible_file is not None else default_bible_file()


def run_and_report(
    transcription_files: Sequence[str | Path],
    bible_file: str | Path | None,
    output_file: str | Path,
    lookup_file: str | Path = "bible_lookup.json",
    optimizer_name: str = "both",
    index_optimizer: str | None = None,
    window_size: int = 20,
    shift: int = 5,
    ngram_size: int = 4,
    max_hits: int = 5,
    noise_level: float = 0.15,
    start_penalty: float = 0.10,
    skip_penalty: float = 0.03,
    jump_penalty: float = 0.02,
    lookback: int = 3,
    max_verse_jump: int = 3,
    baseline_threshold: float | None = 0.15,
    min_reference_words: int = 1,
    missing_verse_tolerance: int = 3,
    index_output_file: str | Path | None = None,
    excluded_books: Sequence[str] = (),
) -> tuple[dict[str, Any], HumanReadableIndex]:
    """Run the pipeline, write detailed JSON and print the page index.

    This is the Python entry point for experiments and evaluations. It
    returns both the complete serialized result and the aggregated index.
    """
    with timed_stage("Full pipeline"):
        resolved_bible_file = resolve_bible_file(bible_file)
        resolved_index_output = (
            Path(index_output_file)
            if index_output_file is not None
            else Path(output_file).with_suffix(".md")
        )
        selected_optimizer = select_index_optimizer(
            requested=index_optimizer,
            pipeline_optimizer=optimizer_name,
        )
        if optimizer_name != "both" and selected_optimizer != optimizer_name:
            raise ValueError(
                f"Cannot index {selected_optimizer!r} when only "
                f"{optimizer_name!r} is run"
            )

        output = run_pipeline(
            transcription_files=transcription_files,
            bible_file=resolved_bible_file,
            output_file=output_file,
            lookup_file=lookup_file,
            optimizer_name=optimizer_name,
            window_size=window_size,
            shift=shift,
            ngram_size=ngram_size,
            max_hits=max_hits,
            noise_level=noise_level,
            start_penalty=start_penalty,
            skip_penalty=skip_penalty,
            jump_penalty=jump_penalty,
            lookback=lookback,
            max_verse_jump=max_verse_jump,
            baseline_threshold=baseline_threshold,
        )
        with timed_stage("Build aggregated index"):
            index = build_aggregated_index(
                pipeline_output=output,
                optimizer_name=selected_optimizer,
                bible_file=resolved_bible_file,
                min_reference_words=min_reference_words,
                missing_verse_tolerance=missing_verse_tolerance,
                excluded_books=excluded_books,
            )
        with timed_stage("Write Markdown index"):
            write_markdown_index(
                index=index,
                output_file=resolved_index_output,
                optimizer_name=selected_optimizer,
                min_reference_words=min_reference_words,
                missing_verse_tolerance=missing_verse_tolerance,
                excluded_books=excluded_books,
            )
        print_pipeline_report(
            output_file=output_file,
            index_output_file=resolved_index_output,
            pipeline_output=output,
            index=index,
            index_optimizer=selected_optimizer,
            min_reference_words=min_reference_words,
            missing_verse_tolerance=missing_verse_tolerance,
            excluded_books=excluded_books,
        )
    return output, index


def run_pipeline(
    transcription_files: Sequence[str | Path],
    bible_file: str | Path | None,
    output_file: str | Path,
    lookup_file: str | Path = "bible_lookup.json",
    optimizer_name: str = "both",
    window_size: int = 20,
    shift: int = 5,
    ngram_size: int = 4,
    max_hits: int = 5,
    noise_level: float = 0.15,
    start_penalty: float = 0.10,
    skip_penalty: float = 0.03,
    jump_penalty: float = 0.02,
    lookback: int = 3,
    max_verse_jump: int = 3,
    baseline_threshold: float | None = None,
) -> dict[str, Any]:
    """Run segmentation, candidate retrieval and path reconstruction."""
    if optimizer_name not in {"global", "naive", "both"}:
        raise ValueError("optimizer_name must be global, naive or both")

    bible_file = resolve_bible_file(bible_file)

    with timed_stage("Load Bible and prepare verse mapper"):
        biblical_text = load_biblical_text(bible_file)
        verse_mapper = prepare_verse_mapper(
            biblical_text=biblical_text,
            lookup_file=lookup_file,
            ngram_size=ngram_size,
        )

    with timed_stage("Segment transcriptions"):
        segmenter = Segmenter(transcription_files)
        segments = segmenter.segment_all(
            shift=shift,
            window_size=window_size,
        )

    with timed_stage("Map segments to verse candidates"):
        segment_mapper = SegmentMapper(
            verse_mapper=verse_mapper,
            max_hits=max_hits,
        )
        mapped_segments = list(segment_mapper.map_segments(segments))

    results: dict[str, Any] = {}

    if optimizer_name in {"global", "both"}:
        with timed_stage("Run global optimizer"):
            optimizer = PathOptimizer(
                reference_lookup=verse_mapper.lookup,
                noise_level=noise_level,
                start_penalty=start_penalty,
                skip_penalty=skip_penalty,
                jump_penalty=jump_penalty,
                lookback=lookback,
                max_verse_jump=max_verse_jump,
            )
            results["global"] = serialize_result(
                optimizer.optimize(mapped_segments)
            )

    if optimizer_name in {"naive", "both"}:
        with timed_stage("Run naive optimizer"):
            baseline = NaiveOptimizer(
                reference_lookup=verse_mapper.lookup,
                threshold=baseline_threshold,
                max_verse_jump=max_verse_jump,
            )
            results["naive"] = serialize_result(
                baseline.optimize(mapped_segments)
            )

    output = {
        "configuration": {
            "transcription_files": [
                str(Path(path)) for path in transcription_files
            ],
            "bible_file": str(Path(bible_file)),
            "lookup_file": str(Path(lookup_file)),
            "optimizer": optimizer_name,
            "window_size": window_size,
            "shift": shift,
            "ngram_size": ngram_size,
            "max_hits": max_hits,
            "noise_level": noise_level,
            "start_penalty": start_penalty,
            "skip_penalty": skip_penalty,
            "jump_penalty": jump_penalty,
            "lookback": lookback,
            "max_verse_jump": max_verse_jump,
            "baseline_threshold": baseline_threshold,
        },
        "statistics": {
            "token_count": len(segmenter.tokens),
            "segment_count": len(segments),
            "mapped_segment_count": sum(
                bool(mapped.candidates) for mapped in mapped_segments
            ),
        },
        "results": results,
    }

    with timed_stage("Write pipeline JSON"):
        write_json(output, output_file)
    return output


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Map ordered HTR transcriptions to biblical readings."
    )
    parser.add_argument(
        "transcriptions",
        nargs="+",
        type=Path,
        help="Transcription files in manuscript reading order.",
    )
    parser.add_argument(
        "--bible",
        type=Path,
        default=None,
        help=(
            "Biblical text JSON. Defaults to the bundled "
            "Sixto-Clementine Vulgate."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("index.json"))
    parser.add_argument(
        "--index-output",
        type=Path,
        default=None,
        help=(
            "Markdown index path. Defaults to the JSON output path with "
            "an .md extension."
        ),
    )
    parser.add_argument(
        "--lookup",
        type=Path,
        default=Path("bible_lookup.json"),
    )
    parser.add_argument(
        "--optimizer",
        choices=("global", "naive", "both"),
        default="both",
    )
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--shift", type=int, default=5)
    parser.add_argument("--ngram-size", type=int, default=3)
    parser.add_argument("--max-hits", type=int, default=5)
    parser.add_argument("--noise-level", type=float, default=0.15)
    parser.add_argument("--start-penalty", type=float, default=0.10)
    parser.add_argument("--skip-penalty", type=float, default=0.03)
    parser.add_argument("--jump-penalty", type=float, default=0.02)
    parser.add_argument("--lookback", type=int, default=3)
    parser.add_argument("--max-verse-jump", type=int, default=3)
    parser.add_argument("--baseline-threshold", type=float, default=None)
    parser.add_argument(
        "--index-optimizer",
        choices=("global", "naive"),
        default=None,
        help="Optimizer result rendered as the aggregated console index.",
    )
    parser.add_argument(
        "--min-reference-words",
        "--min-verses",
        dest="min_reference_words",
        type=int,
        default=1,
        help=(
            "Minimum canonical word count required to retain a reading "
            "(--min-verses is a deprecated compatibility alias)."
        ),
    )
    parser.add_argument("--missing-verse-tolerance", type=int, default=0)
    parser.add_argument(
        "--exclude-book",
        "--exclude",
        dest="excluded_books",
        action="append",
        default=[],
        metavar="BOOK",
        help=(
            "Book to omit from the console and Markdown indexes. May be "
            "repeated; the detailed pipeline JSON remains unchanged."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable informational logging.",
    )
    return parser


def main() -> None:
    """Parse arguments and run the complete pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    if (
        args.index_optimizer is not None
        and args.optimizer != "both"
        and args.index_optimizer != args.optimizer
    ):
        parser.error(
            f"Cannot display {args.index_optimizer!r} when only "
            f"{args.optimizer!r} is run"
        )

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        filename="index.log",
        filemode='a',
        datefmt='%Y-%m-%d %H:%M:%S',
        format="%(levelname)s: %(message)s",
    )

    run_and_report(
        transcription_files=args.transcriptions,
        bible_file=args.bible,
        output_file=args.output,
        lookup_file=args.lookup,
        optimizer_name=args.optimizer,
        index_optimizer=args.index_optimizer,
        window_size=args.window_size,
        shift=args.shift,
        ngram_size=args.ngram_size,
        max_hits=args.max_hits,
        noise_level=args.noise_level,
        start_penalty=args.start_penalty,
        skip_penalty=args.skip_penalty,
        jump_penalty=args.jump_penalty,
        lookback=args.lookback,
        max_verse_jump=args.max_verse_jump,
        baseline_threshold=args.baseline_threshold,
        min_reference_words=args.min_reference_words,
        missing_verse_tolerance=args.missing_verse_tolerance,
        index_output_file=args.index_output,
        excluded_books=args.excluded_books,
    )


if __name__ == "__main__":
    main()
