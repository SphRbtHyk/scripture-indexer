"""Run Kraken HTR and the missal-to-biblical-index pipeline in one command.

The input is a manuscript page image, a PDF, or a directory of such files.
Kraken transcribes every page; the resulting transcription files are then
segmented, mapped to biblical verses and reconstructed into readings. The
aggregated page index is written as YAML. The detailed pipeline result
remains available as JSON in the HTR output directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import yaml

from scripture_indexer.htr import (
    find_inputs,
    iter_page_images,
    run_kraken,
    save_inspection_artifacts,
    select_input,
)
from scripture_indexer.index import HumanReadableIndex
from scripture_indexer.mapper import VerseMapper
from scripture_indexer.run_pipeline import (
    build_aggregated_index,
    load_biblical_text,
    print_pipeline_report,
    resolve_bible_file,
    run_pipeline,
    select_index_optimizer,
    timed_stage,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "htr" / "models" / "catmus-medieval-1.6.0.mlmodel"
DEFAULT_HTR_OUTPUT_DIR = REPO_ROOT / "pipeline_output" / "full_pipeline_output"
DEFAULT_OUTPUT = Path("pipeline_output/index.yaml")


def build_lookup_file(
    bible_file: str | Path | None,
    lookup_file: str | Path,
    ngram_size: int,
) -> None:
    """Build the biblical lookup from the biblical text and exit.

    The lookup is normally created on the first pipeline run and then
    treated as a stable artifact: later runs verify it against the current
    biblical text and refuse to overwrite it. This explicit build is the
    way to (re)create it on purpose, e.g. after changing the Bible file
    or the n-gram size.
    """
    bible_file = resolve_bible_file(bible_file)
    with timed_stage("Build biblical lookup"):
        verse_mapper = VerseMapper(
            biblical_text=load_biblical_text(bible_file),
            n=ngram_size,
            index_output=None,
        )
        verse_mapper.dump_lookup(lookup_file)
    print(f"Wrote {lookup_file} ({len(verse_mapper.lookup)} verses)")


def transcribe_pages(
    input_path: Path,
    *,
    model: Path,
    output_dir: Path,
    segmentation_model: Path | None = None,
    page: str | None = None,
    limit: int = 0,
    pdf_dpi: int = 300,
    device: str = "auto",
    batch_size: int = 8,
    threads: int = 2,
) -> list[Path]:
    """Transcribe every input page with Kraken and return the text files.

    One ``transcription.txt`` file is produced per page under
    ``output_dir/<image stem>/``, in manuscript reading order.
    """
    sources = (
        [select_input(input_path, page)] if page else find_inputs(input_path)
    )
    page_images = iter_page_images(
        sources,
        output_dir / "_rendered_pdf_pages",
        pdf_dpi=pdf_dpi,
    )
    images = list(islice(page_images, limit)) if limit else list(page_images)
    if not images:
        raise ValueError("Aucune image à traiter")

    output_dir.mkdir(parents=True, exist_ok=True)
    transcription_files: list[Path] = []
    for image in images:
        page_dir = output_dir / image.stem
        page_dir.mkdir(parents=True, exist_ok=True)
        alto = page_dir / "recognition.alto.xml"
        run_kraken(
            image,
            alto,
            model=model,
            device=device,
            batch_size=batch_size,
            threads=threads,
            segmentation_model=segmentation_model,
        )
        results = save_inspection_artifacts(image, alto, page_dir)
        transcription_files.append(page_dir / "transcription.txt")
        print(f"{image.name}: {len(results)} ligne(s)", file=sys.stderr)

    return transcription_files


def write_yaml(data: dict[str, Any], output_file: str | Path) -> None:
    """Write pipeline results as UTF-8 YAML."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=100,
        )


def format_index_yaml(
    index: HumanReadableIndex,
    optimizer_name: str,
    min_reference_words: int,
    missing_verse_tolerance: int,
    excluded_books: Sequence[str] = (),
) -> dict[str, Any]:
    """Convert the aggregated page index to a YAML-compatible dictionary."""
    return {
        "optimizer": optimizer_name,
        "min_reference_words": min_reference_words,
        "missing_verse_tolerance": missing_verse_tolerance,
        "excluded_books": list(excluded_books),
        "pages": [
            {
                "page": page.page,
                "readings": [
                    {
                        "reference": reading.reference,
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
        ],
    }


def run_full_pipeline(
    input_path: str | Path,
    bible_file: str | Path | None,
    output_file: str | Path = DEFAULT_OUTPUT,
    model: str | Path = DEFAULT_MODEL,
    htr_output_dir: str | Path = DEFAULT_HTR_OUTPUT_DIR,
    segmentation_model: str | Path | None = None,
    page: str | None = None,
    limit: int = 0,
    pdf_dpi: int = 300,
    device: str = "auto",
    batch_size: int = 8,
    threads: int = 2,
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
    excluded_books: Sequence[str] = (),
) -> dict[str, Any]:
    """Transcribe HTR pages and write the aggregated page index as YAML.

    This is the Python entry point for the integrated pipeline. The YAML
    output contains only the aggregated page index; the detailed pipeline
    result remains available as JSON in the HTR output directory. Returns
    the serialized page index.
    """
    htr_output = Path(htr_output_dir)
    bible_file = resolve_bible_file(bible_file)
    json_output = htr_output / "index.json"
    selected_optimizer = select_index_optimizer(
        requested=index_optimizer,
        pipeline_optimizer=optimizer_name,
    )
    if optimizer_name != "both" and selected_optimizer != optimizer_name:
        raise ValueError(
            f"Cannot index {selected_optimizer!r} when only "
            f"{optimizer_name!r} is run"
        )

    with timed_stage("HTR transcription (Kraken)"):
        transcription_files = transcribe_pages(
            Path(input_path),
            model=Path(model),
            output_dir=htr_output,
            segmentation_model=(
                Path(segmentation_model) if segmentation_model else None
            ),
            page=page,
            limit=limit,
            pdf_dpi=pdf_dpi,
            device=device,
            batch_size=batch_size,
            threads=threads,
        )

    with timed_stage("Indexing pipeline"):
        output = run_pipeline(
            transcription_files=transcription_files,
            bible_file=bible_file,
            output_file=json_output,
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
                bible_file=bible_file,
                min_reference_words=min_reference_words,
                missing_verse_tolerance=missing_verse_tolerance,
                excluded_books=excluded_books,
            )
        with timed_stage("Write YAML index"):
            index_yaml = format_index_yaml(
                index=index,
                optimizer_name=selected_optimizer,
                min_reference_words=min_reference_words,
                missing_verse_tolerance=missing_verse_tolerance,
                excluded_books=excluded_books,
            )
            write_yaml(index_yaml, output_file)
        print_pipeline_report(
            output_file=json_output,
            index_output_file=output_file,
            pipeline_output=output,
            index=index,
            index_optimizer=selected_optimizer,
            min_reference_words=min_reference_words,
            missing_verse_tolerance=missing_verse_tolerance,
            excluded_books=excluded_books,
        )

    return index_yaml


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe manuscript images with Kraken and map the result "
            "to biblical readings, writing a YAML index."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Image, PDF or directory of pages in reading order.",
    )
    parser.add_argument(
        "--page",
        help="Nom d'une seule image ou d'un PDF à traiter (extension facultative)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Modèle de reconnaissance Kraken",
    )
    parser.add_argument(
        "--segmentation-model",
        type=Path,
        help="Modèle de segmentation optionnel",
    )
    parser.add_argument(
        "--htr-output-dir",
        type=Path,
        default=DEFAULT_HTR_OUTPUT_DIR,
        help=(
            "Directory receiving Kraken artifacts (ALTO XML, line crops, "
            "transcriptions) and the detailed JSON and Markdown indexes."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Pages à traiter; 0 = toutes",
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=300,
        help="Résolution de rasterisation des PDF",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda:0, etc.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--bible",
        type=Path,
        default=None,
        help=(
            "Biblical text JSON. Defaults to the bundled "
            "Sixto-Clementine Vulgate."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="YAML output path for the aggregated page index.",
    )
    parser.add_argument(
        "--lookup",
        type=Path,
        default=Path("bible_lookup.json"),
    )
    parser.add_argument(
        "--build-lookup",
        action="store_true",
        help=(
            "Build the biblical lookup from --bible at --lookup using "
            "--ngram-size, then exit. Overwrites an existing lookup file; "
            "without this flag an existing lookup is verified and never "
            "overwritten."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=("global", "naive", "both"),
        default="global",
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
        default="global",
        help="Optimizer result rendered as the aggregated console index.",
    )
    parser.add_argument(
        "--min-reference-words",
        "--min-verses",
        dest="min_reference_words",
        type=int,
        default=30,
        help=(
            "Minimum canonical word count required to retain a reading "
            "(--min-verses is a deprecated compatibility alias)."
        ),
    )
    parser.add_argument("--missing-verse-tolerance", type=int, default=2)
    parser.add_argument(
        "--exclude-book",
        "--exclude",
        dest="excluded_books",
        action="append",
        default=[],
        metavar="BOOK",
        help=(
            "Book to omit from the console and Markdown indexes. May be "
            "repeated; the detailed pipeline outputs remain unchanged."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable informational logging.",
    )
    return parser


def main() -> None:
    """Parse arguments and run the integrated HTR + indexing pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit doit être positif ou nul")
    if args.batch_size < 1 or args.threads < 1:
        parser.error("--batch-size et --threads doivent être strictement positifs")
    if args.pdf_dpi < 72:
        parser.error("--pdf-dpi doit être supérieur ou égal à 72")
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
        filemode="a",
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(levelname)s: %(message)s",
    )

    if args.build_lookup:
        build_lookup_file(
            bible_file=args.bible,
            lookup_file=args.lookup,
            ngram_size=args.ngram_size,
        )
        return

    run_full_pipeline(
        input_path=args.input,
        bible_file=args.bible,
        output_file=args.output,
        model=args.model,
        htr_output_dir=args.htr_output_dir,
        segmentation_model=args.segmentation_model,
        page=args.page,
        limit=args.limit,
        pdf_dpi=args.pdf_dpi,
        device=args.device,
        batch_size=args.batch_size,
        threads=args.threads,
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
        excluded_books=args.excluded_books,
    )


if __name__ == "__main__":
    main()
