"""Kraken HTR pipeline for manuscript images and PDFs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from itertools import islice
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "dataset" / "metz_ms10"
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "catmus-medieval-1.6.0.mlmodel"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "kraken_output"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
PDF_SUFFIXES = {".pdf"}
INPUT_SUFFIXES = IMAGE_SUFFIXES | PDF_SUFFIXES


def natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


def find_inputs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.casefold() not in INPUT_SUFFIXES:
            raise ValueError(f"Extension d'entrée non prise en charge: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Entrée introuvable: {path}")
    return sorted(
        (item for item in path.iterdir() if item.is_file() and item.suffix.casefold() in INPUT_SUFFIXES),
        key=natural_key,
    )


def select_input(input_path: Path, page: str) -> Path:
    """Resolve an exact image/PDF filename (or unique stem) in a directory."""
    if not input_path.is_dir():
        raise ValueError("--page s'utilise avec un dossier d'entrée, pas avec une image")
    exact = input_path / page
    if exact.is_file() and exact.suffix.casefold() in INPUT_SUFFIXES:
        return exact
    matches = [source for source in find_inputs(input_path) if source.stem == page]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Nom de page ambigu: {page}. Donnez le nom complet avec extension.")
    raise FileNotFoundError(f"Page introuvable dans {input_path}: {page}")


def iter_page_images(
    sources: Iterable[Path], rendered_dir: Path, *, pdf_dpi: int
) -> Iterable[Path]:
    """Yield images directly and lazily rasterized PDF pages."""
    for source in sources:
        if source.suffix.casefold() in IMAGE_SUFFIXES:
            yield source
            continue

        try:
            import pymupdf
        except ImportError as exc:
            raise RuntimeError(
                "La prise en charge des PDF nécessite PyMuPDF. Installez-le avec: "
                f"{sys.executable} -m pip install pymupdf"
            ) from exc

        pdf_dir = rendered_dir / source.stem
        pdf_dir.mkdir(parents=True, exist_ok=True)
        try:
            document = pymupdf.open(source)
        except Exception as exc:
            raise RuntimeError(f"Impossible d'ouvrir le PDF {source}: {exc}") from exc
        with document:
            for page_number, page in enumerate(document, start=1):
                image_path = pdf_dir / f"{source.stem}__page_{page_number:04d}.png"
                if not image_path.is_file():
                    pixmap = page.get_pixmap(dpi=pdf_dpi, alpha=False)
                    pixmap.save(image_path)
                yield image_path


def kraken_executable() -> Path:
    candidate = Path(sys.executable).with_name("kraken")
    if candidate.is_file():
        return candidate
    raise RuntimeError(
        "Kraken est absent de cet environnement. Installez-le avec: "
        f"{sys.executable} -m pip install kraken"
    )


def run_kraken(
    image: Path,
    alto: Path,
    *,
    model: Path,
    device: str,
    batch_size: int,
    threads: int,
    segmentation_model: Path | None,
) -> None:
    command = [
        str(kraken_executable()),
        "-a",
        "-i", str(image), str(alto),
        "--device", device,
        "--threads", str(threads),
        "--raise-on-error",
        "segment",
        "--baseline",
    ]
    if segmentation_model:
        command.extend(["--model", str(segmentation_model)])
    command.extend(
        [
            "ocr",
            "--model", str(model),
            "--batch-size", str(batch_size),
            "--num-line-workers", "0",
            "--base-dir", "L",
        ]
    )
    print(f"Kraken: {image.name}", file=sys.stderr)
    subprocess.run(command, check=True)


def _points(value: str | None) -> list[tuple[int, int]]:
    if not value:
        return []
    points = []
    for pair in value.split():
        try:
            x, y = pair.split(",", maxsplit=1)
            points.append((round(float(x)), round(float(y))))
        except ValueError:
            continue
    return points


def _line_text(line: ET.Element) -> tuple[str, float | None]:
    words: list[str] = []
    confidences: list[float] = []
    for element in line.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "String":
            words.append(element.get("CONTENT", ""))
            try:
                confidences.append(float(element.get("WC", "")))
            except ValueError:
                pass
        elif tag == "SP" and words and not words[-1].endswith(" "):
            words.append(" ")
    return "".join(words).strip(), fmean(confidences) if confidences else None


def parse_alto(alto: Path) -> Iterable[tuple[str, list[tuple[int, int]], str, float | None]]:
    root = ET.parse(alto).getroot()
    for index, line in enumerate(root.iterfind(".//{*}TextLine"), start=1):
        polygon = line.find("./{*}Shape/{*}Polygon")
        points = _points(polygon.get("POINTS") if polygon is not None else None)
        if not points:
            left = round(float(line.get("HPOS", "0")))
            top = round(float(line.get("VPOS", "0")))
            width = round(float(line.get("WIDTH", "0")))
            height = round(float(line.get("HEIGHT", "0")))
            points = [(left, top), (left + width, top), (left + width, top + height), (left, top + height)]
        text, confidence = _line_text(line)
        yield line.get("ID", f"line_{index:04d}"), points, text, confidence


def save_inspection_artifacts(
    source: Path, alto: Path, page_dir: Path
) -> list[dict[str, Any]]:
    lines = list(parse_alto(alto))
    crops_dir = page_dir / "lines"
    crops_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with Image.open(source) as original:
        image = original.convert("RGB")
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for number, (line_id, polygon, text, confidence) in enumerate(lines, start=1):
            xs, ys = zip(*polygon, strict=True)
            box = (max(0, min(xs)), max(0, min(ys)), min(image.width, max(xs) + 1), min(image.height, max(ys) + 1))
            crop_path = crops_dir / f"line_{number:04d}.jpg"
            image.crop(box).save(crop_path, quality=95)
            draw.line(polygon + [polygon[0]], fill=(255, 0, 0), width=3)
            draw.text((box[0], max(0, box[1] - 14)), str(number), fill=(255, 0, 0))
            results.append(
                {
                    "source": str(source),
                    "alto": str(alto),
                    "line_id": line_id,
                    "line": number,
                    "polygon": [list(point) for point in polygon],
                    "box": list(box),
                    "crop": str(crop_path),
                    "text": text,
                    "confidence": confidence,
                }
            )
        overlay.save(page_dir / "segmentation_overlay.jpg", quality=90)
    (page_dir / "transcription.txt").write_text(
        "\n".join(result["text"] for result in results) + "\n", encoding="utf-8"
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--page",
        help="Nom d'une seule image ou d'un PDF à traiter (extension facultative)",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Modèle de reconnaissance Kraken")
    parser.add_argument("--segmentation-model", type=Path, help="Modèle de segmentation optionnel")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=5, help="Pages à traiter; 0 = toutes")
    parser.add_argument("--pdf-dpi", type=int, default=300, help="Résolution de rasterisation des PDF")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, etc.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit doit être positif ou nul")
    if args.batch_size < 1 or args.threads < 1:
        parser.error("--batch-size et --threads doivent être strictement positifs")
    if args.pdf_dpi < 72:
        parser.error("--pdf-dpi doit être supérieur ou égal à 72")
    return args


def main() -> int:
    args = parse_args()
    try:
        if not args.model.is_file():
            raise FileNotFoundError(
                f"Modèle absent: {args.model}\n"
                "Téléchargez frolat_medieval_abbr.mlmodel depuis "
                "https://doi.org/10.5281/zenodo.7516308 ou utilisez --model."
            )
        if args.segmentation_model and not args.segmentation_model.is_file():
            raise FileNotFoundError(f"Modèle de segmentation absent: {args.segmentation_model}")
        sources = [select_input(args.input, args.page)] if args.page else find_inputs(args.input)
        page_images = iter_page_images(
            sources,
            args.output_dir / "_rendered_pdf_pages",
            pdf_dpi=args.pdf_dpi,
        )
        images = list(islice(page_images, args.limit)) if args.limit else list(page_images)
        if not images:
            raise ValueError("Aucune image à traiter")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = args.output_dir / "results.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as output:
            for image in images:
                page_dir = args.output_dir / image.stem
                page_dir.mkdir(parents=True, exist_ok=True)
                alto = page_dir / "recognition.alto.xml"
                run_kraken(
                    image,
                    alto,
                    model=args.model,
                    device=args.device,
                    batch_size=args.batch_size,
                    threads=args.threads,
                    segmentation_model=args.segmentation_model,
                )
                results = save_inspection_artifacts(image, alto, page_dir)
                for result in results:
                    serialized = json.dumps(result, ensure_ascii=False)
                    print(serialized)
                    output.write(serialized + "\n")
                output.flush()
                print(f"{image.name}: {len(results)} ligne(s)", file=sys.stderr)
        print(f"Résultats: {jsonl_path}", file=sys.stderr)
    except (FileNotFoundError, OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"erreur: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
