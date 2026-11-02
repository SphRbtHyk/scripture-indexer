"""Tests of the unsupervised auto-hyperparameter calibration."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from scripture_indexer.auto_hyperparam import (
    AutoHyperparamSettings,
    BibleConstants,
    calibrate_hyperparams,
    compute_bible_constants,
    bible_text_digest,
    load_bible_constants,
    save_bible_constants,
)
from scripture_indexer.run_pipeline import (
    build_parser,
    reject_explicit_calibrated_options,
)


SHARED_WORDS = ("et", "in")
SYLLABLES = (
    "ba", "be", "da", "de", "fi", "ga", "la", "lo", "mi",
    "na", "ro", "sa", "ta", "ve", "vo",
)


def make_verse_text(index: int, words_per_verse: int = 12) -> str:
    """Build one distinct pseudo-Latin verse from a per-verse seed."""
    rng = random.Random(1000 + index)
    distinct = [
        "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 3)))
        for _ in range(words_per_verse - len(SHARED_WORDS))
    ]
    words = list(SHARED_WORDS) + distinct
    rng.shuffle(words)
    return " ".join(words)


def make_bible(verse_count: int = 24) -> dict:
    """Build a synthetic Bible of distinct pseudo-Latin verses."""
    verses = [make_verse_text(index) for index in range(verse_count)]
    return {
        "Liber Primus": {
            "1": {str(i): text for i, text in enumerate(verses[:12])}
        },
        "Liber Secundus": {
            "1": {str(i): text for i, text in enumerate(verses[12:])}
        },
    }


def write_pages(
    directory: Path,
    verses: list[str],
    page_count: int,
    rubric: str = "rubrica ordinaria feria secunda",
) -> list[Path]:
    """Write one transcription file per page: rubrics around one verse."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for page in range(page_count):
        verse = verses[page % len(verses)]
        lines = [rubric, rubric, verse, rubric]
        path = directory / f"page_{page:02d}" / "transcription.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def make_settings() -> AutoHyperparamSettings:
    """Return compact calibration settings for the synthetic corpus."""
    return AutoHyperparamSettings(
        verse_sample_size=24,
        candidate_ngrams=(3, 4),
        corruption_rates=(0.05, 0.10, 0.20),
        coarse_ngram_size=3,
        page_blocks=2,
        pages_per_block=2,
        window_size=11,
        sweep_starts=40,
        seed=0,
    )


def make_corpus(tmp_path: Path) -> tuple[dict, list[Path]]:
    """Build a synthetic Bible and matching transcription pages."""
    bible = make_bible()
    verses = [
        verse
        for chapters in bible.values()
        for verse_text in chapters.values()
        for verse in verse_text.values()
    ]
    pages = write_pages(tmp_path / "pages", verses, page_count=12)
    return bible, pages


def test_bible_constants_curves() -> None:
    """The noise floor falls with n; robustness falls with error rate."""
    settings = make_settings()
    constants = compute_bible_constants(make_bible(), settings=settings)

    assert set(constants.j_null) == set(settings.candidate_ngrams)
    assert constants.j_null[3] >= constants.j_null[4]

    robust = constants.j_robust[3]
    rates = sorted(robust)
    for lower, higher in zip(rates, rates[1:]):
        assert robust[lower] >= robust[higher]

    assert constants.margin(3, 0.05) > constants.margin(3, 0.20)


def test_constants_cache_roundtrip(tmp_path: Path) -> None:
    """Cached constants reload identically when the digest matches."""
    bible = make_bible()
    digest = bible_text_digest(bible)
    constants = compute_bible_constants(bible, digest=digest)

    cache_file = tmp_path / "constants.json"
    save_bible_constants(constants, cache_file)
    loaded = load_bible_constants(cache_file, digest)

    assert loaded is not None
    assert loaded.j_null == constants.j_null
    assert loaded.j_robust == constants.j_robust
    assert loaded.digest == digest

    stale = load_bible_constants(cache_file, "other-digest")
    assert stale is None


def test_calibrate_returns_sound_parameters(tmp_path: Path) -> None:
    """Calibration on a clean synthetic manuscript yields sane values."""
    bible, pages = make_corpus(tmp_path)
    result = calibrate_hyperparams(
        transcription_files=pages,
        bible=bible,
        settings=make_settings(),
    )

    assert result.status == "ok"
    parameters = result.parameters
    assert parameters["ngram_size"] in (3, 4)
    assert set(parameters) == {"ngram_size", "noise_level"}
    assert 0.0 < parameters["noise_level"] < 0.5

    diagnostics = result.diagnostics
    assert diagnostics["sampled_token_count"] > 0
    assert diagnostics["error_rate"] in (0.05, 0.10, 0.20)
    assert diagnostics["window_size"] == 11
    assert len(diagnostics["ngram_margins"]) == 2


def test_match_fraction_diagnostics(tmp_path: Path) -> None:
    """Clean synthetic manuscript windows match confidently."""
    bible, pages = make_corpus(tmp_path)
    result = calibrate_hyperparams(
        pages,
        bible=bible,
        settings=make_settings(),
    )

    assert result.diagnostics["match_threshold"] > 0.0
    assert 0.0 < result.diagnostics["match_fraction"] <= 1.0
    assert result.diagnostics["window_mean"] > 0.0


def test_calibration_is_deterministic(tmp_path: Path) -> None:
    """Repeated calibration with the same seed is identical."""
    bible, pages = make_corpus(tmp_path)
    settings = make_settings()
    first = calibrate_hyperparams(pages, bible=bible, settings=settings)
    second = calibrate_hyperparams(pages, bible=bible, settings=settings)

    assert first.parameters == second.parameters
    assert first.diagnostics["match_fraction"] == second.diagnostics["match_fraction"]


def test_calibration_constants_cache(tmp_path: Path) -> None:
    """The constants cache is written once and reused on the second run."""
    bible, pages = make_corpus(tmp_path)
    cache_file = tmp_path / "cache" / "constants.json"

    first = calibrate_hyperparams(
        pages,
        bible=bible,
        settings=make_settings(),
        constants_cache=cache_file,
    )
    assert cache_file.exists()

    second = calibrate_hyperparams(
        pages,
        bible=bible,
        settings=make_settings(),
        constants_cache=cache_file,
    )
    assert first.parameters == second.parameters

    cached = BibleConstants.from_json_dict(
        __import__("json").loads(cache_file.read_text(encoding="utf-8"))
    )
    assert cached.digest == first.diagnostics["constants"]["digest"]


def test_insufficient_sample_falls_back(tmp_path: Path) -> None:
    """A tiny manuscript falls back to the documented defaults."""
    page = tmp_path / "page" / "transcription.txt"
    page.parent.mkdir(parents=True)
    page.write_text("verba paucissima\n", encoding="utf-8")

    result = calibrate_hyperparams(
        [page],
        bible=make_bible(),
        settings=make_settings(),
    )

    assert result.status == "insufficient_sample"
    assert result.parameters == {
        "ngram_size": 3,
        "noise_level": 0.15,
    }


def test_reject_explicit_calibrated_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit calibrated options conflict with --auto-hyperparam."""
    parser = build_parser()

    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--auto-hyperparam", "--ngram-size", "4"],
    )
    with pytest.raises(SystemExit):
        reject_explicit_calibrated_options(parser)

    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--auto-hyperparam", "--noise-level=0.2"],
    )
    with pytest.raises(SystemExit):
        reject_explicit_calibrated_options(parser)

    # Window and shift are fixed parameters: allowed with calibration.
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--auto-hyperparam", "--shift=7", "--verbose"],
    )
    reject_explicit_calibrated_options(parser)
