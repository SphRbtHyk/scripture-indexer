"""Unsupervised, manuscript-level calibration of retrieval hyperparameters.

This module implements the ``--auto-hyperparam`` calibration pass: given
only the HTR transcriptions of one manuscript and the biblical text, it
estimates ``ngram_size`` and ``noise_level`` without any annotated data.
The window and shift belong to the fixed parameters and enter the
calibration as inputs: the noise level is estimated at the operating
window. The method is fully isolated from the rest of the pipeline: it
never mutates optimizer state and only consumes ``Segmenter`` and
``VerseMapper`` primitives, so the pipeline behaviour with explicit
parameters is unchanged.

The calibration is organised in two stages:

Stage A (biblical constants, independent of the manuscript):
    * ``j_null(n)``: mean Jaccard between unrelated verse pairs, the
      chance-similarity floor of the retrieval index.
    * ``j_robust(n, r)``: mean Jaccard between a verse and itself after
      synthetic character corruption at rate ``r``, the expected score
      of a true match under an HTR error rate ``r``.

Stage B (manuscript pass, on a deterministic sample of pages):
    B1. Layout statistics (tokens per page and per line), recorded as
        diagnostics.
    B2. A retrieval sweep over sampled windows of the operating size
        yields a top-1 Jaccard distribution; its Otsu valley separates
        the noise mode from true matches, the noise-mode mean
        estimates ``noise_level``, and the high-mode mean is inverted
        through ``j_robust`` into an HTR error-rate estimate.
    B3. ``ngram_size`` maximises the margin ``j_robust(n, r) - j_null(n)``
        at the estimated error rate.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from scripture_indexer.mapper import VerseMapper
from scripture_indexer.resources import default_bible_file
from scripture_indexer.segmenter import Segmenter


FALLBACK_PARAMETERS: dict[str, Any] = {
    "ngram_size": 3,
    "noise_level": 0.15,
}

CALIBRATED_PARAMETERS = (
    "ngram_size",
    "noise_level",
)

_LATIN_LETTERS = "abcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class AutoHyperparamSettings:
    """Deterministic configuration of the calibration pass.

    Args:
        verse_sample_size: Verses sampled from the Bible for Stage A.
        candidate_ngrams: Character n-gram sizes considered.
        corruption_rates: Character error rates used by the robustness
            curve of Stage A.
        coarse_ngram_size: N-gram size of the first retrieval pass.
        page_blocks: Number of contiguous page blocks sampled.
        pages_per_block: Pages per sampled block.
        window_size: Operating retrieval window; a fixed parameter
            whose value the noise estimation follows.
        sweep_starts: Number of window start offsets sampled per pass.
        seed: Seed of the Stage A verse sampling and corruption.
    """

    verse_sample_size: int = 250
    candidate_ngrams: tuple[int, ...] = (2, 3, 4, 5)
    corruption_rates: tuple[float, ...] = (
        0.03,
        0.06,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
    )
    coarse_ngram_size: int = 3
    page_blocks: int = 6
    pages_per_block: int = 3
    window_size: int = 20
    sweep_starts: int = 120
    seed: int = 0

    def fingerprint(self) -> str:
        """Return a stable digest of every Stage A input."""
        payload = json.dumps(
            {
                "verse_sample_size": self.verse_sample_size,
                "candidate_ngrams": list(self.candidate_ngrams),
                "corruption_rates": list(self.corruption_rates),
                "seed": self.seed,
            },
            sort_keys=True,
        )
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BibleConstants:
    """Stage A curves describing the retrieval index itself.

    Attributes:
        candidate_ngrams: The n-gram sizes covered by the curves.
        corruption_rates: The character error rates covered by the
            robustness curve.
        j_null: Mean Jaccard between unrelated verse pairs per n.
        j_robust: Mean Jaccard between a verse and its corrupted self
            per n and per error rate.
        digest: Fingerprint of the biblical text and settings.
    """

    candidate_ngrams: tuple[int, ...]
    corruption_rates: tuple[float, ...]
    j_null: dict[int, float]
    j_robust: dict[int, dict[float, float]]
    digest: str = ""

    def robust_score(self, n: int, rate: float) -> float:
        """Return the robustness score at the grid rate nearest ``rate``."""
        table = self.j_robust[n]
        nearest = min(table, key=lambda r: abs(r - rate))
        return table[nearest]

    def match_rate(self, n: int, observed: float) -> float:
        """Return the error rate whose robust score is nearest observed."""
        table = self.j_robust[n]
        return min(table, key=lambda r: abs(table[r] - observed))

    def margin(self, n: int, rate: float) -> float:
        """Return the expected true-match margin above the noise floor."""
        return self.robust_score(n, rate) - self.j_null[n]

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, sorted representation."""
        return {
            "digest": self.digest,
            "candidate_ngrams": list(self.candidate_ngrams),
            "corruption_rates": list(self.corruption_rates),
            "j_null": [[n, self.j_null[n]] for n in self.candidate_ngrams],
            "j_robust": [
                [
                    n,
                    [[rate, self.j_robust[n][rate]]
                     for rate in self.corruption_rates],
                ]
                for n in self.candidate_ngrams
            ],
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "BibleConstants":
        """Rebuild constants from their JSON representation."""
        return cls(
            candidate_ngrams=tuple(int(n) for n in data["candidate_ngrams"]),
            corruption_rates=tuple(
                float(rate) for rate in data["corruption_rates"]
            ),
            j_null={int(item[0]): float(item[1]) for item in data["j_null"]},
            j_robust={
                int(item[0]): {
                    float(rate): float(score)
                    for rate, score in item[1]
                }
                for item in data["j_robust"]
            },
            digest=str(data["digest"]),
        )


def _iter_verses(
    biblical_text: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> list[str]:
    """Return every verse text in canonical order."""
    return [
        verse_text
        for chapters in biblical_text.values()
        for verses in chapters.values()
        for verse_text in verses.values()
    ]


def _corrupt(text: str, rate: float, rng: random.Random) -> str:
    """Apply character deletions and substitutions at ``rate``."""
    corrupted: list[str] = []
    for character in text:
        if character.isspace():
            corrupted.append(character)
            continue
        draw = rng.random()
        if draw < rate / 2:
            continue
        if draw < rate:
            corrupted.append(rng.choice(_LATIN_LETTERS))
        else:
            corrupted.append(character)
    return "".join(corrupted)


def _jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    """Return the Jaccard similarity of two n-gram sets."""
    if not first and not second:
        return 0.0
    union_size = len(first | second)
    if union_size == 0:
        return 0.0
    return len(first & second) / union_size


def _sample_verses(
    verses: Sequence[str],
    count: int,
    rng: random.Random,
) -> list[str]:
    """Return a deterministic verse sample, at most one per verse."""
    if len(verses) <= count:
        return list(verses)
    return rng.sample(list(verses), count)


def compute_bible_constants(
    biblical_text: Mapping[str, Mapping[str, Mapping[str, str]]],
    settings: AutoHyperparamSettings | None = None,
    digest: str = "",
) -> BibleConstants:
    """Compute the Stage A chance-similarity and robustness curves.

    Args:
        biblical_text: Nested book/chapter/verse biblical text mapping.
        settings: Calibration settings; defaults are used when omitted.
        digest: Fingerprint stored alongside the constants.

    Returns:
        The biblical constants used by the manuscript calibration pass.
    """
    settings = settings or AutoHyperparamSettings()
    rng = random.Random(settings.seed)
    verses = _sample_verses(
        _iter_verses(biblical_text),
        settings.verse_sample_size,
        rng,
    )

    verse_ngrams = {
        n: [VerseMapper._char_ngrams(verse, n) for verse in verses]
        for n in settings.candidate_ngrams
    }

    j_null: dict[int, float] = {}
    j_robust: dict[int, dict[float, float]] = {}

    for n in settings.candidate_ngrams:
        ngram_sets = verse_ngrams[n]

        pair_offset = 7 if len(ngram_sets) > 7 else 1
        null_scores = [
            _jaccard(
                ngram_sets[index],
                ngram_sets[(index + pair_offset) % len(ngram_sets)],
            )
            for index in range(len(ngram_sets))
            if (index + pair_offset) % len(ngram_sets) != index
        ]
        j_null[n] = (
            sum(null_scores) / len(null_scores) if null_scores else 0.0
        )

        robust: dict[float, float] = {}
        for rate in settings.corruption_rates:
            scores = [
                _jaccard(
                    ngram_sets[index],
                    VerseMapper._char_ngrams(
                        _corrupt(verse, rate, rng),
                        n,
                    ),
                )
                for index, verse in enumerate(verses)
            ]
            robust[rate] = sum(scores) / len(scores) if scores else 0.0
        j_robust[n] = robust

    return BibleConstants(
        candidate_ngrams=tuple(settings.candidate_ngrams),
        corruption_rates=tuple(settings.corruption_rates),
        j_null=j_null,
        j_robust=j_robust,
        digest=digest,
    )


def bible_text_digest(
    biblical_text: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> str:
    """Return a stable digest of the biblical text content."""
    payload = json.dumps(biblical_text, sort_keys=True, ensure_ascii=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def load_bible_constants(
    cache_file: str | Path,
    digest: str,
) -> BibleConstants | None:
    """Load cached constants matching ``digest``, if available."""
    path = Path(cache_file)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("digest") != digest:
        return None
    return BibleConstants.from_json_dict(data)


def save_bible_constants(
    constants: BibleConstants,
    cache_file: str | Path,
) -> None:
    """Write the biblical constants to a JSON cache file."""
    path = Path(cache_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(constants.to_json_dict(), file, ensure_ascii=False, indent=2)


def _sample_page_blocks(
    transcription_files: Sequence[str | Path],
    blocks: int,
    pages_per_block: int,
) -> list[Path]:
    """Return evenly spaced contiguous page blocks, in reading order."""
    paths = [Path(path) for path in transcription_files]
    if not paths:
        return []
    if len(paths) <= pages_per_block:
        return paths

    block_count = max(1, min(blocks, len(paths) // pages_per_block))
    span = len(paths) - pages_per_block
    sampled: list[Path] = []
    seen: set[Path] = set()
    for index in range(block_count):
        if block_count == 1:
            start = 0
        else:
            start = round(index * span / (block_count - 1))
        for path in paths[start : start + pages_per_block]:
            if path not in seen:
                seen.add(path)
                sampled.append(path)
    return sampled


def _median(values: Sequence[float]) -> float:
    """Return the median of a numeric sequence."""
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def _otsu_threshold(values: Sequence[float]) -> float:
    """Return the Otsu threshold of a sample in [0, 1]."""
    if not values:
        return 0.0
    low = min(values)
    high = max(values)
    if high - low < 1e-12:
        return low

    bin_count = 128
    width = (high - low) / bin_count
    counts = [0] * bin_count
    for value in values:
        index = min(bin_count - 1, int((value - low) / width))
        counts[index] += 1

    total = len(values)
    total_sum = sum(values)
    best_threshold = low
    best_variance = -1.0
    weight_background = 0
    sum_background = 0.0

    for index in range(bin_count):
        weight_background += counts[index]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += counts[index] * (low + (index + 0.5) * width)
        mean_background = sum_background / weight_background
        mean_foreground = (total_sum - sum_background) / weight_foreground
        variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )
        if variance > best_variance:
            best_variance = variance
            best_threshold = low + (index + 1) * width

    return best_threshold


def _sweep_starts(
    token_count: int,
    max_window: int,
    count: int,
) -> list[int]:
    """Return evenly spaced window start offsets."""
    span = token_count - max_window
    if span <= 0:
        return [0] if token_count >= 1 else []
    if count <= 1:
        return [0]
    starts: list[int] = []
    seen: set[int] = set()
    for index in range(count):
        start = index * span // (count - 1)
        if start not in seen:
            seen.add(start)
            starts.append(start)
    return starts


def _sweep_scores(
    mapper: VerseMapper,
    token_texts: Sequence[str],
    starts: Sequence[int],
    window_sizes: Sequence[int],
) -> dict[int, list[float]]:
    """Retrieve the top-1 Jaccard per window size and start offset."""
    scores: dict[int, list[float]] = {}
    for window_size in window_sizes:
        window_scores: list[float] = []
        for start in starts:
            window_tokens = token_texts[start : start + window_size]
            candidates = mapper.search(
                " ".join(window_tokens),
                max_hits=1,
            )
            window_scores.append(candidates[0].score if candidates else 0.0)
        scores[window_size] = window_scores
    return scores


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


@dataclass(frozen=True)
class AutoHyperparamResult:
    """Estimated parameters and complete calibration diagnostics.

    Attributes:
        parameters: Values for the calibrated hyperparameters.
        diagnostics: JSON-compatible calibration report.
        status: ``ok`` or ``insufficient_sample``.
    """

    parameters: dict[str, Any]
    diagnostics: dict[str, Any]
    status: str

    def to_json_dict(self) -> dict[str, Any]:
        """Return the full result as a JSON-compatible dictionary."""
        return {
            "status": self.status,
            "parameters": dict(self.parameters),
            "diagnostics": self.diagnostics,
        }


def _insufficient_result(
    reason: str,
    sampled_pages: Sequence[Path],
    settings: AutoHyperparamSettings,
) -> AutoHyperparamResult:
    """Return the fallback result when the sample is unusable."""
    return AutoHyperparamResult(
        parameters=dict(FALLBACK_PARAMETERS),
        diagnostics={
            "status": "insufficient_sample",
            "reason": reason,
            "sampled_pages": [str(path) for path in sampled_pages],
            "settings": _settings_json(settings),
        },
        status="insufficient_sample",
    )


def _settings_json(settings: AutoHyperparamSettings) -> dict[str, Any]:
    """Return the settings as a JSON-compatible dictionary."""
    return {
        "verse_sample_size": settings.verse_sample_size,
        "candidate_ngrams": list(settings.candidate_ngrams),
        "corruption_rates": list(settings.corruption_rates),
        "coarse_ngram_size": settings.coarse_ngram_size,
        "page_blocks": settings.page_blocks,
        "pages_per_block": settings.pages_per_block,
        "window_size": settings.window_size,
        "sweep_starts": settings.sweep_starts,
        "seed": settings.seed,
    }


def _load_biblical_text(
    bible: str | Path | Mapping[str, Any] | None,
) -> Mapping[str, Mapping[str, Mapping[str, str]]]:
    """Accept a loaded mapping, a JSON path or the bundled Bible."""
    if bible is None:
        bible = default_bible_file()
    if isinstance(bible, Mapping):
        return bible
    with Path(bible).open("r", encoding="utf-8") as file:
        return json.load(file)


def calibrate_hyperparams(
    transcription_files: Sequence[str | Path],
    bible: str | Path | Mapping[str, Any] | None = None,
    settings: AutoHyperparamSettings | None = None,
    constants_cache: str | Path | None = None,
) -> AutoHyperparamResult:
    """Estimate retrieval hyperparameters from the manuscript alone.

    Args:
        transcription_files: HTR transcription files in reading order.
        bible: Biblical text mapping, JSON file path, or None for the
            bundled Vulgate.
        settings: Calibration settings; defaults are used when omitted.
        constants_cache: Optional JSON file caching the Stage A
            biblical constants across runs.

    Returns:
        The calibrated parameters with full diagnostics.
    """
    settings = settings or AutoHyperparamSettings()
    if not transcription_files:
        return _insufficient_result(
            "no transcription files",
            [],
            settings,
        )

    sampled_pages = _sample_page_blocks(
        transcription_files,
        settings.page_blocks,
        settings.pages_per_block,
    )
    if not sampled_pages:
        return _insufficient_result(
            "no pages sampled",
            sampled_pages,
            settings,
        )

    biblical_text = _load_biblical_text(bible)
    bible_digest = bible_text_digest(biblical_text)
    digest = sha256(
        (bible_digest + settings.fingerprint()).encode("utf-8")
    ).hexdigest()

    constants = None
    if constants_cache is not None:
        constants = load_bible_constants(constants_cache, digest)
    if constants is None:
        constants = compute_bible_constants(
            biblical_text,
            settings=settings,
            digest=digest,
        )
        if constants_cache is not None:
            save_bible_constants(constants, constants_cache)

    # B1: layout statistics, recorded as diagnostics.
    segmenter = Segmenter(sampled_pages)
    token_texts = [token.text for token in segmenter.tokens]
    tokens_per_page = Counter(
        token.source_id for token in segmenter.tokens
    ).values()
    tokens_per_line = Counter(
        (token.source_id, token.line_start) for token in segmenter.tokens
    ).values()
    median_tokens_per_page = _median(list(tokens_per_page))
    median_tokens_per_line = _median(list(tokens_per_line))

    window_size = settings.window_size
    if not token_texts or window_size >= len(token_texts):
        return _insufficient_result(
            "sampled pages contain too few tokens",
            sampled_pages,
            settings,
        )

    starts = _sweep_starts(
        token_count=len(token_texts),
        max_window=window_size,
        count=settings.sweep_starts,
    )
    if len(starts) < 10:
        return _insufficient_result(
            "sampled pages contain too few windows",
            sampled_pages,
            settings,
        )

    # B2: retrieval sweep at the operating window; its score
    # distribution estimates the noise and the HTR error rate.
    coarse_n = settings.coarse_ngram_size
    if coarse_n not in constants.j_null:
        coarse_n = constants.candidate_ngrams[0]
    mapper = VerseMapper(biblical_text=biblical_text, n=coarse_n, index_output=None)
    sweep = _sweep_scores(mapper, token_texts, starts, (window_size,))
    operating_scores = sweep[window_size]
    noise_threshold = _otsu_threshold(operating_scores)
    high_scores = [
        score for score in operating_scores if score > noise_threshold
    ] or [max(operating_scores)]
    high_mean = _mean(high_scores)
    error_rate = constants.match_rate(coarse_n, high_mean)

    # B3: n-gram size maximizes the expected true-match margin.
    margins = {
        n: constants.margin(n, error_rate)
        for n in constants.candidate_ngrams
    }
    ngram_size = max(margins, key=lambda n: (margins[n], n))

    if ngram_size != coarse_n:
        mapper = VerseMapper(
            biblical_text=biblical_text,
            n=ngram_size,
            index_output=None,
        )
        sweep = _sweep_scores(mapper, token_texts, starts, (window_size,))
        operating_scores = sweep[window_size]
        noise_threshold = _otsu_threshold(operating_scores)
        high_scores = [
            score for score in operating_scores if score > noise_threshold
        ] or [max(operating_scores)]
        high_mean = _mean(high_scores)
        error_rate = constants.match_rate(ngram_size, high_mean)

    # The Stage A true-match threshold and the resulting match share at
    # the operating window are recorded as diagnostics.
    match_threshold = 0.5 * (
        constants.j_null[ngram_size]
        + constants.robust_score(ngram_size, error_rate)
    )
    window_mean = _mean(operating_scores)
    match_fraction = (
        sum(1 for score in operating_scores if score >= match_threshold)
        / len(operating_scores)
    )

    # The noise level is the expected top-1 score without a true match:
    # the mean of the score mode below the Otsu valley.
    noise_valley = noise_threshold
    low_mode = [score for score in operating_scores if score <= noise_valley]
    noise_level = _mean(low_mode) if low_mode else noise_valley
    noise_level = min(0.9, max(0.01, noise_level))

    parameters = {
        "ngram_size": int(ngram_size),
        "noise_level": float(noise_level),
    }

    diagnostics = {
        "status": "ok",
        "sampled_pages": [str(path) for path in sampled_pages],
        "sampled_token_count": len(token_texts),
        "sampled_window_count": len(starts),
        "layout": {
            "median_tokens_per_page": median_tokens_per_page,
            "median_tokens_per_line": median_tokens_per_line,
        },
        "bible_digest": bible_digest,
        "constants": constants.to_json_dict(),
        "error_rate": error_rate,
        "high_mode_mean": high_mean,
        "ngram_margins": [
            [n, margins[n]] for n in sorted(margins)
        ],
        "coarse_ngram_size": coarse_n,
        "window_size": window_size,
        "window_mean": window_mean,
        "match_threshold": match_threshold,
        "match_fraction": match_fraction,
        "noise_valley": noise_valley,
        "noise_level": noise_level,
        "settings": _settings_json(settings),
    }

    return AutoHyperparamResult(
        parameters=parameters,
        diagnostics=diagnostics,
        status="ok",
    )


def format_calibration_report(result: AutoHyperparamResult) -> str:
    """Render a short human-readable calibration report."""
    lines = [
        "Auto hyperparameters:",
        f"  status: {result.status}",
    ]
    if result.status == "ok":
        for name in CALIBRATED_PARAMETERS:
            value = result.parameters[name]
            lines.append(f"  {name}: {value}")
        diagnostics = result.diagnostics
        lines.append(
            f"  estimated HTR error rate: "
            f"{diagnostics['error_rate']:.2f}"
        )
        lines.append(
            f"  sampled pages: "
            f"{len(diagnostics['sampled_pages'])}"
        )
    else:
        lines.append(f"  reason: {result.diagnostics['reason']}")
        for name, value in result.parameters.items():
            lines.append(f"  {name}: {value} (fallback)")
    return "\n".join(lines)


def save_calibration_result(
    result: AutoHyperparamResult,
    output_file: str | Path,
) -> None:
    """Write the calibration result and diagnostics as JSON."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(result.to_json_dict(), file, ensure_ascii=False, indent=2)
