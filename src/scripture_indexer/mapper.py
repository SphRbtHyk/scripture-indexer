"""Retrieve biblical candidates for HTR segments with exact Jaccard."""

from __future__ import annotations

import heapq
import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from scripture_indexer.segmenter import Segment


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BiblicalReference:
    """A structured reference stored in the biblical lookup."""

    verse_id: int
    book: str
    chapter: str
    verse: str

    @property
    def label(self) -> str:
        """Return a compact human-readable reference."""
        return f"{self.book}-{self.chapter}-{self.verse}"


@dataclass(frozen=True, slots=True)
class VerseCandidate:
    """A lightweight candidate returned by retrieval."""

    verse_id: int
    score: float


@dataclass(frozen=True, slots=True)
class MappedSegment:
    """A segment and its ranked biblical candidates."""

    segment: Segment
    candidates: tuple[VerseCandidate, ...]


def load_lookup(
    lookup_file: str | Path,
) -> dict[int, BiblicalReference]:
    """Load a structured biblical lookup from JSON."""
    with Path(lookup_file).open("r", encoding="utf-8") as file:
        serialized = json.load(file)

    return {
        int(verse_id): BiblicalReference(
            verse_id=int(verse_id),
            book=data["book"],
            chapter=data["chapter"],
            verse=data["verse"],
        )
        for verse_id, data in serialized.items()
    }


class VerseMapper:
    """Retrieve verses with an inverted character n-gram index."""

    def __init__(
        self,
        biblical_text: Mapping[str, Mapping[str, Mapping[str, str]]],
        n: int = 4,
        index_output: str | Path | None = "bible_lookup.json",
    ) -> None:
        """Build a stable lookup and an exact Jaccard retrieval index.

        The input mapping must follow canonical biblical order because
        ``verse_id`` is subsequently used as the verse ordinal.
        """
        if n < 1:
            raise ValueError("n must be positive")

        self.n = n
        self.lookup: dict[int, BiblicalReference] = {}
        self.texts: list[str] = []
        self.verse_ngrams: list[frozenset[str]] = []
        self.postings: dict[str, tuple[int, ...]] = {}

        self._build_index(biblical_text)
        if index_output is not None:
            self.dump_lookup(index_output)

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase text, remove punctuation and normalize whitespace."""
        text = re.sub(r"[^\w\s]", "", text.lower())
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _char_ngrams(cls, text: str, n: int) -> frozenset[str]:
        """Return distinct character n-grams from normalized text."""
        normalized = cls._normalize(text)
        if len(normalized) < n:
            return frozenset()

        return frozenset(
            normalized[index : index + n]
            for index in range(len(normalized) - n + 1)
        )

    def _build_index(
        self,
        biblical_text: Mapping[str, Mapping[str, Mapping[str, str]]],
    ) -> None:
        """Precompute verse n-grams and inverted postings."""
        mutable_postings: defaultdict[str, list[int]] = defaultdict(list)

        for book, chapters in biblical_text.items():
            for chapter, verses in chapters.items():
                for verse, text in verses.items():
                    verse_id = len(self.lookup)
                    ngrams = self._char_ngrams(text, self.n)

                    self.lookup[verse_id] = BiblicalReference(
                        verse_id=verse_id,
                        book=str(book),
                        chapter=str(chapter),
                        verse=str(verse),
                    )
                    self.texts.append(text)
                    self.verse_ngrams.append(ngrams)

                    for ngram in ngrams:
                        mutable_postings[ngram].append(verse_id)

        self.postings = {
            ngram: tuple(verse_ids)
            for ngram, verse_ids in mutable_postings.items()
        }

        logger.info(
            "Built index containing %d verses and %d distinct %d-grams",
            len(self.lookup),
            len(self.postings),
            self.n,
        )

    def dump_lookup(self, output_file: str | Path) -> None:
        """Write the structured reference lookup to JSON."""
        serialized = {
            str(verse_id): {
                "book": reference.book,
                "chapter": reference.chapter,
                "verse": reference.verse,
            }
            for verse_id, reference in self.lookup.items()
        }

        with Path(output_file).open("w", encoding="utf-8") as file:
            json.dump(serialized, file, ensure_ascii=False, indent=2)

        logger.info("Dumped biblical lookup to %s", output_file)

    def _score(
        self,
        query_size: int,
        verse_id: int,
        intersection_size: int,
    ) -> float:
        """Compute Jaccard from precomputed set cardinalities."""
        union_size = (
            query_size
            + len(self.verse_ngrams[verse_id])
            - intersection_size
        )
        return intersection_size / union_size if union_size else 0.0

    def search(
        self,
        input_text: str,
        max_hits: int | None = 5,
    ) -> tuple[VerseCandidate, ...]:
        """Return candidates ranked by exact character n-gram Jaccard."""
        if max_hits is not None and max_hits < 1:
            raise ValueError("max_hits must be positive or None")

        input_ngrams = self._char_ngrams(input_text, self.n)
        if not input_ngrams:
            return ()

        intersections: defaultdict[int, int] = defaultdict(int)
        for ngram in input_ngrams:
            for verse_id in self.postings.get(ngram, ()):
                intersections[verse_id] += 1

        if max_hits is None:
            verse_ids = list(self.lookup)
        elif not intersections:
            return ()
        else:
            verse_ids = heapq.nlargest(
                max_hits,
                intersections,
                key=lambda verse_id: (
                    self._score(
                        len(input_ngrams),
                        verse_id,
                        intersections[verse_id],
                    ),
                    -verse_id,
                ),
            )

        candidates = [
            VerseCandidate(
                verse_id=verse_id,
                score=self._score(
                    len(input_ngrams),
                    verse_id,
                    intersections.get(verse_id, 0),
                ),
            )
            for verse_id in verse_ids
        ]

        if max_hits is None:
            candidates.sort(key=lambda item: (-item.score, item.verse_id))

        return tuple(candidates)

    def compute_jaccard(
        self,
        input_text: str,
        max_hits: int | None = 5,
    ) -> dict[str, float]:
        """Return the human-readable reference-to-score representation."""
        return {
            self.lookup[candidate.verse_id].label: candidate.score
            for candidate in self.search(input_text, max_hits=max_hits)
        }


class SegmentMapper:
    """Attach ranked biblical candidates to HTR segments."""

    def __init__(self, verse_mapper: VerseMapper, max_hits: int = 5) -> None:
        if max_hits < 1:
            raise ValueError("max_hits must be positive")

        self.verse_mapper = verse_mapper
        self.max_hits = max_hits

    def map_segment(self, segment: Segment) -> MappedSegment:
        """Map one segment to at most ``max_hits`` candidates."""
        return MappedSegment(
            segment=segment,
            candidates=self.verse_mapper.search(
                segment.text,
                max_hits=self.max_hits,
            ),
        )

    def map_segments(
        self,
        segments: Iterable[Segment],
    ) -> Iterator[MappedSegment]:
        """Map an ordered segment stream lazily."""
        for segment in segments:
            yield self.map_segment(segment)
