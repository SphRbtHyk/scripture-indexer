"""Independent window-level argmax baseline, without grouping or gap filling.

Each nonempty candidate list produces exactly one singleton LectioPath.
LectioPath is only a compatibility container, not a reconstructed reading.
Empty candidate lists produce no prediction. Ties prefer the smallest verse ID.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import warnings

from scripture_indexer.mapper import BiblicalReference, MappedSegment
from scripture_indexer.optimizer import LectioPath, OptimizationResult, SelectedStep


class NaiveOptimizer:
    """Return the highest-scoring verse independently for each window."""

    def __init__(
        self,
        reference_lookup: Mapping[int, BiblicalReference],
        threshold: float | None = None,
        max_verse_jump: int | None = None,
    ) -> None:
        """Initialize the baseline.

        Only reference_lookup affects predictions. The other arguments are
        accepted for compatibility with existing pipeline callers and ignored.
        There is no score threshold, continuity constraint, or length filter.
        """
        self.reference_lookup = reference_lookup
        if threshold is not None or max_verse_jump is not None:
            warnings.warn(
                "NaiveOptimizer ignores threshold and max_verse_jump: "
                "it now returns an independent argmax for every window.",
                UserWarning,
                stacklevel=2,
            )

    def optimize(
        self,
        mapped_segments: Sequence[MappedSegment],
    ) -> OptimizationResult:
        """Return one singleton result per window with at least one candidate.

        Input order and segment coordinates are preserved. Repeated predictions
        remain separate. Every step has transition='start' and coverage_weight=1
        because overlapping windows are not combined. Scores are descriptive
        sums of local similarities, not the global optimizer's objective.
        """
        lectiones: list[LectioPath] = []
        for mapped in mapped_segments:
            if mapped.segment.token_count < 1:
                raise ValueError("Segments cannot be empty")
            if not mapped.candidates:
                continue
            for candidate in mapped.candidates:
                if candidate.verse_id not in self.reference_lookup:
                    raise KeyError(f"Unknown verse_id: {candidate.verse_id}")
                if not 0.0 <= candidate.score <= 1.0:
                    raise ValueError("Candidate scores must be between 0 and 1")

            winner = max(
                mapped.candidates,
                key=lambda candidate: (candidate.score, -candidate.verse_id),
            )
            step = SelectedStep(
                segment=mapped.segment,
                candidate=winner,
                reference=self.reference_lookup[winner.verse_id],
                transition="start",
                coverage_weight=1.0,
                contribution=winner.score,
            )
            lectiones.append(LectioPath(steps=(step,), score=winner.score))

        return OptimizationResult(
            lectiones=tuple(lectiones),
            total_score=sum(lectio.score for lectio in lectiones),
        )
