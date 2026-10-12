"""Find optimal biblical paths among scored HTR segments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from scripture_indexer.mapper import BiblicalReference, \
    MappedSegment, VerseCandidate
from scripture_indexer.segmenter import Segment


Transition = Literal["start", "continue"]


@dataclass(frozen=True, slots=True)
class SelectedStep:
    """One candidate selected by the global optimizer."""

    segment: Segment
    candidate: VerseCandidate
    reference: BiblicalReference
    transition: Transition
    coverage_weight: float
    contribution: float


@dataclass(frozen=True, slots=True)
class LectioPath:
    """One continuous biblical reading reconstructed from candidate nodes."""

    steps: tuple[SelectedStep, ...]
    score: float

    @property
    def reference_start(self) -> str:
        """Return the first selected verse label."""
        return self.steps[0].reference.label

    @property
    def reference_end(self) -> str:
        """Return the last selected verse label."""
        return self.steps[-1].reference.label

    @property
    def mean_jaccard(self) -> float:
        """Return coverage-weighted Jaccard for reporting, not decoding."""
        weight = sum(step.coverage_weight for step in self.steps)
        if weight == 0:
            return 0.0

        return sum(
            step.coverage_weight * step.candidate.score
            for step in self.steps
        ) / weight


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Globally selected lectiones and their total objective score."""

    lectiones: tuple[LectioPath, ...]
    total_score: float


@dataclass(frozen=True, slots=True)
class _BackPointer:
    """Internal link to the preceding selected candidate."""

    transition: Transition
    previous_window: int | None
    previous_candidate: int | None
    coverage_weight: float
    contribution: float


@dataclass(frozen=True, slots=True)
class _State:
    """Best accumulated score ending at one candidate."""

    score: float
    pointer: _BackPointer


class PathOptimizer:
    """Select globally optimal, non-conflicting biblical paths."""

    def __init__(
        self,
        reference_lookup: Mapping[int, BiblicalReference],
        noise_level: float,
        start_penalty: float,
        skip_penalty: float = 0.0,
        jump_penalty: float = 0.0,
        lookback: int = 3,
        max_verse_jump: int = 3,
    ) -> None:
        """Configure the additive path objective.

        Args:
            reference_lookup: Stable mapping from verse IDs to references.
            noise_level: Jaccard baseline subtracted from every emission.
            start_penalty: Cost of starting each new lectio.
            skip_penalty: Cost per missing window within a lectio.
            jump_penalty: Cost per skipped verse beyond the next verse.
            lookback: Previous windows eligible for continuation.
            max_verse_jump: Largest forward verse jump allowed.
        """
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError("noise_level must be between 0 and 1")
        if min(start_penalty, skip_penalty, jump_penalty) < 0:
            raise ValueError("penalties must be non-negative")
        if lookback < 1:
            raise ValueError("lookback must be positive")
        if max_verse_jump < 0:
            raise ValueError("max_verse_jump cannot be negative")

        self.reference_lookup = reference_lookup
        self.noise_level = noise_level
        self.start_penalty = start_penalty
        self.skip_penalty = skip_penalty
        self.jump_penalty = jump_penalty
        self.lookback = lookback
        self.max_verse_jump = max_verse_jump

    def _emission(self, candidate: VerseCandidate) -> float:
        """Center a Jaccard score relative to the noise level."""
        return candidate.score - self.noise_level

    @staticmethod
    def _coverage_weight(previous: Segment, current: Segment) -> float:
        """Return the proportion of current tokens not previously covered."""
        new_tokens = current.token_end - max(
            current.token_start,
            previous.token_end,
        )
        new_tokens = max(0, min(new_tokens, current.token_count))
        return new_tokens / current.token_count

    def _compatible(
        self,
        previous: VerseCandidate,
        current: VerseCandidate,
    ) -> bool:
        """Test whether candidates may continue the same lectio."""
        previous_ref = self.reference_lookup[previous.verse_id]
        current_ref = self.reference_lookup[current.verse_id]
        verse_jump = current.verse_id - previous.verse_id

        return (
            previous_ref.book == current_ref.book
            and 0 <= verse_jump <= self.max_verse_jump
        )

    def _continuation_penalty(
        self,
        previous_window: int,
        current_window: int,
        previous: VerseCandidate,
        current: VerseCandidate,
    ) -> float:
        """Penalize missing windows and discontinuous verse progression."""
        missing_windows = current_window - previous_window - 1
        verse_jump = current.verse_id - previous.verse_id

        return (
            self.skip_penalty * missing_windows
            + self.jump_penalty * max(0, verse_jump - 1)
        )

    @staticmethod
    def _choose(current: _State, score: float, pointer: _BackPointer) -> _State:
        """Keep the strictly better state for deterministic tie handling."""
        if score > current.score:
            return _State(score=score, pointer=pointer)
        return current

    def _validate(self, mapped_segments: Sequence[MappedSegment]) -> None:
        """Validate coordinates, scores and lookup coverage."""
        for mapped in mapped_segments:
            if mapped.segment.token_count < 1:
                raise ValueError("Segments cannot be empty")

            for candidate in mapped.candidates:
                if candidate.verse_id not in self.reference_lookup:
                    raise KeyError(f"Unknown verse_id: {candidate.verse_id}")
                if not 0.0 <= candidate.score <= 1.0:
                    raise ValueError("Candidate scores must be between 0 and 1")

        for previous, current in zip(mapped_segments, mapped_segments[1:]):
            if current.segment.token_start <= previous.segment.token_start:
                raise ValueError("Segment token starts must strictly increase")
            if current.segment.token_end < previous.segment.token_end:
                raise ValueError("Segment token ends must not decrease")

    def _update_nonoverlap(
        self,
        mapped_segments: Sequence[MappedSegment],
        states: Sequence[Sequence[_State]],
        current_i: int,
        cursor: int,
        current_best: tuple[int, int, _State] | None,
    ) -> tuple[tuple[int, int, _State] | None, int]:
        """Add newly non-overlapping windows to the completed-state pool."""
        current_start = mapped_segments[current_i].segment.token_start

        while (
            cursor < current_i
            and mapped_segments[cursor].segment.token_end <= current_start
        ):
            for candidate_j, state in enumerate(states[cursor]):
                if current_best is None or state.score > current_best[2].score:
                    current_best = (cursor, candidate_j, state)
            cursor += 1

        return current_best, cursor

    def _start_state(self, emission: float) -> _State:
        """Create the state for the first selected candidate."""
        contribution = emission - self.start_penalty
        return _State(
            score=contribution,
            pointer=_BackPointer(
                transition="start",
                previous_window=None,
                previous_candidate=None,
                coverage_weight=1.0,
                contribution=contribution,
            ),
        )

    def _start_after(
        self,
        best: _State,
        previous_i: int,
        previous_j: int,
        previous_state: _State,
        emission: float,
        coverage: float,
    ) -> _State:
        """Evaluate a new lectio following a selected candidate."""
        contribution = coverage * emission - self.start_penalty
        return self._choose(
            best,
            previous_state.score + contribution,
            _BackPointer(
                transition="start",
                previous_window=previous_i,
                previous_candidate=previous_j,
                coverage_weight=coverage,
                contribution=contribution,
            ),
        )

    def _continue_after(
        self,
        best: _State,
        previous_i: int,
        previous_j: int,
        previous_state: _State,
        previous: VerseCandidate,
        current_i: int,
        current: VerseCandidate,
        emission: float,
        coverage: float,
    ) -> _State:
        """Evaluate continuation from one compatible predecessor."""
        if not self._compatible(previous, current):
            return best

        penalty = self._continuation_penalty(
            previous_window=previous_i,
            current_window=current_i,
            previous=previous,
            current=current,
        )
        contribution = coverage * emission - penalty

        return self._choose(
            best,
            previous_state.score + contribution,
            _BackPointer(
                transition="continue",
                previous_window=previous_i,
                previous_candidate=previous_j,
                coverage_weight=coverage,
                contribution=contribution,
            ),
        )

    def _best_candidate_state(
        self,
        mapped_segments: Sequence[MappedSegment],
        states: Sequence[Sequence[_State]],
        current_i: int,
        current_j: int,
        first_overlap: int,
        best_nonoverlap: tuple[int, int, _State] | None,
    ) -> _State:
        """Compare every valid way of reaching one candidate."""
        mapped = mapped_segments[current_i]
        candidate = mapped.candidates[current_j]
        emission = self._emission(candidate)
        best = self._start_state(emission)

        if best_nonoverlap is not None:
            previous_i, previous_j, previous_state = best_nonoverlap
            best = self._start_after(
                best,
                previous_i,
                previous_j,
                previous_state,
                emission,
                coverage=1.0,
            )

        for previous_i in range(first_overlap, current_i):
            previous_mapped = mapped_segments[previous_i]
            coverage = self._coverage_weight(
                previous_mapped.segment,
                mapped.segment,
            )
            for previous_j, previous_state in enumerate(states[previous_i]):
                best = self._start_after(
                    best,
                    previous_i,
                    previous_j,
                    previous_state,
                    emission,
                    coverage,
                )

        first_previous = max(0, current_i - self.lookback)
        for previous_i in range(first_previous, current_i):
            previous_mapped = mapped_segments[previous_i]
            coverage = self._coverage_weight(
                previous_mapped.segment,
                mapped.segment,
            )
            for previous_j, previous_candidate in enumerate(
                previous_mapped.candidates
            ):
                best = self._continue_after(
                    best,
                    previous_i,
                    previous_j,
                    states[previous_i][previous_j],
                    previous_candidate,
                    current_i,
                    candidate,
                    emission,
                    coverage,
                )

        return best

    @staticmethod
    def _best_final_state(
        states: Sequence[Sequence[_State]],
    ) -> tuple[int, int, _State] | None:
        """Return the best state ending anywhere in the document."""
        best: tuple[int, int, _State] | None = None

        for window_i, window_states in enumerate(states):
            for candidate_j, state in enumerate(window_states):
                if best is None or state.score > best[2].score:
                    best = (window_i, candidate_j, state)

        return best

    def _backtrace(
        self,
        mapped_segments: Sequence[MappedSegment],
        states: Sequence[Sequence[_State]],
        end_window: int,
        end_candidate: int,
    ) -> tuple[LectioPath, ...]:
        """Reconstruct selected candidates and group them into lectiones."""
        reversed_steps: list[SelectedStep] = []
        window_i = end_window
        candidate_j = end_candidate

        while True:
            state = states[window_i][candidate_j]
            pointer = state.pointer
            mapped = mapped_segments[window_i]
            candidate = mapped.candidates[candidate_j]

            reversed_steps.append(
                SelectedStep(
                    segment=mapped.segment,
                    candidate=candidate,
                    reference=self.reference_lookup[candidate.verse_id],
                    transition=pointer.transition,
                    coverage_weight=pointer.coverage_weight,
                    contribution=pointer.contribution,
                )
            )

            if (
                pointer.previous_window is None
                or pointer.previous_candidate is None
            ):
                break

            window_i = pointer.previous_window
            candidate_j = pointer.previous_candidate

        return self._group_lectiones(reversed(reversed_steps))

    @staticmethod
    def _group_lectiones(
        steps: Iterable[SelectedStep],
    ) -> tuple[LectioPath, ...]:
        """Split a forward step iterator at start transitions."""
        lectiones: list[LectioPath] = []
        current: list[SelectedStep] = []

        for step in steps:
            if step.transition == "start" and current:
                lectiones.append(PathOptimizer._make_lectio(current))
                current = []
            current.append(step)

        if current:
            lectiones.append(PathOptimizer._make_lectio(current))

        return tuple(lectiones)

    @staticmethod
    def _make_lectio(steps: Sequence[SelectedStep]) -> LectioPath:
        """Build one immutable lectio from selected steps."""
        return LectioPath(
            steps=tuple(steps),
            score=sum(step.contribution for step in steps),
        )

    def optimize(
        self,
        mapped_segments: Sequence[MappedSegment],
    ) -> OptimizationResult:
        """Return the globally optimal set of biblical lectiones."""
        if not mapped_segments:
            return OptimizationResult(lectiones=(), total_score=0.0)

        self._validate(mapped_segments)
        states: list[list[_State]] = []
        best_nonoverlap: tuple[int, int, _State] | None = None
        nonoverlap_cursor = 0

        for current_i, mapped in enumerate(mapped_segments):
            best_nonoverlap, nonoverlap_cursor = self._update_nonoverlap(
                mapped_segments,
                states,
                current_i,
                nonoverlap_cursor,
                best_nonoverlap,
            )

            states.append(
                [
                    self._best_candidate_state(
                        mapped_segments,
                        states,
                        current_i,
                        current_j,
                        nonoverlap_cursor,
                        best_nonoverlap,
                    )
                    for current_j in range(len(mapped.candidates))
                ]
            )

        end = self._best_final_state(states)
        if end is None or end[2].score <= 0:
            return OptimizationResult(lectiones=(), total_score=0.0)

        return OptimizationResult(
            lectiones=self._backtrace(
                mapped_segments,
                states,
                end_window=end[0],
                end_candidate=end[1],
            ),
            total_score=end[2].score,
        )
