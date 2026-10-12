"""Segment ordered HTR transcriptions into overlapping token windows."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Token:
    """A token and its location in a transcription file."""

    text: str
    source_id: str
    line_start: int
    line_end: int


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Line interval covered by a segment in one source."""

    source_id: str
    line_start: int
    line_end: int


@dataclass(frozen=True, slots=True)
class Segment:
    """An overlapping token window with global and source coordinates."""

    window_id: int
    text: str
    token_start: int
    token_end: int
    source_start: str
    line_start: int
    source_end: str
    line_end: int
    source_spans: tuple[SourceSpan, ...] = ()

    @property
    def token_count(self) -> int:
        """Return the number of tokens in the half-open token interval."""
        return self.token_end - self.token_start


class Segmenter:
    """Read ordered HTR files as one continuous token sequence."""

    def __init__(
        self,
        transcription_files: Sequence[str | Path],
        source_id_fn: Callable[[Path], str] | None = None,
    ) -> None:
        """Load files in the supplied order and retain token provenance.

        Args:
            transcription_files: HTR files in manuscript reading order.
            source_id_fn: Optional function deriving a page/source identifier
                from a path. The filename stem is used by default.
        """
        if not transcription_files:
            raise ValueError("transcription_files cannot be empty")

        self.paths = tuple(Path(path) for path in transcription_files)
        self.source_id_fn = source_id_fn or\
            (lambda path: str(path.parent.name))
        self.tokens = self._load_tokens()

    @staticmethod
    def _clean_token(token: str) -> str:
        """Remove HTR hyphenation marks and surrounding whitespace."""
        return token.replace("-", "").strip()

    @classmethod
    def _tokenize_source(cls, text: str, source_id: str) -> list[Token]:
        """Tokenize one source and join words hyphenated across lines."""
        tokens: list[Token] = []
        carry_text: str | None = None
        carry_line: int | None = None

        for line_number, line in enumerate(text.splitlines(), start=1):
            words = line.split()
            if not words:
                continue

            start_lines = [line_number] * len(words)

            if carry_text is not None:
                words[0] = carry_text + words[0]
                start_lines[0] = carry_line or line_number
                carry_text = None
                carry_line = None

            if words[-1].endswith("-"):
                carried_word = words.pop()
                carry_text = cls._clean_token(carried_word)
                carry_line = start_lines.pop()

            for word, start_line in zip(words, start_lines):
                cleaned = cls._clean_token(word)
                if cleaned:
                    tokens.append(
                        Token(
                            text=cleaned,
                            source_id=source_id,
                            line_start=start_line,
                            line_end=line_number,
                        )
                    )

        if carry_text:
            final_line = carry_line or 1
            tokens.append(
                Token(
                    text=carry_text,
                    source_id=source_id,
                    line_start=final_line,
                    line_end=final_line,
                )
            )

        return tokens

    def _load_tokens(self) -> list[Token]:
        """Load every source into one ordered token list."""
        tokens: list[Token] = []
        source_ids: set[str] = set()

        for path in self.paths:
            source_id = self.source_id_fn(path)
            if source_id in source_ids:
                raise ValueError(
                    f"Duplicate source identifier: {source_id!r}. "
                    "Provide a source_id_fn producing unique identifiers."
                )

            source_ids.add(source_id)
            with path.open("r", encoding="utf-8") as file:
                tokens.extend(self._tokenize_source(file.read(), source_id))

        return tokens

    @staticmethod
    def _window_starts(
        token_count: int,
        window_size: int,
        shift: int,
    ) -> list[int]:
        """Return starts while ensuring coverage of the document tail."""
        if token_count == 0:
            return []
        if token_count <= window_size:
            return [0]

        starts = list(range(0, token_count - window_size + 1, shift))
        final_start = token_count - window_size
        if starts[-1] != final_start:
            starts.append(final_start)

        return starts

    @staticmethod
    def _source_spans(tokens: Sequence[Token]) -> tuple[SourceSpan, ...]:
        """Collect the line interval covered in every traversed source."""
        spans: list[SourceSpan] = []

        for token in tokens:
            if spans and spans[-1].source_id == token.source_id:
                previous = spans[-1]
                spans[-1] = SourceSpan(
                    source_id=previous.source_id,
                    line_start=min(previous.line_start, token.line_start),
                    line_end=max(previous.line_end, token.line_end),
                )
            else:
                spans.append(
                    SourceSpan(
                        source_id=token.source_id,
                        line_start=token.line_start,
                        line_end=token.line_end,
                    )
                )

        return tuple(spans)

    def iter_segments(
        self,
        shift: int = 5,
        window_size: int = 20,
    ) -> Iterator[Segment]:
        """Yield ordered windows, including windows spanning two sources."""
        if shift < 1:
            raise ValueError("shift must be positive")
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if shift > window_size:
            raise ValueError("shift cannot exceed window_size")

        starts = self._window_starts(
            token_count=len(self.tokens),
            window_size=window_size,
            shift=shift,
        )

        for window_id, start in enumerate(starts):
            end = min(start + window_size, len(self.tokens))
            window_tokens = self.tokens[start:end]
            first = window_tokens[0]
            last = window_tokens[-1]
            source_spans = self._source_spans(window_tokens)

            yield Segment(
                window_id=window_id,
                text=" ".join(token.text for token in window_tokens),
                token_start=start,
                token_end=end,
                source_start=first.source_id,
                line_start=first.line_start,
                source_end=last.source_id,
                line_end=last.line_end,
                source_spans=source_spans,
            )

    def segment_all(
        self,
        shift: int = 5,
        window_size: int = 20,
    ) -> list[Segment]:
        """Return all windows as a list."""
        return list(self.iter_segments(shift=shift, window_size=window_size))
