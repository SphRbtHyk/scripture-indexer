# How it works

This page explains the scientific ideas. A full scientific paper with details and benchmark results will follow soon.

```
pages ──► 1. Transcription ──► 2. Token stream ──► 3. Sliding windows
                                                        │
 index ◄── 6. Reading ◄──────── 5. Optimization ◄────────┤
           reconstruction        (2 variants)     4. Verse retrieval
```

## 1. Transcription (HTR)

Module: `src/scripture_indexer/htr.py`

Every page image is passed to [Kraken](https://kraken.re), a handwriting
recognition engine. Kraken segments the page
into text lines and recognizes each line, producing an ALTO XML file.
Scripture Indexer then extracts the plain text and writes one
`transcription.txt` per page, in reading order.

PDFs are rasterized page by page with PyMuPDF (300 DPI by default) before
transcription.

## 2. Token stream

Module: `src/scripture_indexer/segmenter.py` (tokenization part)

All transcription files are concatenated in the order you supplied The text is
then split into tokens (whitespace-separated words), with a small normalization step: words hyphenated across a line break are joined back into one word (`quo-` + `niam` becomes `quoniam`).

The result is one long, ordered list of tokens covering the whole
manuscript.

## 3. Sliding windows

Module: `src/scripture_indexer/segmenter.py` (windowing part)

A biblical reading rarely starts and ends at a window boundary that we
could know in advance. So instead of cutting the token stream into fixed
blocks, Scripture Indexer slides a window across it:

- each window contains `--window-size` consecutive tokens (default 20);
- the window then advances by `--shift` tokens (default 5), so consecutive
  windows overlap heavily;
- a final window is placed at the very end so the document tail is covered.

Every window keeps its position both in the token stream and in the
manuscript (which pages and lines it covers). A window may straddle a page
boundary.

## 4. Verse retrieval

Module: `src/scripture_indexer/mapper.py`

Each window must be compared to all ~31,000 verses of the Vulgate. Doing
this with exact string matching would fail immediately (HTR text is never
exact), and comparing character by character with every verse would be far
too slow. Scripture Indexer solves both problems with an **inverted index
of character n-grams**:

1. Each biblical verse is normalized (lowercased, punctuation removed,
   whitespace collapsed) and broken into its set of character n-grams
   (`--ngram-size`, default 3 in the CLI). For `in principio`, the 3-grams
   include `in `, `n p`, ` pr`, `pri`, ...
2. An inverted index maps each n-gram to the verses containing it. This is
   built once, from the biblical text, and saved as `bible_lookup.json`
   together with the verse references.
3. For each window, the n-grams of its text are looked up in the index.
   Only verses sharing at least one n-gram are considered — everything
   else is skipped without ever being compared.
4. Each surviving verse is scored with the **Jaccard similarity** between
   its n-gram set and the window's n-gram set:

   ```
   score = |shared n-grams| / |union of n-grams|
   ```

   The score is 1.0 for identical texts and decreases as texts diverge.
   Because n-grams are short substrings, a word that is misspelled in a few
   characters still shares most of its n-grams with the correct one, making our pipeline resilient to possible HTR noise.

Each window keeps only the `--max-hits` best verses (default 5). Windows
whose best score is very low are effectively noise; the optimizer in the
next stage decides what to do with them.

## 5. Optimization

Modules: `src/scripture_indexer/optimizer.py` (global) and
`src/scripture_indexer/naive_optimizer.py` (naive)

We now have, for every window, a short list of candidate verses with
scores. The goal of the optimizer is to pick, out of all candidates, the chain that best explains the manuscript as a sequence of readings.

Scripture Indexer implements two strategies, that can be selected using the `--optimizer` flag and can run both in one pass (`--optimizer both`, the default):

### Global optimizer

The global optimizer (`--optimizer global`) is a dynamic-programming
search over the sequence of windows, which enables to determine at each step the best explanation of the found reading, according to an internal score. It walks the windows in order, carrying the best answer forward, and then backtracks at the end to recover the full explanation, our index.

The "goodness" of an explanation is an additive score. When a window
selects a verse, it earns:

```
contribution = coverage × (jaccard − noise_level) − penalties
```

Piece by piece:

- **jaccard − noise_level**: the emission. Subtracting
  `--noise-level` (default 0.15) discounts the similarity that any random
  Latin text would show against the Bible.
- **coverage**: when windows overlap, a verse selected for a second window
  mostly re-asserts text that was already explained. Only the fraction of
  *new* tokens in the window counts fully. Without this, ten overlapping
  windows quoting the same verse would earn ten rewards.
- **penalties**: three optional costs,
  - `--start-penalty` (default 0.10): paid each time a *new* reading
    begins. Keeps the optimizer from declaring a new reading at every
    window.
  - `--skip-penalty` (default 0.03): paid for each window skipped inside a
    reading. A reading that jumps over many unmapped windows is suspect.
  - `--jump-penalty` (default 0.02): paid for each verse skipped inside
    the Bible. A reading that quotes verse 1 then verse 9 is suspicious
    unless the score gain justifies it.

A candidate may **continue** a previous reading only under hard constraints:

- the previous window is at most `--lookback` windows back (default 3);
- both verses are in the same book;
- the verse ID moves forward by at most `--max-verse-jump` (default 3) —
  readings always progress forward through the Bible.

Otherwise, the candidate must **start** a new reading (paying
`--start-penalty`) or simply not be selected at all.

Because the score is additive and each window's best states depend only on
previous windows, dynamic programming finds the **globally optimal** set of
readings — the single selection of (window, verse) pairs with the highest
total score — without trying every combination. Backtracking through the
choices and splitting at each "start" transition yields the readings, each with its verse chain, its page provenance, and its score.

### Naive optimizer

The naive optimizer (`--optimizer naive`) is used as the baseline, and should in most cases return a less performant answer: for every window with at least one candidate, it returns the single best-scoring verse, with no continuity constraint, no grouping and no gap filling. Each result is one isolated window prediction.
Use `--index-optimizer naive` to see the baseline rendered as an index next to
the global one.

## 6. Reading reconstruction and the page index

Module: `src/scripture_indexer/index.py`

The optimizer's raw output is a list of verse chains with provenance.
The index builder turns it into something a human can read, applying four
rules:

**Deduplication.** Each verse appears once per reading, in canonical
biblical order, regardless of how many windows selected it.

**Fragment merging.** A reading that spans a page break is often found as
two separate fragments (one per page). Fragments are merged when they sit
on the same page or on consecutive pages **and** their biblical intervals
overlap or nearly connect, up to `--missing-verse-tolerance` missing
verses (default 0: no gap filling unless you allow it).

**Pruning.** A reading is retained only if the canonical text it
references contains at least `--min-reference-words` words (default 1). A high word number will ensure liturgical readings are filtered, and a low one will lead to every reading being detected.

**Formatting.** References are rendered in the usual notation, with gaps
kept visible:

```
Genesis 1:1–5          one run within a chapter
Genesis 1:1–2:3        a run spanning two chapters
Genesis 1:1–5; 3:1     two runs, separated by a real gap
```

Finally, each reading is split across the pages it touches. A page entry
marks whether the reading `(continued)` from the previous page or
`(continues)` on the next one. The result, displayed per page, is
printed to the console and written as YAML (`--output`), Markdown, or
plain text, depending on the command. The exact file formats are described
in [Output formats](outputs.md).

## Tuning guidelines

| Symptom | Try |
|---|---|
| Too many one-verse readings | raise `--min-reference-words`; raise `--start-penalty` |
| Readings broken into fragments | raise `--missing-verse-tolerance`; raise `--lookback` |
| Readings skipping verses | raise `--jump-penalty` or lower `--max-verse-jump` |
| Plain text misread as scripture | raise `--noise-level` |
| Missed readings (low recall) | lower `--noise-level`; raise `--window-size` or lower `--shift` |
| Slow retrieval | raise `--ngram-size` (fewer, more specific n-grams) or lower `--max-hits` |

The defaults were tuned on a corpus of medieval missals; treat them as a starting point that you need to adjust to your own work, not as constants.
