# Quickstart

This page walks you through a first complete run, then shows the most useful
variations. The full option list is in the [CLI reference](cli.md).

## Run the full pipeline

With the `htr` extra installed and a Kraken model in place, you can run the indexation on the given example:

```bash
scripture-indexer example/page1.pdf
```

What happens:

1. Every page of the pdf is transcribed with Kraken. Transcription results are stored by default in `pipeline_output/full_pipeline_output/`.
2. The transcriptions are indexed against the Vulgate JSON.
3. The page index is written by default to `pipeline_output/index.yaml`, as well as is printed to the console.

The console output ends with something like:
```
Wrote htr/full_pipeline_output/index.json
Wrote index.yaml
global: 12 lectiones

Aggregated index (global, min_reference_words=1, missing_verse_tolerance=0, excluded_books=[])
001
  John 1:1–14
  Psalms 22:1–9 (continues)
002
  Psalms 22:9–10 (continued)
  Genesis 1:1–5
```

`(continues)` means the reading spills over onto the next page;
`(continued)` means it started on a previous page.

## Useful parameters
Full parameters of the CLI are available in the description of the [CLI Parameters](cli.md)

**Try a few pages first.** Transcription is the slow stage. Limit the run
while you are tuning:

```bash
scripture-indexer dataset/metz_ms10 --limit 5
```

**Write the index somewhere specific:**

```bash
scripture-indexer dataset/metz_ms10 \
    --output my_index.yaml
```

**Filter out noise.** Raise the minimum reading size and allow small gaps
inside readings (see
[How it works](how_it_works.md#6-reading-reconstruction-and-the-page-index) for what these mean):

```bash
scripture-indexer dataset/metz_ms10 \
    --min-reference-words 10 --missing-verse-tolerance 3
```

⚠️ A low minimum reference words is more likely to return liturgical readings, while a higher reading will filter out shorter liturgical readings but might miss short lectiones.

**Exclude a book from the display index** (e.g. Psalms, when they dominate):

```bash
scripture-indexer dataset/metz_ms10 \
    --exclude-book "Psalms"
```

## Index existing transcriptions

If the text is already transcribed (for example if you want to use another engine than Kraken), run the indexing pipeline directly. Pass the transcription files in manuscript
reading order:

```bash
python -m scripture_indexer.run_pipeline \
    htr/full_pipeline_output/001/transcription.txt \
    htr/full_pipeline_output/002/transcription.txt \
    --output index.json
```

## Inspecting transcriptions

Each processed page gets its own folder under the HTR output directory:

```
htr/full_pipeline_output/
└── 001/
    ├── recognition.alto.xml      # Kraken ALTO XML output
    ├── transcription.txt        # plain text, one line per manuscript line
    ├── segmentation_overlay.jpg # page image with detected baselines drawn
    └── lines/                   # one crop per detected line
```

If you have poor results, you should check out `transcription.txt` and `segmentation_overlay.jpg`, as poor HTR quality is probably the culprit.

## The lookup file

The first indexing run builds `bible_lookup.json` next to your working
directory. It maps internal verse IDs (position in the biblical text) to
biblical references, and later runs reuse it instead of rebuilding it.

Runs never overwrite it silently: if it no longer matches the current
biblical text or n-gram size, the pipeline refuses to run rather than
corrupt the stable artifact. Rebuild it deliberately with:

```bash
scripture-indexer --build-lookup --ngram-size 4
```

Use `--bible my_bible.json` in the same command to build the lookup from a
different biblical text.