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

The console output will run:

```bash
Wrote pipeline_output/index.yaml
global: 4 lectiones

Aggregated index (global, min_reference_words=30, missing_verse_tolerance=2, excluded_books=[])
page1__page_0001
  Acts of Apostles 8:14–17
  John 10:1–10
[2026-10-12T16:40:25.991+02:00] Indexing pipeline finished in 00:00:02.960
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

> ⚠️ A low minimum reference words is more likely to return liturgical readings, while a higher reading will filter out shorter liturgical readings but might miss short lectiones.

**Exclude a book from the display index** (e.g. Psalms, if you want to include psalms from the filter):

```bash
scripture-indexer dataset/metz_ms10 \
    --exclude-book "Psalms"
```

## Index existing transcriptions

If the text is already transcribed (for example if you want to use another engine than Kraken), you can run the indexing pipeline directly on your scans. Pass the transcription files in manuscript
reading order (this is important as it conditions the optimization algorithm):

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
