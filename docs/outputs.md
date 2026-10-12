# Output formats

Scripture Indexer writes several files per run. Which ones appear depends on
the command used. This page describes each format so you can parse them,
post-process them, or feed them into other tools.

| File | Written by | Format |
|---|---|---|
| `index.yaml` | `scripture-indexer` | Aggregated page index |
| `index.json` | `scripture-indexer`, `run_pipeline` | Detailed pipeline result |
| `index.log` | all commands | Log file |
| `bible_lookup.json` | first run of any indexing command | Verse reference lookup |
| `recognition.alto.xml`, `transcription.txt`, ... | HTR stage | Per-page Kraken artifacts |

## `index.yaml` — aggregated page index

Written by `scripture-indexer` to `--output`. One page list per manuscript
page. This is the file to read if you want only the results of the pipeline.

```yaml
optimizer: global
min_reference_words: 1
missing_verse_tolerance: 0
excluded_books: []
pages:
  - page: 001
    readings:
      - reference: John 1:1–14
        continues_from_previous_page: false
        continues_on_next_page: false
```

- `page` is the page folder name (transcription source ID).
- `reference` is the biblical passage formatted in standard notation; gaps
  inside a reading are shown with `;`, never hidden.
- The two `continues_*` flags mark readings crossing page boundaries.
- The header repeats the settings that shaped the index.

## `index.json` — detailed pipeline result

Written by `scripture-indexer` (into the HTR output directory) and by
`run_pipeline` (to `--output`). This is the file to read if you want to
know **why** a reference was detected and better understand the reasons for bugs.

```jsonc
{
  "configuration": {
    // everything the run used: files, optimizer, window size,
    // n-gram size, penalties, ...
  },
  "statistics": {
    "token_count": 48213,          // tokens in the whole stream
    "segment_count": 9641,          // sliding windows
    "mapped_segment_count": 8712    // windows with at least one candidate
  },
  "results": {
    "global": { /* see below */ },
    "naive": { /* same shape, one singleton lectio per window */ }
  }
}
```

Each optimizer result contains the reconstructed readings:

```jsonc
{
  "total_score": 87.31,
  "lectio_count": 12,
  "lectiones": [
    {
      "lectio_id": 1,
      "book": "John",
      "chapter_start": "1", "verse_start": "1",
      "chapter_end": "1",  "verse_end": "14",
      "source_start": "001", "line_start": 1,
      "source_end": "001",   "line_end": 42,
      "path_score": 7.94,
      "mean_jaccard": 0.41,
      "boundary_precision": "window-level",
      "steps": [
        {
          "window_id": 0,
          "token_start": 0, "token_end": 20,
          "verse_id": 23803,
          "reference": "John-1-1",
          "jaccard": 0.52,
          "transition": "start",        // "start" or "continue"
          "coverage_weight": 1.0,
          "contribution": 0.27,
          "book": "John", "chapter": "1", "verse": "1",
          "source_spans": [
            { "source_id": "001", "line_start": 1, "line_end": 4 }
          ]
        }
        // one entry per selected (window, verse) pair
      ]
    }
  ]
}
```

Important fields for debugging:
- `steps[].jaccard` — similarity between the window text and the verse.
- `steps[].transition` — `start` begins a new reading, `continue` extends
  the previous one.
- `steps[].contribution` — what this step added to the global score, after
  coverage weighting and penalties.
- `source_spans` — page and line ranges the window covers; the union of
  the spans of a reading's steps locates it exactly in the manuscript.
- `path_score` and `mean_jaccard` — overall quality indicators for the
  reading.

## HTR artifacts

Per processed page, under the HTR output directory:

```
htr/full_pipeline_output/
├── _rendered_pdf_pages/        # PDF pages rasterized to PNG
└── 001/
    ├── recognition.alto.xml    # Kraken ALTO XML (text, coordinates, confidence)
    ├── transcription.txt       # plain text, one line per manuscript line
    ├── segmentation_overlay.jpg# page with detected baselines drawn
    ├── lines/                  # line_0001.jpg, ... one crop per line
    └── results.jsonl (htr.py only)  # line-level JSON records
```

The transcription text is the input consumed by the indexing pipeline.
Poor results can often be explained by the quality of the HTR output. I would suggest checking first the quality of the HTR as a culprit for poor results.
