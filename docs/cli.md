# CLI reference

The indexation uses the `scripture-indexer` command, that performs the three steps: page transcription using Kraken, transcription indexing using matching, and finally writes the aggregated index as a YAML file.

## Minimal example

Using the provided example page, run:
```bash
scripture-indexer example/page1.pdf
```

### Input and HTR options

| Option | Default | Description |
|---|---|---|
| `input` (positional) | `None | Image, PDF, or directory of pages, in reading order |
| `--page NAME` | — | Process a single page or PDF from the input directory; extension optional |
| `--model PATH` | `htr/models/catmus-medieval-1.6.0.mlmodel` | Kraken recognition model |
| `--segmentation-model PATH` | — | Optional Kraken segmentation model |
| `--limit N` | 0 | Process at most N pages; 0 = all |
| `--pdf-dpi N` | 300 | Rasterization resolution for PDFs (minimum 72) |
| `--device DEV` | auto | `auto`, `cpu`, `cuda:0`, ... |
| `--batch-size N` | 8 | Kraken recognition batch size |
| `--threads N` | 2 | Kraken worker threads |

### Indexing options

| Option | Default | Description |
|---|---|---|
| `--bible PATH` | bundled Vulgate | Biblical text JSON (see [Installation](installation.md#the-bible-file)) |
| `--output PATH` | `index.yaml` | Where to write the aggregated page index |
| `--lookup PATH` | `bible_lookup.json` | |
| `--optimizer {global,naive,both}` | both | Which optimizer(s) to run |
| `--index-optimizer {global,naive}` | global | Which optimizer's result is rendered as the aggregated index |
| `--window-size N` | 20 | Tokens per sliding window |
| `--shift N` | 5 | Token step between consecutive windows |
| `--ngram-size N` | 3 | Character n-gram length for retrieval |
| `--max-hits N` | 5 | Candidate verses kept per window |
| `--noise-level F` | 0.15 | Jaccard baseline subtracted from every match |
| `--start-penalty F` | 0.10 | Cost of starting a new reading |
| `--skip-penalty F` | 0.03 | Cost per skipped window inside a reading |
| `--jump-penalty F` | 0.02 | Cost per skipped verse inside a reading |
| `--lookback N` | 3 | Windows eligible for continuing a reading |
| `--max-verse-jump N` | 3 | Largest forward verse jump allowed |
| `--baseline-threshold F` | — | Accepted (and ignored) by the naive optimizer, kept for compatibility |

### Index shaping options

| Option | Default | Description |
|---|---|---|
| `--min-reference-words N` | 1 | Minimum canonical word count to retain a reading (`--min-verses` is a deprecated alias) |
| `--missing-verse-tolerance N` | 0 | Missing verses tolerated when merging connected fragments |
| `--exclude-book BOOK` | — | Omit a book from the YAML index; repeatable (`--exclude` alias) |

### Other

| Option | Description |
|---|---|
| `--verbose` | Log informational messages to `index.log` |
| `--htr-output-dir PATH` | `pipeline_output/full_pipeline_output` | Directory for Kraken artifacts and the detailed `index.json` |

**Lookup file behavior.** On the first run, the lookup is built from the
biblical text. On later runs it is verified against the current text; if it
does not match, the run stops rather than silently overwriting the lookup.
Use `--build-lookup` to rebuild it on purpose, e.g. after changing
`--bible` or `--ngram-size`.

**Incompatibility check.** `--index-optimizer` must name an optimizer that
is actually run: you cannot display the `naive` index while running only
`global`, and vice versa.
