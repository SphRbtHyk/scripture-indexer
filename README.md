# Scripture Indexer

<p align="center">
  <img src="public/logo.jpg" alt="Scripture Indexer logo" width="480">
</p>

**Scripture Indexer** automatically builds an index of biblical quotations from
photographs or scans of Latin manuscripts, such as missals and lectionaries.

You point it at a set of page images (or a PDF), and it:

1. transcribes the pages with the [Kraken](https://kraken.re) handwriting
   recognition engine,
2. compares the transcription against a biblical text of your choice
   (the Sixto-Clementine Vulgate text is used as default),
3. indentifies the biblical readings,
4. writes a page-by-page index of biblical references, ready to read as a YAML file.

## General idea

The transcription is cut into overlapping windows of consecutive words. Each
window is compared to every verse of the biblical text using character n-gram
similarity. A dynamic-programming optimizer then walks through the
windows in order and selects the most plausible chain of verses: it rewards
similarities, penalizes gaps and implausible jumps, and tries to group the verses into the longest possible sequence of readings. The retained readings are finally grouped per page and formatted as standard biblical references.

The full algorithm is described in [docs/how_it_works.md](docs/how_it_works.md).

## Installation

Requires Python 3.8 or newer.

```bash
pip install .
```

To also transcribe page images (the HTR step), install the `htr` extra, which
pulls in Kraken, Pillow and PyMuPDF:

```bash
pip install ".[htr]"
```

Go to [docs/installation.md](docs/installation.md) to see the installation of Kraken models.

## Usage

The installed command `scripture-indexer` runs everything in one go:

```bash
scripture-indexer path/to/pages
```

This writes `index.yaml` with the page index. 

If you already have transcriptions and only want the indexing part:

```bash
python -m scripture_indexer.run_pipeline page1/transcription.txt page2/transcription.txt \
    --output index.json
```

To make sure everything is working on your system, try out the minimal example bundled with the repo (excerpt from Toulouse MS91):

```bash
scripture-indexer example/page1.pdf
```

This should return in your console and in the file `pipeline_output/index.yaml` the following index:

```yaml
optimizer: global
min_reference_words: 30
missing_verse_tolerance: 2
excluded_books: []
pages:
- page: page1__page_0001
  readings:
  - reference: Acts of Apostles 8:14–17
    continues_from_previous_page: false
    continues_on_next_page: false
  - reference: John 10:1–10
    continues_from_previous_page: false
    continues_on_next_page: false
```

More information: [docs/quickstart.md](docs/quickstart.md).
All commands and options: [docs/cli.md](docs/cli.md).

## Documentation

| Document | Contents |
|---|---|
| [Overview](docs/index.md) | What the tool does and how the pieces fit together |
| [Installation](docs/installation.md) | Requirements, extras, models |
| [Quickstart](docs/quickstart.md) | First run, step by step |
| [How it works](docs/how_it_works.md) | The algorithm, explained for humans |
| [CLI reference](docs/cli.md) | Every command and option |
| [Output formats](docs/outputs.md) | What each output file contains |

## License

MIT — see `LICENSE`.
