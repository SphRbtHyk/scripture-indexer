# Installation

## Requirements

- Python 3.8 or newer.
- A biblical text in JSON format (see [the Bible file](#the-bible-file)
  below). The Sixto-Clementine Vulgate is bundled with the package.
- Kraken and a recognition model, **only** if you want to transcribe page
  images. If you already have transcriptions, you can skip this.

## Install the package

From the repository root:

```bash
pip install .
```

This installs the `scripture-indexer` command and the `scripture_indexer`
Python package.

### Extras

Two optional groups are defined in `pyproject.toml`:

```bash
pip install ".[htr]"    # Kraken, Pillow, PyMuPDF — if you want the library to perform the HTR for you
pip install ".[test]"   # pytest — needed to run the test suite
```

## HTR models

The full pipeline needs a Kraken recognition model. By default it looks for
`htr/models/catmus-medieval-1.6.0.mlmodel`; point `--model` at another model
file if you prefer. Kraken models for Latin scripts (e.g. Catmus, e-print)
are available on [Zenodo](https://zenodo.org).

A separate segmentation model can be passed with `--segmentation-model`;
otherwise Kraken uses its built-in baseline segmentation.

Check that the model file exists before running it:

```bash
scripture-indexer --help
ls htr/models/
```

## The Bible file

The biblical text must be a JSON file with three nesting levels:

```json
{
  "Genesis": {
    "1": {
      "1": "in principio creauit deus caelum et terram ",
      "2": "...",
      "3": "..."
    },
    "2": { "1": "..." }
  },
  "Exodus": { "1": { "1": "..." } }
}
```

Structure: `book → chapter → verse → text`.

Two rules matter:

- **Order is meaningful.** Books, chapters and verses must appear in
  canonical biblical order. Verses are numbered internally by their position
  in the file, and the optimizer uses that number to decide whether two
  verses can follow each other.
- **Chapters and verses are keys, not ranges.** Each verse has its own entry.

The Sixto-Clementine Vulgate is bundled with the package
(`src/scripture_indexer/data/vulgata.json`, 73 books) and follows these
rules, so the indexer works without any `--bible` option:

```bash
scripture-indexer path/to/pages
```

Pass `--bible` to use a different biblical text instead.

## Sanity check

Checks that everything is properly installed by running the example in your root folder:

```bash
scripture-indexer example/page1.pdf
```

