# Scripture Indexer — Documentation

<p align="center">
  <img src="assets/logo.jpg" alt="Scripture Indexer logo" width="320">
</p>

Scripture Indexer turns photographs of Latin manuscripts into a page-by-page
index of the biblical passages they quote.

It was built for missals and lectionaries — books whose content is largely
made of biblical readings — but it works on any manuscript that quotes the
Bible extensively.

## General workflow

Given page images (JPEG, PNG, TIFF, ...) or a PDF, Scripture Indexer:

1. **Transcribes** each page with the Kraken handwriting recognition engine.
2. **Searches** every piece of the transcription against a biblical text you
   provide (the Sixto-Clementine Vulgate is used as
   the default).
3. **Reconstructs** the readings (*lectiones*) — continuous biblical passages such as "John 1:1–14" using the most plausible path in the missal.
4. **Writes an index** listing, for each manuscript page, which biblical
   passages appear on it.

You can run the whole chain in one command, or use the transcription-only
pipeline when the text is already available.

## Pipeline summary

```
 page images ──► HTR (Kraken) ──► transcription.txt ──► sliding windows
                                                            │
                                    character n-gram Jaccard ▼
                                                            │
 page index ◄── reading ◄── global optimizer ◄── verse candidates
 (YAML/JSON/  reconstruction      (dynamic         (top-5 per window)
  Markdown)                          programming)
```

Each stage is described in detail in
[How it works](how_it_works.md).

## Where to start

| If you want to... | Read |
|---|---|
| install and run it once | [Installation](installation.md), then [Quickstart](quickstart.md) |
| understand the algorithm | [How it works](how_it_works.md) |
| find a specific option | [CLI reference](cli.md) |
| parse or post-process the results | [Output formats](outputs.md) |
