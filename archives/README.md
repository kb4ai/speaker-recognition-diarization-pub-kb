# Archives Directory

This directory contains supplementary reference materials that support the main knowledge base.

## Structure

```
archives/
├── bibliography/    # BibTeX citation files
│   ├── embeddings.bib        # Speaker embedding papers
│   ├── diarization.bib       # Diarization papers
│   └── datasets.bib          # Dataset papers
└── papers/          # PDF copies (gitignored for size)
    └── .gitkeep
```

## Bibliography Files

The `.bib` files contain BibTeX entries for academic papers referenced throughout the knowledge base. These can be used for:

* Generating citations in academic writing
* Cross-referencing with paper entries in `papers/`
* Building comprehensive reading lists

## Paper PDFs

The `papers/` directory is for local storage of PDF copies. These are **not committed to git** (see `.gitignore`) due to size and licensing concerns.

To populate locally:

```bash
# Download papers manually or use scripts
./scripts/download-papers.py  # If available
```

## Usage

Reference bibliography entries in YAML files:

```yaml
sources:
  - arxiv: "2005.07143"
    bibtex-key: "desplanques2020ecapa"
    accessed: "2026-01-06"
```
