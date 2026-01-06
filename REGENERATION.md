# Content Regeneration Guide

This document explains how to regenerate auto-generated content in this knowledge base.

## Overview

This repository uses **data-driven documentation**. Tables, indexes, and summaries are generated from YAML source files using scripts.

```
YAML Data → Scripts → Generated Markdown
```

## What Gets Generated

| Content | Source | Command |
|---------|--------|---------|
| `data/*/README.md` | YAML files in each directory | `make readme` |
| `papers/README.md` | `papers/*.paper.yaml` | `make readme` |
| `knowledge/README.md` | Markdown files in knowledge/ | `make readme` |
| `comparisons/auto-generated.md` | All YAML files | `make tables` |

## Quick Commands

```bash
# Validate all YAML files
make validate

# Generate comparison tables
make tables

# Generate per-directory README files
make readme

# Full rebuild (validate + tables + readme)
make all

# Clean generated content
make clean
```

## Regeneration Workflow

### After Adding New Entries

```bash
# 1. Validate the new entry
./scripts/check-yaml.py data/tools/my-new-tool.tool.yaml

# 2. Regenerate affected README
make readme

# 3. Regenerate comparison tables
make tables

# 4. Commit changes
git add data/tools/my-new-tool.tool.yaml
git add data/tools/README.md
git add comparisons/auto-generated.md
git commit -m "Add my-new-tool entry"
```

### After Modifying Entries

```bash
# Validate changes
make validate

# Regenerate all content
make all

# Review changes
git diff

# Commit
git add -A
git commit -m "Update tool entries"
```

### Full Clean Rebuild

```bash
# Remove all generated content
make clean

# Regenerate everything
make all
```

## Script Details

### `scripts/check-yaml.py`

Validates YAML files against schemas in `schemas/`.

```bash
# Validate single file
./scripts/check-yaml.py data/tools/pyannote--pyannote-audio.tool.yaml

# Validate all files (default)
./scripts/check-yaml.py

# Validate with verbose output
./scripts/check-yaml.py --verbose
```

### `scripts/generate-readme.py`

Generates README.md files for each directory.

```bash
# Generate all READMEs
./scripts/generate-readme.py

# Generate specific target
./scripts/generate-readme.py tools
./scripts/generate-readme.py knowledge

# List available targets
./scripts/generate-readme.py --list
```

**Targets:**

* `tools` → `data/tools/README.md`
* `algorithms` → `data/algorithms/README.md`
* `models` → `data/models/README.md`
* `datasets` → `data/datasets/README.md`
* `papers` → `papers/README.md`
* `schemas` → `schemas/README.md`
* `knowledge` → `knowledge/README.md`
* `printouts` → `printouts/README.md`
* `ramblings` → `ramblings/README.md`
* `sources` → `sources/README.md`

### `scripts/generate-tables.py`

Generates comparison tables from YAML data.

```bash
./scripts/generate-tables.py > comparisons/auto-generated.md
```

### `scripts/verify-sources.py`

Checks that source URLs are accessible.

```bash
./scripts/verify-sources.py
```

### `scripts/clone-all.sh`

Clones tracked repositories to `tmp/`.

```bash
./scripts/clone-all.sh
```

### `scripts/generate-bib.py`

Generates BibTeX bibliography from paper YAML entries.

```bash
# Generate to stdout
./scripts/generate-bib.py

# Save to file
./scripts/generate-bib.py > archives/bibliography/papers-auto.bib

# Include abstracts
./scripts/generate-bib.py --format=full
```

### `scripts/extract-citations.py`

Extracts and analyzes citations from markdown files in `knowledge/`.

```bash
# Show all citations found
./scripts/extract-citations.py

# Show only unmatched citations
./scripts/extract-citations.py --unmatched

# Verify all citations have paper entries (CI-friendly)
./scripts/extract-citations.py --verify
```

### `scripts/update-stats.py`

Fetches current GitHub stats for tool repositories.

```bash
# Dry run - show current vs remote stats
./scripts/update-stats.py

# Update YAML files with new stats
./scripts/update-stats.py --update

# Check specific tool
./scripts/update-stats.py --tool pyannote
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    YAML Source Files                        │
│  data/tools/*.tool.yaml                                     │
│  data/algorithms/**/*.algorithm.yaml                        │
│  data/models/**/*.model.yaml                                │
│  papers/*.paper.yaml                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Python Scripts                           │
│  scripts/generate-readme.py                                 │
│  scripts/generate-tables.py                                 │
│  scripts/check-yaml.py                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Generated Markdown                         │
│  data/*/README.md                                           │
│  comparisons/auto-generated.md                              │
└─────────────────────────────────────────────────────────────┘
```

## Adding New Generators

To add a new README generator:

### 1. Add Generator Function

In `scripts/generate-readme.py`:

```python
def generate_mydir_readme():
    """Generate mydir/README.md"""
    data = load_yaml_files("mydir/*.mytype.yaml")

    lines = [
        "# My Directory",
        "",
        f"**Total: {len(data)} entries**",
        "",
        # ... generate content
    ]

    return "\n".join(lines)
```

### 2. Register in GENERATORS

```python
GENERATORS = {
    # ... existing generators
    'mydir': ('mydir/README.md', generate_mydir_readme),
}
```

### 3. Update Makefile (if needed)

Add cleanup target:

```makefile
clean:
    # ... existing
    rm -f mydir/README.md
```

## Common Issues

### Validation Errors

```
Error: data/tools/my-tool.tool.yaml: missing required field 'name'
```

**Fix:** Add the required field to your YAML file.

### Import Errors

```
Error: PyYAML not installed
```

**Fix:** `pip install pyyaml` or `make install`

### Stale Generated Content

If README doesn't reflect recent changes:

```bash
make clean
make all
```

## Makefile Reference

Run `make help` for the full list of targets:

```
Core:
  make validate      - Validate all YAML files
  make tables        - Generate comparison tables
  make readme        - Generate per-directory README files
  make all           - Run validate + tables + readme

Bibliography:
  make bib           - Generate BibTeX from paper entries
  make citations     - Analyze citations in knowledge articles
  make citations-verify - Verify citations have paper entries

External:
  make clone         - Clone all tracked repositories
  make sources       - Verify source URLs
  make stats         - Check GitHub stats (dry run)
  make stats-update  - Update GitHub stats in YAML files

Maintenance:
  make clean         - Remove generated files
  make install       - Install Python dependencies
```

## See Also

* [CONTRIBUTING.md](CONTRIBUTING.md) - Adding new entries
* [TRACEABILITY.md](TRACEABILITY.md) - Source attribution
* [schemas/](schemas/) - YAML schema specifications

---

*Last updated: 2026-01-06*
