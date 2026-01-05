# Contributing Guidelines

This document explains how to add, update, and maintain entries in this knowledge base while ensuring traceability and auditability.

## Core Principles

1. **Every fact has a source** - Never add unsourced claims
2. **Timestamps on everything** - All entries have `last-update` dates
3. **Precision in code references** - Use commit hashes, file paths, line numbers
4. **Archive transient content** - Web pages go in `printouts/`
5. **Machine-readable first** - YAML for data, Markdown for prose

## Adding New Entries

### Step 1: Identify Entry Type

| Type | Extension | Location | When to Use |
|------|-----------|----------|-------------|
| Tool | `.tool.yaml` | `data/tools/` | Open-source frameworks, libraries |
| Algorithm | `.algorithm.yaml` | `data/algorithms/{category}/` | Named algorithms, techniques |
| Model | `.model.yaml` | `data/models/{category}/` | Pre-trained models on HuggingFace/NGC |
| Dataset | `.dataset.yaml` | `data/datasets/` | Training/evaluation corpora |
| Paper | `.paper.yaml` | `papers/` | Academic publications |

### Step 2: File Naming

Follow strict naming conventions:

* **Tools**: `{github-owner}--{repo-name}.tool.yaml`
  * Example: `pyannote--pyannote-audio.tool.yaml`
  * Example: `speechbrain--speechbrain.tool.yaml`

* **Algorithms**: `{algorithm-name}.algorithm.yaml`
  * Example: `ecapa-tdnn.algorithm.yaml`
  * Place in category subdirectory: `embeddings/`, `clustering/`, `vad/`, `end-to-end/`

* **Models**: `{provider}--{model-id}.model.yaml`
  * Example: `speechbrain--spkrec-ecapa-voxceleb.model.yaml`

* **Datasets**: `{dataset-name}.dataset.yaml`
  * Example: `voxceleb1.dataset.yaml`

* **Papers**: `{short-key}-{year}.paper.yaml`
  * Example: `ecapa-tdnn-2020.paper.yaml`

### Step 3: Follow Schema

Every entry type has a specification in `schemas/`:

```bash
cat schemas/tool.spec.yaml      # Tool schema
cat schemas/algorithm.spec.yaml # Algorithm schema
cat schemas/model.spec.yaml     # Model schema
cat schemas/dataset.spec.yaml   # Dataset schema
cat schemas/paper.spec.yaml     # Paper schema
cat schemas/source.spec.yaml    # Source citation schema
```

### Step 4: Required Fields

All entries MUST include:

```yaml
last-update: "YYYY-MM-DD"  # When this YAML was created/updated

sources:
  - url: "https://..."
    accessed: "YYYY-MM-DD"
    source-type: primary   # primary | secondary | tertiary
```

### Step 5: Validate

```bash
./scripts/check-yaml.py your-new-file.tool.yaml
# Or validate all:
make validate
```

## Source Citation Standards

### Minimum Citation (Required)

```yaml
sources:
  - url: "https://example.com/documentation"
    accessed: "2026-01-06"
```

### Standard Citation (Recommended)

```yaml
sources:
  - url: "https://github.com/org/repo"
    accessed: "2026-01-06"
    source-type: primary
    title: "Repository README"
```

### Comprehensive Citation (For Code References)

```yaml
sources:
  - url: "https://github.com/org/repo"
    accessed: "2026-01-06"
    commit: "abc123def456789"
    file-path: "src/models/ecapa_tdnn.py"
    line-range: "L45-L120"
    source-type: primary
    confidence: confirmed
    verified-date: "2026-01-06"
    notes: "Main model implementation"
```

### Academic Paper Citation

```yaml
sources:
  - arxiv: "2005.07143"
    title: "ECAPA-TDNN: Emphasized Channel Attention..."
    accessed: "2026-01-06"
    source-type: primary
    doi: "10.21437/Interspeech.2020-2650"
```

### Source Types

| Type | Use For |
|------|---------|
| `primary` | Official docs, original papers, source code |
| `secondary` | Tutorials, blog posts, third-party analysis |
| `tertiary` | Aggregated content (Wikipedia, Perplexity) |
| `benchmark` | Performance evaluation results |
| `code` | Source code implementation details |

### Confidence Levels

| Level | Meaning |
|-------|---------|
| `confirmed` | Verified from primary source |
| `likely` | From reputable secondary source |
| `uncertain` | Needs verification |

## Archiving Web Content

For web pages that might change or disappear:

### 1. Create Archive Directory

```bash
mkdir -p printouts/YYYY-MM-DD--{descriptive-name}/
```

### 2. Save Content

```bash
# Save page as markdown
curl -s "https://example.com/page" | pandoc -f html -t markdown > content.md
```

### 3. Create SOURCE.yaml

```yaml
# printouts/2026-01-06--example-page/SOURCE.yaml
source:
  url: "https://example.com/page"
  accessed: "2026-01-06"
  archived-date: "2026-01-06"
  source-type: "secondary"

description: |
  Brief description of what this archive contains.

files:
  - name: "content.md"
    description: "Main page content"
```

### 4. Reference in Entries

```yaml
sources:
  - url: "https://example.com/page"
    accessed: "2026-01-06"
    local-path: "printouts/2026-01-06--example-page/content.md"
```

## Example Entries

### Complete Tool Entry

```yaml
# File: data/tools/resemble-ai--resemblyzer.tool.yaml

last-update: "2026-01-06"
repo-url: "https://github.com/resemble-ai/Resemblyzer"
repo-commit: "abc123def456"

name: "Resemblyzer"
description: "Voice similarity and speaker identification using d-vectors"
language: "Python"
license: "Apache-2.0"

category: "voice-fingerprinting"

capabilities:
  diarization: true
  speaker-embedding: true
  speaker-verification: true
  streaming: false

performance:
  rtf: 0.2

embedding:
  architecture: "d-vector"
  dimension: 256
  pretrained-available: true

installation:
  pip: "resemblyzer"

documentation:
  readme: true
  examples: true

features:
  - "GE2E d-vector embeddings"
  - "Simple voice comparison API"
  - "Pre-trained on LibriSpeech + VoxCeleb"

sources:
  - url: "https://github.com/resemble-ai/Resemblyzer"
    accessed: "2026-01-06"
    source-type: primary

  - url: "https://pypi.org/project/Resemblyzer/"
    accessed: "2026-01-06"
    source-type: primary
```

### Complete Paper Entry

```yaml
# File: papers/ecapa-tdnn-2020.paper.yaml

last-update: "2026-01-06"

title: "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification"
short-title: "ECAPA-TDNN"
year: 2020

authors:
  - "Brecht Desplanques"
  - "Jenthe Thienpondt"
  - "Kris Demuynck"

affiliations:
  - "IDLab, Ghent University - imec"

venue: "Interspeech 2020"
venue-type: "conference"

arxiv: "2005.07143"
doi: "10.21437/Interspeech.2020-2650"

pdf-url: "https://arxiv.org/pdf/2005.07143.pdf"
code-url: "https://github.com/speechbrain/speechbrain"

topics:
  - "speaker-verification"
  - "speaker-embedding"
  - "TDNN"
  - "attention-mechanism"

contributions:
  - "Res2Net-style multi-scale feature extraction"
  - "Squeeze-and-Excitation channel attention"
  - "Attentive statistics pooling"
  - "State-of-the-art VoxCeleb1 results"

reported-metrics:
  voxceleb1-eer: 0.87
  voxceleb1-dcf: 0.089

implementations:
  - tool: "speechbrain"
    url: "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb"
    official: false

sources:
  - arxiv: "2005.07143"
    accessed: "2026-01-06"
    source-type: primary
```

## Updating Existing Entries

1. Update the `last-update` field to today's date
2. Update `repo-commit` if analyzing a new version
3. Add new sources for new information
4. Run validation before committing
5. Document what changed in your commit message

## Research Notes

For exploration and research findings, use the `ramblings/` directory:

```
ramblings/YYYY-MM-DD--{descriptive-topic}.md
```

Example: `ramblings/2026-01-06--streaming-diarization-comparison.md`

These are informal notes that may later be formalized into proper entries.

## Regenerating Content

After adding or updating entries:

```bash
# Regenerate comparison tables
make tables

# Full validation + regeneration
make all
```

## Commit Guidelines

* Use descriptive commit messages
* Reference what type of content changed
* Include `last-update` date in commit message for batch updates

Example:

```
Add Resemblyzer tool entry

- New tool: resemble-ai--resemblyzer.tool.yaml
- Voice fingerprinting using d-vectors
- Sources: GitHub repo, PyPI

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

## Quality Checklist

Before submitting:

- [ ] `last-update` field is set to today's date
- [ ] At least one source citation with `url` and `accessed`
- [ ] File follows naming convention
- [ ] `make validate` passes without errors
- [ ] Links are accessible
- [ ] Information is accurate and verifiable

## Getting Help

* Check `schemas/` for field definitions
* Look at existing entries as examples
* Open an issue for questions
