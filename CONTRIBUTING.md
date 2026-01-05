# Contributing Guidelines

## Adding New Entries

### Step 1: Identify Entry Type

| Type | Extension | Location | Example |
|------|-----------|----------|---------|
| Tool | `.tool.yaml` | `data/tools/` | `pyannote--pyannote-audio.tool.yaml` |
| Algorithm | `.algorithm.yaml` | `data/algorithms/{category}/` | `ecapa-tdnn.algorithm.yaml` |
| Model | `.model.yaml` | `data/models/{category}/` | `pyannote--embedding.model.yaml` |
| Dataset | `.dataset.yaml` | `data/datasets/` | `voxceleb1.dataset.yaml` |
| Paper | `.paper.yaml` | `papers/` | `ecapa-tdnn-2020.paper.yaml` |

### Step 2: Create YAML File

Follow the naming convention:

* **Tools**: `{owner}--{repo}.tool.yaml`
* **Algorithms**: `{name}.algorithm.yaml`
* **Models**: `{provider}--{model-id}.model.yaml`
* **Datasets**: `{name}.dataset.yaml`
* **Papers**: `{key}-{year}.paper.yaml`

### Step 3: Follow Schema

Reference the corresponding spec file in `schemas/`:

```bash
cat schemas/tool.spec.yaml      # For tools
cat schemas/algorithm.spec.yaml # For algorithms
```

### Step 4: Required Fields

All entries must include:

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
```

## Example: Adding a New Tool

```yaml
# File: data/tools/resemblyzer--resemblyzer.tool.yaml

last-update: "2026-01-05"
repo-url: "https://github.com/resemble-ai/Resemblyzer"
repo-commit: "abc123"

name: "Resemblyzer"
description: "Voice similarity and diarization using d-vectors"
language: "Python"
license: "Apache-2.0"
stars: 1500

capabilities:
  diarization: true
  speaker-embedding: true
  streaming: false
  overlap-detection: false

performance:
  rtf: 0.2

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
    accessed: "2026-01-05"
    source-type: primary
```

## Updating Existing Entries

1. Update the `last-update` field to today's date
2. Update `repo-commit` if analyzing a new version
3. Add new sources if information comes from new references
4. Run validation before committing

## Research Notes

Add research notes to `ramblings/` with dated filenames:

```
ramblings/2026-01-05--streaming-diarization-comparison.md
```

## Regenerating Tables

After adding entries, regenerate comparison tables:

```bash
./scripts/generate-tables.py > comparisons/auto-generated.md
```
