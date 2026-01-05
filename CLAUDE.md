# Speaker Recognition & Diarization Knowledge Base

This repository is a public knowledge base (`-pub-kb`) for speaker diarization, speaker recognition, and voice fingerprinting topics.

## Repository Structure

### File Naming Convention

**Type-encoded extensions:** `{filename}.{type}.yaml`

Each YAML file encodes its schema type in the extension:

* Tools: `{owner}--{repo}.tool.yaml`
* Algorithms: `{name}.algorithm.yaml`
* Models: `{id}.model.yaml`
* Datasets: `{name}.dataset.yaml`
* Papers: `{key}.paper.yaml`
* Sources: `{id}.source.yaml`

**Spec files are self-referential:** `{type}.spec.yaml`

* The spec file's type is "spec", so `tool.spec.yaml` follows the pattern!
* Located in `schemas/` directory

### Directory Layout

* `schemas/` - YAML schema specifications (`*.spec.yaml`)
* `data/tools/` - Tool/implementation entries (`*.tool.yaml`)
* `data/algorithms/` - Algorithm entries by category (`*.algorithm.yaml`)
* `data/models/` - Pre-trained model entries (`*.model.yaml`)
* `data/datasets/` - Dataset entries (`*.dataset.yaml`)
* `papers/` - Paper metadata entries (`*.paper.yaml`)
* `sources/` - Source tracking entries (`*.source.yaml`)
* `scripts/` - Validation and generation scripts
* `comparisons/` - Auto-generated comparison tables
* `knowledge/` - Educational articles (fundamentals, math, concepts)
* `printouts/` - Archived web content
* `ramblings/` - Research notes (YYYY-MM-DD--topic.md)
* `tmp/` - Cloned repositories (gitignored)

## Working with This Repository

### Adding New Entries

1. Identify the correct data type (tool, algorithm, model, dataset, paper)
2. Create YAML file with appropriate extension: `{name}.{type}.yaml`
3. Follow the schema in `schemas/{type}.spec.yaml`
4. Include required tracking fields: `last-update`, source citations
5. Validate with `./scripts/check-yaml.py`

### Validation

```bash
./scripts/check-yaml.py                    # Validate all files
./scripts/check-yaml.py data/tools/*.yaml  # Validate specific files
```

### Generating Comparison Tables

```bash
./scripts/generate-tables.py > comparisons/auto-generated.md
```

## Domain Terminology

* **Diarization**: "Who spoke when" - segmenting audio by speaker identity
* **Speaker Embedding**: Fixed-dimensional vector representing voice characteristics
* **DER**: Diarization Error Rate - primary evaluation metric
* **EER**: Equal Error Rate - speaker verification metric
* **VAD**: Voice Activity Detection - detecting speech vs silence
* **RTTM**: Rich Transcription Time Marked - standard annotation format
