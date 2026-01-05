# Speaker Recognition & Diarization Knowledge Base

A comprehensive, machine-readable knowledge base for speaker diarization, speaker recognition, and voice fingerprinting technologies.

## Overview

This repository provides structured YAML data about:

* **Tools & Frameworks** - Open-source implementations (pyannote, SpeechBrain, NeMo, Kaldi)
* **Algorithms** - Embedding extraction, clustering, VAD, end-to-end approaches
* **Pre-trained Models** - Ready-to-use speaker embedding and diarization models
* **Datasets** - Benchmarking corpora (VoxCeleb, AMI, CALLHOME, DIHARD)
* **Research Papers** - Key publications in the field

## Quick Navigation

| Resource | Description |
|----------|-------------|
| [Tools](data/tools/) | Speaker diarization frameworks and libraries |
| [Algorithms](data/algorithms/) | Core algorithms by category |
| [Models](data/models/) | Pre-trained embedding and pipeline models |
| [Datasets](data/datasets/) | Training and evaluation datasets |
| [Comparisons](comparisons/) | Auto-generated comparison tables |
| [Knowledge](knowledge/) | Educational articles |

## File Naming Convention

All YAML files use type-encoded extensions: `{name}.{type}.yaml`

```
data/tools/pyannote--pyannote-audio.tool.yaml
data/algorithms/embeddings/ecapa-tdnn.algorithm.yaml
data/datasets/voxceleb1.dataset.yaml
schemas/tool.spec.yaml
```

## Getting Started

### Validate Data

```bash
./scripts/check-yaml.py
```

### Generate Comparison Tables

```bash
./scripts/generate-tables.py > comparisons/auto-generated.md
```

### Clone All Tracked Repositories

```bash
./scripts/clone-all.sh
```

## Key Concepts

### Speaker Diarization Pipeline

```
Audio → VAD → Segmentation → Embedding Extraction → Clustering → Output
```

1. **VAD (Voice Activity Detection)**: Identify speech vs non-speech
2. **Segmentation**: Detect speaker change points
3. **Embedding Extraction**: Extract speaker embeddings (x-vectors, ECAPA-TDNN)
4. **Clustering**: Group segments by speaker (AHC, spectral, VBx)

### Evaluation Metrics

* **DER (Diarization Error Rate)**: Primary metric = (FA + Miss + Confusion) / Total
* **EER (Equal Error Rate)**: Speaker verification accuracy
* **RTF (Real-Time Factor)**: Processing speed relative to audio duration

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding new entries.

## License

This knowledge base is released under MIT License.
