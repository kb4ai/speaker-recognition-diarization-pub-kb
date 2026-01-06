# Speaker Recognition & Diarization: Comparison Tables

*Auto-generated from YAML data files*

---

## Summary Statistics

* **Tools**: 10
* **Algorithms**: 10
* **Models**: 10
* **Datasets**: 5
* **Last updated**: 2026-01-06

### Tools by Category

* asr-framework: 1
* diarization-framework: 2
* diarization-wrapper: 1
* embedding-toolkit: 2
* speech-toolkit: 1
* streaming-diarization: 1
* transcription-diarization: 1
* voice-fingerprinting: 1

## Tools Overview

*Sorted by GitHub stars*

| Tool | Stars | Language | Category | Capabilities |
|------|------:|----------|----------|--------------|
| [Simple Diarizer](https://github.com/cvqluu/simple_diarizer) | ? | Python | diarization wrapper | diarization |
| [ESPnet](https://github.com/espnet/espnet) | ? | Python | speech toolkit | diarization, speaker-embedding, speaker-verification, vad, training, asr |
| [diart](https://github.com/juanmc2005/diart) | ? | Python | streaming diarization | diarization, speaker-embedding, vad, overlap-detection, streaming |
| [Kaldi](https://github.com/kaldi-asr/kaldi) | ? | C++ | asr framework | diarization, speaker-embedding, speaker-verification, speaker-identification, vad, training |
| [WhisperX](https://github.com/m-bain/whisperX) | ? | Python | transcription diarization | diarization, vad |
| [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) | ? | Python | diarization framework | diarization, speaker-embedding, speaker-verification, speaker-identification, vad, overlap-detection, streaming, training |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | ? | Python | diarization framework | diarization, speaker-embedding, speaker-verification, speaker-identification, vad, overlap-detection, streaming, training |
| [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) | ? | Python | voice fingerprinting | diarization, speaker-embedding, speaker-verification, speaker-identification |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) | ? | Python | embedding toolkit | diarization, speaker-embedding, speaker-verification, speaker-identification, vad, training |
| [WeSpeaker](https://github.com/wenet-e2e/wespeaker) | ? | Python | embedding toolkit | diarization, speaker-embedding, speaker-verification, speaker-identification, vad, training |

## Tools by Accuracy

*Sorted by DER on AMI corpus (lower is better)*

| Tool | DER (AMI) | EER (VoxCeleb) | RTF | Streaming |
|------|----------:|---------------:|----:|:---------:|
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | 10.0% | 0.8% | 0.15 | ✓ |

## Algorithms by Category

### Embedding Extraction

| Algorithm | Year | Output Dim | Key Features |
|-----------|-----:|-----------:|--------------|
| ECAPA-TDNN | 2020 | 192 | Squeeze-and-Excitation (SE) blocks, Res2Net multi-scale feature extraction, Attentive statistics pooling |
| x-vectors | 2018 | 512 | Time Delay Neural Network (TDNN), Statistics pooling (mean and std), Segment-level processing |
| i-vectors | 2011 | 400 | Factor analysis, Universal Background Model (UBM), Total Variability matrix (T-matrix) |

### Clustering

| Algorithm | Year | Output Dim | Key Features |
|-----------|-----:|-----------:|--------------|
| VBx | 2021 | ? | Variational Bayes inference, Automatic speaker count estimation, Probabilistic clustering |
| AHC | 1967 | ? | Bottom-up hierarchical clustering, Distance metrics (cosine, euclidean), Linkage criteria (average, complete, ward) |
| Spectral Clustering | ? | ? | Graph-based clustering, Affinity matrix construction, Eigenvalue analysis |

### Vad

| Algorithm | Year | Output Dim | Key Features |
|-----------|-----:|-----------:|--------------|
| Neural VAD | 2018 | ? | Deep neural networks, Binary classification, Temporal modeling |
| Energy-based VAD | ? | ? | Signal energy thresholding, Short-time energy analysis, Zero-crossing rate |

### End To End

| Algorithm | Year | Output Dim | Key Features |
|-----------|-----:|-----------:|--------------|
| Sortformer | 2024 | ? | Transformer architecture, Sort-based permutation, Multi-speaker ASR integration |
| EEND | 2019 | ? | Multi-label classification, Permutation Invariant Training (PIT), Self-attention mechanism |


## Embedding Models

*Sorted by EER on VoxCeleb1 (lower is better)*

| Model | Architecture | Dimension | EER (VoxCeleb1) | Provider |
|-------|--------------|----------:|----------------:|----------|
| [TitaNet Large](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large) | TitaNet | 192 | 0.66% | nvidia |
| [ECAPA-TDNN Speaker Embedding](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | ECAPA-TDNN | 192 | 0.8% | speechbrain |
| [WavLM Base Plus Speaker Verification](https://huggingface.co/microsoft/wavlm-base-plus-sv) | WavLM (Transformer) | 512 | 0.84% | microsoft |
| [WeSpeaker ResNet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) | ResNet34 | 256 | 0.84% | pyannote |
| [CAM++ Speaker Embedding](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM) | CAM++ (TDNN + Multi-scale Aggregation) | 192 | 0.87% | wespeaker |
| TitaNet Small | TitaNet (SE-Res2Net) | 192 | 1.48% | nvidia |
| [SpeechBrain x-vector Speaker Embedding](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) | x-vector (TDNN) | 512 | 1.9% | speechbrain |

## Datasets Comparison

| Dataset | Type | Hours | Speakers | Languages | License |
|---------|------|------:|---------:|-----------|---------|
| AMI Corpus |  | ? | ? |  | CC BY 4.0 |
| CALLHOME |  | ? | ? |  | LDC License |
| DIHARD |  | ? | ? |  | LDC License |
| VoxCeleb1 |  | ? | ? |  | CC BY-SA 4.0 |
| VoxCeleb2 |  | ? | ? |  | CC BY-SA 4.0 |
