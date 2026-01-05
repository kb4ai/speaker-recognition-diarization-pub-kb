# Speaker Diarization Pipeline Architecture

## Overview

Speaker diarization—answering "who spoke when" in audio recordings—follows a well-established multi-stage pipeline. Modern systems employ either a **cascaded (modular) pipeline** or an **end-to-end neural approach**.

## Cascaded Pipeline

The cascaded system remains dominant in production environments and consists of four main stages:

```
Audio → [VAD] → [Segmentation] → [Embedding] → [Clustering] → Speaker Labels
         ↓           ↓                ↓              ↓
      Speech      Speaker         Fixed-dim       Cluster
      Regions     Boundaries      Vectors         Labels
```

### Stage 1: Voice Activity Detection (VAD)

**Voice Activity Detection** identifies speech versus non-speech regions, filtering out silence, background noise, and non-verbal sounds. This binary classification task distinguishes when "someone is speaking" without identifying who.

**Types:**

* **Energy-based VAD**: Traditional approach using signal energy thresholds
* **Neural VAD**: Deep learning models (e.g., MarbleNet, PyanNet, Silero)
* **Overlap-aware VAD**: Detects simultaneous speech from multiple speakers

**Output:** Timestamps of voiced regions at millisecond resolution.

### Stage 2: Speaker Segmentation

**Speaker segmentation** divides continuous audio into homogeneous segments where only one speaker is active, detecting **speaker change points** (boundaries where speaker identity changes).

**Methods:**

1. **Bayesian Information Criterion (BIC)**: Classical statistical approach comparing Gaussian models
2. **Sliding window approach**: Analyzes adjacent time windows to detect acoustic changes
3. **Neural segmentation**: Learned speaker change detection

**Key parameters:**

* **Window size**: Duration of analysis segments (typically 1.5-3 seconds)
* **Shift/hop length**: Overlap between consecutive windows (e.g., 0.25-0.5 seconds)

### Stage 3: Speaker Embedding Extraction

This stage extracts **speaker embeddings**—fixed-dimensional vector representations capturing voice characteristics.

**Feature extraction:**

* **Mel-Frequency Cepstral Coefficients (MFCC)**: 13-39 coefficients per frame
* **Mel-spectrogram/log-Mel features**: Direct log-scaled filterbank outputs
* **Fbank features**: Filterbank energies without DCT transformation

**Embedding architectures:**

1. **i-vectors** (2011): Generative approach using factor analysis, 400-600 dimensions
2. **x-vectors** (2018): TDNN with statistics pooling, typically 512 dimensions
3. **ECAPA-TDNN** (2020): State-of-the-art with attention mechanisms, 192 dimensions

### Stage 4: Clustering

**Clustering** groups segments with similar embeddings, assuming each cluster corresponds to one speaker.

**Algorithms:**

1. **Agglomerative Hierarchical Clustering (AHC)**: Bottom-up merging with cosine distance
2. **Spectral Clustering**: Graph-based approach using affinity matrix eigenvalues
3. **VBx (Variational Bayes)**: Bayesian clustering with automatic speaker count estimation

**Distance metrics:**

* **Cosine similarity**: Most common for speaker embeddings
* **PLDA scoring**: Probabilistic approach accounting for variance

## End-to-End Neural Diarization (EEND)

An alternative that directly predicts speaker activities from audio using neural networks.

**Key concepts:**

* **Multi-label classification**: Each frame assigned multiple binary labels (one per speaker)
* **Permutation Invariant Training (PIT)**: Solves label ambiguity by trying all speaker permutations
* **Power Set Encoding (PSE)**: Single-label classification over speaker combinations

**Advantages:**

* Naturally handles overlapping speech
* Joint optimization of all components
* No separate clustering step

**Limitations:**

* Fixed maximum speaker count during training
* Requires large training data

## Typical Hyperparameters

| Parameter | Typical Value |
|-----------|---------------|
| Embedding dimension | 192-512 |
| Clustering threshold | 0.5-1.5 (cosine) |
| Minimum segment duration | 0.3-0.5 seconds |
| VAD threshold | Tuned per dataset |
| Window size | 1.5-3 seconds |
| Hop length | 0.25-0.5 seconds |

## References

* NVIDIA NeMo Speaker Diarization Documentation
* Pyannote.audio Framework
* "End-to-End Neural Speaker Diarization with Permutation-Free Objectives" (Interspeech 2019)
