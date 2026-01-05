# Speaker Embeddings

## Definition

A **speaker embedding** (also called **voiceprint**, **x-vector**, **d-vector**, or **speaker profile**) is a fixed-dimensional numerical vector (typically 192-512 dimensions) that captures unique acoustic characteristics of a speaker's voice—including speech rhythm, pitch, intonation, formant frequencies, and speaking style.

## Properties

* **Uniqueness**: Each speaker has statistically distinct embedding characteristics
* **Irreversibility**: Cannot reconstruct original audio from embedding
* **Privacy-preserving**: Contains only statistical voice patterns, not linguistic content
* **Compact**: Typical size 1-4 KB (much smaller than audio)
* **Comparison-ready**: Optimized for cosine similarity or Euclidean distance metrics

## Evolution of Embedding Architectures

### i-vectors (2011)

**Identity vectors** use a generative approach based on factor analysis.

* **Universal Background Model (UBM)**: GMM trained on diverse speakers
* **Total Variability matrix (T-matrix)**: Projects supervectors to low-dimensional space
* **Dimension**: 400-600
* Captures both speaker and channel variability

### x-vectors (2018)

**x-vectors** use discriminative deep neural network embeddings.

* **Architecture**: Time Delay Neural Network (TDNN) with statistics pooling
* **Statistics pooling layer**: Computes mean and standard deviation across time
* **Segment-level processing**: Typically 1.5-3 second chunks
* **Dimension**: 512

### ECAPA-TDNN (2020)

**Emphasized Channel Attention, Propagation and Aggregation TDNN** is current state-of-the-art.

* **Res2Net blocks**: Multi-scale feature extraction with hierarchical connections
* **Squeeze-and-Excitation (SE) blocks**: Channel-wise attention mechanism
* **Attentive statistics pooling**: Weighted mean/std using learned attention
* **Dimension**: 192
* **EER on VoxCeleb1**: ~0.80%

## Feature Extraction

### Acoustic Features

**Mel-Frequency Cepstral Coefficients (MFCC):**

```
Audio → Fourier Transform → Mel-scale Filterbank → Log → DCT → MFCCs
```

* Typically 13-39 coefficients per frame
* Delta and delta-delta coefficients: First and second-order temporal derivatives

**Alternative features:**

* **Mel-spectrogram**: Log-scaled filterbank outputs
* **Fbank features**: Filterbank energies without DCT

### Frame vs Segment Level

* **Frame-level features**: Extracted per audio frame (e.g., 25ms window, 10ms shift)
* **Segment-level embeddings**: Aggregated representation of longer audio chunks

## Speaker Enrollment

**Speaker enrollment** creates a reference template for a known speaker:

1. **Audio collection**: Capture speech samples from target speaker
   * Minimum: 30 seconds of net speech (excluding silence)
   * Recommended: 3+ recordings of 10-12 seconds each

2. **Feature extraction**: Process audio through embedding model

3. **Profile creation**: Store embedding with speaker metadata
   * Speaker ID/name
   * Enrollment audio duration
   * Model version used

4. **Database storage**: Save voiceprint for future matching

## Comparison Metrics

### Cosine Similarity

Most common metric for speaker embeddings (range: -1 to 1):

```
similarity = (A · B) / (||A|| × ||B||)
```

**Threshold guidelines:**

* High (0.8-0.9): Low false acceptance, may reject genuine speakers
* Medium (0.6-0.8): Balanced trade-off
* Low (0.4-0.6): High recall, more false acceptances

### PLDA Scoring

**Probabilistic Linear Discriminant Analysis** accounts for within/between-speaker variance:

* Models speaker and session variability separately
* Produces log-likelihood ratio scores
* Better performance in challenging conditions

## Speaker Recognition Tasks

### Speaker Identification (1:N)

"Which enrolled speaker is this?"

* Compare test embedding against all N enrolled voiceprints
* Return closest match above threshold or "unknown"
* **Closed-set**: Speaker guaranteed to be in database
* **Open-set**: May reject if no satisfactory match

### Speaker Verification (1:1)

"Is this the claimed speaker?"

* Compare test embedding against single claimed identity
* Binary decision: accept/reject

## Pre-trained Models

| Model | Dimension | EER (VoxCeleb1) | Framework |
|-------|-----------|-----------------|-----------|
| speechbrain/spkrec-ecapa-voxceleb | 192 | 0.80% | SpeechBrain |
| nvidia/titanet_large | 512 | ~1.0% | NeMo |
| pyannote/embedding | 512 | N/A | Pyannote |
| wespeaker-voxceleb-resnet34-LM | 256 | ~2.5% | WeSpeaker |

## References

* "ECAPA-TDNN: Emphasized Channel Attention..." (Interspeech 2020)
* "X-vectors: Robust DNN Embeddings for Speaker Recognition" (ICASSP 2018)
* "Front-End Factor Analysis for Speaker Verification" (IEEE 2011)
