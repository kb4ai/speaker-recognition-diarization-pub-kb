# Embedding Architecture Comparison

Comparing speaker embedding architectures: i-vectors, x-vectors, d-vectors, ECAPA-TDNN, and WavLM.

## Architecture Overview

| Architecture | Year | Type | Dimension | Approach |
|--------------|------|------|-----------|----------|
| **i-vector** | 2011 | Generative | 400-600 | Factor analysis on GMM supervectors |
| **d-vector** | 2014 | Discriminative | 256-512 | LSTM/DNN with GE2E loss |
| **x-vector** | 2018 | Discriminative | 512 | TDNN with statistics pooling |
| **ECAPA-TDNN** | 2020 | Discriminative | 192 | Enhanced TDNN with attention |
| **WavLM** | 2022 | Self-supervised | 512 | Transformer with SSL pre-training |

## Quick Comparison

| Factor | i-vector | d-vector | x-vector | ECAPA-TDNN | WavLM |
|--------|----------|----------|----------|------------|-------|
| **EER (VoxCeleb1)** | ~5% | ~3% | ~2% | ~0.8% | ~0.8% |
| **Training data needed** | Medium | Large | Large | Large | Massive |
| **Computation (training)** | Low | High | High | High | Very High |
| **Computation (inference)** | Medium | Low | Low | Low | High |
| **Handles short audio** | Poor | Good | Good | Best | Good |
| **Pre-trained available** | Few | Some | Many | Many | Growing |

## i-vectors (2011)

### Architecture

```
Audio → MFCC → UBM (GMM) → Supervector → T-matrix → i-vector
                                              ↓
                                        400-600 dim
```

### How It Works

1. **Universal Background Model (UBM)**: GMM trained on diverse speakers
2. **Supervector**: Concatenated GMM means adapted to utterance
3. **Total Variability Matrix (T)**: Projects supervector to low-dimensional space
4. **i-vector**: `w = T^T Σ^{-1} (s - m)` where s is supervector, m is UBM mean

### Strengths

* Well-understood probabilistic model
* Works with limited training data
* Interpretable latent factors
* No GPU required for training

### Weaknesses

* Captures channel variability along with speaker
* Requires PLDA backend for best results
* Poor with short utterances (<3s)
* Dated performance on modern benchmarks

### When to Use

* Legacy systems requiring compatibility
* Limited computational resources
* Small training datasets
* When interpretability matters

## d-vectors (2014)

### Architecture

```
Audio → Mel-spectrogram → LSTM/DNN → Frame embeddings → Pooling → d-vector
                                                            ↓
                                                       256-512 dim
```

### How It Works

1. **Frame-level processing**: LSTM processes spectrogram frames
2. **Temporal pooling**: Average over all frames
3. **GE2E Loss**: Generalized End-to-End loss for training

```python
# GE2E Loss concept
for each speaker:
    centroid = mean(speaker_embeddings)
    positive_sim = cosine(embedding, centroid)
    negative_sim = max(cosine(embedding, other_centroids))
    loss = softmax_loss(positive_sim, negative_sim)
```

### Strengths

* Simple architecture
* Works well for text-dependent verification
* Good real-time performance
* Popular in voice cloning (e.g., Resemblyzer)

### Weaknesses

* Lower accuracy than x-vectors/ECAPA
* Less robust to noise
* Requires careful hyperparameter tuning

### When to Use

* Voice cloning pipelines
* Text-dependent speaker verification
* Real-time applications with tight latency
* When using Resemblyzer ecosystem

## x-vectors (2018)

### Architecture

```
Audio → MFCC → TDNN Layers → Statistics Pooling → Dense → x-vector
                    ↓                 ↓                    ↓
              Frame-level      Mean + Std           512 dim
```

### Key Components

**Time Delay Neural Network (TDNN)**:

```
Layer 1: context [-2, -1, 0, 1, 2]     → 512 channels
Layer 2: context [-2, 0, 2]            → 512 channels
Layer 3: context [-3, 0, 3]            → 512 channels
Layer 4: context [0]                   → 512 channels
Layer 5: context [0]                   → 1500 channels
```

**Statistics Pooling**:

```python
# Aggregate frame embeddings
mean = torch.mean(frame_embeddings, dim=1)
std = torch.std(frame_embeddings, dim=1)
pooled = torch.cat([mean, std], dim=1)  # 3000 dim
```

### Strengths

* Major accuracy improvement over i-vectors
* Robust to noise and channel variations
* Many pre-trained models available
* Well-established training recipes

### Weaknesses

* Large embedding dimension (512)
* No explicit attention mechanism
* Single-scale temporal processing
* Superseded by ECAPA-TDNN

### When to Use

* Production systems requiring proven reliability
* When ECAPA-TDNN is overkill
* Compatibility with existing x-vector systems
* Kaldi-based pipelines

## ECAPA-TDNN (2020)

### Architecture

```
Audio → Mel-filterbank → SE-Res2Net Blocks → Attentive Pooling → ECAPA embedding
                              ↓                     ↓                  ↓
                        Multi-scale          Weighted stats       192 dim
```

### Key Innovations

**1. Res2Net Multi-Scale Processing**:

```python
# Hierarchical residual connections
y1 = x1
y2 = conv(x2)
y3 = conv(x3 + y2)      # Cascaded
y4 = conv(x4 + y3)      # Cascaded
output = concat(y1, y2, y3, y4)
```

**2. Squeeze-and-Excitation (SE) Blocks**:

```python
# Channel attention
z = global_avg_pool(features)           # Squeeze
s = sigmoid(fc2(relu(fc1(z))))          # Excitation
output = features * s                    # Rescale
```

**3. Attentive Statistics Pooling**:

```python
# Learned attention weights
attention = softmax(linear(tanh(linear(frames))))
weighted_mean = sum(attention * frames)
weighted_std = sqrt(sum(attention * (frames - mean)^2))
output = concat(weighted_mean, weighted_std)
```

### Strengths

* State-of-the-art accuracy
* Compact embeddings (192-dim)
* Multi-scale feature extraction
* Channel and temporal attention
* Excellent short-utterance performance

### Weaknesses

* More complex architecture
* Slightly higher compute than x-vectors
* Requires careful training (augmentation, AAM-Softmax)

### When to Use

* Best accuracy is required
* Modern speaker recognition systems
* Short audio clips (<2s)
* GPU inference available

## Performance Comparison

### VoxCeleb1 Benchmark

| Model | EER (%) | minDCF | Embedding Dim |
|-------|---------|--------|---------------|
| i-vector + PLDA | 5.3 | 0.49 | 400 |
| d-vector (GE2E) | 3.1 | 0.30 | 256 |
| x-vector | 2.0 | 0.20 | 512 |
| ECAPA-TDNN | 0.87 | 0.089 | 192 |

### Computational Requirements

| Model | Training | Inference (RTF) | Memory |
|-------|----------|-----------------|--------|
| i-vector | CPU OK | 0.5 | Low |
| d-vector | GPU | 0.1 | Medium |
| x-vector | GPU | 0.1 | Medium |
| ECAPA-TDNN | GPU | 0.12 | Medium |

### Robustness

| Condition | i-vector | d-vector | x-vector | ECAPA-TDNN |
|-----------|----------|----------|----------|------------|
| Clean audio | Good | Good | Very Good | Excellent |
| Noisy audio | Poor | Fair | Good | Very Good |
| Short (<2s) | Poor | Fair | Good | Excellent |
| Reverb | Poor | Fair | Good | Very Good |

## Implementation Examples

### i-vector (Kaldi)

```bash
# Extract i-vectors using Kaldi
steps/online/nnet2/extract_ivectors_online.sh \
    --nj 40 \
    data/test \
    exp/extractor \
    exp/ivectors_test
```

### d-vector (Resemblyzer)

```python
from resemblyzer import VoiceEncoder, preprocess_wav

encoder = VoiceEncoder()
wav = preprocess_wav("audio.wav")
embedding = encoder.embed_utterance(wav)  # (256,)
```

### x-vector (SpeechBrain)

```python
from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb"
)
embedding = classifier.encode_batch(waveform)  # (1, 512)
```

### ECAPA-TDNN (SpeechBrain)

```python
from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)
embedding = classifier.encode_batch(waveform)  # (1, 192)
```

## Migration Guide

### i-vector → x-vector

1. Replace GMM-UBM with TDNN
2. Use statistics pooling instead of T-matrix
3. Train with softmax/AAM-softmax loss
4. Optional: Keep PLDA backend for scoring

### x-vector → ECAPA-TDNN

1. Replace TDNN with SE-Res2Net blocks
2. Use attentive statistics pooling
3. Reduce embedding dimension (512 → 192)
4. Apply stronger augmentation during training

## Self-Supervised Embeddings (2021+)

### WavLM / wav2vec 2.0 / HuBERT

A new paradigm using self-supervised learning (SSL) from unlabeled audio.

```
Audio → CNN encoder → Transformer → [CLS] or pooling → Embedding
             ↓                           ↓
     Pre-trained on         Fine-tuned on labeled
      94k+ hours               speaker data
```

### How It Works

1. **Self-supervised pre-training**: Masked speech prediction on massive unlabeled data
2. **Fine-tuning**: Add classification head, train on VoxCeleb
3. **Embedding extraction**: Use hidden states or add x-vector style pooling

### Key Models

| Model | Pre-train Data | Params | VoxCeleb1 EER |
|-------|---------------|--------|---------------|
| WavLM Base+ | 94k hours | 94M | 0.84% |
| WavLM Large | 94k hours | 316M | ~0.70% |
| wav2vec 2.0 | 60k hours | 317M | 2.65% |
| HuBERT Large | 60k hours | 316M | 1.08% |

### Code Example (WavLM)

```python
from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor

model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")

inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
embeddings = model(**inputs).embeddings  # (1, 512)
```

### Strengths

* State-of-the-art accuracy (sub-1% EER)
* Robust to noise and domain shift
* Benefits from scale (more pre-training data = better)
* Multi-task capability (ASR, SV, diarization)

### Weaknesses

* Large model size (94M-316M parameters)
* Slow inference without optimization
* Requires significant compute for fine-tuning
* Newer, less mature tooling

### When to Use

* When accuracy is paramount
* Offline processing with GPU available
* Cross-domain applications (noisy, diverse audio)
* When leveraging other SSL capabilities

## Recommendations

| Scenario | Recommended | Rationale |
|----------|-------------|-----------|
| New project (accuracy) | WavLM/ECAPA-TDNN | Best accuracy |
| New project (balanced) | ECAPA-TDNN | Good accuracy, efficient |
| Real-time streaming | d-vector | Lowest latency |
| Edge deployment | TitaNet Small | Compact, fast |
| Legacy integration | x-vector | Widely supported |
| No GPU | i-vector | CPU-friendly |
| Voice cloning | d-vector | GE2E training, Resemblyzer ecosystem |

## References

* Dehak et al. "Front-End Factor Analysis for Speaker Verification" (2011) - i-vectors
* Wan et al. "Generalized End-to-End Loss for Speaker Verification" (2018) - d-vectors
* Snyder et al. "X-vectors: Robust DNN Embeddings for Speaker Recognition" (2018)
* Desplanques et al. "ECAPA-TDNN" (Interspeech 2020)
* Chen et al. "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing" (2022)
* Hsu et al. "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units" (2021)

---

*Last updated: 2026-01-06*
