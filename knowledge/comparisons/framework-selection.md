# Framework Selection Guide

Choosing the right speaker diarization framework depends on your use case, performance requirements, and deployment constraints.

## Quick Decision Matrix

| If you need... | Use |
|----------------|-----|
| Best accuracy, production ready | **pyannote.audio** |
| GPU training & enterprise scale | **NVIDIA NeMo** |
| Research flexibility | **SpeechBrain** |
| Real-time streaming | **diart** |
| Simple voice matching | **Resemblyzer** |
| ASR + diarization combo | **WhisperX** |

## Framework Comparison

### pyannote.audio

**Best for:** Production diarization systems, industry-standard accuracy

```
Strengths:              Limitations:
✓ State-of-the-art DER  ✗ HuggingFace auth required
✓ Active development    ✗ Larger memory footprint
✓ Great documentation   ✗ Python only
✓ Pre-trained models
✓ Handles overlap
```

**Performance:**

* DER on AMI: ~10%
* RTF on GPU: 0.15
* EER on VoxCeleb1: 0.8%

**Use when:**

* Building production diarization system
* Need best accuracy out-of-box
* Python ecosystem is acceptable

**Avoid when:**

* Need real-time streaming (use diart instead)
* Working in non-Python environment

### NVIDIA NeMo

**Best for:** Enterprise deployments, GPU-accelerated training

```
Strengths:              Limitations:
✓ NVIDIA optimization   ✗ Heavy dependencies
✓ Full pipeline tools   ✗ NVIDIA GPU required
✓ Pre-trained models    ✗ Steeper learning curve
✓ Active development
✓ Enterprise support
```

**Performance:**

* Optimized for NVIDIA GPUs
* RTF: 0.1 (with TensorRT)
* Multi-GPU training support

**Use when:**

* Enterprise/commercial deployment
* Have NVIDIA GPUs
* Need conversational AI integration
* Training custom models at scale

**Avoid when:**

* Limited hardware resources
* Simple use case
* Non-NVIDIA GPU

### SpeechBrain

**Best for:** Research, custom model development, education

```
Strengths:              Limitations:
✓ Modular architecture  ✗ Less production-ready
✓ Research-friendly     ✗ Smaller community
✓ Many pretrained       ✗ Documentation gaps
✓ PyTorch native
✓ Recipes & tutorials
```

**Performance:**

* ECAPA-TDNN EER: 0.8%
* Flexible embedding extraction
* Good VoxCeleb results

**Use when:**

* Researching new architectures
* Need to modify components
* Educational purposes
* Custom training pipelines

**Avoid when:**

* Need production-ready system
* Want minimal setup

### diart

**Best for:** Real-time streaming diarization

```
Strengths:              Limitations:
✓ True streaming        ✗ Lower accuracy vs offline
✓ Built on pyannote     ✗ Latency-accuracy tradeoff
✓ Simple API            ✗ Smaller community
✓ Microphone support
```

**Performance:**

* RTF: 0.3 (real-time capable)
* Uses pyannote models
* Streaming buffer support

**Use when:**

* Real-time requirements
* Live transcription systems
* Latency-sensitive applications

**Avoid when:**

* Offline batch processing
* Need highest accuracy

### Resemblyzer

**Best for:** Simple voice matching, embedding extraction

```
Strengths:              Limitations:
✓ Simple API            ✗ No diarization pipeline
✓ Lightweight           ✗ Lower accuracy
✓ Voice comparison      ✗ Limited features
✓ Pre-trained d-vectors
```

**Performance:**

* 256-dim d-vector embeddings
* GE2E trained
* Good for voice matching

**Use when:**

* Simple voice comparison
* Building custom pipeline
* Lightweight requirements
* Learning speaker embeddings

**Avoid when:**

* Need full diarization
* Require best accuracy

### WhisperX

**Best for:** Combined transcription + diarization

```
Strengths:              Limitations:
✓ ASR + diarization     ✗ Not specialized
✓ Word-level timestamps ✗ Moderate DER
✓ Uses Whisper models   ✗ Higher latency
✓ Speaker-attributed
```

**Performance:**

* Leverages Whisper ASR
* RTF: 0.1 (with batching)
* Word-level timestamps

**Use when:**

* Need transcription + speaker labels
* Speaker-attributed transcripts
* Meeting/podcast transcription

**Avoid when:**

* Diarization-only needed
* Real-time requirements

## Detailed Comparison Table

| Feature | pyannote | NeMo | SpeechBrain | diart | Resemblyzer | WhisperX |
|---------|----------|------|-------------|-------|-------------|----------|
| Diarization | ✓ | ✓ | ✓ | ✓ | Partial | ✓ |
| Embedding | ✓ | ✓ | ✓ | ✓ | ✓ | Via pyannote |
| VAD | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| Streaming | Via diart | ✓ | ✗ | ✓ | ✗ | ✗ |
| Training | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| ASR | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ |
| Pre-trained | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| GPU Required | Recommended | Yes | Recommended | No | No | Recommended |

## Performance Benchmarks

### Diarization Error Rate (DER)

| Framework | AMI | CALLHOME | DIHARD |
|-----------|-----|----------|--------|
| pyannote 3.1 | 10% | 11% | 18% |
| NeMo | 11% | 12% | 19% |
| diart | 15% | 14% | 22% |

*Lower is better. Results approximate, depend on configuration.*

### Real-Time Factor (RTF)

| Framework | CPU | GPU |
|-----------|-----|-----|
| pyannote | 0.6 | 0.15 |
| NeMo | N/A | 0.1 |
| diart | 0.5 | 0.3 |
| WhisperX | 0.4 | 0.1 |

*Lower is better. RTF < 1 means faster than real-time.*

## Deployment Considerations

### Cloud Deployment

| Framework | Docker | Kubernetes | Serverless |
|-----------|--------|------------|------------|
| pyannote | ✓ | ✓ | Possible |
| NeMo | ✓ | ✓ | Hard |
| diart | ✓ | ✓ | Possible |

### Edge Deployment

| Framework | Mobile | Embedded | ONNX Export |
|-----------|--------|----------|-------------|
| pyannote | Hard | Hard | Possible |
| NeMo | Via TensorRT | Possible | ✓ |
| Resemblyzer | Possible | ✓ | ✓ |

## Migration Paths

### Starting Simple → Production

```
Resemblyzer (prototype) → pyannote (production)
```

### Research → Deployment

```
SpeechBrain (research) → NeMo or pyannote (deploy)
```

### Offline → Streaming

```
pyannote (offline) → diart (streaming)
```

## Cost Considerations

| Factor | pyannote | NeMo | SpeechBrain | Resemblyzer |
|--------|----------|------|-------------|-------------|
| License | MIT | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| GPU Required | Recommended | Yes | Recommended | No |
| Memory (inference) | 2-4 GB | 4-8 GB | 2-4 GB | <1 GB |
| Setup Complexity | Medium | High | Medium | Low |

## Recommendations by Scenario

### Meeting Transcription Service

**Recommendation:** pyannote.audio + WhisperX

```
- pyannote for accurate speaker labels
- Whisper for transcription
- Combine for speaker-attributed transcript
```

### Real-time Call Center

**Recommendation:** diart + NeMo

```
- diart for streaming diarization
- NeMo for enterprise features
- Custom clustering threshold per use case
```

### Research Project

**Recommendation:** SpeechBrain

```
- Modular, hackable
- Good documentation
- PyTorch native
- Many pretrained models
```

### MVP/Prototype

**Recommendation:** Resemblyzer or pyannote

```
- Fast to get started
- Minimal dependencies
- Good enough for demos
```

## Getting Started Code

### pyannote (Recommended Default)

```python
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1"
)
diarization = pipeline("audio.wav")
```

### NeMo

```python
from nemo.collections.asr.models import ClusteringDiarizer

cfg = ClusteringDiarizer.from_pretrained("diar_msdd_telephonic")
diarization = cfg.diarize("audio.wav")
```

### SpeechBrain

```python
from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)
embedding = classifier.encode_batch(waveform)
```

---

*Last updated: 2026-01-06*
