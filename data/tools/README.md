# Tools & Frameworks

Open-source tools and frameworks for speaker diarization, recognition, and embedding extraction.

**Total: 10 tools**

## Quick Links

| Tool | Category | Description |
|------|----------|-------------|
| [ESPnet](https://github.com/espnet/espnet) | speech toolkit | End-to-end speech processing toolkit with speaker recognitio... |
| [Kaldi](https://github.com/kaldi-asr/kaldi) | asr framework | Speech recognition toolkit with x-vector speaker embeddings |
| [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) | diarization framework | Enterprise-scale conversational AI framework with speaker di... |
| [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) | voice fingerprinting | High-quality voice similarity and speaker diarization using ... |
| [Simple Diarizer](https://github.com/cvqluu/simple_diarizer) | diarization wrapper | Simplified speaker diarization wrapper using pretrained mode... |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) | embedding toolkit | PyTorch-based speech toolkit for research and production |
| [WeSpeaker](https://github.com/wenet-e2e/wespeaker) | embedding toolkit | Production-ready research and production oriented speaker em... |
| [WhisperX](https://github.com/m-bain/whisperX) | transcription diarization | Fast automatic speech recognition with word-level timestamps... |
| [diart](https://github.com/juanmc2005/diart) | streaming diarization | Real-time speaker diarization with streaming support |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | diarization framework | Neural building blocks for speaker diarization - industry-st... |

## By Category

### Asr Framework

* **[Kaldi](kaldi-asr--kaldi.tool.yaml)** - Speech recognition toolkit with x-vector speaker embeddings

### Diarization Framework

* **[NVIDIA NeMo](nvidia--nemo.tool.yaml)** - Enterprise-scale conversational AI framework with speaker diarization
* **[pyannote.audio](pyannote--pyannote-audio.tool.yaml)** - Neural building blocks for speaker diarization - industry-standard Python librar

### Diarization Wrapper

* **[Simple Diarizer](cvqluu--simple-diarizer.tool.yaml)** - Simplified speaker diarization wrapper using pretrained models

### Embedding Toolkit

* **[SpeechBrain](speechbrain--speechbrain.tool.yaml)** - PyTorch-based speech toolkit for research and production
* **[WeSpeaker](wenet-e2e--wespeaker.tool.yaml)** - Production-ready research and production oriented speaker embedding toolkit

### Speech Toolkit

* **[ESPnet](espnet--espnet.tool.yaml)** - End-to-end speech processing toolkit with speaker recognition and diarization

### Streaming Diarization

* **[diart](juanmc2005--diart.tool.yaml)** - Real-time speaker diarization with streaming support

### Transcription Diarization

* **[WhisperX](m-bain--whisperx.tool.yaml)** - Fast automatic speech recognition with word-level timestamps and speaker diariza

### Voice Fingerprinting

* **[Resemblyzer](resemble-ai--resemblyzer.tool.yaml)** - High-quality voice similarity and speaker diarization using d-vectors

## Capabilities Matrix

| Tool | Diarization | Embedding | VAD | Streaming | Training |
|------|:-----------:|:---------:|:---:|:---------:|:--------:|
| ESPnet | ✓ | ✓ | ✓ |  | ✓ |
| Kaldi | ✓ | ✓ | ✓ |  | ✓ |
| NVIDIA NeMo | ✓ | ✓ | ✓ | ✓ | ✓ |
| Resemblyzer | ✓ | ✓ |  |  |  |
| Simple Diarizer | ✓ |  |  |  |  |
| SpeechBrain | ✓ | ✓ | ✓ |  | ✓ |
| WeSpeaker | ✓ | ✓ | ✓ |  | ✓ |
| WhisperX | ✓ |  | ✓ |  |  |
| diart | ✓ | ✓ | ✓ | ✓ |  |
| pyannote.audio | ✓ | ✓ | ✓ | ✓ | ✓ |

## Performance Comparison

| Tool | DER (AMI) | EER (VoxCeleb) | RTF |
|------|----------:|---------------:|----:|
| NVIDIA NeMo |  |  | 0.1 |
| SpeechBrain |  | 0.8% |  |
| WeSpeaker |  | 0.72% |  |
| WhisperX |  |  | 0.1 |
| diart |  |  | 0.3 |
| pyannote.audio | 10.0% | 0.8% | 0.15 |

---

*Auto-generated on 2026-01-06. See [CONTRIBUTING.md](../CONTRIBUTING.md) for update instructions.*