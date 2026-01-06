# Pre-trained Models

Pre-trained speaker embedding and diarization models available on HuggingFace and NGC.

**Total: 10 models**

## Model Types

* [Diarization Pipeline](diarization/) (3)
* [Embedding Model](embedding/) (7)

## Embedding Models

| Model | Architecture | Dim | EER | Provider |
|-------|--------------|----:|----:|----------|
| [WavLM Base Plus Speaker Verification](https://huggingface.co/microsoft/wavlm-base-plus-sv) | WavLM (Transformer) | 512 | 0.84% | microsoft |
| [TitaNet Large](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large) | TitaNet | 192 | 0.66% | nvidia |
| [TitaNet Small]() | TitaNet (SE-Res2Net) | 192 | 1.48% | nvidia |
| [WeSpeaker ResNet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) | ResNet34 | 256 | 0.84% | pyannote |
| [ECAPA-TDNN Speaker Embedding](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | ECAPA-TDNN | 192 | 0.8% | speechbrain |
| [SpeechBrain x-vector Speaker Embedding](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) | x-vector (TDNN) | 512 | 1.9% | speechbrain |
| [CAM++ Speaker Embedding](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM) | CAM++ (TDNN + Multi-scale Aggregation) | 192 | 0.87% | wespeaker |

## Diarization Pipelines

| Model | DER (AMI) | Provider |
|-------|----------:|----------|
| [NVIDIA MSDD (Multi-Scale Diarization Decoder)]() | 11.5% | nvidia |
| [NVIDIA Sortformer]() | 8.5% | NVIDIA |
| [pyannote Speaker Diarization 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) | 10.0% | pyannote |

## Usage Examples

See individual model files for code examples.

---

*Auto-generated on 2026-01-06*