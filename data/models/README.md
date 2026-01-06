# Pre-trained Models

Pre-trained speaker embedding and diarization models available on HuggingFace and NGC.

**Total: 8 models**

## Model Types

* [Diarization Pipeline](diarization/) (3)
* [Embedding Model](embedding/) (5)

## Embedding Models

| Model | Architecture | Dim | EER | Provider |
|-------|--------------|----:|----:|----------|
| [TitaNet Large](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large) | TitaNet | 192 | 0.66% | nvidia |
| [TitaNet Small]() | TitaNet (SE-Res2Net) | 192 | 1.48% | nvidia |
| [WeSpeaker ResNet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) | ResNet34 | 256 | 0.84% | pyannote |
| [ECAPA-TDNN Speaker Embedding](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | ECAPA-TDNN | 192 | 0.8% | speechbrain |
| [SpeechBrain x-vector Speaker Embedding](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) | x-vector (TDNN) | 512 | 1.9% | speechbrain |

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