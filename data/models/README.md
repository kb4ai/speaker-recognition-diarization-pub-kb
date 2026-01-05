# Pre-trained Models

Pre-trained speaker embedding and diarization models available on HuggingFace and NGC.

**Total: 4 models**

## Model Types

* [Diarization Pipeline](diarization/) (1)
* [Embedding Model](embedding/) (3)

## Embedding Models

| Model | Architecture | Dim | EER | Provider |
|-------|--------------|----:|----:|----------|
| [TitaNet Large](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large) | TitaNet | 192 | 0.66% | nvidia |
| [WeSpeaker ResNet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) | ResNet34 | 256 | 0.84% | pyannote |
| [ECAPA-TDNN Speaker Embedding](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | ECAPA-TDNN | 192 | 0.8% | speechbrain |

## Diarization Pipelines

| Model | DER (AMI) | Provider |
|-------|----------:|----------|
| [pyannote Speaker Diarization 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) | 10.0% | pyannote |

## Usage Examples

See individual model files for code examples.

---

*Auto-generated on 2026-01-06*