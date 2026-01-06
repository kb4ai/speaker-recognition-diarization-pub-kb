# Training Custom Speaker Embeddings

Train speaker embedding models on your own data for domain-specific applications.

## When to Train Custom Embeddings

Pre-trained models work well for general speech, but custom training helps when:

* **Domain mismatch**: Call center audio, medical dictation, specific languages
* **Unique acoustic conditions**: Noisy environments, specific microphones
* **Privacy requirements**: Cannot use models trained on public data
* **Specialized speakers**: Children, elderly, accented speech

## Prerequisites

* GPU with 8+ GB VRAM (16+ GB recommended)
* 50+ hours of labeled speaker data (more is better)
* Python 3.8+ with PyTorch

## Option 1: SpeechBrain Training

SpeechBrain provides the most accessible training pipeline.

### Step 1: Prepare Data

Organize audio files:

```
data/
├── train/
│   ├── speaker001/
│   │   ├── utt001.wav
│   │   ├── utt002.wav
│   │   └── ...
│   ├── speaker002/
│   │   └── ...
│   └── ...
├── dev/
│   └── (same structure)
└── test/
    └── (same structure)
```

Create manifest CSV:

```csv
ID,duration,wav,spk_id
utt001,3.5,data/train/speaker001/utt001.wav,speaker001
utt002,4.2,data/train/speaker001/utt002.wav,speaker001
```

### Step 2: Install SpeechBrain

```bash
pip install speechbrain
git clone https://github.com/speechbrain/speechbrain
cd speechbrain/recipes/VoxCeleb/SpeakerRec
```

### Step 3: Configure Training

Edit `train_ecapa_tdnn.yaml`:

```yaml
# Data paths
data_folder: /path/to/your/data
train_annotation: /path/to/train.csv
valid_annotation: /path/to/dev.csv

# Model architecture
embedding_model: !new:speechbrain.lobes.models.ECAPA_TDNN.ECAPA_TDNN
    input_size: 80
    channels: [512, 512, 512, 512, 1536]
    kernel_sizes: [5, 3, 3, 3, 1]
    dilations: [1, 2, 3, 4, 1]
    attention_channels: 128
    lin_neurons: 192

# Training hyperparameters
number_of_epochs: 20
batch_size: 32
lr: 0.001
```

### Step 4: Train

```bash
python train.py train_ecapa_tdnn.yaml \
    --data_folder=/path/to/data \
    --output_folder=results/my_ecapa
```

### Step 5: Export Model

```python
from speechbrain.pretrained import EncoderClassifier

# Load trained model
classifier = EncoderClassifier.from_hparams(
    source="results/my_ecapa",
    savedir="my_model"
)

# Test inference
embedding = classifier.encode_batch(waveform)
```

## Option 2: WeSpeaker Training

WeSpeaker offers production-oriented training.

### Step 1: Install WeSpeaker

```bash
git clone https://github.com/wenet-e2e/wespeaker
cd wespeaker
pip install -r requirements.txt
```

### Step 2: Prepare Kaldi-style Data

```
data/
├── wav.scp      # utterance_id /path/to/audio.wav
├── utt2spk      # utterance_id speaker_id
└── spk2utt      # speaker_id utt1 utt2 utt3 ...
```

### Step 3: Configure and Train

```bash
# Edit config
vim examples/voxceleb/v2/conf/ecapa_tdnn.yaml

# Run training
bash examples/voxceleb/v2/run.sh --stage 1 --stop_stage 4
```

## Option 3: NVIDIA NeMo Training

For enterprise-scale training with NVIDIA GPUs.

### Step 1: Install NeMo

```bash
pip install nemo_toolkit[asr]
```

### Step 2: Prepare Manifest

```json
{"audio_filepath": "audio1.wav", "duration": 3.5, "label": "speaker001"}
{"audio_filepath": "audio2.wav", "duration": 4.2, "label": "speaker001"}
```

### Step 3: Configure Training

```python
import nemo.collections.asr as nemo_asr

# Create config
config = {
    "model": {
        "train_ds": {
            "manifest_filepath": "train_manifest.json",
            "batch_size": 32,
        },
        "encoder": {
            "_target_": "nemo.collections.asr.modules.ConvASREncoder",
            "feat_in": 80,
            "activation": "relu",
        },
        "decoder": {
            "_target_": "nemo.collections.asr.modules.SpeakerDecoder",
            "feat_in": 1536,
            "num_classes": num_speakers,
        }
    }
}
```

### Step 4: Train

```python
from nemo.collections.asr.models import EncDecSpeakerLabelModel

model = EncDecSpeakerLabelModel(cfg=config)
trainer = pl.Trainer(max_epochs=20, gpus=1)
trainer.fit(model)
```

## Data Augmentation

Augmentation is critical for good generalization:

### Speed Perturbation

```python
# SpeechBrain augmentation
from speechbrain.processing.speech_augmentation import SpeedPerturb

speed_perturb = SpeedPerturb(
    orig_freq=16000,
    speeds=[90, 100, 110]  # 0.9x, 1.0x, 1.1x
)
augmented = speed_perturb(waveform)
```

### Noise Addition

```python
from speechbrain.processing.speech_augmentation import AddNoise

add_noise = AddNoise(
    csv_file="noise_manifest.csv",
    snr_low=5,
    snr_high=20
)
noisy = add_noise(waveform, lengths)
```

### Room Impulse Response

```python
from speechbrain.processing.speech_augmentation import AddReverb

add_reverb = AddReverb(
    csv_file="rir_manifest.csv"
)
reverbed = add_reverb(waveform, lengths)
```

### SpecAugment

```python
from speechbrain.lobes.augment import SpecAugment

spec_augment = SpecAugment(
    time_warp=False,
    freq_mask_width=(0, 20),
    time_mask_width=(0, 100)
)
augmented_features = spec_augment(features)
```

## Loss Functions

### AAM-Softmax (Recommended)

Additive Angular Margin loss provides best results:

```python
from speechbrain.nnet.losses import LogSoftmaxWrapper, AdditiveAngularMargin

aam_loss = LogSoftmaxWrapper(
    loss_fn=AdditiveAngularMargin(margin=0.2, scale=30)
)
```

### Parameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| `margin` | 0.2 | Higher = harder training, better separation |
| `scale` | 30 | Higher = sharper probability distribution |

## Evaluation During Training

### Track EER on Validation Set

```python
from speechbrain.utils.metric_stats import EER

eer_computer = EER()

for batch in dev_loader:
    embeddings = model.encode_batch(batch.sig)
    scores = compute_cosine_scores(embeddings)
    eer_computer.append(scores, labels)

eer = eer_computer.summarize()
print(f"Validation EER: {eer:.2%}")
```

## Hyperparameter Recommendations

| Hyperparameter | Recommended Value | Notes |
|----------------|-------------------|-------|
| Batch size | 64-256 | Larger is better with enough GPU memory |
| Learning rate | 1e-3 to 1e-4 | With Adam/AdamW |
| Epochs | 20-50 | Monitor validation EER |
| Embedding dim | 192 | ECAPA-TDNN default |
| Segment length | 2-3 seconds | During training |
| Augmentation | All of the above | Critical for generalization |

## Common Issues

### Overfitting

* **Symptoms**: Low training loss, poor validation EER
* **Solutions**: More augmentation, dropout, fewer epochs

### Underfitting

* **Symptoms**: High loss, not improving
* **Solutions**: Longer training, higher learning rate, simpler model

### Speaker Imbalance

* **Symptoms**: Model biased toward frequent speakers
* **Solutions**: Balanced sampling, class weights

## Transfer Learning

Start from pre-trained model for faster convergence:

```python
# Load pre-trained ECAPA-TDNN
pretrained = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

# Replace classifier head
pretrained.classifier = nn.Linear(192, num_your_speakers)

# Fine-tune with lower learning rate
optimizer = Adam(pretrained.parameters(), lr=1e-5)
```

## Export for Production

### ONNX Export

```python
import torch.onnx

# Export to ONNX
dummy_input = torch.randn(1, 16000)
torch.onnx.export(
    model.mods.embedding_model,
    dummy_input,
    "speaker_embedding.onnx",
    input_names=["audio"],
    output_names=["embedding"],
    dynamic_axes={"audio": {1: "length"}}
)
```

## Resources

* [SpeechBrain VoxCeleb Recipe](https://github.com/speechbrain/speechbrain/tree/main/recipes/VoxCeleb)
* [WeSpeaker Training Guide](https://github.com/wenet-e2e/wespeaker/blob/master/docs/training.md)
* [NeMo Speaker Recognition](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/intro.html)

---

*Last updated: 2026-01-06*
