# Pyannote.audio Quickstart Guide

Get speaker diarization running in 5 minutes with pyannote.audio, the industry-standard Python framework.

## Prerequisites

* Python 3.8+
* HuggingFace account (free)
* Accept pyannote model terms at HuggingFace

## Installation

```bash
pip install pyannote.audio
```

## Step 1: Accept Model Terms

Before using the pre-trained models, you must accept the terms:

1. Go to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. Click "Agree and access repository"
3. Do the same for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

## Step 2: Get HuggingFace Token

```bash
# Login via CLI
pip install huggingface_hub
huggingface-cli login
# Enter your token when prompted
```

Or set environment variable:

```bash
export HF_TOKEN="hf_your_token_here"
```

## Step 3: Basic Diarization

```python
from pyannote.audio import Pipeline
import torch

# Load pre-trained pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HF_TOKEN"
)

# Send to GPU if available
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))

# Run diarization
diarization = pipeline("audio.wav")

# Print results
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}s - {turn.end:.1f}s: {speaker}")
```

**Output:**

```
0.2s - 3.5s: SPEAKER_00
3.8s - 7.2s: SPEAKER_01
7.5s - 12.1s: SPEAKER_00
...
```

## Step 4: Export to RTTM

```python
# Save as RTTM file (standard diarization format)
with open("output.rttm", "w") as f:
    diarization.write_rttm(f)
```

**RTTM format:**

```
SPEAKER audio 1 0.200 3.300 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER audio 1 3.800 3.400 <NA> <NA> SPEAKER_01 <NA> <NA>
```

## Common Use Cases

### Processing Multiple Files

```python
from pathlib import Path

audio_dir = Path("audio_files/")

for audio_file in audio_dir.glob("*.wav"):
    diarization = pipeline(str(audio_file))

    rttm_path = audio_file.with_suffix(".rttm")
    with open(rttm_path, "w") as f:
        diarization.write_rttm(f)
```

### Specifying Number of Speakers

```python
# If you know the speaker count
diarization = pipeline("audio.wav", num_speakers=2)

# Or specify a range
diarization = pipeline("audio.wav", min_speakers=2, max_speakers=5)
```

### Getting Speaker Embeddings

```python
from pyannote.audio import Inference

# Load embedding model
inference = Inference(
    "pyannote/embedding",
    use_auth_token="YOUR_HF_TOKEN"
)

# Extract embedding for audio segment
embedding = inference.crop("audio.wav", {"start": 0.5, "end": 3.0})
# Shape: (1, 512)
```

### Voice Activity Detection Only

```python
from pyannote.audio import Pipeline

vad_pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection",
    use_auth_token="YOUR_HF_TOKEN"
)

vad = vad_pipeline("audio.wav")

for speech in vad.get_timeline().support():
    print(f"Speech: {speech.start:.1f}s - {speech.end:.1f}s")
```

## Pipeline Configuration

### Adjusting Hyperparameters

```python
# Get optimal hyperparameters for your use case
pipeline.instantiate({
    "clustering": {
        "method": "centroid",
        "min_cluster_size": 12,
        "threshold": 0.7,
    },
    "segmentation": {
        "min_duration_off": 0.0,
    }
})

diarization = pipeline("audio.wav")
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `clustering.threshold` | Merge threshold (lower = more speakers) | Auto-tuned |
| `clustering.min_cluster_size` | Minimum segments per speaker | 12 |
| `segmentation.min_duration_off` | Minimum silence between segments | 0.0s |

## Evaluation

### Using pyannote.metrics

```python
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment

# Load reference
reference = Annotation()
reference[Segment(0.0, 3.5)] = "A"
reference[Segment(3.5, 8.0)] = "B"

# Compute DER
metric = DiarizationErrorRate()
der = metric(reference, diarization)
print(f"DER: {der:.2%}")
```

### Component Breakdown

```python
# Detailed error breakdown
report = metric.report(display=True)
# Shows: FA, Miss, Confusion, Total
```

## Troubleshooting

### CUDA Out of Memory

```python
# Process in chunks for long audio
from pyannote.audio.pipelines.utils import get_devices

# Use CPU if GPU memory limited
pipeline.to(torch.device("cpu"))
```

### Slow Processing

```python
# Enable GPU
pipeline.to(torch.device("cuda"))

# Check RTF (should be < 0.2 on GPU)
import time
start = time.time()
diarization = pipeline("audio.wav")
duration = get_audio_duration("audio.wav")
rtf = (time.time() - start) / duration
print(f"RTF: {rtf:.2f}")
```

### Too Many/Few Speakers

```python
# Adjust clustering threshold
# Lower threshold = more speakers
# Higher threshold = fewer speakers

pipeline.instantiate({
    "clustering": {"threshold": 0.5}  # More speakers
})

pipeline.instantiate({
    "clustering": {"threshold": 0.9}  # Fewer speakers
})
```

## Integration Examples

### With Whisper (Speech-to-Text)

```python
import whisper
from pyannote.audio import Pipeline

# Transcribe
model = whisper.load_model("base")
result = model.transcribe("audio.wav")

# Diarize
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipeline("audio.wav")

# Combine: assign speakers to transcript segments
for segment in result["segments"]:
    start, end = segment["start"], segment["end"]
    # Find speaker at this time
    speaker = diarization.crop(Segment(start, end)).argmax()
    print(f"[{speaker}] {segment['text']}")
```

### Streaming with diart

For real-time diarization, see the [diart](https://github.com/juanmc2005/diart) library which builds on pyannote.

## Performance Tips

1. **Use GPU**: 5-10x faster than CPU
2. **Batch processing**: Process multiple files in parallel
3. **Pre-load model**: Initialize pipeline once, reuse for many files
4. **Downsample if needed**: 16kHz is sufficient for diarization

## References

* [pyannote.audio documentation](https://pyannote.github.io/)
* [HuggingFace model hub](https://huggingface.co/pyannote)
* [GitHub repository](https://github.com/pyannote/pyannote-audio)

---

*Last updated: 2026-01-06*
