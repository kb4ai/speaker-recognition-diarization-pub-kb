# Evaluation Metrics for Speaker Diarization

## Diarization Error Rate (DER)

**DER** is the primary evaluation metric for speaker diarization systems.

### Formula

```
DER = (False Alarm + Missed Detection + Speaker Confusion) / Total Speech Duration
```

Expressed as a percentage.

### Components

| Error Type | Description |
|------------|-------------|
| **False Alarm (FA)** | Non-speech classified as speech |
| **Missed Detection (MS)** | Speech not detected (missed speech) |
| **Speaker Confusion (SC)** | Speech attributed to wrong speaker |

### Example Calculation

Given:

* Total speech duration: 100 seconds
* False alarms: 5 seconds
* Missed speech: 3 seconds
* Speaker confusion: 7 seconds

```
DER = (5 + 3 + 7) / 100 = 15%
```

### Collar

A **collar** is a forgiveness window around reference segment boundaries where errors are not counted.

* Typical collar: 0.25 seconds (250ms)
* Some challenges use 0 collar (stricter evaluation)
* Accounts for annotation ambiguity at boundaries

### Overlap Handling

DER can be calculated with or without overlap regions:

* **Including overlap**: Harder, counts errors in multi-speaker regions
* **Excluding overlap**: Easier, ignores overlapping speech
* DIHARD challenge uses 0 collar and includes overlap (strictest)

## Equal Error Rate (EER)

**EER** is the primary metric for speaker verification systems.

### Definition

The point where **False Acceptance Rate (FAR)** equals **False Rejection Rate (FRR)**:

```
EER = FAR = FRR (at threshold where they're equal)
```

### Terminology

| Metric | Description |
|--------|-------------|
| **False Acceptance Rate (FAR)** | Impostor accepted as genuine speaker |
| **False Rejection Rate (FRR)** | Genuine speaker rejected |
| **Detection Cost Function (DCF)** | Weighted combination of FAR and FRR |

### Interpretation

* Lower EER = better performance
* EER of 0.80% means ~1 in 125 trials is an error
* State-of-the-art systems achieve <1% EER on VoxCeleb

## Real-Time Factor (RTF)

**RTF** measures processing speed relative to audio duration.

### Formula

```
RTF = Processing Time / Audio Duration
```

### Interpretation

| RTF | Meaning |
|-----|---------|
| RTF < 1.0 | Faster than real-time |
| RTF = 1.0 | Real-time processing |
| RTF > 1.0 | Slower than real-time |

### Examples

* RTF = 0.1 means 10 seconds of audio processed in 1 second
* RTF = 0.15 typical for GPU-accelerated diarization
* RTF < 0.3 generally acceptable for batch processing

## Jaccard Error Rate (JER)

**JER** is an alternative to DER that accounts for speaker-specific errors.

### Formula

Per-speaker Jaccard distance averaged across all speakers:

```
JER = (1/N) × Σ (1 - Intersection(Ref_i, Hyp_i) / Union(Ref_i, Hyp_i))
```

### Advantages Over DER

* Treats each speaker equally regardless of speaking time
* Better for imbalanced speaker scenarios

## Benchmarks by Dataset

| Dataset | Typical DER | Notes |
|---------|-------------|-------|
| AMI Headset Mix | 10-15% | Meeting recordings |
| CALLHOME | 8-12% | Telephone, 2 speakers |
| VoxCeleb1 | EER 0.8-2% | Speaker verification |
| DIHARD III | 11-20% | Challenging domains, 0 collar |

## Evaluation Tools

### dscore

Standard toolkit for computing DER:

```bash
pip install dscore
dscore -r reference.rttm -s hypothesis.rttm
```

### pyannote.metrics

Comprehensive metrics for pyannote ecosystem:

```python
from pyannote.metrics.diarization import DiarizationErrorRate
metric = DiarizationErrorRate()
der = metric(reference, hypothesis)
```

### simpleDER

Lightweight Python package:

```bash
pip install simpleder
```

```python
from simpleder import DER
error = DER(ref, hyp)
```

## Best Practices

1. **Report collar**: Always specify collar used (0.0 or 0.25s)
2. **Overlap handling**: State whether overlap is included/excluded
3. **Multiple metrics**: Report both DER and component errors
4. **Confidence intervals**: Report variance across test files
5. **RTF with hardware**: Specify GPU/CPU and batch size

## References

* NIST Rich Transcription Evaluation Plan
* DIHARD Challenge Guidelines
* SpeechBrain DER Documentation
