# Building a Voice Fingerprint System

Create a speaker identification system with enrollment and matching capabilities.

## System Overview

A voice fingerprint system has two phases:

```
┌─────────────────────────────────────────────────────────────┐
│                    ENROLLMENT PHASE                         │
│  Audio → Preprocessing → Embedding → Store in Database      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   MATCHING PHASE                            │
│  Audio → Preprocessing → Embedding → Compare → Decision     │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

```bash
pip install speechbrain torch torchaudio numpy
```

## Step 1: Load Embedding Model

```python
from speechbrain.inference.speaker import EncoderClassifier
import torch

# Load pre-trained ECAPA-TDNN
encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)
```

## Step 2: Audio Preprocessing

```python
import torchaudio

def load_audio(filepath, target_sr=16000):
    """Load and preprocess audio file."""
    waveform, sr = torchaudio.load(filepath)

    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform

def extract_embedding(encoder, waveform):
    """Extract speaker embedding from audio."""
    with torch.no_grad():
        embedding = encoder.encode_batch(waveform)
    return embedding.squeeze().cpu().numpy()
```

## Step 3: Speaker Database

```python
import json
import numpy as np
from pathlib import Path

class SpeakerDatabase:
    """Simple speaker embedding database."""

    def __init__(self, db_path="speaker_db.json"):
        self.db_path = Path(db_path)
        self.speakers = {}
        self.load()

    def load(self):
        """Load database from disk."""
        if self.db_path.exists():
            with open(self.db_path) as f:
                data = json.load(f)
                self.speakers = {
                    k: np.array(v["embedding"])
                    for k, v in data.items()
                }

    def save(self):
        """Save database to disk."""
        data = {
            k: {"embedding": v.tolist()}
            for k, v in self.speakers.items()
        }
        with open(self.db_path, "w") as f:
            json.dump(data, f)

    def enroll(self, speaker_id, embedding):
        """Add or update speaker."""
        self.speakers[speaker_id] = embedding
        self.save()

    def get_all_speakers(self):
        """Get all enrolled speaker IDs."""
        return list(self.speakers.keys())

    def get_embedding(self, speaker_id):
        """Get embedding for a speaker."""
        return self.speakers.get(speaker_id)
```

## Step 4: Enrollment Process

```python
def enroll_speaker(encoder, db, speaker_id, audio_files):
    """
    Enroll a speaker with multiple audio samples.

    Args:
        encoder: Embedding model
        db: Speaker database
        speaker_id: Unique identifier for speaker
        audio_files: List of audio file paths

    Returns:
        Average embedding for the speaker
    """
    embeddings = []

    for audio_path in audio_files:
        # Load and process audio
        waveform = load_audio(audio_path)

        # Extract embedding
        embedding = extract_embedding(encoder, waveform)
        embeddings.append(embedding)

    # Average embeddings (more robust than single sample)
    avg_embedding = np.mean(embeddings, axis=0)

    # Normalize to unit length
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

    # Store in database
    db.enroll(speaker_id, avg_embedding)

    print(f"Enrolled speaker: {speaker_id} ({len(audio_files)} samples)")
    return avg_embedding

# Example usage
db = SpeakerDatabase()

enroll_speaker(
    encoder, db,
    speaker_id="john_doe",
    audio_files=[
        "enrollment/john_sample1.wav",
        "enrollment/john_sample2.wav",
        "enrollment/john_sample3.wav"
    ]
)
```

## Step 5: Speaker Matching

```python
def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def identify_speaker(encoder, db, audio_path, threshold=0.5):
    """
    Identify speaker from audio (1:N matching).

    Args:
        encoder: Embedding model
        db: Speaker database
        audio_path: Path to test audio
        threshold: Minimum similarity for match

    Returns:
        (speaker_id, similarity) or (None, best_score)
    """
    # Extract embedding from test audio
    waveform = load_audio(audio_path)
    test_embedding = extract_embedding(encoder, waveform)
    test_embedding = test_embedding / np.linalg.norm(test_embedding)

    best_match = None
    best_score = -1

    # Compare against all enrolled speakers
    for speaker_id in db.get_all_speakers():
        enrolled_embedding = db.get_embedding(speaker_id)
        score = cosine_similarity(test_embedding, enrolled_embedding)

        if score > best_score:
            best_score = score
            best_match = speaker_id

    # Apply threshold
    if best_score >= threshold:
        return best_match, best_score
    else:
        return None, best_score

def verify_speaker(encoder, db, audio_path, claimed_id, threshold=0.5):
    """
    Verify claimed speaker identity (1:1 matching).

    Args:
        encoder: Embedding model
        db: Speaker database
        audio_path: Path to test audio
        claimed_id: Speaker ID being claimed
        threshold: Minimum similarity for acceptance

    Returns:
        (is_verified, similarity_score)
    """
    enrolled_embedding = db.get_embedding(claimed_id)
    if enrolled_embedding is None:
        return False, 0.0

    # Extract test embedding
    waveform = load_audio(audio_path)
    test_embedding = extract_embedding(encoder, waveform)
    test_embedding = test_embedding / np.linalg.norm(test_embedding)

    # Compute similarity
    score = cosine_similarity(test_embedding, enrolled_embedding)

    return score >= threshold, score
```

## Step 6: Complete Application

```python
class VoiceFingerprintSystem:
    """Complete voice fingerprint system."""

    def __init__(self, db_path="speaker_db.json"):
        # Load embedding model
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )

        # Initialize database
        self.db = SpeakerDatabase(db_path)

        # Thresholds
        self.identification_threshold = 0.5
        self.verification_threshold = 0.6

    def enroll(self, speaker_id, audio_files):
        """Enroll a new speaker."""
        return enroll_speaker(self.encoder, self.db, speaker_id, audio_files)

    def identify(self, audio_path):
        """Identify speaker from audio."""
        return identify_speaker(
            self.encoder, self.db, audio_path,
            self.identification_threshold
        )

    def verify(self, audio_path, claimed_id):
        """Verify claimed identity."""
        return verify_speaker(
            self.encoder, self.db, audio_path, claimed_id,
            self.verification_threshold
        )

    def list_speakers(self):
        """List all enrolled speakers."""
        return self.db.get_all_speakers()

# Usage example
system = VoiceFingerprintSystem()

# Enroll speakers
system.enroll("alice", ["alice1.wav", "alice2.wav"])
system.enroll("bob", ["bob1.wav", "bob2.wav"])

# Identify unknown speaker
speaker, score = system.identify("unknown.wav")
print(f"Identified as: {speaker} (score: {score:.2f})")

# Verify claimed identity
is_verified, score = system.verify("test.wav", "alice")
print(f"Verification: {'PASS' if is_verified else 'FAIL'} (score: {score:.2f})")
```

## Threshold Selection

Choose thresholds based on your security requirements:

| Use Case | Verification Threshold | Identification Threshold |
|----------|----------------------|-------------------------|
| High security | 0.7 - 0.8 | 0.6 - 0.7 |
| Balanced | 0.5 - 0.6 | 0.5 |
| Convenience | 0.4 - 0.5 | 0.4 |

### Threshold Calibration

```python
def calibrate_threshold(encoder, db, test_pairs):
    """
    Calibrate threshold using labeled test pairs.

    Args:
        test_pairs: List of (audio_path, speaker_id, is_same) tuples
    """
    scores_same = []
    scores_diff = []

    for audio_path, speaker_id, is_same in test_pairs:
        _, score = verify_speaker(encoder, db, audio_path, speaker_id, 0.0)

        if is_same:
            scores_same.append(score)
        else:
            scores_diff.append(score)

    # Find threshold that minimizes EER
    # (simplified: use mean of distributions)
    threshold = (np.mean(scores_same) + np.mean(scores_diff)) / 2

    return threshold
```

## Production Considerations

### Multiple Enrollment Samples

```python
def enroll_with_quality_check(encoder, db, speaker_id, audio_files, min_duration=3.0):
    """Enroll with audio quality validation."""
    valid_embeddings = []

    for audio_path in audio_files:
        waveform = load_audio(audio_path)
        duration = waveform.shape[1] / 16000

        # Skip too-short samples
        if duration < min_duration:
            print(f"Skipping {audio_path}: too short ({duration:.1f}s)")
            continue

        embedding = extract_embedding(encoder, waveform)
        valid_embeddings.append(embedding)

    if len(valid_embeddings) < 2:
        raise ValueError("Need at least 2 valid samples for enrollment")

    # Check consistency between samples
    similarities = []
    for i in range(len(valid_embeddings)):
        for j in range(i+1, len(valid_embeddings)):
            sim = cosine_similarity(valid_embeddings[i], valid_embeddings[j])
            similarities.append(sim)

    if np.mean(similarities) < 0.7:
        raise ValueError("Enrollment samples are inconsistent - may be different speakers")

    avg_embedding = np.mean(valid_embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    db.enroll(speaker_id, avg_embedding)

    return avg_embedding
```

### Batch Processing

```python
def identify_batch(encoder, db, audio_files, threshold=0.5):
    """Identify speakers for multiple audio files."""
    results = []

    # Extract all embeddings
    embeddings = []
    for audio_path in audio_files:
        waveform = load_audio(audio_path)
        embedding = extract_embedding(encoder, waveform)
        embeddings.append(embedding / np.linalg.norm(embedding))

    # Get all enrolled embeddings
    enrolled = {
        sid: db.get_embedding(sid)
        for sid in db.get_all_speakers()
    }

    # Compute all similarities at once
    for i, test_emb in enumerate(embeddings):
        best_match, best_score = None, -1

        for speaker_id, enrolled_emb in enrolled.items():
            score = cosine_similarity(test_emb, enrolled_emb)
            if score > best_score:
                best_score = score
                best_match = speaker_id

        if best_score >= threshold:
            results.append((audio_files[i], best_match, best_score))
        else:
            results.append((audio_files[i], None, best_score))

    return results
```

### REST API Example

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
system = VoiceFingerprintSystem()

@app.route("/enroll", methods=["POST"])
def api_enroll():
    speaker_id = request.form["speaker_id"]
    audio_file = request.files["audio"]
    audio_path = f"/tmp/{audio_file.filename}"
    audio_file.save(audio_path)

    system.enroll(speaker_id, [audio_path])
    return jsonify({"status": "enrolled", "speaker_id": speaker_id})

@app.route("/identify", methods=["POST"])
def api_identify():
    audio_file = request.files["audio"]
    audio_path = f"/tmp/{audio_file.filename}"
    audio_file.save(audio_path)

    speaker, score = system.identify(audio_path)
    return jsonify({"speaker": speaker, "confidence": float(score)})

@app.route("/verify", methods=["POST"])
def api_verify():
    claimed_id = request.form["speaker_id"]
    audio_file = request.files["audio"]
    audio_path = f"/tmp/{audio_file.filename}"
    audio_file.save(audio_path)

    verified, score = system.verify(audio_path, claimed_id)
    return jsonify({"verified": verified, "confidence": float(score)})
```

## Security Considerations

1. **Replay attacks**: Consider adding liveness detection
2. **Template protection**: Encrypt stored embeddings
3. **Audio quality**: Validate minimum SNR and duration
4. **Update embeddings**: Re-enroll periodically for voice changes

## References

* [SpeechBrain Speaker Recognition](https://speechbrain.github.io/)
* [Speaker Embeddings Explained](../fundamentals/speaker-embeddings.md)
* [Evaluation Metrics](../fundamentals/evaluation-metrics.md)

---

*Last updated: 2026-01-06*
