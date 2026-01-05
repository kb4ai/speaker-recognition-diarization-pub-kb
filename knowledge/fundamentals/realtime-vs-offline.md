# Real-Time vs Offline Diarization

## Overview

The choice between online (streaming) and offline (batch) processing involves fundamental trade-offs in accuracy, latency, computational resources, and system complexity.

## Offline/Batch Diarization

### Characteristics

* Full audio file available before processing
* Can analyze bidirectionally (past and future context)
* Iterative refinement permitted
* Higher computational budget acceptable

### Accuracy Advantages

* **90-95% typical accuracy** vs 75-90% for streaming
* Complete context enables better speaker boundary detection
* Multiple passes for refinement (initial clustering → re-segmentation → overlap handling)
* Sophisticated global optimization (spectral clustering, VBx)
* Better handling of speaker overlap through retroactive analysis

### Processing Characteristics

| Metric | Typical Value |
|--------|---------------|
| Latency | Minutes to hours |
| RTF (GPU) | 0.1-0.5x |
| DER | 10-15% |

### Best Use Cases

* Meeting transcription and documentation
* Broadcast media archiving
* Legal/compliance recordings
* Research and analysis
* When accuracy > immediacy

### Recommended Tools

* **Pyannote 3.1**: 10% DER, state-of-the-art accuracy
* **NVIDIA NeMo**: Enterprise-grade, GPU-optimized
* **Kaldi**: Academic baseline, proven techniques

## Online/Streaming Diarization

### Characteristics

* Processes audio incrementally as it arrives
* Limited or no future context
* Immediate output required
* Strict latency constraints

### Technical Challenges

* **Causal processing**: Cannot use future information
* **Speaker count uncertainty**: New speakers may appear anytime
* **Limited embedding context**: Short audio chunks for low-latency extraction
* **Boundary precision**: Less context for accurate speaker transitions
* **Memory constraints**: Must maintain speaker history efficiently

### Modern Approaches

**Frame-wise Neural Diarization (FS-EEND):**

* Processes frame-by-frame with look-ahead (0.5-2 seconds)
* Non-autoregressive attractor decoder
* Achieves near-offline accuracy with controlled latency

**Multi-stage Clustering:**

* Fallback clusterer: Short inputs (<10 seconds)
* Main clusterer: Medium-length (10-60 seconds)
* Pre-clusterer: Compresses long-form audio

**Speaker Cache/Tracking:**

* Rolling buffer of speaker embeddings
* Arrival-order speaker indexing (spk_0 = first detected)
* Dynamic cache updates based on confidence

### Accuracy vs Latency Trade-off

| Latency | Typical Accuracy | Notes |
|---------|------------------|-------|
| 2-5 seconds | 85-90% | Better speaker transitions |
| 1-2 seconds | 80-85% | Acceptable for most applications |
| <1 second | 75-80% | May miss quick speaker changes |

### Recommended Tools

* **NVIDIA Streaming Sortformer**: 2-4+ speaker tracking, frame-level timestamps
* **diart**: Python real-time framework, WebRTC integration
* **pyannote.audio streaming**: Real-time capabilities in recent versions

## Hybrid Workflow

The most sophisticated production systems combine offline and online processing.

### Initial Offline Phase (Training/Enrollment)

1. Process historical recordings with high-accuracy offline diarization
2. Extract high-quality speaker embeddings from labeled segments
3. Manually correct errors and verify speaker identities
4. Build speaker database with verified voiceprints
5. Train speaker-specific models or adaptation layers

### Subsequent Online Phase (Inference)

1. Stream new audio through real-time diarization
2. Extract embeddings from detected speech segments
3. Compare against enrolled voiceprint database using cosine similarity
4. Match segments to known speakers (or mark as "unknown")
5. Apply online enrollment for new speakers who introduce themselves

### Incremental Learning and Adaptation

* **Online enrollment**: Register new speakers on-the-fly
* **Speaker profile updates**: Continuously refine voiceprints
* **Domain adaptation**: Adjust models to acoustic conditions
* **Few-shot adaptation**: Quickly adapt with limited samples

## Comparison Table

| Aspect | Offline | Online | Hybrid |
|--------|---------|--------|--------|
| **Accuracy** | 90-95% | 75-90% | 85-95% |
| **Latency** | Minutes+ | <5 seconds | Variable |
| **Speaker count** | Any | Limited | Any |
| **Overlap handling** | Excellent | Limited | Good |
| **New speakers** | Discovered | Must be enrolled | Both |
| **Computational cost** | High | Low-Medium | Medium-High |
| **Use case** | Archives | Live streams | Production |

## System Design Considerations

### For Offline Archival Processing

* Use highest-accuracy models (Pyannote 3.1, ensemble methods)
* Apply multiple refinement passes
* Leverage GPU clusters for batch processing
* Optimize DER with extensive hyperparameter tuning

### For Real-Time Applications

* Choose low-latency models (Streaming Sortformer, FS-EEND)
* Implement speaker caching for consistent labeling
* Balance latency vs accuracy based on use case
* Consider edge deployment constraints (mobile, embedded)

### For Hybrid Systems

* Build speaker database from high-quality offline diarization
* Deploy lightweight online models for matching against database
* Implement human-in-the-loop corrections
* Continuously update voiceprints with verified samples

## Accuracy Factors

| Factor | Impact |
|--------|--------|
| **Talk time** | Speakers need 30+ seconds for reliable detection |
| **Audio quality** | Clean audio dramatically improves accuracy |
| **Speaker similarity** | Similar-sounding voices increase confusion |
| **Overlap handling** | Simultaneous speech remains challenging |
| **Channel consistency** | Different microphones affect embeddings |

## References

* "Streaming Speaker Diarization with Neural Attractors" (2023)
* NVIDIA Streaming Sortformer Documentation
* "Frame-wise Streaming End-to-End Speaker Diarization" (ICASSP 2024)
