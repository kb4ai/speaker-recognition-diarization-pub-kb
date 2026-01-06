# Speaker Diarization Glossary

A comprehensive terminology reference for speaker diarization, recognition, and embedding concepts.

## A

**AHC (Agglomerative Hierarchical Clustering)**
: Bottom-up clustering algorithm that iteratively merges closest segments. Commonly used for speaker clustering with cosine distance.

**ASR (Automatic Speech Recognition)**
: Technology for converting spoken audio into text. Often combined with diarization for speaker-attributed transcripts.

**Attentive Statistics Pooling**
: Learned attention mechanism that weights temporal statistics (mean/std) before aggregation. Key innovation in ECAPA-TDNN.

## B

**BIC (Bayesian Information Criterion)**
: Statistical criterion used for speaker change detection. Compares model fits to identify acoustic boundaries.

## C

**Collar**
: Forgiveness window (typically 0.25s) around reference segment boundaries where errors are not counted in DER calculation.

**Cosine Similarity**
: Distance metric for comparing embeddings: `sim = (A·B)/(||A||×||B||)`. Range: -1 to 1, higher = more similar.

**Clustering Threshold**
: Cutoff value for determining when embeddings belong to same vs different speakers. Typically 0.5-1.5 for cosine distance.

## D

**DCF (Detection Cost Function)**
: Weighted combination of false acceptance and false rejection rates. Alternative to EER for speaker verification.

**DER (Diarization Error Rate)**
: Primary evaluation metric: `(FA + MS + SC) / Total Speech Duration`. Lower is better.

**Diarization**
: The task of partitioning audio into speaker-homogeneous segments, answering "who spoke when."

**d-vector**
: Speaker embedding from GE2E (Generalized End-to-End) training. Precursor to x-vectors. Used in Resemblyzer.

## E

**ECAPA-TDNN**
: Emphasized Channel Attention, Propagation and Aggregation TDNN. State-of-the-art (2020) embedding architecture with 192 dimensions and ~0.80% EER on VoxCeleb1.

**EER (Equal Error Rate)**
: Point where False Acceptance Rate equals False Rejection Rate. Primary metric for speaker verification.

**EEND (End-to-End Neural Diarization)**
: Neural approach that directly predicts speaker activities from audio, handling overlap natively.

**Embedding**
: Fixed-dimensional vector representation of speaker voice characteristics. See: Speaker Embedding.

**Enrollment**
: Process of registering a speaker by collecting audio samples and creating a reference embedding/voiceprint.

## F

**FA (False Alarm)**
: Non-speech incorrectly classified as speech in diarization evaluation.

**FAR (False Acceptance Rate)**
: Rate of incorrectly accepting impostors in speaker verification.

**Fbank Features**
: Filterbank energies without DCT transformation. Alternative to MFCCs for neural models.

**FRR (False Rejection Rate)**
: Rate of incorrectly rejecting genuine speakers in speaker verification.

## G

**GMM (Gaussian Mixture Model)**
: Statistical model used in i-vector extraction via Universal Background Model.

## H

**Hop Length**
: Step size between consecutive analysis windows. Typical: 10ms for features, 0.25-0.5s for segmentation.

## I

**i-vector**
: Identity vector. Generative speaker embedding from factor analysis (2011). 400-600 dimensions.

## J

**JER (Jaccard Error Rate)**
: Alternative to DER that treats each speaker equally regardless of speaking time.

## M

**Mel-spectrogram**
: Time-frequency representation with frequency axis warped to mel scale for human auditory perception.

**MFCC (Mel-Frequency Cepstral Coefficients)**
: Classic acoustic features: `Audio → FFT → Mel-filterbank → Log → DCT`. Typically 13-39 coefficients.

**MS (Missed Speech)**
: Speech not detected in diarization evaluation. Also called Missed Detection.

## N

**Neural VAD**
: Deep learning approach to voice activity detection. Examples: MarbleNet, PyanNet, Silero.

## O

**Overlap**
: Regions where multiple speakers talk simultaneously. Challenging for traditional cascaded systems.

**Overlap-aware VAD**
: VAD systems that detect and label overlapping speech regions.

## P

**PIT (Permutation Invariant Training)**
: Training objective that solves speaker label ambiguity by considering all speaker permutations.

**PLDA (Probabilistic Linear Discriminant Analysis)**
: Scoring method that models within/between-speaker variance. Produces log-likelihood ratio scores.

**PSE (Power Set Encoding)**
: Single-label classification over all speaker combinations. Used in some EEND variants.

## R

**Res2Net**
: Multi-scale feature extraction with hierarchical residual connections. Core component of ECAPA-TDNN.

**RTF (Real-Time Factor)**
: `Processing Time / Audio Duration`. RTF < 1 means faster than real-time.

**RTTM (Rich Transcription Time Marked)**
: Standard annotation format for diarization. Fields: `SPEAKER file 1 start duration <NA> <NA> speaker_id <NA> <NA>`

## S

**SC (Speaker Confusion)**
: Speech attributed to wrong speaker in diarization evaluation.

**SE Block (Squeeze-and-Excitation)**
: Channel-wise attention mechanism that recalibrates feature maps. Used in ECAPA-TDNN.

**Segmentation**
: Dividing audio into speaker-homogeneous chunks by detecting change points.

**Speaker Change Detection**
: Identifying timestamps where speaker identity changes.

**Speaker Embedding**
: Fixed-dimensional vector (192-512 dim) capturing unique voice characteristics. Properties: compact, comparison-ready, irreversible.

**Speaker Identification (1:N)**
: "Which enrolled speaker is this?" Compare against all N enrolled speakers.

**Speaker Verification (1:1)**
: "Is this the claimed speaker?" Binary accept/reject decision.

**Spectral Clustering**
: Graph-based clustering using affinity matrix eigenvalues. Alternative to AHC.

**Statistics Pooling**
: Aggregation layer computing mean and standard deviation across temporal dimension.

## T

**TDNN (Time Delay Neural Network)**
: Neural architecture with dilated convolutions for capturing temporal context. Base of x-vectors and ECAPA-TDNN.

**T-matrix (Total Variability Matrix)**
: Projection matrix in i-vector extraction that maps supervectors to low-dimensional space.

## U

**UBM (Universal Background Model)**
: GMM trained on diverse speakers. Foundation for i-vector extraction.

## V

**VAD (Voice Activity Detection)**
: Binary classification of speech vs non-speech regions. First stage of diarization pipeline.

**VBx (Variational Bayes x-vectors)**
: Bayesian clustering approach with automatic speaker count estimation.

**Voiceprint**
: Colloquial term for speaker embedding. Analogous to fingerprint for voice.

**VoxCeleb**
: Large-scale speaker recognition datasets. VoxCeleb1: 1,251 speakers. VoxCeleb2: 6,112 speakers.

## W

**Window Size**
: Duration of audio analyzed at once. Typical: 25ms for features, 1.5-3s for segmentation.

## X

**x-vector**
: Discriminative DNN embedding from TDNN with statistics pooling (2018). 512 dimensions. Foundation for modern speaker recognition.

---

## Cross-References

| Concept | Related Articles |
|---------|-----------------|
| Pipeline stages | [Pipeline Architecture](pipeline-architecture.md) |
| DER, EER, RTF | [Evaluation Metrics](evaluation-metrics.md) |
| Embedding architectures | [Speaker Embeddings](speaker-embeddings.md) |
| Streaming vs batch | [Real-time vs Offline](realtime-vs-offline.md) |
| Annotation format | [RTTM Format](rttm-format.md) |

---

*Last updated: 2026-01-06*
