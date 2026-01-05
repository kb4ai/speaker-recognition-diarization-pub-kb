# KB-ROUTING: Query Resolution Logic

## Query Type → Path Resolution

```pseudocode
if query.type == "tool" or query.mentions(["library", "framework", "implementation"]):
    → data/tools/*.tool.yaml

    if query.mentions(["pyannote", "speechbrain", "nemo", "kaldi", "diart"]):
        → specific tool file

elif query.type == "algorithm":
    → data/algorithms/

    if query.mentions(["embedding", "x-vector", "ecapa", "d-vector", "i-vector"]):
        → data/algorithms/embeddings/*.algorithm.yaml

    elif query.mentions(["clustering", "ahc", "spectral", "vbx", "plda"]):
        → data/algorithms/clustering/*.algorithm.yaml

    elif query.mentions(["vad", "voice activity", "speech detection"]):
        → data/algorithms/vad/*.algorithm.yaml

    elif query.mentions(["segmentation", "change point", "bic"]):
        → data/algorithms/segmentation/*.algorithm.yaml

    elif query.mentions(["end-to-end", "eend", "sortformer", "ts-vad"]):
        → data/algorithms/end-to-end/*.algorithm.yaml

elif query.type == "model" or query.mentions(["pretrained", "huggingface", "checkpoint"]):
    → data/models/

    if query.mentions(["embedding model"]):
        → data/models/embeddings/*.model.yaml

    elif query.mentions(["pipeline", "full diarization"]):
        → data/models/diarization-pipelines/*.model.yaml

    elif query.mentions(["vad model"]):
        → data/models/vad/*.model.yaml

elif query.type == "dataset" or query.mentions(["corpus", "benchmark", "training data"]):
    → data/datasets/*.dataset.yaml

elif query.type == "comparison" or query.mentions(["compare", "which is better", "vs"]):
    → comparisons/*.md

elif query.type == "concept" or query.mentions(["what is", "explain", "how does"]):
    → knowledge/

    if query.mentions(["pipeline", "architecture", "process"]):
        → knowledge/fundamentals/

    elif query.mentions(["math", "equation", "formula", "mfcc", "cosine"]):
        → knowledge/math/

    elif query.mentions(["voiceprint", "enrollment", "verification"]):
        → knowledge/concepts/

elif query.type == "paper" or query.mentions(["paper", "publication", "arxiv"]):
    → papers/*.paper.yaml
```

## Common Queries

### "Best open-source diarization tool"
→ `comparisons/tools-overview.md` → sort by stars or accuracy

### "How does ECAPA-TDNN work?"
→ `data/algorithms/embeddings/ecapa-tdnn.algorithm.yaml`
→ `knowledge/fundamentals/speaker-embeddings.md`

### "VoxCeleb dataset details"
→ `data/datasets/voxceleb1.dataset.yaml`
→ `data/datasets/voxceleb2.dataset.yaml`

### "Real-time vs offline diarization"
→ `comparisons/real-time-vs-offline.md`
→ `knowledge/concepts/online-vs-offline-diarization.md`

### "Speaker embedding dimension comparison"
→ `comparisons/embedding-models-comparison.md`

## Search Patterns

```bash
# Find all Python tools
grep -l 'language: "Python"' data/tools/*.tool.yaml

# Find tools with streaming capability
grep -l 'streaming: true' data/tools/*.tool.yaml

# Find algorithms from 2020+
grep -l 'year-introduced: 202' data/algorithms/**/*.algorithm.yaml

# Find models with EER < 1%
grep -B5 'eer-voxceleb:' data/models/**/*.model.yaml | grep -A5 '0\.'
```

## Schema Inheritance

```
schemas/source.spec.yaml          # Base source citation schema
    ↓ (referenced by)
schemas/tool.spec.yaml            # Tools include sources[]
schemas/algorithm.spec.yaml       # Algorithms include sources[]
schemas/model.spec.yaml           # Models include sources[]
schemas/dataset.spec.yaml         # Datasets include sources[]
schemas/paper.spec.yaml           # Papers include sources[]
```
