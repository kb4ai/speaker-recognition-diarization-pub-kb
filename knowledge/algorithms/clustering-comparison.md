# Clustering Algorithms for Speaker Diarization

Clustering is the final stage of the cascaded diarization pipeline, grouping speaker embeddings into discrete speaker identities.

## Overview

| Algorithm | Type | Speaker Count | Complexity | Best For |
|-----------|------|---------------|------------|----------|
| AHC | Hierarchical | Pre-set threshold | O(n² log n) | General use |
| Spectral | Graph-based | Eigenvalue analysis | O(n³) | Known speaker count |
| VBx | Bayesian | Automatic | O(n²) per iteration | Unknown speakers |

## Agglomerative Hierarchical Clustering (AHC)

### How It Works

AHC performs bottom-up clustering:

```
1. Start: Each segment = 1 cluster
2. Compute pairwise distances
3. Merge two closest clusters
4. Repeat until threshold reached
```

### Algorithm

```python
def ahc_clustering(embeddings, threshold):
    # Initialize: each embedding is its own cluster
    clusters = [[i] for i in range(len(embeddings))]

    while True:
        # Find closest pair of clusters
        min_dist = infinity
        for i, j in all_pairs(clusters):
            dist = linkage_distance(clusters[i], clusters[j], embeddings)
            if dist < min_dist:
                min_dist, merge_i, merge_j = dist, i, j

        # Stop if distance exceeds threshold
        if min_dist > threshold:
            break

        # Merge closest clusters
        clusters[merge_i].extend(clusters[merge_j])
        del clusters[merge_j]

    return clusters
```

### Linkage Methods

| Method | Formula | Use Case |
|--------|---------|----------|
| Single | min(d(a,b)) for a∈A, b∈B | Elongated clusters |
| Complete | max(d(a,b)) for a∈A, b∈B | Compact clusters |
| Average | mean(d(a,b)) for a∈A, b∈B | **Recommended** |
| Ward | Minimize variance increase | Equal-sized clusters |

### Distance Metrics

**Cosine Distance** (most common):

```
d(a, b) = 1 - (a · b) / (||a|| × ||b||)
```

**PLDA Score** (probabilistic):

```
d(a, b) = -log P(same speaker | a, b)
```

### Threshold Selection

| Dataset | Typical Threshold | Collar |
|---------|------------------|--------|
| AMI | 0.5 - 0.7 | 0.25s |
| CALLHOME | 0.4 - 0.6 | 0.25s |
| DIHARD | 0.3 - 0.5 | 0.0s |

**Grid search recommended** on development set.

### Pros & Cons

**Advantages:**

* Simple to implement
* No assumption on cluster shape
* Works well with cosine distance
* Deterministic results

**Limitations:**

* Requires threshold tuning per dataset
* Sensitive to outliers
* Cannot undo bad early merges
* O(n² log n) complexity

## Spectral Clustering

### How It Works

Spectral clustering uses graph theory:

```
1. Build affinity matrix from embeddings
2. Compute graph Laplacian
3. Find eigenvectors (spectral decomposition)
4. Cluster in eigenspace using k-means
```

### Algorithm Steps

**Step 1: Affinity Matrix**

```
A[i,j] = exp(-d(eᵢ, eⱼ)² / (2σ²))
```

Where `d` is cosine distance and `σ` is bandwidth parameter.

**Step 2: Graph Laplacian**

```
D = diag(sum(A, axis=1))  # Degree matrix
L = D - A                  # Unnormalized Laplacian
L_norm = D^(-1/2) L D^(-1/2)  # Normalized (preferred)
```

**Step 3: Eigenvector Decomposition**

Find `k` smallest eigenvectors of L_norm:

```
L_norm v = λ v
V = [v₁, v₂, ..., vₖ]  # Stack eigenvectors
```

**Step 4: k-means in Eigenspace**

```
Y = row_normalize(V)
labels = kmeans(Y, k)
```

### Determining Speaker Count

**Eigengap heuristic:**

```python
eigenvalues = sorted(eigenvalues)
gaps = [eigenvalues[i+1] - eigenvalues[i] for i in range(len-1)]
n_speakers = argmax(gaps) + 1
```

### Pros & Cons

**Advantages:**

* Can find non-convex clusters
* Natural speaker count estimation via eigengap
* Robust to cluster shape

**Limitations:**

* O(n³) eigendecomposition
* Requires good affinity function
* k-means step is stochastic
* Sensitive to bandwidth σ

## VBx (Variational Bayes x-vectors)

### Overview

VBx is a Bayesian approach that:

* Models speaker and session variability
* Automatically estimates speaker count
* Handles uncertainty in assignments

### Generative Model

```
For each segment s:
  1. Draw speaker z_s ~ Categorical(π)
  2. Draw latent factor y_s ~ N(0, I)
  3. Generate embedding: x_s = μ + Vy_s + ε

Where:
  π = speaker prior probabilities
  μ = global mean embedding
  V = speaker variability subspace
  ε ~ N(0, Σ) = residual noise
```

### VB-EM Algorithm

**E-step:** Update soft speaker assignments

```
γ(z_s = k) ∝ π_k × N(x_s | μ_k, Σ_k)
```

**M-step:** Update speaker parameters

```
π_k = (1/N) Σ_s γ(z_s = k)
μ_k = (Σ_s γ(z_s = k) x_s) / (Σ_s γ(z_s = k))
```

### Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `Fa` | Between-speaker scaling | 0.3-0.5 |
| `Fb` | Within-speaker scaling | 17-19 |
| `loopP` | Self-loop probability | 0.9-0.99 |
| `max_speakers` | Upper bound | 10-20 |

### Integration with HMM

VBx often uses an HMM to model speaker turns:

```
States: Speaker identities (1 to K)
Transitions: P(speaker_t | speaker_{t-1})
Emissions: P(embedding | speaker)
```

**Self-loop probability** (`loopP`) controls minimum segment duration.

### Pros & Cons

**Advantages:**

* Automatic speaker count estimation
* Uncertainty quantification
* Handles overlapping speech (soft assignments)
* State-of-the-art performance

**Limitations:**

* More complex implementation
* Requires PLDA model training
* Multiple hyperparameters
* Iterative (slower than AHC)

## Comparison Summary

### Performance (DER on AMI)

| Algorithm | DER | Notes |
|-----------|-----|-------|
| AHC | 12-15% | With tuned threshold |
| Spectral | 13-16% | With eigengap |
| VBx | 10-13% | With PLDA backend |

### When to Use Each

**Use AHC when:**

* Simple pipeline needed
* Dataset has consistent characteristics
* Development set available for threshold tuning

**Use Spectral when:**

* Speaker count known or estimable
* Non-convex clusters expected
* Eigengap gives good estimates

**Use VBx when:**

* Best accuracy required
* Speaker count varies widely
* PLDA model available
* Handling overlap is important

## Implementation Examples

### AHC with scikit-learn

```python
from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.5,
    metric='cosine',
    linkage='average'
)
labels = clustering.fit_predict(embeddings)
```

### Spectral with pyannote

```python
from pyannote.audio.pipelines.clustering import AgglomerativeClustering

clustering = AgglomerativeClustering(
    metric="cosine",
    method="centroid",
    threshold=0.7
)
```

### VBx with BUT toolkit

```python
# Using BUT VBx implementation
from vbx import VBx

vbx = VBx(
    Fa=0.4,
    Fb=17,
    loopP=0.9
)
labels = vbx.fit_predict(embeddings, plda_model)
```

## References

* Park et al. "VBx: Neural Speaker Diarization" (ICASSP 2022)
* Ng et al. "On Spectral Clustering" (NeurIPS 2001)
* Day & Edelsbrunner "Efficient Algorithms for AHC" (J. Classification 1984)

---

*Last updated: 2026-01-06*
