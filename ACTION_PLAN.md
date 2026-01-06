# Speaker Recognition & Diarization Knowledge Base - Action Plan

This document outlines the implementation roadmap to transform this repository into a comprehensive, well-structured knowledge base with proper source attribution, archived knowledge, and machine-readable artifacts.

## Current State Assessment

### What Exists (Updated 2026-01-06)

* **10 Tools**: pyannote, speechbrain, nvidia-nemo, kaldi, diart, resemblyzer, whisperx, wespeaker, simple-diarizer, espnet
* **10 Algorithms**: 3 embedding, 3 clustering, 2 VAD, 2 end-to-end
* **5 Models**: speechbrain ECAPA, pyannote diarization 3.1, nvidia titanet, wespeaker resnet34, nvidia sortformer
* **5 Datasets**: VoxCeleb1/2, AMI, CALLHOME, DIHARD
* **6 Papers**: ECAPA-TDNN, x-vectors, pyannote, VoxCeleb1, VoxCeleb2, EEND
* **6 Schema Specs**: tool, algorithm, model, dataset, paper, source
* **Scripts**: validation, table generation, README generation, source verification, clone-all
* **Makefile**: validate, tables, readme, clone, sources, clean targets
* **10 Auto-generated READMEs**: One per directory with data extraction
* **Bibliography**: 16 BibTeX entries in archives/bibliography/
* **Archived Research**: Perplexity research in printouts/

### What's Missing

* **More Papers**: i-vectors, clustering papers, EEND-EDA
* **More Models**: more speechbrain models, more NeMo models
* **Phase 4/5**: README/CONTRIBUTING enhancements, automation scripts

## Implementation Phases

### Phase 1: Foundation & Traceability (Priority: High)

#### 1.1 Archive Source Research

* [x] Move Perplexity research to `printouts/2026-01-06--perplexity-speaker-diarization-research/`
* [x] Create SOURCE.yaml with provenance metadata
* [x] Extract key citations to `archives/bibliography/speaker-diarization.bib` (16 BibTeX entries)

#### 1.2 Fix Validation Issues

* [x] Fix `annotations` fields in dataset files (list → object)
* [x] Fix `splits` field in voxceleb2 (list → object)
* [x] Fix validation script (`language` removed from array_fields)

#### 1.3 Directory Structure Enhancement

```
archives/
├── bibliography/           # BibTeX files by topic
│   ├── embeddings.bib
│   ├── diarization.bib
│   └── speaker-diarization.bib
└── papers/                 # PDF copies (gitignored, local only)
    └── .gitkeep
```

### Phase 2: Content Population (Priority: High)

#### 2.1 Pre-trained Model Entries

Create entries in `data/models/`:

* [x] `embedding/pyannote--wespeaker-voxceleb-resnet34.model.yaml`
* [x] `embedding/speechbrain--spkrec-ecapa-voxceleb.model.yaml`
* [x] `embedding/nvidia--titanet-large.model.yaml`
* [x] `diarization/pyannote--speaker-diarization-3.1.model.yaml`
* [x] `diarization/nvidia--sortformer.model.yaml`

#### 2.2 Research Paper Entries

Create entries in `papers/`:

* [x] `ecapa-tdnn-2020.paper.yaml` - ECAPA-TDNN architecture
* [x] `x-vectors-2018.paper.yaml` - X-vector embeddings
* [x] `eend-2019.paper.yaml` - End-to-end neural diarization
* [x] `pyannote-2020.paper.yaml` - Pyannote.audio paper
* [x] `voxceleb1-2017.paper.yaml` - VoxCeleb1 paper
* [x] `voxceleb2-2018.paper.yaml` - VoxCeleb2 paper

#### 2.3 Additional Tool Entries

From research, add:

* [x] `resemble-ai--resemblyzer.tool.yaml` - Voice fingerprinting
* [x] `m-bain--whisperx.tool.yaml` - ASR + diarization
* [x] `cvqluu--simple-diarizer.tool.yaml` - Simple wrapper
* [x] `wenet-e2e--wespeaker.tool.yaml` - Speaker embedding toolkit
* [x] `espnet--espnet.tool.yaml` - ESPnet speech toolkit with speaker module

### Phase 3: Knowledge Articles (Priority: High - COMPLETE)

The `knowledge/` directory provides educational content for practitioners learning speaker diarization.

**Status:** 14 articles created across 4 subdirectories.

#### 3.1 Directory Structure

```
knowledge/
├── README.md                    # Auto-generated index (✓)
├── fundamentals/                # Core concepts - 6 articles
│   ├── pipeline-architecture.md (✓)
│   ├── evaluation-metrics.md (✓)
│   ├── speaker-embeddings.md (✓)
│   ├── glossary.md (✓)
│   ├── realtime-vs-offline.md (✓)
│   └── rttm-format.md (✓)
├── algorithms/                  # Deep dives - 2 articles
│   ├── ecapa-tdnn-explained.md (✓)
│   └── clustering-comparison.md (✓)
├── tutorials/                   # Step-by-step guides - 1 article
│   └── pyannote-quickstart.md (✓)
└── comparisons/                 # Decision guides - 1 article
    └── framework-selection.md (✓)
```

#### 3.2 Fundamentals (Priority: Highest) - COMPLETE

Created in `knowledge/fundamentals/`:

* [x] `pipeline-architecture.md` - Overview of 4-stage pipeline
  - VAD → Segmentation → Embedding → Clustering
  - Diagram of data flow
  - End-to-end vs modular approaches

* [x] `evaluation-metrics.md` - DER, EER, RTF explained
  - Formula and interpretation for each metric
  - Benchmark values by dataset
  - Collar and overlap handling

* [x] `speaker-embeddings.md` - What embeddings are and how they work
  - i-vectors, x-vectors, ECAPA-TDNN evolution
  - Cosine similarity and PLDA
  - Pre-trained model comparison

* [x] `glossary.md` - Terminology reference
  - 40+ terms with definitions
  - Cross-references to detailed articles
  - Organized alphabetically

* [x] `realtime-vs-offline.md` - Streaming considerations
* [x] `rttm-format.md` - Annotation format specification

#### 3.3 Algorithm Deep Dives - COMPLETE

Created in `knowledge/algorithms/`:

* [x] `ecapa-tdnn-explained.md` - Architecture and innovations
  - Res2Net blocks with diagrams
  - Squeeze-and-Excitation mechanism
  - Attentive statistics pooling
  - Mathematical formulations

* [x] `clustering-comparison.md` - AHC vs Spectral vs VBx
  - Algorithm descriptions with code
  - Complexity analysis
  - When to use each approach

#### 3.4 Tutorials (Priority: Medium) - COMPLETE

Created in `knowledge/tutorials/`:

* [x] `pyannote-quickstart.md` - Get started in 5 minutes
* [x] `custom-embedding-training.md` - Train on your data
* [x] `building-voice-fingerprint-system.md` - Enrollment + matching

#### 3.5 Comparisons (Priority: Medium) - COMPLETE

Created in `knowledge/comparisons/`:

* [x] `framework-selection.md` - Which tool for which use case
* [x] `embedding-architectures.md` - ECAPA vs x-vector vs d-vector

### Phase 4: Documentation Excellence (Priority: High)

#### 4.1 README.md Enhancement

Transform into comprehensive entry point:

* [x] Clear scope statement
* [x] Visual diagram of knowledge structure (Mermaid)
* [x] Quick start for different user types (Learning Path)
* [x] Navigation guide to all sections
* [x] Auditability and traceability principles

#### 4.2 CONTRIBUTING.md Enhancement

Add sections for:

* [x] Source citation requirements with examples
* [x] Archival procedures (printouts, bibliography)
* [x] Git commit conventions for this KB
* [x] Quality checklist for new entries
* [x] How to cite repository references (repo+commit+path+line)

#### 4.3 New Documentation Files

* [x] `TRACEABILITY.md` - Source tracking principles
* [x] `REGENERATION.md` - How to regenerate tables/reports
* [x] `RESEARCH_NOTES.md` - Guide to ramblings/ directory

### Phase 5: Automation & Generation (Priority: Medium)

#### 5.1 Enhanced Table Generation

* [ ] Tool comparison matrix (capabilities, performance)
* [ ] Algorithm comparison by category
* [ ] Dataset statistics summary
* [ ] Model performance leaderboard

#### 5.2 Scripts Enhancement

* [ ] `scripts/extract-citations.py` - Parse markdown for citations
* [ ] `scripts/generate-bib.py` - Create .bib from paper entries
* [ ] `scripts/archive-url.py` - Fetch and archive web content
* [ ] `scripts/update-stats.py` - Refresh GitHub stats for tools

### Phase 6: Repository Cloning & Analysis (Priority: Low)

#### 6.1 Clone Infrastructure

* [ ] Enhance `clone-all.sh` with commit pinning
* [ ] Add `scripts/analyze-repo.py` for automated analysis
* [ ] Create `tmp/README.md` explaining directory purpose

#### 6.2 Deep Analysis Support

For each cloned repo, support:

* Documentation extraction
* API surface analysis
* Dependency mapping
* License verification

## Source Citation Standards

### Minimum Citation (Required)

```yaml
sources:
  - url: "https://example.com/doc"
    accessed: "2026-01-06"
```

### Comprehensive Citation (Recommended)

```yaml
sources:
  - url: "https://github.com/org/repo"
    accessed: "2026-01-06"
    commit: "abc123def456"
    file-path: "src/model.py"
    line-range: "L45-L67"
    source-type: primary
    confidence: confirmed
    verified-date: "2026-01-06"
```

### For Papers

```yaml
sources:
  - arxiv: "2005.07143"
    title: "ECAPA-TDNN: Emphasized Channel Attention..."
    accessed: "2026-01-06"
    source-type: primary
```

## Regeneration Commands

```bash
# Validate all YAML files
make validate

# Generate comparison tables
make tables

# Clone all tracked repositories
make clone

# Verify source URLs are accessible
make sources

# Full rebuild
make all

# Clean generated content
make clean
```

## User Personas & Navigation

### Casual Explorer

**Entry**: README.md → Quick Navigation table
**Goal**: Understand what tools exist
**Path**: `comparisons/auto-generated.md` → specific tool YAML

### Practitioner

**Entry**: README.md → Knowledge articles
**Goal**: Implement speaker diarization
**Path**: `knowledge/fundamentals/` → tool entries → code examples

### Researcher

**Entry**: papers/ directory
**Goal**: Understand algorithms deeply
**Path**: paper entries → algorithm entries → original citations

### Contributor

**Entry**: CONTRIBUTING.md
**Goal**: Add new content
**Path**: Schema specs → examples → validation

### Hacker/Builder

**Entry**: README.md → Clone section
**Goal**: Modify or combine tools
**Path**: `make clone` → `tmp/` repos → deep analysis

## Quality Principles

1. **Every fact has a source** - No unsourced claims
2. **Timestamps everywhere** - All content dated
3. **Git precision** - Commit hashes for code references
4. **Machine-readable** - YAML for data, Markdown for prose
5. **Regenerable** - Generated content has clear regeneration path
6. **Archival-minded** - Web content preserved in printouts/

## Milestones

| Milestone | Description | Target |
|-----------|-------------|--------|
| M1 | Foundation complete (Phase 1) | Immediate |
| M2 | 10+ models, 10+ papers | Week 1 |
| M3 | Knowledge articles complete | Week 2 |
| M4 | Documentation excellence | Week 2 |
| M5 | Full automation | Week 3 |

## Notes

* This is a living document - update as work progresses
* Check off items as completed
* Add new items discovered during implementation
* Use ramblings/ for research notes during implementation
