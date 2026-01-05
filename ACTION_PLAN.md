# Speaker Recognition & Diarization Knowledge Base - Action Plan

This document outlines the implementation roadmap to transform this repository into a comprehensive, well-structured knowledge base with proper source attribution, archived knowledge, and machine-readable artifacts.

## Current State Assessment

### What Exists

* **5 Tools**: pyannote, speechbrain, nvidia-nemo, kaldi, diart
* **10 Algorithms**: 3 embedding, 3 clustering, 2 VAD, 2 end-to-end
* **5 Datasets**: VoxCeleb1/2, AMI, CALLHOME, DIHARD
* **6 Schema Specs**: tool, algorithm, model, dataset, paper, source
* **Scripts**: validation, table generation, source verification, clone-all
* **Makefile**: validate, tables, clone, sources, clean targets

### What's Missing

* **Models**: No pre-trained model entries (pyannote, speechbrain, nvidia)
* **Papers**: No paper entries (need ECAPA-TDNN, x-vector, EEND papers)
* **Archived Content**: printouts/ mostly empty
* **Bibliography**: No .bib collection
* **Knowledge Articles**: knowledge/ directory skeleton only

## Implementation Phases

### Phase 1: Foundation & Traceability (Priority: High)

#### 1.1 Archive Source Research

* [x] Move Perplexity research to `printouts/2026-01-06--perplexity-speaker-diarization-research/`
* [x] Create SOURCE.yaml with provenance metadata
* [ ] Extract key citations to `archives/bibliography/speaker-diarization.bib`

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

* [ ] `embedding/pyannote--wespeaker-resnet34.model.yaml`
* [ ] `embedding/speechbrain--ecapa-voxceleb.model.yaml`
* [ ] `embedding/nvidia--titanet-large.model.yaml`
* [ ] `diarization/pyannote--speaker-diarization-3.1.model.yaml`
* [ ] `diarization/nvidia--sortformer.model.yaml`

#### 2.2 Research Paper Entries

Create entries in `papers/`:

* [ ] `ecapa-tdnn-2020.paper.yaml` - ECAPA-TDNN architecture
* [ ] `x-vectors-2018.paper.yaml` - X-vector embeddings
* [ ] `eend-2019.paper.yaml` - End-to-end neural diarization
* [ ] `pyannote-2019.paper.yaml` - Pyannote.audio paper
* [ ] `voxceleb-2017.paper.yaml` - VoxCeleb1 paper
* [ ] `voxceleb2-2018.paper.yaml` - VoxCeleb2 paper

#### 2.3 Additional Tool Entries

From research, add:

* [ ] `resemble-ai--resemblyzer.tool.yaml` - Voice fingerprinting
* [ ] `m-bain--whisperx.tool.yaml` - ASR + diarization
* [ ] `cvqluu--simple-diarizer.tool.yaml` - Simple wrapper
* [ ] `wenet-e2e--wespeaker.tool.yaml` - Speaker embedding toolkit
* [ ] `espnet--espnet-spk.tool.yaml` - ESPnet speaker module

### Phase 3: Knowledge Articles (Priority: Medium)

#### 3.1 Fundamentals

Create in `knowledge/fundamentals/`:

* [ ] `speaker-diarization-pipeline.md` - Overview of 4-stage pipeline
* [ ] `evaluation-metrics.md` - DER, EER, RTF explained
* [ ] `speaker-embeddings.md` - What embeddings are and how they work
* [ ] `glossary.md` - Terminology reference

#### 3.2 Algorithms Deep Dives

Create in `knowledge/algorithms/`:

* [ ] `ecapa-tdnn-explained.md` - Architecture and innovations
* [ ] `clustering-comparison.md` - AHC vs Spectral vs VBx
* [ ] `real-time-vs-offline.md` - Trade-offs and approaches

### Phase 4: Documentation Excellence (Priority: High)

#### 4.1 README.md Enhancement

Transform into comprehensive entry point:

* [ ] Clear scope statement
* [ ] Visual diagram of knowledge structure
* [ ] Quick start for different user types
* [ ] Navigation guide to all sections
* [ ] Auditability and traceability principles

#### 4.2 CONTRIBUTING.md Enhancement

Add sections for:

* [ ] Source citation requirements with examples
* [ ] Archival procedures (printouts, bibliography)
* [ ] Git commit conventions for this KB
* [ ] Quality checklist for new entries
* [ ] How to cite repository references (repo+commit+path+line)

#### 4.3 New Documentation Files

* [ ] `TRACEABILITY.md` - Source tracking principles
* [ ] `REGENERATION.md` - How to regenerate tables/reports
* [ ] `RESEARCH_NOTES.md` - Guide to ramblings/ directory

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
