# Traceability & Source Attribution

This document explains the source tracking principles used throughout this knowledge base.

## Core Philosophy

**Every fact has a source.** No claims should exist in this repository without traceable attribution.

## Why Traceability Matters

1. **Verification**: Readers can verify claims against original sources
2. **Updates**: When sources change, we know which facts to review
3. **Trust**: Clear provenance builds confidence in the content
4. **Reproducibility**: Others can replicate our research process

## Source Types

| Type | Use For | Example |
|------|---------|---------|
| `primary` | Original source (paper, official docs, source code) | ArXiv paper, GitHub repo |
| `secondary` | Analysis or summary of primary sources | Blog posts, tutorials |
| `tertiary` | Aggregated from multiple sources | Wikipedia, Perplexity AI |
| `benchmark` | Performance evaluation results | Leaderboards, papers |
| `code` | Implementation details from source code | GitHub file + line numbers |

## Citation Standards

### Minimum Citation (Required)

Every YAML entry must include at least:

```yaml
sources:
  - url: "https://example.com/doc"
    accessed: "2026-01-06"
```

### Standard Citation (Recommended)

```yaml
sources:
  - url: "https://github.com/org/repo"
    accessed: "2026-01-06"
    source-type: primary
    title: "Repository README"
```

### Comprehensive Citation (For Code References)

```yaml
sources:
  - url: "https://github.com/org/repo"
    accessed: "2026-01-06"
    commit: "abc123def456789"
    file-path: "src/models/ecapa_tdnn.py"
    line-range: "L45-L120"
    source-type: code
    confidence: confirmed
    verified-date: "2026-01-06"
    notes: "Main model implementation"
```

### Academic Paper Citation

```yaml
sources:
  - arxiv: "2005.07143"
    title: "ECAPA-TDNN: Emphasized Channel Attention..."
    accessed: "2026-01-06"
    source-type: primary
    doi: "10.21437/Interspeech.2020-2650"
```

## Confidence Levels

| Level | Meaning | When to Use |
|-------|---------|-------------|
| `confirmed` | Verified directly from primary source | After reading original paper/code |
| `likely` | From reputable secondary source | Trusted blog, documentation |
| `uncertain` | Needs verification | Initial research, may change |

## Archival Procedures

Web content can disappear or change. For important sources:

### 1. Create Archive Directory

```bash
mkdir -p printouts/YYYY-MM-DD--descriptive-name/
```

### 2. Save Content

Save the page content (HTML, PDF, or markdown conversion).

### 3. Create SOURCE.yaml

```yaml
# printouts/2026-01-06--example-page/SOURCE.yaml
source:
  url: "https://example.com/page"
  accessed: "2026-01-06"
  archived-date: "2026-01-06"
  source-type: "secondary"

description: |
  Brief description of what this archive contains.

files:
  - name: "content.md"
    description: "Main page content"
```

### 4. Reference in Entries

```yaml
sources:
  - url: "https://example.com/page"
    accessed: "2026-01-06"
    local-path: "printouts/2026-01-06--example-page/content.md"
```

## Code References

When citing specific code:

```yaml
sources:
  - url: "https://github.com/pyannote/pyannote-audio"
    commit: "3a7b8c9d"
    file-path: "pyannote/audio/pipelines/speaker_diarization.py"
    line-range: "L156-L200"
    accessed: "2026-01-06"
    source-type: code
```

This allows:

* **Precise verification**: Exact code location
* **Version tracking**: Specific commit, not "latest"
* **Change detection**: Know when referenced code changes

## Bibliography Management

Academic citations go in `archives/bibliography/`:

```bibtex
@inproceedings{desplanques2020ecapa,
  title={ECAPA-TDNN: Emphasized Channel Attention...},
  author={Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris},
  booktitle={Interspeech 2020},
  year={2020}
}
```

Reference in YAML:

```yaml
sources:
  - arxiv: "2005.07143"
    bibtex-key: "desplanques2020ecapa"
    accessed: "2026-01-06"
```

## Timestamps

All entries require:

```yaml
last-update: "YYYY-MM-DD"  # When this YAML was created/updated
```

All sources require:

```yaml
accessed: "YYYY-MM-DD"  # When the source was accessed
```

## Verification Workflow

### Adding New Facts

1. Find primary source (paper, official docs, code)
2. Read and understand the source
3. Add fact with full citation
4. Set `confidence: confirmed` if verified directly

### Updating Existing Facts

1. Check if source URL still works
2. Verify fact is still accurate
3. Update `last-update` date
4. Update `accessed` date on sources
5. Note changes if significant

### Periodic Review

Run `make sources` to verify all URLs are accessible.

## Quality Checklist

Before committing:

- [ ] All facts have at least one source
- [ ] `last-update` field is current
- [ ] Source URLs are accessible
- [ ] `accessed` dates are accurate
- [ ] Commit hashes used for code references
- [ ] Confidence levels are appropriate

## See Also

* [CONTRIBUTING.md](CONTRIBUTING.md) - How to add new entries
* [REGENERATION.md](REGENERATION.md) - How to regenerate content
* [schemas/source.spec.yaml](schemas/source.spec.yaml) - Source citation schema

---

*Last updated: 2026-01-06*
