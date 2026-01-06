#!/usr/bin/env python3
"""
Generate README.md files for each directory in the knowledge base.

This script reads YAML data files and generates markdown documentation
with auto-generated tables and indexes.

Usage:
    ./scripts/generate-readme.py           # Generate all READMEs
    ./scripts/generate-readme.py tools     # Generate only tools README
    ./scripts/generate-readme.py --list    # List available targets
"""

import sys
import os
from pathlib import Path
from datetime import date

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def load_yaml_files(pattern):
    """Load all YAML files matching pattern."""
    files = list(PROJECT_ROOT.glob(pattern))
    data = []
    for f in sorted(files):
        try:
            with open(f) as fp:
                content = yaml.safe_load(fp)
                if content:
                    content['_file'] = f.name
                    content['_path'] = str(f.relative_to(PROJECT_ROOT))
                    data.append(content)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}", file=sys.stderr)
    return data


def generate_tools_readme():
    """Generate data/tools/README.md"""
    tools = load_yaml_files("data/tools/*.tool.yaml")

    # Group by category
    by_category = {}
    for t in tools:
        cat = t.get('category', 'other')
        by_category.setdefault(cat, []).append(t)

    lines = [
        "# Tools & Frameworks",
        "",
        "Open-source tools and frameworks for speaker diarization, recognition, and embedding extraction.",
        "",
        f"**Total: {len(tools)} tools**",
        "",
        "## Quick Links",
        "",
        "| Tool | Category | Description |",
        "|------|----------|-------------|",
    ]

    for t in sorted(tools, key=lambda x: x.get('name', '')):
        name = t.get('name', 'Unknown')
        cat = t.get('category', '').replace('-', ' ')
        desc = t.get('description', '')[:60]
        if len(t.get('description', '')) > 60:
            desc += '...'
        repo = t.get('repo-url', '')
        lines.append(f"| [{name}]({repo}) | {cat} | {desc} |")

    lines.extend([
        "",
        "## By Category",
        "",
    ])

    for cat in sorted(by_category.keys()):
        cat_tools = by_category[cat]
        cat_display = cat.replace('-', ' ').title()
        lines.append(f"### {cat_display}")
        lines.append("")
        for t in cat_tools:
            name = t.get('name', 'Unknown')
            fname = t['_file']
            lines.append(f"* **[{name}]({fname})** - {t.get('description', '')[:80]}")
        lines.append("")

    lines.extend([
        "## Capabilities Matrix",
        "",
        "| Tool | Diarization | Embedding | VAD | Streaming | Training |",
        "|------|:-----------:|:---------:|:---:|:---------:|:--------:|",
    ])

    for t in sorted(tools, key=lambda x: x.get('name', '')):
        name = t.get('name', 'Unknown')
        caps = t.get('capabilities', {})
        diar = "✓" if caps.get('diarization') else ""
        emb = "✓" if caps.get('speaker-embedding') else ""
        vad = "✓" if caps.get('vad') else ""
        stream = "✓" if caps.get('streaming') else ""
        train = "✓" if caps.get('training') else ""
        lines.append(f"| {name} | {diar} | {emb} | {vad} | {stream} | {train} |")

    lines.extend([
        "",
        "## Performance Comparison",
        "",
        "| Tool | DER (AMI) | EER (VoxCeleb) | RTF |",
        "|------|----------:|---------------:|----:|",
    ])

    for t in sorted(tools, key=lambda x: x.get('name', '')):
        name = t.get('name', 'Unknown')
        perf = t.get('performance', {})
        der = perf.get('der-ami', '')
        if der:
            der = f"{der}%"
        eer = perf.get('eer-voxceleb', '')
        if eer:
            eer = f"{eer}%"
        rtf = perf.get('rtf', '')
        if der or eer or rtf:
            lines.append(f"| {name} | {der} | {eer} | {rtf} |")

    lines.extend([
        "",
        "---",
        "",
        f"*Auto-generated on {date.today()}. See [CONTRIBUTING.md](../CONTRIBUTING.md) for update instructions.*",
    ])

    return "\n".join(lines)


def generate_algorithms_readme():
    """Generate data/algorithms/README.md"""
    algorithms = load_yaml_files("data/algorithms/**/*.algorithm.yaml")

    # Group by category
    by_category = {}
    for a in algorithms:
        cat = a.get('category', 'other')
        by_category.setdefault(cat, []).append(a)

    lines = [
        "# Algorithms",
        "",
        "Core algorithms for speaker diarization and recognition, organized by category.",
        "",
        f"**Total: {len(algorithms)} algorithms**",
        "",
        "## Categories",
        "",
    ]

    for cat in sorted(by_category.keys()):
        cat_display = cat.replace('-', ' ').title()
        count = len(by_category[cat])
        lines.append(f"* [{cat_display}]({cat}/) ({count})")

    lines.extend([
        "",
        "## Algorithm Index",
        "",
    ])

    for cat in sorted(by_category.keys()):
        cat_algorithms = by_category[cat]
        cat_display = cat.replace('-', ' ').title()
        lines.append(f"### {cat_display}")
        lines.append("")
        lines.append("| Algorithm | Year | Key Innovation |")
        lines.append("|-----------|-----:|----------------|")

        for a in sorted(cat_algorithms, key=lambda x: x.get('year-introduced', 0), reverse=True):
            name = a.get('name', 'Unknown')
            year = a.get('year-introduced', '')
            features = a.get('features', [])
            innovation = features[0] if features else ''
            fpath = a['_path']
            lines.append(f"| [{name}]({fpath.split('/')[-1]}) | {year} | {innovation[:50]} |")

        lines.append("")

    lines.extend([
        "## See Also",
        "",
        "* [Tools](../tools/) - Implementations of these algorithms",
        "* [Models](../models/) - Pre-trained models using these algorithms",
        "* [Papers](../../papers/) - Original research publications",
        "",
        "---",
        "",
        f"*Auto-generated on {date.today()}*",
    ])

    return "\n".join(lines)


def generate_models_readme():
    """Generate data/models/README.md"""
    models = load_yaml_files("data/models/**/*.model.yaml")

    # Group by type
    by_type = {}
    for m in models:
        mtype = m.get('model-type', 'other')
        by_type.setdefault(mtype, []).append(m)

    lines = [
        "# Pre-trained Models",
        "",
        "Pre-trained speaker embedding and diarization models available on HuggingFace and NGC.",
        "",
        f"**Total: {len(models)} models**",
        "",
        "## Model Types",
        "",
    ]

    for mtype in sorted(by_type.keys()):
        type_display = mtype.replace('-', ' ').title()
        count = len(by_type[mtype])
        subdir = mtype.split('-')[0] if '-' in mtype else mtype
        lines.append(f"* [{type_display}]({subdir}/) ({count})")

    lines.extend([
        "",
        "## Embedding Models",
        "",
        "| Model | Architecture | Dim | EER | Provider |",
        "|-------|--------------|----:|----:|----------|",
    ])

    for m in models:
        if m.get('model-type') != 'embedding-model':
            continue
        name = m.get('name', 'Unknown')
        arch = m.get('architecture', '')
        dim = m.get('embedding-dimension', '')
        benchmarks = m.get('benchmarks', {})
        eer = benchmarks.get('voxceleb1-eer', '')
        if eer:
            eer = f"{eer}%"
        provider = m.get('provider', '')
        url = m.get('huggingface-url', '')
        lines.append(f"| [{name}]({url}) | {arch} | {dim} | {eer} | {provider} |")

    lines.extend([
        "",
        "## Diarization Pipelines",
        "",
        "| Model | DER (AMI) | Provider |",
        "|-------|----------:|----------|",
    ])

    for m in models:
        if m.get('model-type') != 'diarization-pipeline':
            continue
        name = m.get('name', 'Unknown')
        benchmarks = m.get('benchmarks', {})
        der = benchmarks.get('ami-der', '')
        if der:
            der = f"{der}%"
        provider = m.get('provider', '')
        url = m.get('huggingface-url', '')
        lines.append(f"| [{name}]({url}) | {der} | {provider} |")

    lines.extend([
        "",
        "## Usage Examples",
        "",
        "See individual model files for code examples.",
        "",
        "---",
        "",
        f"*Auto-generated on {date.today()}*",
    ])

    return "\n".join(lines)


def generate_datasets_readme():
    """Generate data/datasets/README.md"""
    datasets = load_yaml_files("data/datasets/*.dataset.yaml")

    lines = [
        "# Datasets",
        "",
        "Benchmark datasets for speaker verification, diarization, and recognition.",
        "",
        f"**Total: {len(datasets)} datasets**",
        "",
        "## Dataset Overview",
        "",
        "| Dataset | Type | Hours | Speakers | License |",
        "|---------|------|------:|---------:|---------|",
    ]

    for d in sorted(datasets, key=lambda x: x.get('name', '')):
        name = d.get('name', 'Unknown')
        dtype = d.get('category', '')
        stats = d.get('statistics', {})
        hours = stats.get('hours', stats.get('total-hours', ''))
        speakers = stats.get('speakers', stats.get('total-speakers', ''))
        access = d.get('access', {})
        license_info = access.get('license', '')
        fname = d['_file']
        lines.append(f"| [{name}]({fname}) | {dtype} | {hours} | {speakers} | {license_info} |")

    lines.extend([
        "",
        "## Annotations Available",
        "",
        "| Dataset | Speaker Labels | Transcripts | Overlap | VAD |",
        "|---------|:--------------:|:-----------:|:-------:|:---:|",
    ])

    for d in sorted(datasets, key=lambda x: x.get('name', '')):
        name = d.get('name', 'Unknown')
        ann = d.get('annotations', {})
        spk = "✓" if ann.get('speaker-labels') else ""
        trans = "✓" if ann.get('transcripts') else ""
        overlap = "✓" if ann.get('overlap-annotations') else ""
        vad = "✓" if ann.get('vad-labels') else ""
        lines.append(f"| {name} | {spk} | {trans} | {overlap} | {vad} |")

    lines.extend([
        "",
        "## Access Information",
        "",
    ])

    for d in sorted(datasets, key=lambda x: x.get('name', '')):
        name = d.get('name', 'Unknown')
        access = d.get('access', {})
        url = access.get('url', '')
        reg = "Yes" if access.get('requires-registration') else "No"
        lines.append(f"* **{name}**: [Website]({url}) - Registration: {reg}")

    lines.extend([
        "",
        "---",
        "",
        f"*Auto-generated on {date.today()}*",
    ])

    return "\n".join(lines)


def generate_papers_readme():
    """Generate papers/README.md"""
    papers = load_yaml_files("papers/*.paper.yaml")

    # Group by topic
    by_topic = {}
    for p in papers:
        topics = p.get('topics', ['other'])
        for topic in topics[:1]:  # Use first topic as primary
            by_topic.setdefault(topic, []).append(p)

    lines = [
        "# Research Papers",
        "",
        "Academic publications on speaker diarization, recognition, and related topics.",
        "",
        f"**Total: {len(papers)} papers**",
        "",
        "## Paper Index",
        "",
        "| Paper | Year | Venue | Citations |",
        "|-------|-----:|-------|----------:|",
    ]

    for p in sorted(papers, key=lambda x: x.get('year', 0), reverse=True):
        title = p.get('short-title', p.get('title', 'Unknown'))[:40]
        year = p.get('year', '')
        venue = p.get('venue', '')[:20]
        citations = p.get('citation-count', '')
        fname = p['_file']
        lines.append(f"| [{title}]({fname}) | {year} | {venue} | {citations} |")

    lines.extend([
        "",
        "## By Topic",
        "",
    ])

    for topic in sorted(by_topic.keys()):
        topic_papers = by_topic[topic]
        topic_display = topic.replace('-', ' ').title()
        lines.append(f"### {topic_display}")
        lines.append("")
        for p in topic_papers:
            title = p.get('short-title', p.get('title', 'Unknown'))
            year = p.get('year', '')
            arxiv = p.get('arxiv', '')
            link = f"https://arxiv.org/abs/{arxiv}" if arxiv else ''
            fname = p['_file']
            lines.append(f"* [{title}]({fname}) ({year})" + (f" - [arXiv]({link})" if link else ""))
        lines.append("")

    lines.extend([
        "## Adding Papers",
        "",
        "See [CONTRIBUTING.md](../CONTRIBUTING.md) for the paper entry format.",
        "",
        "---",
        "",
        f"*Auto-generated on {date.today()}*",
    ])

    return "\n".join(lines)


def generate_schemas_readme():
    """Generate schemas/README.md"""
    schemas = list(PROJECT_ROOT.glob("schemas/*.spec.yaml"))

    lines = [
        "# Schema Specifications",
        "",
        "YAML schema definitions for each entry type in the knowledge base.",
        "",
        "## Available Schemas",
        "",
        "| Schema | Purpose | File Pattern |",
        "|--------|---------|--------------|",
        "| [tool.spec.yaml](tool.spec.yaml) | Open-source tools | `data/tools/{owner}--{repo}.tool.yaml` |",
        "| [algorithm.spec.yaml](algorithm.spec.yaml) | Algorithms & techniques | `data/algorithms/{cat}/{name}.algorithm.yaml` |",
        "| [model.spec.yaml](model.spec.yaml) | Pre-trained models | `data/models/{cat}/{provider}--{id}.model.yaml` |",
        "| [dataset.spec.yaml](dataset.spec.yaml) | Datasets & corpora | `data/datasets/{name}.dataset.yaml` |",
        "| [paper.spec.yaml](paper.spec.yaml) | Research papers | `papers/{key}-{year}.paper.yaml` |",
        "| [source.spec.yaml](source.spec.yaml) | Source citations | Embedded in all types |",
        "",
        "## Schema Structure",
        "",
        "Each spec file defines:",
        "",
        "* **required** - Fields that must be present",
        "* **fields** - All available fields with types and descriptions",
        "* **enums** - Valid values for categorical fields",
        "",
        "## Using Schemas",
        "",
        "### Validation",
        "",
        "```bash",
        "./scripts/check-yaml.py data/tools/my-tool.tool.yaml",
        "```",
        "",
        "### Reference",
        "",
        "```bash",
        "# View tool schema",
        "cat schemas/tool.spec.yaml",
        "",
        "# List required fields",
        "yq '.required' schemas/tool.spec.yaml",
        "```",
        "",
        "## Common Patterns",
        "",
        "### Source Citations",
        "",
        "All entry types support the `sources` array:",
        "",
        "```yaml",
        "sources:",
        "  - url: \"https://...\"",
        "    accessed: \"2026-01-06\"",
        "    source-type: primary",
        "```",
        "",
        "### Timestamps",
        "",
        "All entries require `last-update: \"YYYY-MM-DD\"`",
        "",
        "---",
        "",
        f"*Last updated: {date.today()}*",
    ]

    return "\n".join(lines)


def generate_knowledge_readme():
    """Generate knowledge/README.md with dynamic article indexing"""
    # Scan knowledge subdirectories for markdown files
    knowledge_dir = PROJECT_ROOT / "knowledge"

    subdirs = {
        'fundamentals': {
            'title': 'Fundamentals',
            'description': 'Core concepts, pipeline architecture, and terminology'
        },
        'algorithms': {
            'title': 'Algorithm Deep-Dives',
            'description': 'Detailed explanations of key algorithms'
        },
        'tutorials': {
            'title': 'Tutorials',
            'description': 'Step-by-step guides and quickstarts'
        },
        'comparisons': {
            'title': 'Comparisons',
            'description': 'Framework and technology comparisons'
        },
        'concepts': {
            'title': 'Concepts',
            'description': 'Advanced conceptual topics'
        },
        'math': {
            'title': 'Mathematical Foundations',
            'description': 'Mathematical background and formulations'
        }
    }

    def extract_title(filepath):
        """Extract title from markdown file (first H1 header)"""
        try:
            with open(filepath) as f:
                for line in f:
                    if line.startswith('# '):
                        return line[2:].strip()
        except:
            pass
        # Fallback to filename
        return filepath.stem.replace('-', ' ').title()

    def count_articles(subdir):
        """Count markdown articles in subdirectory"""
        articles = list((knowledge_dir / subdir).glob("*.md"))
        return [a for a in articles if a.name != "README.md"]

    # Count total articles
    total = 0
    articles_by_section = {}
    for subdir in subdirs:
        subdir_path = knowledge_dir / subdir
        if subdir_path.exists():
            articles = count_articles(subdir)
            articles_by_section[subdir] = articles
            total += len(articles)

    lines = [
        "# Knowledge Articles",
        "",
        "Educational content explaining speaker diarization concepts, algorithms, and best practices.",
        "",
        f"**Total: {total} articles**",
        "",
        "## Structure",
        "",
        "```",
        "knowledge/",
    ]

    for subdir, info in subdirs.items():
        count = len(articles_by_section.get(subdir, []))
        lines.append(f"├── {subdir}/       # {info['description']} ({count})")

    lines.extend([
        "```",
        "",
        "## Quick Start",
        "",
        "New to speaker diarization? Start with:",
        "",
        "1. [Pipeline Architecture](fundamentals/pipeline-architecture.md) - Overview of the 4-stage pipeline",
        "2. [Speaker Embeddings](fundamentals/speaker-embeddings.md) - Understanding voice representations",
        "3. [Evaluation Metrics](fundamentals/evaluation-metrics.md) - DER, EER, and how to measure quality",
        "4. [Glossary](fundamentals/glossary.md) - Terminology reference",
        "",
    ])

    # Generate article index for each section
    for subdir, info in subdirs.items():
        articles = articles_by_section.get(subdir, [])
        if articles:
            lines.extend([
                f"## {info['title']}",
                "",
                f"*{info['description']}*",
                "",
            ])

            for article in sorted(articles, key=lambda x: x.name):
                title = extract_title(article)
                rel_path = f"{subdir}/{article.name}"
                lines.append(f"* [{title}]({rel_path})")

            lines.append("")

    lines.extend([
        "## Learning Path",
        "",
        "| Level | Topics | Articles |",
        "|-------|--------|----------|",
        "| Beginner | Pipeline, Embeddings, Metrics | fundamentals/ |",
        "| Intermediate | ECAPA-TDNN, Clustering, Streaming | algorithms/ |",
        "| Practical | Pyannote, Framework Selection | tutorials/, comparisons/ |",
        "",
        "## Contributing",
        "",
        "Add articles as markdown files in the appropriate subdirectory.",
        "See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.",
        "",
        "---",
        "",
        f"*Auto-generated on {date.today()}*",
    ])

    return "\n".join(lines)


def generate_printouts_readme():
    """Generate printouts/README.md"""
    archives = list(PROJECT_ROOT.glob("printouts/*/SOURCE.yaml"))

    lines = [
        "# Archived Web Content",
        "",
        "Web pages and documents archived for reference and traceability.",
        "",
        "## Purpose",
        "",
        "Web content can change or disappear. This directory preserves important",
        "reference material with full provenance tracking.",
        "",
        "## Archive Structure",
        "",
        "Each archive is in a dated directory:",
        "",
        "```",
        "printouts/",
        "└── YYYY-MM-DD--descriptive-name/",
        "    ├── SOURCE.yaml    # Provenance metadata",
        "    ├── content.md     # Main content",
        "    └── ...            # Additional files",
        "```",
        "",
        "## Current Archives",
        "",
    ]

    if archives:
        lines.append("| Archive | Date | Source |")
        lines.append("|---------|------|--------|")
        for source_file in sorted(archives):
            dir_name = source_file.parent.name
            try:
                with open(source_file) as f:
                    data = yaml.safe_load(f)
                    source = data.get('source', {})
                    url = source.get('url', '')[:50]
                    date_str = source.get('archived-date', '')
                    lines.append(f"| [{dir_name}]({dir_name}/) | {date_str} | {url} |")
            except:
                lines.append(f"| [{dir_name}]({dir_name}/) | ? | ? |")
    else:
        lines.append("*No archives yet.*")

    lines.extend([
        "",
        "## Adding Archives",
        "",
        "See [CONTRIBUTING.md](../CONTRIBUTING.md) for archival procedures.",
        "",
        "---",
        "",
        f"*Auto-generated on {date.today()}*",
    ])

    return "\n".join(lines)


def generate_ramblings_readme():
    """Generate ramblings/README.md"""
    notes = list(PROJECT_ROOT.glob("ramblings/*.md"))
    notes = [n for n in notes if n.name != "README.md"]

    lines = [
        "# Research Notes & Ramblings",
        "",
        "Informal research notes, explorations, and ideas that may later be formalized.",
        "",
        "## Naming Convention",
        "",
        "```",
        "YYYY-MM-DD--descriptive-topic.md",
        "```",
        "",
        "## Notes Index",
        "",
    ]

    if notes:
        for note in sorted(notes, reverse=True):
            name = note.stem
            parts = name.split('--', 1)
            date_str = parts[0] if len(parts) > 1 else ''
            topic = parts[1].replace('-', ' ').title() if len(parts) > 1 else name
            lines.append(f"* [{topic}]({note.name}) ({date_str})")
    else:
        lines.append("*No notes yet.*")

    lines.extend([
        "",
        "## Purpose",
        "",
        "This is a scratchpad for:",
        "",
        "* Initial research findings",
        "* Tool comparisons and experiments",
        "* Ideas for new entries",
        "* Questions to investigate",
        "",
        "Content here is informal and may be incomplete.",
        "",
        "---",
        "",
        f"*Auto-generated on {date.today()}*",
    ])

    return "\n".join(lines)


def generate_sources_readme():
    """Generate sources/README.md"""
    lines = [
        "# Source Tracking",
        "",
        "Standalone source entries for frequently referenced materials.",
        "",
        "## Purpose",
        "",
        "While most source citations are embedded in entry files,",
        "this directory holds standalone source entries for:",
        "",
        "* Frequently referenced materials",
        "* Complex multi-file sources",
        "* Sources cited by multiple entries",
        "",
        "## File Format",
        "",
        "```yaml",
        "# {id}.source.yaml",
        "",
        "id: unique-identifier",
        "url: https://...",
        "accessed: \"2026-01-06\"",
        "source-type: primary",
        "",
        "# Additional metadata",
        "title: \"Document Title\"",
        "author: \"Author Name\"",
        "```",
        "",
        "## Current Sources",
        "",
        "*No standalone sources yet.*",
        "",
        "---",
        "",
        f"*Last updated: {date.today()}*",
    ]

    return "\n".join(lines)


GENERATORS = {
    'tools': ('data/tools/README.md', generate_tools_readme),
    'algorithms': ('data/algorithms/README.md', generate_algorithms_readme),
    'models': ('data/models/README.md', generate_models_readme),
    'datasets': ('data/datasets/README.md', generate_datasets_readme),
    'papers': ('papers/README.md', generate_papers_readme),
    'schemas': ('schemas/README.md', generate_schemas_readme),
    'knowledge': ('knowledge/README.md', generate_knowledge_readme),
    'printouts': ('printouts/README.md', generate_printouts_readme),
    'ramblings': ('ramblings/README.md', generate_ramblings_readme),
    'sources': ('sources/README.md', generate_sources_readme),
}


def main():
    args = sys.argv[1:]

    if '--list' in args:
        print("Available targets:")
        for name in sorted(GENERATORS.keys()):
            print(f"  {name}")
        return 0

    targets = args if args else list(GENERATORS.keys())

    for target in targets:
        if target not in GENERATORS:
            print(f"Unknown target: {target}", file=sys.stderr)
            continue

        path, generator = GENERATORS[target]
        output_path = PROJECT_ROOT / path

        print(f"Generating {path}...")
        content = generator()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)

    print(f"Generated {len(targets)} README file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
