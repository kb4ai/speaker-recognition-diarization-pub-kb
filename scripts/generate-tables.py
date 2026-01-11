#!/usr/bin/env python3
"""
Generate comparison tables from YAML data files.

Usage:
    ./scripts/generate-tables.py                    # Generate all tables
    ./scripts/generate-tables.py --tools            # Tools comparison only
    ./scripts/generate-tables.py --algorithms       # Algorithms comparison only
    ./scripts/generate-tables.py --datasets         # Datasets comparison only
    ./scripts/generate-tables.py --json             # Output as JSON
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
TOOLS_DIR = DATA_DIR / "tools"
ALGORITHMS_DIR = DATA_DIR / "algorithms"
MODELS_DIR = DATA_DIR / "models"
DATASETS_DIR = DATA_DIR / "datasets"


def load_yaml_files(directory, pattern="*.yaml"):
    """Load all YAML files from a directory."""
    items = []
    if not directory.exists():
        return items
    for yaml_file in sorted(directory.rglob(pattern)):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
                if data:
                    data['_filename'] = yaml_file.stem
                    data['_filepath'] = str(yaml_file.relative_to(PROJECT_ROOT))
                    items.append(data)
        except Exception as e:
            print(f"Warning: Could not load {yaml_file}: {e}", file=sys.stderr)
    return items


def format_capabilities(caps):
    """Format capabilities as checkmarks."""
    if not caps:
        return ""
    enabled = [k for k, v in caps.items() if v]
    return ", ".join(enabled)


def generate_tools_overview(tools):
    """Generate tools overview table sorted by stars."""
    sorted_tools = sorted(
        tools,
        key=lambda p: (p.get('stars') is not None, p.get('stars') or 0),
        reverse=True
    )

    lines = []
    lines.append("## Tools Overview")
    lines.append("")
    lines.append("*Sorted by GitHub stars*")
    lines.append("")
    lines.append("| Tool | Stars | Language | Category | Capabilities |")
    lines.append("|------|------:|----------|----------|--------------|")

    for t in sorted_tools:
        name = t.get('name', t.get('_filename', 'Unknown'))
        repo_url = t.get('repo-url', '')
        stars = t.get('stars', '?')
        lang = t.get('language', '')
        category = t.get('category', '').replace('-', ' ')
        caps = format_capabilities(t.get('capabilities', {}))

        name_cell = f"[{name}]({repo_url})" if repo_url else name
        lines.append(f"| {name_cell} | {stars} | {lang} | {category} | {caps} |")

    return "\n".join(lines)


def generate_tools_by_accuracy(tools):
    """Generate tools table sorted by DER."""
    tools_with_der = [t for t in tools if t.get('performance', {}).get('der-ami')]
    sorted_tools = sorted(
        tools_with_der,
        key=lambda p: p.get('performance', {}).get('der-ami', 999)
    )

    lines = []
    lines.append("## Tools by Accuracy")
    lines.append("")
    lines.append("*Sorted by DER on AMI corpus (lower is better)*")
    lines.append("")
    lines.append("| Tool | DER (AMI) | EER (VoxCeleb) | RTF | Streaming |")
    lines.append("|------|----------:|---------------:|----:|:---------:|")

    for t in sorted_tools:
        name = t.get('name', t.get('_filename', 'Unknown'))
        repo_url = t.get('repo-url', '')
        perf = t.get('performance', {})
        der = perf.get('der-ami', '?')
        eer = perf.get('eer-voxceleb', '?')
        rtf = perf.get('rtf', '?')
        streaming = "✓" if t.get('capabilities', {}).get('streaming') else ""

        name_cell = f"[{name}]({repo_url})" if repo_url else name
        lines.append(f"| {name_cell} | {der}% | {eer}% | {rtf} | {streaming} |")

    if not sorted_tools:
        lines.append("| *No tools with DER benchmarks yet* | | | | |")

    return "\n".join(lines)


def generate_algorithms_matrix(algorithms):
    """Generate algorithms comparison by category."""
    by_category = defaultdict(list)
    for a in algorithms:
        cat = a.get('category', 'uncategorized')
        by_category[cat].append(a)

    lines = []
    lines.append("## Algorithms by Category")
    lines.append("")

    category_order = ['embedding-extraction', 'clustering', 'vad', 'segmentation', 'end-to-end', 'scoring']

    for category in category_order + [c for c in sorted(by_category.keys()) if c not in category_order]:
        if category not in by_category:
            continue

        cat_algos = sorted(
            by_category[category],
            key=lambda a: a.get('year-introduced', 0),
            reverse=True
        )

        lines.append(f"### {category.replace('-', ' ').title()}")
        lines.append("")
        lines.append("| Algorithm | Year | Output Dim | Key Features |")
        lines.append("|-----------|-----:|-----------:|--------------|")

        for a in cat_algos:
            name = a.get('name', a.get('_filename', 'Unknown'))
            year = a.get('year-introduced', '?')
            dim = a.get('output', {}).get('dimension', '?')
            features = ", ".join(a.get('mathematical-basis', [])[:3])

            lines.append(f"| {name} | {year} | {dim} | {features} |")

        lines.append("")

    return "\n".join(lines)


def generate_datasets_comparison(datasets):
    """Generate datasets comparison table."""
    sorted_datasets = sorted(
        datasets,
        key=lambda d: d.get('statistics', {}).get('total-hours', 0),
        reverse=True
    )

    lines = []
    lines.append("## Datasets Comparison")
    lines.append("")
    lines.append("| Dataset | Type | Hours | Speakers | Languages | License |")
    lines.append("|---------|------|------:|---------:|-----------|---------|")

    for d in sorted_datasets:
        name = d.get('name', d.get('_filename', 'Unknown'))
        url = d.get('url', '')
        dtype = d.get('type', '')
        stats = d.get('statistics', {})
        hours = stats.get('total-hours', '?')
        speakers = stats.get('num-speakers', '?')
        langs = ", ".join(d.get('language', [])[:2])
        license_info = d.get('access', {}).get('license', '?')

        name_cell = f"[{name}]({url})" if url else name
        lines.append(f"| {name_cell} | {dtype} | {hours} | {speakers} | {langs} | {license_info} |")

    return "\n".join(lines)


def generate_embedding_models(models):
    """Generate embedding models comparison."""
    embedding_models = [m for m in models if m.get('model-type') == 'embedding-model']
    sorted_models = sorted(
        embedding_models,
        key=lambda m: m.get('benchmarks', {}).get('voxceleb1-eer', 999)
    )

    lines = []
    lines.append("## Embedding Models")
    lines.append("")
    lines.append("*Sorted by EER on VoxCeleb1 (lower is better)*")
    lines.append("")
    lines.append("| Model | Architecture | Dimension | EER (VoxCeleb1) | Provider |")
    lines.append("|-------|--------------|----------:|----------------:|----------|")

    for m in sorted_models:
        name = m.get('name', m.get('model-id', m.get('_filename', 'Unknown')))
        url = m.get('huggingface-url', '')
        arch = m.get('architecture', '?')
        dim = m.get('embedding-dimension', '?')
        eer = m.get('benchmarks', {}).get('voxceleb1-eer', '?')
        provider = m.get('provider', '?')

        name_cell = f"[{name}]({url})" if url else name
        lines.append(f"| {name_cell} | {arch} | {dim} | {eer}% | {provider} |")

    if not sorted_models:
        lines.append("| *No embedding models yet* | | | | |")

    return "\n".join(lines)


def generate_tools_hardware(tools):
    """Generate tools hardware requirements table."""
    lines = []
    lines.append("## Tools: Backend & Hardware Requirements")
    lines.append("")
    lines.append("*Framework and hardware requirements for deployment planning*")
    lines.append("")
    lines.append("| Tool | Framework | GPU Required | Min VRAM | Realtime | Inference Engines |")
    lines.append("|------|-----------|:------------:|----------|:--------:|-------------------|")

    for t in sorted(tools, key=lambda x: x.get('name', '')):
        name = t.get('name', t.get('_filename', 'Unknown'))
        repo_url = t.get('repo-url', '')

        # Backend info
        backend = t.get('backend', {})
        framework = backend.get('framework', '?')
        if framework == 'pytorch':
            framework = 'PyTorch'
        elif framework == 'tensorflow':
            framework = 'TensorFlow'
        elif framework == 'kaldi':
            framework = 'Kaldi'

        # Add additional frameworks
        additional = backend.get('additional-frameworks', [])
        if additional:
            extra = [f.title() if f not in ['tensorrt', 'onnx', 'ctranslate2'] else f.upper() if f == 'onnx' else f.replace('tensorrt', 'TensorRT').replace('ctranslate2', 'CTranslate2') for f in additional[:1]]
            if extra:
                framework = f"{framework}+{extra[0]}"

        # Hardware info
        hardware = t.get('hardware', {})
        gpu_required = "**Yes**" if hardware.get('gpu-required', False) else "No"
        min_vram = hardware.get('min-vram-gb', '?')
        if min_vram != '?':
            min_vram = f"{min_vram} GB"

        # Realtime support
        realtime = hardware.get('realtime-processing', {})
        realtime_supported = "Yes" if realtime.get('supported', False) else "-"

        # Inference engines
        engines = backend.get('inference-engine', [])
        engines_str = ", ".join([e.upper() if e == 'onnx' else e.title() for e in engines[:3]]) if engines else "?"

        name_cell = f"[{name}]({repo_url})" if repo_url else name
        lines.append(f"| {name_cell} | {framework} | {gpu_required} | {min_vram} | {realtime_supported} | {engines_str} |")

    return "\n".join(lines)


def generate_statistics(tools, algorithms, models, datasets):
    """Generate summary statistics."""
    lines = []
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append(f"* **Tools**: {len(tools)}")
    lines.append(f"* **Algorithms**: {len(algorithms)}")
    lines.append(f"* **Models**: {len(models)}")
    lines.append(f"* **Datasets**: {len(datasets)}")
    lines.append(f"* **Last updated**: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")

    # Tools by category
    if tools:
        by_cat = defaultdict(int)
        for t in tools:
            by_cat[t.get('category', 'uncategorized')] += 1
        lines.append("### Tools by Category")
        lines.append("")
        for cat, count in sorted(by_cat.items()):
            lines.append(f"* {cat}: {count}")
        lines.append("")

    return "\n".join(lines)


def main():
    args = sys.argv[1:]

    # Load all data
    tools = load_yaml_files(TOOLS_DIR, "*.tool.yaml")
    algorithms = load_yaml_files(ALGORITHMS_DIR, "*.algorithm.yaml")
    models = load_yaml_files(MODELS_DIR, "*.model.yaml")
    datasets = load_yaml_files(DATASETS_DIR, "*.dataset.yaml")

    if '--json' in args:
        output = {
            'tools': tools,
            'algorithms': algorithms,
            'models': models,
            'datasets': datasets,
        }
        print(json.dumps(output, indent=2, default=str))
        return

    # Generate markdown output
    sections = []

    sections.append("# Speaker Recognition & Diarization: Comparison Tables")
    sections.append("")
    sections.append("*Auto-generated from YAML data files*")
    sections.append("")
    sections.append("---")
    sections.append("")

    if '--tools' in args or not any(a.startswith('--') for a in args):
        sections.append(generate_statistics(tools, algorithms, models, datasets))
        sections.append(generate_tools_overview(tools))
        sections.append("")
        sections.append(generate_tools_by_accuracy(tools))
        sections.append("")
        sections.append(generate_tools_hardware(tools))

    if '--algorithms' in args or not any(a.startswith('--') for a in args):
        sections.append("")
        sections.append(generate_algorithms_matrix(algorithms))

    if '--models' in args or not any(a.startswith('--') for a in args):
        sections.append("")
        sections.append(generate_embedding_models(models))

    if '--datasets' in args or not any(a.startswith('--') for a in args):
        sections.append("")
        sections.append(generate_datasets_comparison(datasets))

    print("\n".join(sections))


if __name__ == "__main__":
    main()
