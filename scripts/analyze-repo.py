#!/usr/bin/env python3
"""
Analyze a cloned repository and extract useful information.

Extracts:
- README content summary
- License information (with SPDX verification)
- Dependencies (requirements.txt, pyproject.toml, setup.py)
- Main modules and classes
- Git statistics

Deep analysis mode (--deep) additionally extracts:
- API surface (classes, functions, entry points)
- README sections (title, description, installation, usage, features)
- Categorized dependencies (ML frameworks, audio libraries)

Usage:
    ./scripts/analyze-repo.py tmp/pyannote-audio    # Analyze specific repo
    ./scripts/analyze-repo.py --all                 # Analyze all in tmp/
    ./scripts/analyze-repo.py tmp/repo --json       # Output as JSON
    ./scripts/analyze-repo.py tmp/repo --deep       # Deep analysis
    ./scripts/analyze-repo.py --all --deep --json   # Deep analysis, all repos, JSON
"""

import sys
import os
import re
import subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TMP_DIR = PROJECT_ROOT / "tmp"
TOOLS_DIR = PROJECT_ROOT / "data" / "tools"


def run_git_command(repo_path, *args):
    """Run a git command in the repository."""
    try:
        result = subprocess.run(
            ['git', '-C', str(repo_path)] + list(args),
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_git_stats(repo_path):
    """Get git repository statistics."""
    stats = {}

    # Current commit
    stats['current_commit'] = run_git_command(repo_path, 'rev-parse', '--short', 'HEAD')
    stats['current_commit_full'] = run_git_command(repo_path, 'rev-parse', 'HEAD')

    # Last commit date
    last_date = run_git_command(repo_path, 'log', '-1', '--format=%ci')
    if last_date:
        stats['last_commit_date'] = last_date[:10]

    # Commit count (may not work for shallow clones)
    count = run_git_command(repo_path, 'rev-list', '--count', 'HEAD')
    if count:
        try:
            stats['commit_count'] = int(count)
        except ValueError:
            pass

    # Contributors (may not work for shallow clones)
    contributors = run_git_command(repo_path, 'shortlog', '-sn', 'HEAD')
    if contributors:
        stats['contributor_count'] = len(contributors.strip().split('\n'))

    return stats


def find_readme(repo_path):
    """Find and read README file."""
    readme_names = ['README.md', 'README.rst', 'README.txt', 'README', 'readme.md']

    for name in readme_names:
        readme_path = repo_path / name
        if readme_path.exists():
            try:
                with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return {
                    'file': name,
                    'length': len(content),
                    'first_lines': content[:500],
                }
            except Exception:
                pass

    return None


def find_license(repo_path):
    """Find and identify license."""
    license_names = ['LICENSE', 'LICENSE.md', 'LICENSE.txt', 'COPYING', 'license']

    for name in license_names:
        license_path = repo_path / name
        if license_path.exists():
            try:
                with open(license_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()[:1000]

                # Try to identify license type
                license_type = 'Unknown'
                if 'MIT License' in content or 'Permission is hereby granted' in content:
                    license_type = 'MIT'
                elif 'Apache License' in content:
                    license_type = 'Apache-2.0'
                elif 'GNU GENERAL PUBLIC LICENSE' in content:
                    if 'Version 3' in content:
                        license_type = 'GPL-3.0'
                    else:
                        license_type = 'GPL'
                elif 'BSD' in content:
                    license_type = 'BSD'

                return {'file': name, 'type': license_type}
            except Exception:
                pass

    return None


def find_dependencies(repo_path):
    """Find dependency specifications."""
    deps = {}

    # requirements.txt
    req_path = repo_path / 'requirements.txt'
    if req_path.exists():
        try:
            with open(req_path) as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
                deps['requirements.txt'] = lines[:20]  # Limit to first 20
        except Exception:
            pass

    # pyproject.toml
    pyproject_path = repo_path / 'pyproject.toml'
    if pyproject_path.exists():
        try:
            with open(pyproject_path) as f:
                content = f.read()
            # Extract dependencies section (simple parsing)
            if 'dependencies' in content:
                deps['pyproject.toml'] = True
        except Exception:
            pass

    # setup.py
    setup_path = repo_path / 'setup.py'
    if setup_path.exists():
        deps['setup.py'] = True

    # setup.cfg
    setupcfg_path = repo_path / 'setup.cfg'
    if setupcfg_path.exists():
        deps['setup.cfg'] = True

    return deps


def find_python_modules(repo_path):
    """Find main Python modules and classes."""
    modules = []

    # Look for Python packages
    for item in repo_path.iterdir():
        if item.is_dir() and (item / '__init__.py').exists():
            modules.append({
                'name': item.name,
                'type': 'package',
                'files': len(list(item.glob('**/*.py')))
            })

    # Look for main source directories
    for src_dir in ['src', 'lib']:
        src_path = repo_path / src_dir
        if src_path.exists():
            for item in src_path.iterdir():
                if item.is_dir() and (item / '__init__.py').exists():
                    modules.append({
                        'name': f"{src_dir}/{item.name}",
                        'type': 'package',
                        'files': len(list(item.glob('**/*.py')))
                    })

    return modules[:10]  # Limit to top 10


def extract_api_surface(repo_path):
    """Extract public API surface from Python packages."""
    api = {
        'classes': [],
        'functions': [],
        'entry_points': []
    }

    # Find __init__.py files with __all__ exports
    for init_file in repo_path.rglob('__init__.py'):
        try:
            with open(init_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Look for __all__ definition
            all_match = re.search(r'__all__\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if all_match:
                exports = re.findall(r'[\'"](\w+)[\'"]', all_match.group(1))
                for export in exports[:20]:  # Limit exports
                    api['entry_points'].append({
                        'name': export,
                        'module': str(init_file.parent.relative_to(repo_path))
                    })
        except Exception:
            pass

    # Find class definitions
    for py_file in repo_path.rglob('*.py'):
        # Skip test files and examples
        if 'test' in str(py_file).lower() or 'example' in str(py_file).lower():
            continue

        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Find class definitions (simple pattern)
            classes = re.findall(r'^class\s+([A-Z]\w+)\s*[:\(]', content, re.MULTILINE)
            for cls in classes[:5]:
                api['classes'].append({
                    'name': cls,
                    'file': str(py_file.relative_to(repo_path))
                })

            # Find top-level function definitions
            functions = re.findall(r'^def\s+([a-z_]\w+)\s*\(', content, re.MULTILINE)
            for func in functions[:5]:
                if not func.startswith('_'):  # Skip private
                    api['functions'].append({
                        'name': func,
                        'file': str(py_file.relative_to(repo_path))
                    })
        except Exception:
            pass

        # Stop after finding enough
        if len(api['classes']) >= 30 and len(api['functions']) >= 30:
            break

    # Deduplicate and limit
    api['classes'] = api['classes'][:20]
    api['functions'] = api['functions'][:20]
    api['entry_points'] = api['entry_points'][:15]

    return api


def extract_readme_sections(repo_path):
    """Extract key sections from README for documentation extraction."""
    readme_names = ['README.md', 'README.rst', 'readme.md']
    sections = {}

    for name in readme_names:
        readme_path = repo_path / name
        if readme_path.exists():
            try:
                with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Extract title
                title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                if title_match:
                    sections['title'] = title_match.group(1).strip()

                # Extract description (first paragraph after title)
                desc_match = re.search(r'^#[^#].*?\n\n(.+?)(?:\n\n|\n#)', content, re.DOTALL)
                if desc_match:
                    desc = desc_match.group(1).strip()[:500]
                    sections['description'] = desc

                # Look for Installation section
                install_match = re.search(r'##?\s*Installation\s*\n(.*?)(?:\n##|\Z)', content, re.DOTALL | re.IGNORECASE)
                if install_match:
                    install = install_match.group(1).strip()[:1000]
                    sections['installation'] = install

                # Look for Usage/Quick Start section
                usage_match = re.search(r'##?\s*(?:Usage|Quick\s*Start|Getting\s*Started)\s*\n(.*?)(?:\n##|\Z)', content, re.DOTALL | re.IGNORECASE)
                if usage_match:
                    usage = usage_match.group(1).strip()[:1000]
                    sections['usage'] = usage

                # Look for Features section
                features_match = re.search(r'##?\s*Features?\s*\n(.*?)(?:\n##|\Z)', content, re.DOTALL | re.IGNORECASE)
                if features_match:
                    features = features_match.group(1).strip()[:500]
                    sections['features'] = features

                break
            except Exception:
                pass

    return sections


def verify_license_spdx(repo_path, license_info):
    """Verify license and provide SPDX identifier."""
    if not license_info or 'type' not in license_info:
        return None

    # SPDX license mapping
    spdx_map = {
        'MIT': {'spdx': 'MIT', 'url': 'https://opensource.org/licenses/MIT'},
        'Apache-2.0': {'spdx': 'Apache-2.0', 'url': 'https://www.apache.org/licenses/LICENSE-2.0'},
        'GPL-3.0': {'spdx': 'GPL-3.0-or-later', 'url': 'https://www.gnu.org/licenses/gpl-3.0.html'},
        'GPL': {'spdx': 'GPL-2.0-or-later', 'url': 'https://www.gnu.org/licenses/gpl-2.0.html'},
        'BSD': {'spdx': 'BSD-3-Clause', 'url': 'https://opensource.org/licenses/BSD-3-Clause'},
        'LGPL': {'spdx': 'LGPL-3.0-or-later', 'url': 'https://www.gnu.org/licenses/lgpl-3.0.html'},
    }

    license_type = license_info.get('type', 'Unknown')
    if license_type in spdx_map:
        return spdx_map[license_type]

    return {'spdx': 'Unknown', 'url': None}


def map_dependencies_detailed(repo_path):
    """Create detailed dependency mapping with categories."""
    deps = {
        'core': [],
        'ml_frameworks': [],
        'audio': [],
        'dev': [],
        'files': []
    }

    # ML framework patterns
    ml_patterns = ['torch', 'tensorflow', 'jax', 'keras', 'onnx', 'torchaudio', 'transformers']
    audio_patterns = ['librosa', 'soundfile', 'audioread', 'pyaudio', 'pydub', 'scipy.io.wavfile']

    # requirements.txt
    req_path = repo_path / 'requirements.txt'
    if req_path.exists():
        deps['files'].append('requirements.txt')
        try:
            with open(req_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # Extract package name
                    pkg = re.split(r'[<>=!~\[]', line)[0].strip().lower()
                    if any(ml in pkg for ml in ml_patterns):
                        deps['ml_frameworks'].append(pkg)
                    elif any(audio in pkg for audio in audio_patterns):
                        deps['audio'].append(pkg)
                    else:
                        deps['core'].append(pkg)
        except Exception:
            pass

    # requirements-dev.txt or dev-requirements.txt
    for dev_req in ['requirements-dev.txt', 'dev-requirements.txt', 'test-requirements.txt']:
        dev_path = repo_path / dev_req
        if dev_path.exists():
            deps['files'].append(dev_req)
            try:
                with open(dev_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        pkg = re.split(r'[<>=!~\[]', line)[0].strip()
                        deps['dev'].append(pkg)
            except Exception:
                pass

    # pyproject.toml
    if (repo_path / 'pyproject.toml').exists():
        deps['files'].append('pyproject.toml')

    # setup.py
    if (repo_path / 'setup.py').exists():
        deps['files'].append('setup.py')

    # Deduplicate
    for key in ['core', 'ml_frameworks', 'audio', 'dev']:
        deps[key] = list(dict.fromkeys(deps[key]))[:25]

    return deps


def find_docs(repo_path):
    """Check for documentation."""
    docs = {}

    # Check for docs directory
    for doc_dir in ['docs', 'doc', 'documentation']:
        doc_path = repo_path / doc_dir
        if doc_path.exists():
            docs['docs_dir'] = doc_dir
            docs['docs_files'] = len(list(doc_path.rglob('*')))

    # Check for specific doc files
    for doc_file in ['CONTRIBUTING.md', 'CHANGELOG.md', 'HISTORY.md']:
        if (repo_path / doc_file).exists():
            docs[doc_file.lower().replace('.md', '')] = True

    return docs


def analyze_repo(repo_path, deep=False):
    """Analyze a repository and return findings.

    Args:
        repo_path: Path to the repository
        deep: If True, perform deep analysis (API surface, detailed deps)
    """
    repo_path = Path(repo_path)

    if not repo_path.exists():
        return {'error': f"Repository not found: {repo_path}"}

    if not (repo_path / '.git').exists():
        return {'error': f"Not a git repository: {repo_path}"}

    analysis = {
        'name': repo_path.name,
        'path': str(repo_path),
        'analyzed_at': datetime.now().isoformat(),
        'deep_analysis': deep,
    }

    # Git stats
    analysis['git'] = get_git_stats(repo_path)

    # README
    readme = find_readme(repo_path)
    if readme:
        analysis['readme'] = readme

    # License
    license_info = find_license(repo_path)
    if license_info:
        analysis['license'] = license_info
        # Add SPDX verification
        spdx = verify_license_spdx(repo_path, license_info)
        if spdx:
            analysis['license']['spdx'] = spdx['spdx']
            analysis['license']['spdx_url'] = spdx['url']

    # Dependencies (basic)
    deps = find_dependencies(repo_path)
    if deps:
        analysis['dependencies'] = deps

    # Python modules
    modules = find_python_modules(repo_path)
    if modules:
        analysis['modules'] = modules

    # Documentation
    docs = find_docs(repo_path)
    if docs:
        analysis['documentation'] = docs

    # Deep analysis features
    if deep:
        # API surface extraction
        api = extract_api_surface(repo_path)
        if api and (api['classes'] or api['functions'] or api['entry_points']):
            analysis['api_surface'] = api

        # README section extraction
        readme_sections = extract_readme_sections(repo_path)
        if readme_sections:
            analysis['readme_sections'] = readme_sections

        # Detailed dependency mapping
        detailed_deps = map_dependencies_detailed(repo_path)
        if detailed_deps:
            analysis['dependencies_detailed'] = detailed_deps

    return analysis


def format_analysis(analysis):
    """Format analysis as readable text."""
    if 'error' in analysis:
        return f"Error: {analysis['error']}"

    lines = []
    lines.append(f"# Repository Analysis: {analysis['name']}")
    lines.append(f"")
    lines.append(f"Analyzed: {analysis['analyzed_at']}")
    if analysis.get('deep_analysis'):
        lines.append(f"Mode: Deep Analysis")
    lines.append(f"")

    # Git info
    if 'git' in analysis:
        git = analysis['git']
        lines.append(f"## Git")
        lines.append(f"")
        if git.get('current_commit'):
            lines.append(f"* Current commit: {git['current_commit']}")
        if git.get('last_commit_date'):
            lines.append(f"* Last commit: {git['last_commit_date']}")
        if git.get('commit_count'):
            lines.append(f"* Total commits: {git['commit_count']}")
        if git.get('contributor_count'):
            lines.append(f"* Contributors: {git['contributor_count']}")
        lines.append(f"")

    # License
    if 'license' in analysis:
        lic = analysis['license']
        lines.append(f"## License")
        lines.append(f"")
        lines.append(f"* Type: {lic.get('type', 'Unknown')}")
        if lic.get('spdx'):
            lines.append(f"* SPDX: {lic['spdx']}")
        lines.append(f"* File: {lic.get('file', 'N/A')}")
        lines.append(f"")

    # Dependencies
    if 'dependencies' in analysis:
        deps = analysis['dependencies']
        lines.append(f"## Dependencies")
        lines.append(f"")
        for file, content in deps.items():
            if isinstance(content, list):
                lines.append(f"### {file}")
                lines.append(f"")
                for dep in content[:10]:
                    lines.append(f"* {dep}")
                if len(content) > 10:
                    lines.append(f"* ... and {len(content) - 10} more")
            else:
                lines.append(f"* {file}: present")
        lines.append(f"")

    # Modules
    if 'modules' in analysis:
        lines.append(f"## Python Modules")
        lines.append(f"")
        for mod in analysis['modules']:
            lines.append(f"* {mod['name']} ({mod['files']} files)")
        lines.append(f"")

    # Documentation
    if 'documentation' in analysis:
        docs = analysis['documentation']
        lines.append(f"## Documentation")
        lines.append(f"")
        if 'docs_dir' in docs:
            lines.append(f"* Docs directory: {docs['docs_dir']}/ ({docs.get('docs_files', '?')} files)")
        for key in ['contributing', 'changelog', 'history']:
            if docs.get(key):
                lines.append(f"* {key.upper()}.md: present")
        lines.append(f"")

    # Deep analysis sections
    if analysis.get('deep_analysis'):
        # README sections
        if 'readme_sections' in analysis:
            sections = analysis['readme_sections']
            lines.append(f"## Extracted Documentation")
            lines.append(f"")
            if sections.get('title'):
                lines.append(f"**Title:** {sections['title']}")
                lines.append(f"")
            if sections.get('description'):
                lines.append(f"**Description:**")
                lines.append(f"")
                lines.append(sections['description'][:300])
                lines.append(f"")
            if sections.get('features'):
                lines.append(f"**Features:**")
                lines.append(f"")
                lines.append(sections['features'][:300])
                lines.append(f"")

        # API surface
        if 'api_surface' in analysis:
            api = analysis['api_surface']
            lines.append(f"## API Surface")
            lines.append(f"")

            if api.get('entry_points'):
                lines.append(f"### Entry Points (__all__)")
                lines.append(f"")
                for ep in api['entry_points'][:10]:
                    lines.append(f"* {ep['name']} ({ep['module']})")
                lines.append(f"")

            if api.get('classes'):
                lines.append(f"### Classes")
                lines.append(f"")
                for cls in api['classes'][:10]:
                    lines.append(f"* {cls['name']} ({cls['file']})")
                lines.append(f"")

            if api.get('functions'):
                lines.append(f"### Functions")
                lines.append(f"")
                for func in api['functions'][:10]:
                    lines.append(f"* {func['name']} ({func['file']})")
                lines.append(f"")

        # Detailed dependencies
        if 'dependencies_detailed' in analysis:
            detailed = analysis['dependencies_detailed']
            lines.append(f"## Dependency Categories")
            lines.append(f"")
            if detailed.get('ml_frameworks'):
                lines.append(f"### ML Frameworks")
                lines.append(f"")
                for dep in detailed['ml_frameworks']:
                    lines.append(f"* {dep}")
                lines.append(f"")
            if detailed.get('audio'):
                lines.append(f"### Audio Libraries")
                lines.append(f"")
                for dep in detailed['audio']:
                    lines.append(f"* {dep}")
                lines.append(f"")

    return '\n'.join(lines)


def main():
    args = sys.argv[1:]

    if not args or '--help' in args or '-h' in args:
        print(__doc__)
        print("\nFlags:")
        print("  --all       Analyze all repos in tmp/")
        print("  --deep      Enable deep analysis (API surface, detailed deps)")
        print("  --json      Output as JSON")
        print("  --update    Update YAML files with findings (not implemented)")
        sys.exit(0)

    output_json = '--json' in args
    analyze_all = '--all' in args
    update_yaml = '--update' in args
    deep_mode = '--deep' in args

    # Filter out flags
    repo_args = [a for a in args if not a.startswith('--')]

    if analyze_all:
        # Analyze all repos in tmp/
        if TMP_DIR.exists():
            repos = [d for d in TMP_DIR.iterdir() if d.is_dir() and (d / '.git').exists()]
        else:
            repos = []
            print(f"Warning: {TMP_DIR} does not exist. Run 'make clone' first.", file=sys.stderr)
    elif repo_args:
        repos = [Path(r) for r in repo_args]
    else:
        print("Error: Specify repository path or use --all")
        sys.exit(1)

    if not repos:
        print("No repositories to analyze.")
        sys.exit(0)

    results = []

    for repo_path in repos:
        analysis = analyze_repo(repo_path, deep=deep_mode)
        results.append(analysis)

        if not output_json:
            print(format_analysis(analysis))
            print("=" * 60)
            print()

    if output_json:
        import json
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
