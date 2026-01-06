#!/usr/bin/env python3
"""
Analyze a cloned repository and extract useful information.

Extracts:
- README content summary
- License information
- Dependencies (requirements.txt, pyproject.toml, setup.py)
- Main modules and classes
- Git statistics

Usage:
    ./scripts/analyze-repo.py tmp/pyannote-audio    # Analyze specific repo
    ./scripts/analyze-repo.py --all                 # Analyze all in tmp/
    ./scripts/analyze-repo.py tmp/repo --json       # Output as JSON
    ./scripts/analyze-repo.py tmp/repo --update     # Update YAML with findings
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


def analyze_repo(repo_path):
    """Analyze a repository and return findings."""
    repo_path = Path(repo_path)

    if not repo_path.exists():
        return {'error': f"Repository not found: {repo_path}"}

    if not (repo_path / '.git').exists():
        return {'error': f"Not a git repository: {repo_path}"}

    analysis = {
        'name': repo_path.name,
        'path': str(repo_path),
        'analyzed_at': datetime.now().isoformat(),
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

    # Dependencies
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

    return analysis


def format_analysis(analysis):
    """Format analysis as readable text."""
    if 'error' in analysis:
        return f"Error: {analysis['error']}"

    lines = []
    lines.append(f"# Repository Analysis: {analysis['name']}")
    lines.append(f"")
    lines.append(f"Analyzed: {analysis['analyzed_at']}")
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

    return '\n'.join(lines)


def main():
    args = sys.argv[1:]

    if not args or '--help' in args or '-h' in args:
        print(__doc__)
        sys.exit(0)

    output_json = '--json' in args
    analyze_all = '--all' in args
    update_yaml = '--update' in args

    # Filter out flags
    repo_args = [a for a in args if not a.startswith('--')]

    if analyze_all:
        # Analyze all repos in tmp/
        repos = [d for d in TMP_DIR.iterdir() if d.is_dir() and (d / '.git').exists()]
    elif repo_args:
        repos = [Path(r) for r in repo_args]
    else:
        print("Error: Specify repository path or use --all")
        sys.exit(1)

    results = []

    for repo_path in repos:
        analysis = analyze_repo(repo_path)
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
