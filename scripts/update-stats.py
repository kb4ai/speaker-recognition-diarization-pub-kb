#!/usr/bin/env python3
"""
Update GitHub repository statistics for tool entries.

Fetches current stars, forks, and other metrics from GitHub API.

Usage:
    ./scripts/update-stats.py                  # Show current vs remote stats
    ./scripts/update-stats.py --update         # Update YAML files in place
    ./scripts/update-stats.py --json           # Output as JSON
    ./scripts/update-stats.py --tool pyannote  # Check specific tool

Requirements:
    - gh CLI installed and authenticated (recommended)
    - OR: Set GITHUB_TOKEN environment variable
"""

import sys
import re
import subprocess
import json
from pathlib import Path
from datetime import datetime

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TOOLS_DIR = PROJECT_ROOT / "data" / "tools"


def parse_github_url(url):
    """Extract owner and repo from GitHub URL."""
    if not url:
        return None, None

    # Handle various GitHub URL formats
    patterns = [
        r'github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/|$)',
        r'github\.com:([^/]+)/([^/]+?)(?:\.git)?$',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1), match.group(2)

    return None, None


def fetch_github_stats(owner, repo):
    """Fetch repository stats using gh CLI or API."""
    # Try gh CLI first (handles auth automatically)
    try:
        result = subprocess.run(
            ['gh', 'api', f'repos/{owner}/{repo}', '--jq',
             '{stars: .stargazers_count, forks: .forks_count, '
             'open_issues: .open_issues_count, last_push: .pushed_at, '
             'description: .description, license: .license.spdx_id}'],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)
            return {
                'stars': data.get('stars'),
                'forks': data.get('forks'),
                'open_issues': data.get('open_issues'),
                'last_push': data.get('last_push', '')[:10] if data.get('last_push') else None,
                'description': data.get('description'),
                'license': data.get('license'),
                'source': 'gh-cli'
            }
    except FileNotFoundError:
        print("Note: gh CLI not found, trying curl...", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print(f"Warning: Timeout fetching {owner}/{repo}", file=sys.stderr)
    except json.JSONDecodeError:
        pass

    # Fallback to curl with GITHUB_TOKEN
    import os
    token = os.environ.get('GITHUB_TOKEN', '')

    try:
        headers = ['-H', f'Authorization: token {token}'] if token else []
        result = subprocess.run(
            ['curl', '-s', '-f'] + headers +
            [f'https://api.github.com/repos/{owner}/{repo}'],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)
            return {
                'stars': data.get('stargazers_count'),
                'forks': data.get('forks_count'),
                'open_issues': data.get('open_issues_count'),
                'last_push': data.get('pushed_at', '')[:10] if data.get('pushed_at') else None,
                'description': data.get('description'),
                'license': data.get('license', {}).get('spdx_id') if data.get('license') else None,
                'source': 'api'
            }
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    return None


def load_tools():
    """Load all tool YAML files."""
    tools = []
    if not TOOLS_DIR.exists():
        return tools

    for yaml_file in sorted(TOOLS_DIR.glob("*.tool.yaml")):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
                if data:
                    data['_filepath'] = yaml_file
                    tools.append(data)
        except Exception as e:
            print(f"Warning: Could not load {yaml_file}: {e}", file=sys.stderr)

    return tools


def update_yaml_file(filepath, updates):
    """Update specific fields in a YAML file preserving formatting."""
    with open(filepath) as f:
        content = f.read()

    for field, value in updates.items():
        if value is None:
            continue

        # Pattern to match field: value
        pattern = rf'^({field}:\s*).*$'
        replacement = rf'\g<1>{value}'

        if re.search(pattern, content, re.MULTILINE):
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        else:
            # Field doesn't exist, we'd need to add it - skip for now
            print(f"  Note: {field} not in file, skipping", file=sys.stderr)

    # Update last-update date
    today = datetime.now().strftime('%Y-%m-%d')
    content = re.sub(
        r'^(last-update:\s*)["\']?\d{4}-\d{2}-\d{2}["\']?$',
        rf'\g<1>"{today}"',
        content,
        flags=re.MULTILINE
    )

    with open(filepath, 'w') as f:
        f.write(content)


def main():
    args = sys.argv[1:]
    do_update = '--update' in args
    output_json = '--json' in args

    # Filter by tool name if specified
    tool_filter = None
    for i, arg in enumerate(args):
        if arg == '--tool' and i + 1 < len(args):
            tool_filter = args[i + 1].lower()

    tools = load_tools()

    if not tools:
        print("No tool files found in data/tools/", file=sys.stderr)
        sys.exit(1)

    results = []

    for tool in tools:
        name = tool.get('name', 'Unknown')
        repo_url = tool.get('repo-url', '')

        if tool_filter and tool_filter not in name.lower():
            continue

        owner, repo = parse_github_url(repo_url)

        if not owner or not repo:
            results.append({
                'name': name,
                'error': 'Could not parse GitHub URL',
                'url': repo_url
            })
            continue

        print(f"Fetching {owner}/{repo}...", file=sys.stderr)
        stats = fetch_github_stats(owner, repo)

        if not stats:
            results.append({
                'name': name,
                'error': 'Failed to fetch stats',
                'url': repo_url
            })
            continue

        # Compare with current values
        current_stars = tool.get('stars')
        current_forks = tool.get('forks')

        result = {
            'name': name,
            'owner': owner,
            'repo': repo,
            'current_stars': current_stars,
            'remote_stars': stats['stars'],
            'current_forks': current_forks,
            'remote_forks': stats['forks'],
            'last_push': stats['last_push'],
            'stars_changed': current_stars != stats['stars'] if current_stars else True,
            'filepath': str(tool['_filepath'])
        }

        results.append(result)

        if do_update and (result.get('stars_changed') or current_forks != stats['forks']):
            print(f"  Updating {tool['_filepath'].name}...", file=sys.stderr)
            update_yaml_file(tool['_filepath'], {
                'stars': stats['stars'],
                'forks': stats['forks']
            })

    # Output
    if output_json:
        print(json.dumps(results, indent=2, default=str))
        return

    print(f"\n# GitHub Stats Report")
    print(f"")
    print(f"**Tools checked:** {len(results)}")
    print(f"**Updated:** {'Yes' if do_update else 'No (dry run)'}")
    print(f"")
    print(f"| Tool | Current Stars | Remote Stars | Forks | Last Push |")
    print(f"|------|-------------:|-------------:|------:|-----------|")

    for r in results:
        if 'error' in r:
            print(f"| {r['name']} | - | Error: {r['error']} | - | - |")
        else:
            stars_diff = ''
            if r.get('current_stars') and r.get('remote_stars'):
                diff = r['remote_stars'] - r['current_stars']
                if diff != 0:
                    stars_diff = f" ({'+' if diff > 0 else ''}{diff})"

            print(f"| {r['name']} | {r.get('current_stars', '?')} | "
                  f"{r.get('remote_stars', '?')}{stars_diff} | "
                  f"{r.get('remote_forks', '?')} | {r.get('last_push', '?')} |")

    if not do_update:
        print(f"\nRun with --update to apply changes to YAML files.")


if __name__ == "__main__":
    main()
