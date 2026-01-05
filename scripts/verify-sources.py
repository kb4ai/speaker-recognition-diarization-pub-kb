#!/usr/bin/env python3
"""
Verify source URLs are accessible.

Usage:
    ./scripts/verify-sources.py           # Check all sources
    ./scripts/verify-sources.py --report  # Generate detailed report
"""

import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between requests
TIMEOUT = 10  # seconds per request


def load_all_sources():
    """Extract all sources from YAML files."""
    sources = []

    for yaml_file in DATA_DIR.rglob("*.yaml"):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
                if not data:
                    continue

                # Check for sources array
                if 'sources' in data and isinstance(data['sources'], list):
                    for source in data['sources']:
                        if isinstance(source, dict) and 'url' in source:
                            sources.append({
                                'url': source['url'],
                                'file': str(yaml_file.relative_to(PROJECT_ROOT)),
                                'accessed': source.get('accessed', '?'),
                            })

                # Check repo-url
                if 'repo-url' in data:
                    sources.append({
                        'url': data['repo-url'],
                        'file': str(yaml_file.relative_to(PROJECT_ROOT)),
                        'accessed': data.get('last-update', '?'),
                    })

        except Exception as e:
            print(f"Warning: Could not load {yaml_file}: {e}", file=sys.stderr)

    return sources


def check_url(url):
    """Check if a URL is accessible."""
    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urlopen(req, timeout=TIMEOUT)
        return True, response.status
    except HTTPError as e:
        return False, e.code
    except URLError as e:
        return False, str(e.reason)
    except Exception as e:
        return False, str(e)


def main():
    args = sys.argv[1:]
    report_mode = '--report' in args

    sources = load_all_sources()
    unique_urls = list(set(s['url'] for s in sources))

    print(f"Found {len(sources)} source references ({len(unique_urls)} unique URLs)")
    print("")

    # Check URLs
    results = {}
    failed = []

    for i, url in enumerate(unique_urls):
        print(f"Checking [{i+1}/{len(unique_urls)}]: {url[:60]}...", end=" ", flush=True)
        ok, status = check_url(url)

        if ok:
            print(f"✓ ({status})")
            results[url] = ('ok', status)
        else:
            print(f"✗ ({status})")
            results[url] = ('failed', status)
            failed.append(url)

        time.sleep(REQUEST_DELAY)

    # Summary
    print("")
    print("=" * 50)
    print(f"Total URLs checked: {len(unique_urls)}")
    print(f"Accessible: {len(unique_urls) - len(failed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("")
        print("Failed URLs:")
        for url in failed:
            status = results[url][1]
            # Find files referencing this URL
            files = [s['file'] for s in sources if s['url'] == url]
            print(f"  ✗ {url}")
            print(f"    Status: {status}")
            print(f"    Referenced in: {', '.join(files[:3])}")

    if report_mode:
        print("")
        print("=" * 50)
        print("Full Report:")
        for source in sources:
            url = source['url']
            status, code = results.get(url, ('unknown', '?'))
            icon = "✓" if status == 'ok' else "✗"
            print(f"  {icon} {source['file']}")
            print(f"      URL: {url}")
            print(f"      Accessed: {source['accessed']}")
            print(f"      Status: {code}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
