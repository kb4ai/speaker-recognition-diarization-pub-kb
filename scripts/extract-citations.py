#!/usr/bin/env python3
"""
Extract and analyze citations from markdown files.

Parses knowledge/ articles to find:
- Author et al. citations: "Desplanques et al. (2020)"
- Inline paper references: "ECAPA-TDNN paper"
- arXiv references: arXiv:2005.07143
- DOI references: doi:10.1234/...

Cross-references with papers/*.paper.yaml entries.

Usage:
    ./scripts/extract-citations.py                  # Report all citations
    ./scripts/extract-citations.py --unmatched     # Show only unmatched
    ./scripts/extract-citations.py --json          # Output as JSON
    ./scripts/extract-citations.py --verify        # Check against papers/
"""

import sys
import re
from pathlib import Path
from collections import defaultdict

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
PAPERS_DIR = PROJECT_ROOT / "papers"


# Citation patterns
PATTERNS = {
    'author_year': re.compile(
        r'(?P<authors>[\w\-]+(?:\s+et\s+al\.)?)'
        r'\s*'
        r'["\u201c](?P<title>[^"\u201d]+)["\u201d]'
        r'\s*'
        r'\((?P<venue_year>[^)]+)\)',
        re.IGNORECASE
    ),
    'author_paren': re.compile(
        r'(?P<authors>[\w\-]+(?:\s+et\s+al\.)?)'
        r'\s*\((?P<year>\d{4})\)',
        re.IGNORECASE
    ),
    'arxiv': re.compile(
        r'arXiv[:\s]*(?P<id>\d{4}\.\d{4,5}(?:v\d+)?)',
        re.IGNORECASE
    ),
    'doi': re.compile(
        r'doi[:\s]*(?P<id>10\.\d{4,}/[\w\.\-/]+)',
        re.IGNORECASE
    ),
}


def load_papers():
    """Load all paper YAML files for cross-referencing."""
    papers = {}
    if not PAPERS_DIR.exists():
        return papers

    for yaml_file in PAPERS_DIR.glob("*.paper.yaml"):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
                if data:
                    key = yaml_file.stem.replace('.paper', '')
                    papers[key] = data
        except Exception as e:
            print(f"Warning: Could not load {yaml_file}: {e}", file=sys.stderr)

    return papers


def extract_citations_from_file(filepath):
    """Extract all citations from a markdown file."""
    citations = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
        return citations

    lines = content.split('\n')

    for line_num, line in enumerate(lines, 1):
        # Skip code blocks
        if line.strip().startswith('```'):
            continue

        # Author + "Title" (Year/Venue) pattern
        for match in PATTERNS['author_year'].finditer(line):
            citations.append({
                'type': 'full_citation',
                'authors': match.group('authors'),
                'title': match.group('title'),
                'venue_year': match.group('venue_year'),
                'file': str(filepath.relative_to(PROJECT_ROOT)),
                'line': line_num,
                'raw': match.group(0)
            })

        # arXiv references
        for match in PATTERNS['arxiv'].finditer(line):
            citations.append({
                'type': 'arxiv',
                'arxiv_id': match.group('id'),
                'file': str(filepath.relative_to(PROJECT_ROOT)),
                'line': line_num,
                'raw': match.group(0)
            })

        # DOI references
        for match in PATTERNS['doi'].finditer(line):
            citations.append({
                'type': 'doi',
                'doi': match.group('id'),
                'file': str(filepath.relative_to(PROJECT_ROOT)),
                'line': line_num,
                'raw': match.group(0)
            })

    return citations


def match_citation_to_paper(citation, papers):
    """Try to match a citation to a paper YAML entry."""
    if citation['type'] == 'arxiv':
        arxiv_id = citation['arxiv_id']
        for key, paper in papers.items():
            if paper.get('arxiv') == arxiv_id:
                return key, paper
        return None, None

    if citation['type'] == 'doi':
        doi = citation['doi']
        for key, paper in papers.items():
            if paper.get('doi') == doi:
                return key, paper
        return None, None

    if citation['type'] == 'full_citation':
        # Try to match by title
        title = citation['title'].lower()
        for key, paper in papers.items():
            paper_title = paper.get('title', '').lower()
            short_title = paper.get('short-title', '').lower()

            # Fuzzy match: check if key words overlap
            if title in paper_title or paper_title in title:
                return key, paper
            if short_title and (short_title in title or title in short_title):
                return key, paper

        # Try to match by author
        author = citation['authors'].split()[0].lower()
        for key, paper in papers.items():
            authors = paper.get('authors', [])
            for paper_author in authors:
                if author in paper_author.lower():
                    return key, paper

        return None, None

    return None, None


def main():
    args = sys.argv[1:]
    show_unmatched = '--unmatched' in args
    output_json = '--json' in args
    verify_mode = '--verify' in args

    # Load papers for cross-referencing
    papers = load_papers()

    # Find all markdown files
    md_files = []
    if KNOWLEDGE_DIR.exists():
        md_files.extend(KNOWLEDGE_DIR.rglob("*.md"))

    if not md_files:
        print("No markdown files found in knowledge/", file=sys.stderr)
        sys.exit(0)

    # Extract all citations
    all_citations = []
    for md_file in sorted(md_files):
        citations = extract_citations_from_file(md_file)
        all_citations.extend(citations)

    # Match citations to papers
    matched = []
    unmatched = []

    for citation in all_citations:
        paper_key, paper = match_citation_to_paper(citation, papers)
        if paper_key:
            citation['matched_paper'] = paper_key
            matched.append(citation)
        else:
            unmatched.append(citation)

    # Output
    if output_json:
        import json
        output = {
            'total_citations': len(all_citations),
            'matched': len(matched),
            'unmatched': len(unmatched),
            'citations': unmatched if show_unmatched else all_citations
        }
        print(json.dumps(output, indent=2))
        return

    # Text output
    print(f"# Citation Analysis Report")
    print(f"")
    print(f"**Files scanned:** {len(md_files)}")
    print(f"**Total citations found:** {len(all_citations)}")
    print(f"**Matched to papers/:** {len(matched)}")
    print(f"**Unmatched:** {len(unmatched)}")
    print(f"")

    if verify_mode or show_unmatched:
        if unmatched:
            print(f"## Unmatched Citations")
            print(f"")
            print(f"These citations have no corresponding entry in papers/:")
            print(f"")

            # Group by file
            by_file = defaultdict(list)
            for c in unmatched:
                by_file[c['file']].append(c)

            for file, cites in sorted(by_file.items()):
                print(f"### {file}")
                print(f"")
                for c in cites:
                    print(f"* Line {c['line']}: `{c['raw']}`")
                print(f"")
        else:
            print("All citations matched to papers/")

    else:
        # Show all citations grouped by type
        by_type = defaultdict(list)
        for c in all_citations:
            by_type[c['type']].append(c)

        for ctype, cites in sorted(by_type.items()):
            print(f"## {ctype.replace('_', ' ').title()} ({len(cites)})")
            print(f"")
            for c in cites[:10]:  # Show first 10
                status = "✓" if 'matched_paper' in c else "?"
                print(f"* {status} `{c['raw'][:60]}...` ({c['file']}:{c['line']})")
            if len(cites) > 10:
                print(f"* ... and {len(cites) - 10} more")
            print(f"")

    if verify_mode:
        sys.exit(1 if unmatched else 0)


if __name__ == "__main__":
    main()
