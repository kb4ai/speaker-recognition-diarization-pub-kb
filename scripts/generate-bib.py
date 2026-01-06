#!/usr/bin/env python3
"""
Generate BibTeX entries from paper YAML files.

Reads all papers/*.paper.yaml files and outputs valid BibTeX entries.

Usage:
    ./scripts/generate-bib.py                     # Output to stdout
    ./scripts/generate-bib.py > papers.bib        # Save to file
    ./scripts/generate-bib.py --format=full       # Include abstracts
    ./scripts/generate-bib.py --key-style=short   # Use short keys
"""

import sys
import re
from pathlib import Path
from datetime import datetime

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PAPERS_DIR = PROJECT_ROOT / "papers"


def load_papers():
    """Load all paper YAML files."""
    papers = []
    if not PAPERS_DIR.exists():
        return papers
    for yaml_file in sorted(PAPERS_DIR.glob("*.paper.yaml")):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
                if data:
                    data['_filename'] = yaml_file.stem
                    papers.append(data)
        except Exception as e:
            print(f"Warning: Could not load {yaml_file}: {e}", file=sys.stderr)
    return papers


def generate_bibtex_key(paper, style='standard'):
    """Generate a BibTeX citation key.

    Standard format: {first_author_lastname}{year}{short_title_word}
    Short format: {first_author_lastname}{year}
    """
    # Get first author's last name
    authors = paper.get('authors', ['Unknown'])
    first_author = authors[0] if authors else 'Unknown'
    last_name = first_author.split()[-1].lower()
    # Remove special characters
    last_name = re.sub(r'[^a-z]', '', last_name)

    year = paper.get('year', 'XXXX')

    if style == 'short':
        return f"{last_name}{year}"

    # Get short title or first word of title
    short_title = paper.get('short-title', '')
    if short_title:
        key_suffix = re.sub(r'[^a-zA-Z]', '', short_title).lower()
    else:
        title = paper.get('title', 'unknown')
        # Get first significant word (skip common words)
        words = title.split()
        skip_words = {'a', 'an', 'the', 'on', 'for', 'in', 'of', 'to', 'with'}
        key_suffix = 'paper'
        for word in words:
            clean = re.sub(r'[^a-zA-Z]', '', word).lower()
            if clean and clean not in skip_words:
                key_suffix = clean
                break

    return f"{last_name}{year}{key_suffix}"


def escape_bibtex(text):
    """Escape special characters for BibTeX."""
    if not text:
        return ''
    # Protect uppercase letters in titles with braces
    # Replace & with \&
    text = text.replace('&', r'\&')
    return text


def format_authors_bibtex(authors):
    """Format author list for BibTeX (LastName, FirstName and ...)."""
    if not authors:
        return 'Unknown'

    formatted = []
    for author in authors:
        parts = author.split()
        if len(parts) >= 2:
            # LastName, FirstName MiddleName...
            last = parts[-1]
            first = ' '.join(parts[:-1])
            formatted.append(f"{last}, {first}")
        else:
            formatted.append(author)

    return ' and '.join(formatted)


def generate_bibtex_entry(paper, include_abstract=False, key_style='standard'):
    """Generate a single BibTeX entry."""
    venue_type = paper.get('venue-type', 'conference')
    entry_type = 'inproceedings' if venue_type == 'conference' else 'article'

    key = generate_bibtex_key(paper, key_style)

    lines = [f"@{entry_type}{{{key},"]

    # Title (protect capitalization with braces)
    title = paper.get('title', 'Unknown Title')
    # Protect specific terms that should stay capitalized (use word boundaries)
    protected_title = title
    terms = ['ECAPA-TDNN', 'TDNN', 'EEND', 'DNN', 'VoxCeleb', 'DIHARD', 'AMI']
    for term in terms:
        # Only replace if not already protected
        if '{' + term + '}' not in protected_title:
            protected_title = protected_title.replace(term, '{' + term + '}')
    lines.append(f"  title={{{protected_title}}},")

    # Authors
    authors = format_authors_bibtex(paper.get('authors', []))
    lines.append(f"  author={{{authors}}},")

    # Year
    year = paper.get('year', 'XXXX')
    lines.append(f"  year={{{year}}},")

    # Venue
    venue = paper.get('venue', '')
    if venue:
        if entry_type == 'inproceedings':
            lines.append(f"  booktitle={{{escape_bibtex(venue)}}},")
        else:
            lines.append(f"  journal={{{escape_bibtex(venue)}}},")

    # DOI
    doi = paper.get('doi', '')
    if doi:
        lines.append(f"  doi={{{doi}}},")

    # arXiv
    arxiv = paper.get('arxiv', '')
    if arxiv:
        lines.append(f"  eprint={{{arxiv}}},")
        lines.append(f"  archiveprefix={{arXiv}},")

    # URL
    pdf_url = paper.get('pdf-url', '')
    if pdf_url and not arxiv:  # Don't duplicate if arXiv
        lines.append(f"  url={{{pdf_url}}},")

    # Abstract (optional)
    if include_abstract:
        abstract = paper.get('abstract', '')
        if abstract:
            # Clean up multiline abstract
            clean_abstract = ' '.join(abstract.split())
            lines.append(f"  abstract={{{clean_abstract}}},")

    # Keywords
    keywords = paper.get('keywords', [])
    if keywords:
        lines.append(f"  keywords={{{', '.join(keywords)}}},")

    # Remove trailing comma from last field
    lines[-1] = lines[-1].rstrip(',')
    lines.append("}")

    return '\n'.join(lines)


def main():
    args = sys.argv[1:]

    include_abstract = '--format=full' in args
    key_style = 'short' if '--key-style=short' in args else 'standard'

    papers = load_papers()

    if not papers:
        print("% No papers found in papers/ directory", file=sys.stderr)
        sys.exit(0)

    # Generate header
    print(f"% Speaker Recognition & Diarization Bibliography")
    print(f"% Auto-generated from papers/*.paper.yaml")
    print(f"% Generated: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"% Entries: {len(papers)}")
    print()

    # Sort by year (newest first), then by key
    sorted_papers = sorted(papers, key=lambda p: (-p.get('year', 0), p.get('short-title', '')))

    # Group by category based on topics
    categories = {
        'embedding': [],
        'diarization': [],
        'dataset': [],
        'other': []
    }

    for paper in sorted_papers:
        topics = paper.get('topics', [])
        topics_str = ' '.join(topics).lower()

        if 'embedding' in topics_str or 'verification' in topics_str:
            categories['embedding'].append(paper)
        elif 'diarization' in topics_str or 'end-to-end' in topics_str:
            categories['diarization'].append(paper)
        elif 'dataset' in topics_str or 'corpus' in topics_str:
            categories['dataset'].append(paper)
        else:
            categories['other'].append(paper)

    # Output by category
    category_titles = {
        'embedding': 'SPEAKER EMBEDDING & VERIFICATION',
        'diarization': 'SPEAKER DIARIZATION',
        'dataset': 'DATASETS & BENCHMARKS',
        'other': 'OTHER'
    }

    for cat_key in ['embedding', 'diarization', 'dataset', 'other']:
        papers_in_cat = categories[cat_key]
        if not papers_in_cat:
            continue

        print(f"% {'=' * 50}")
        print(f"% {category_titles[cat_key]}")
        print(f"% {'=' * 50}")
        print()

        for paper in papers_in_cat:
            entry = generate_bibtex_entry(paper, include_abstract, key_style)
            print(entry)
            print()

    print(f"% End of auto-generated bibliography")
    print(f"% Source: {PAPERS_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
