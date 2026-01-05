#!/usr/bin/env python3
"""
Validate YAML files against type-specific schemas.

File naming convention: {name}.{type}.yaml
Spec file location: schemas/{type}.spec.yaml

Usage:
    ./scripts/check-yaml.py                              # Check all files
    ./scripts/check-yaml.py data/tools/pyannote*.yaml    # Check specific files
    ./scripts/check-yaml.py --strict                     # Fail on warnings too
"""

import sys
import re
from pathlib import Path
from datetime import datetime

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = PROJECT_ROOT / "papers"


def get_yaml_type(filepath):
    """Extract type from filename: name.{type}.yaml -> type"""
    name = filepath.name
    parts = name.rsplit('.', 2)
    if len(parts) >= 3 and parts[-1] == 'yaml':
        return parts[-2]
    return None


def load_spec(yaml_type):
    """Load the spec for a given type."""
    spec_file = SCHEMAS_DIR / f"{yaml_type}.spec.yaml"
    if not spec_file.exists():
        return None
    with open(spec_file) as f:
        return yaml.safe_load(f)


def validate_date(value, field_name):
    """Validate YYYY-MM-DD date format."""
    if not isinstance(value, str):
        return f"{field_name}: expected string, got {type(value).__name__}"
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', value):
        return f"{field_name}: invalid date format '{value}', expected YYYY-MM-DD"
    try:
        datetime.strptime(value, '%Y-%m-%d')
    except ValueError as e:
        return f"{field_name}: invalid date '{value}': {e}"
    return None


def validate_url(value, field_name):
    """Validate URL format."""
    if not isinstance(value, str):
        return f"{field_name}: expected string, got {type(value).__name__}"
    if not re.match(r'^https?://', value):
        return f"{field_name}: invalid URL '{value}', must start with http(s)://"
    return None


def validate_yaml_file(filepath, spec, yaml_type):
    """Validate a single YAML file against its spec."""
    errors = []
    warnings = []

    try:
        with open(filepath) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return [f"YAML parse error: {e}"], []
    except Exception as e:
        return [f"Error reading file: {e}"], []

    if data is None:
        return ["File is empty"], []

    if not isinstance(data, dict):
        return ["File must contain a YAML mapping (dictionary)"], []

    # Check required fields from spec
    if spec:
        required = spec.get('required', [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")

    # Common validation: last-update
    if 'last-update' in data:
        err = validate_date(data['last-update'], 'last-update')
        if err:
            errors.append(err)

    # Validate URL fields
    url_fields = ['repo-url', 'url', 'huggingface-url', 'pdf-url', 'code-url', 'project-page']
    for field in url_fields:
        if field in data and data[field]:
            err = validate_url(data[field], field)
            if err:
                errors.append(err)

    # Validate date fields
    date_fields = ['last-commit', 'created', 'accessed', 'archived-date', 'verified-date', 'citation-count-date']
    for field in date_fields:
        if field in data and data[field]:
            err = validate_date(data[field], field)
            if err:
                errors.append(err)

    # Validate repo-commit format (should be hex string)
    if 'repo-commit' in data and data['repo-commit']:
        commit = data['repo-commit']
        if not isinstance(commit, str):
            errors.append(f"repo-commit: expected string, got {type(commit).__name__}")
        elif not re.match(r'^[a-fA-F0-9]+$', commit):
            warnings.append(f"repo-commit: '{commit}' doesn't look like a git commit hash")

    # Validate category enum if spec provides it
    if spec and 'category' in data:
        fields = spec.get('fields', {})
        category_spec = fields.get('category', {})
        valid_categories = category_spec.get('enum', [])
        if valid_categories and data['category'] not in valid_categories:
            warnings.append(f"category: '{data['category']}' not in known categories")

    # Validate numeric fields
    numeric_fields = ['stars', 'forks', 'contributors', 'year', 'year-introduced',
                      'embedding-dimension', 'citation-count']
    for field in numeric_fields:
        if field in data and data[field] is not None:
            if not isinstance(data[field], (int, float)):
                errors.append(f"{field}: expected number, got {type(data[field]).__name__}")

    # Validate boolean fields
    bool_fields = ['reputable-source', 'archived', 'requires-token', 'commercial-use',
                   'fine-tunable', 'registration-required']
    for field in bool_fields:
        if field in data and data[field] is not None:
            if not isinstance(data[field], bool):
                errors.append(f"{field}: expected boolean, got {type(data[field]).__name__}")

    # Validate nested objects
    object_fields = ['capabilities', 'performance', 'embedding', 'installation',
                     'requirements', 'documentation', 'training', 'benchmarks',
                     'usage', 'statistics', 'collection', 'access', 'splits',
                     'annotations', 'original-paper', 'input', 'output', 'complexity']
    for field in object_fields:
        if field in data and data[field] is not None:
            if not isinstance(data[field], dict):
                errors.append(f"{field}: expected mapping, got {type(data[field]).__name__}")

    # Validate arrays
    array_fields = ['features', 'notes', 'sources', 'implementations', 'variants',
                    'advantages', 'limitations', 'authors', 'keywords', 'topics',
                    'contributions', 'language', 'benchmark-usage', 'related-datasets',
                    'related-algorithms', 'languages', 'mathematical-basis', 'key-equations']
    for field in array_fields:
        if field in data and data[field] is not None:
            if not isinstance(data[field], list):
                errors.append(f"{field}: expected list, got {type(data[field]).__name__}")

    # Validate sources array structure
    if 'sources' in data and isinstance(data['sources'], list):
        for i, source in enumerate(data['sources']):
            if isinstance(source, dict):
                if 'url' not in source and 'arxiv' not in source:
                    warnings.append(f"sources[{i}]: missing 'url' or 'arxiv'")
                if 'accessed' not in source:
                    warnings.append(f"sources[{i}]: missing 'accessed' date")

    return errors, warnings


def find_yaml_files():
    """Find all type-encoded YAML files in the repository."""
    files = []

    # Data directory files
    if DATA_DIR.exists():
        for yaml_file in DATA_DIR.rglob("*.yaml"):
            if get_yaml_type(yaml_file):
                files.append(yaml_file)

    # Papers directory
    if PAPERS_DIR.exists():
        for yaml_file in PAPERS_DIR.glob("*.paper.yaml"):
            files.append(yaml_file)

    return files


def main():
    args = sys.argv[1:]
    strict = '--strict' in args
    args = [a for a in args if a != '--strict']

    # Determine which files to check
    if args:
        files = [Path(a) for a in args if Path(a).exists()]
    else:
        files = find_yaml_files()

    if not files:
        print("No YAML files found to validate.")
        print("Expected files matching: *.{type}.yaml")
        sys.exit(0)

    total_errors = 0
    total_warnings = 0
    files_by_type = {}

    for filepath in sorted(files):
        yaml_type = get_yaml_type(filepath)
        if not yaml_type:
            print(f"\n{filepath}:")
            print(f"  ⚠️  WARNING: Cannot determine type from filename (expected: name.type.yaml)")
            total_warnings += 1
            continue

        spec = load_spec(yaml_type)
        if spec is None:
            print(f"\n{filepath}:")
            print(f"  ⚠️  WARNING: No spec found for type '{yaml_type}' at schemas/{yaml_type}.spec.yaml")
            total_warnings += 1

        errors, warnings = validate_yaml_file(filepath, spec, yaml_type)

        if errors or warnings:
            print(f"\n{filepath}:")
            for err in errors:
                print(f"  ❌ ERROR: {err}")
            for warn in warnings:
                print(f"  ⚠️  WARNING: {warn}")

        total_errors += len(errors)
        total_warnings += len(warnings)

        # Track by type
        if yaml_type not in files_by_type:
            files_by_type[yaml_type] = 0
        files_by_type[yaml_type] += 1

    print(f"\n{'='*50}")
    print(f"Validated {len(files)} file(s)")
    print(f"By type: {dict(files_by_type)}")
    print(f"Errors: {total_errors}, Warnings: {total_warnings}")

    if total_errors > 0:
        sys.exit(1)
    if strict and total_warnings > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
