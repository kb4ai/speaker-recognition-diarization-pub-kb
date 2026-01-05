#!/bin/bash
#
# Clone all tracked tool repositories to tmp/
#
# Usage:
#   ./scripts/clone-all.sh           # Clone all repos
#   ./scripts/clone-all.sh --update  # Update existing clones
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TOOLS_DIR="$PROJECT_ROOT/data/tools"
TMP_DIR="$PROJECT_ROOT/tmp"

UPDATE_MODE=false
if [[ "$1" == "--update" ]]; then
    UPDATE_MODE=true
fi

# Create tmp directory if needed
mkdir -p "$TMP_DIR"

# Check for yq or python for YAML parsing
if command -v yq &> /dev/null; then
    PARSER="yq"
elif command -v python3 &> /dev/null; then
    PARSER="python"
else
    echo "Error: Requires yq or python3 to parse YAML"
    exit 1
fi

echo "Scanning for tool YAML files..."
count=0

for yaml_file in "$TOOLS_DIR"/*.tool.yaml; do
    [ -f "$yaml_file" ] || continue

    # Extract repo-url
    if [[ "$PARSER" == "yq" ]]; then
        repo_url=$(yq '.repo-url // ""' "$yaml_file")
    else
        repo_url=$(python3 -c "
import yaml, sys
with open('$yaml_file') as f:
    data = yaml.safe_load(f)
    print(data.get('repo-url', '') or '')
")
    fi

    if [[ -z "$repo_url" || "$repo_url" == "null" ]]; then
        echo "Skipping $(basename "$yaml_file"): no repo-url"
        continue
    fi

    # Extract repo name from URL
    repo_name=$(basename "$yaml_file" .tool.yaml)
    clone_dir="$TMP_DIR/$repo_name"

    if [[ -d "$clone_dir" ]]; then
        if $UPDATE_MODE; then
            echo "Updating: $repo_name"
            (cd "$clone_dir" && git pull --rebase 2>/dev/null || echo "  (pull failed, may have local changes)")
        else
            echo "Exists: $repo_name"
        fi
    else
        echo "Cloning: $repo_name"
        git clone --depth 1 "$repo_url" "$clone_dir" 2>/dev/null || echo "  (clone failed)"
    fi

    ((count++))
done

echo ""
echo "Processed $count repositories"
echo "Clones location: $TMP_DIR"
