#!/bin/bash
#
# Clone all tracked tool repositories to tmp/
#
# Reads repo-url and optional repo-commit from tool YAML files.
# If repo-commit is specified, checks out that specific commit.
#
# Usage:
#   ./scripts/clone-all.sh              # Clone all repos
#   ./scripts/clone-all.sh --update     # Update existing clones
#   ./scripts/clone-all.sh --pinned     # Clone and checkout pinned commits
#   ./scripts/clone-all.sh --full       # Full clone (not shallow)
#   ./scripts/clone-all.sh --tool NAME  # Clone specific tool only
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TOOLS_DIR="$PROJECT_ROOT/data/tools"
TMP_DIR="$PROJECT_ROOT/tmp"

UPDATE_MODE=false
PINNED_MODE=false
FULL_CLONE=false
TOOL_FILTER=""

show_help() {
    cat << EOF
Clone all tracked tool repositories to tmp/

Usage: $0 [OPTIONS]

Options:
  --update     Update existing clones (fetch + pull/checkout)
  --pinned     Clone and checkout repo-commit from YAML files
  --full       Full clone instead of shallow (--depth 1)
  --tool NAME  Only process tools matching NAME
  --help       Show this help message

Examples:
  $0                        # Shallow clone all repos
  $0 --pinned               # Clone and checkout pinned commits
  $0 --tool pyannote        # Clone only pyannote repos
  $0 --update --pinned      # Update and checkout pinned commits

The script reads data/tools/*.tool.yaml files and extracts:
  - repo-url: Repository URL to clone
  - repo-commit: Optional commit hash to checkout (with --pinned)
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --update)
            UPDATE_MODE=true
            shift
            ;;
        --pinned)
            PINNED_MODE=true
            FULL_CLONE=true  # Need full clone to checkout specific commits
            shift
            ;;
        --full)
            FULL_CLONE=true
            shift
            ;;
        --tool)
            TOOL_FILTER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

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

    # Extract repo name from filename
    repo_name=$(basename "$yaml_file" .tool.yaml)

    # Apply tool filter if specified
    if [[ -n "$TOOL_FILTER" && "$repo_name" != *"$TOOL_FILTER"* ]]; then
        continue
    fi

    # Extract repo-url and repo-commit
    if [[ "$PARSER" == "yq" ]]; then
        repo_url=$(yq '.repo-url // ""' "$yaml_file")
        repo_commit=$(yq '.repo-commit // ""' "$yaml_file")
    else
        read -r repo_url repo_commit <<< $(python3 -c "
import yaml, sys
with open('$yaml_file') as f:
    data = yaml.safe_load(f)
    url = data.get('repo-url', '') or ''
    commit = data.get('repo-commit', '') or ''
    print(url, commit)
")
    fi

    if [[ -z "$repo_url" || "$repo_url" == "null" ]]; then
        echo "Skipping $repo_name: no repo-url"
        continue
    fi

    clone_dir="$TMP_DIR/$repo_name"

    if [[ -d "$clone_dir" ]]; then
        if $UPDATE_MODE; then
            echo "Updating: $repo_name"
            (cd "$clone_dir" && git fetch origin 2>/dev/null)
            if $PINNED_MODE && [[ -n "$repo_commit" && "$repo_commit" != "null" ]]; then
                echo "  Checking out pinned commit: $repo_commit"
                (cd "$clone_dir" && git checkout "$repo_commit" 2>/dev/null || echo "  (checkout failed)")
            else
                (cd "$clone_dir" && git pull --rebase 2>/dev/null || echo "  (pull failed)")
            fi
        else
            echo "Exists: $repo_name"
            # Show current commit
            current=$(cd "$clone_dir" && git rev-parse --short HEAD 2>/dev/null)
            if [[ -n "$repo_commit" && "$repo_commit" != "null" ]]; then
                echo "  Current: $current (pinned: $repo_commit)"
            else
                echo "  Current: $current"
            fi
        fi
    else
        echo "Cloning: $repo_name"
        if $FULL_CLONE; then
            git clone "$repo_url" "$clone_dir" 2>/dev/null || { echo "  (clone failed)"; continue; }
        else
            git clone --depth 1 "$repo_url" "$clone_dir" 2>/dev/null || { echo "  (clone failed)"; continue; }
        fi

        # Checkout pinned commit if specified
        if $PINNED_MODE && [[ -n "$repo_commit" && "$repo_commit" != "null" ]]; then
            echo "  Checking out pinned commit: $repo_commit"
            (cd "$clone_dir" && git checkout "$repo_commit" 2>/dev/null || echo "  (checkout failed)")
        fi
    fi

    ((count++))
done

echo ""
echo "====================================="
echo "Processed $count repositories"
echo "Clones location: $TMP_DIR"
echo ""
echo "Options used:"
$UPDATE_MODE && echo "  - Update mode (fetch/pull)"
$PINNED_MODE && echo "  - Pinned mode (checkout repo-commit)"
$FULL_CLONE && echo "  - Full clone (not shallow)"
[[ -n "$TOOL_FILTER" ]] && echo "  - Tool filter: $TOOL_FILTER"
echo ""
echo "Useful commands:"
echo "  cd $TMP_DIR/<repo> && git log -1  # Check current commit"
echo "  cd $TMP_DIR/<repo> && git pull    # Update to latest"
