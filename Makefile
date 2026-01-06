.PHONY: all validate tables readme clone sources clean help bib citations stats

# Default target - full regeneration
all: validate tables readme

# Validate all YAML files against schemas
validate:
	@echo "Validating YAML files..."
	./scripts/check-yaml.py

# Generate comparison tables
tables:
	@echo "Generating comparison tables..."
	./scripts/generate-tables.py > comparisons/auto-generated.md
	@echo "Generated: comparisons/auto-generated.md"

# Generate per-directory README files
readme:
	@echo "Generating README files..."
	./scripts/generate-readme.py
	@echo "Generated all README files"

# Clone all tracked repositories to tmp/
clone:
	@echo "Cloning repositories..."
	./scripts/clone-all.sh

# Verify source URLs are accessible
sources:
	@echo "Verifying source URLs..."
	./scripts/verify-sources.py

# Clean generated files and cloned repos
clean:
	rm -rf tmp/*
	rm -f comparisons/auto-generated.md
	rm -f data/*/README.md
	rm -f papers/README.md
	rm -f schemas/README.md
	rm -f knowledge/README.md
	rm -f printouts/README.md
	rm -f ramblings/README.md
	rm -f sources/README.md

# Generate BibTeX from paper entries
bib:
	@echo "Generating BibTeX bibliography..."
	./scripts/generate-bib.py > archives/bibliography/papers-auto.bib
	@echo "Generated: archives/bibliography/papers-auto.bib"

# Extract and analyze citations from markdown
citations:
	@echo "Analyzing citations in knowledge articles..."
	./scripts/extract-citations.py

# Check citations against papers (fail if unmatched)
citations-verify:
	@echo "Verifying all citations have paper entries..."
	./scripts/extract-citations.py --verify

# Update GitHub stats for tools (dry run)
stats:
	@echo "Checking GitHub stats (dry run)..."
	./scripts/update-stats.py

# Update GitHub stats and save to YAML files
stats-update:
	@echo "Updating GitHub stats..."
	./scripts/update-stats.py --update

# Install Python dependencies
install:
	pip install -r scripts/requirements.txt

# Show help
help:
	@echo "Available targets:"
	@echo ""
	@echo "  Core:"
	@echo "    make validate      - Validate all YAML files"
	@echo "    make tables        - Generate comparison tables"
	@echo "    make readme        - Generate per-directory README files"
	@echo "    make all           - Run validate + tables + readme"
	@echo ""
	@echo "  Bibliography:"
	@echo "    make bib           - Generate BibTeX from paper entries"
	@echo "    make citations     - Analyze citations in knowledge articles"
	@echo "    make citations-verify - Verify citations have paper entries"
	@echo ""
	@echo "  External:"
	@echo "    make clone         - Clone all tracked repositories"
	@echo "    make sources       - Verify source URLs"
	@echo "    make stats         - Check GitHub stats (dry run)"
	@echo "    make stats-update  - Update GitHub stats in YAML files"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make clean         - Remove generated files"
	@echo "    make install       - Install Python dependencies"
