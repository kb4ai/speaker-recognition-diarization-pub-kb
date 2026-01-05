.PHONY: all validate tables clone sources clean help

# Default target
all: validate tables

# Validate all YAML files against schemas
validate:
	@echo "Validating YAML files..."
	./scripts/check-yaml.py

# Generate comparison tables
tables:
	@echo "Generating comparison tables..."
	./scripts/generate-tables.py > comparisons/auto-generated.md
	@echo "Generated: comparisons/auto-generated.md"

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

# Install Python dependencies
install:
	pip install -r scripts/requirements.txt

# Show help
help:
	@echo "Available targets:"
	@echo "  make validate  - Validate all YAML files"
	@echo "  make tables    - Generate comparison tables"
	@echo "  make clone     - Clone all tracked repositories"
	@echo "  make sources   - Verify source URLs"
	@echo "  make clean     - Remove generated files"
	@echo "  make install   - Install Python dependencies"
	@echo "  make all       - Run validate + tables"
