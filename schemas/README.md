# Schema Specifications

YAML schema definitions for each entry type in the knowledge base.

## Available Schemas

| Schema | Purpose | File Pattern |
|--------|---------|--------------|
| [tool.spec.yaml](tool.spec.yaml) | Open-source tools | `data/tools/{owner}--{repo}.tool.yaml` |
| [algorithm.spec.yaml](algorithm.spec.yaml) | Algorithms & techniques | `data/algorithms/{cat}/{name}.algorithm.yaml` |
| [model.spec.yaml](model.spec.yaml) | Pre-trained models | `data/models/{cat}/{provider}--{id}.model.yaml` |
| [dataset.spec.yaml](dataset.spec.yaml) | Datasets & corpora | `data/datasets/{name}.dataset.yaml` |
| [paper.spec.yaml](paper.spec.yaml) | Research papers | `papers/{key}-{year}.paper.yaml` |
| [source.spec.yaml](source.spec.yaml) | Source citations | Embedded in all types |

## Schema Structure

Each spec file defines:

* **required** - Fields that must be present
* **fields** - All available fields with types and descriptions
* **enums** - Valid values for categorical fields

## Using Schemas

### Validation

```bash
./scripts/check-yaml.py data/tools/my-tool.tool.yaml
```

### Reference

```bash
# View tool schema
cat schemas/tool.spec.yaml

# List required fields
yq '.required' schemas/tool.spec.yaml
```

## Common Patterns

### Source Citations

All entry types support the `sources` array:

```yaml
sources:
  - url: "https://..."
    accessed: "2026-01-06"
    source-type: primary
```

### Timestamps

All entries require `last-update: "YYYY-MM-DD"`

---

*Last updated: 2026-01-06*