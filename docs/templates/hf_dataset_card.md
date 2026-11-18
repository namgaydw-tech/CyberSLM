---
pretty_name: CyberSLM Synthetic Demonstration Fixture
license: other
task_categories:
  - text-generation
language:
  - en
tags:
  - synthetic
  - cybersecurity
  - coursework
---

# CyberSLM Synthetic Demonstration Fixture

This is an unmistakably fictional, independently created demonstration dataset. It is **not**
the restricted THEIA-derived Phase 4F dataset and was not observed in a real incident.

## Dataset summary

- Total records: {{TOTAL_RECORDS}}
- Train records: {{TRAIN_RECORDS}}
- Validation records: {{VALIDATION_RECORDS}}
- Test records: {{TEST_RECORDS}}
- Manifest SHA-256: `{{MANIFEST_SHA256}}`

The fixture uses reserved example domains and documentation IP ranges. Fictional CVE numbers,
entities, alerts, and reports must not be presented as real observations.

## Schema

| Field | Meaning |
| --- | --- |
| `document_id` | Fictional source-group identifier used for group-aware splitting |
| `chunk_id` | Fictional record identifier |
| `text` | Synthetic cybersecurity-themed text |
| `source_url` | Reserved `.invalid` provenance URL |
| `publisher` | Fictional publisher |
| `report_year` | Future fictional year |
| `license` | Explicit synthetic-fixture status |
| `split` | Frozen train, validation, or test partition |

## Splitting

Records are grouped by `document_id`; no document crosses partitions. The sorted document IDs
are deterministically assigned to train, validation, and test.

## Important distinction

This publishable fixture is intended to test loading, schemas, and demonstrations. It is not
identical to the private training corpus and must not be used to claim Phase 4F model accuracy.

## Loading

```python
from datasets import load_dataset

dataset = load_dataset("{{REPO_ID}}")
```
