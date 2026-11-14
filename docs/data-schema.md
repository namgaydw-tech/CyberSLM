# Proposed Parquet schema

One row represents a text chunk. Chunking must retain a stable document identity so all chunks from one source remain in the same split.

| Column | Type | Required | Purpose |
|---|---|---:|---|
| `text` | UTF-8 string | yes | Training text after reversible, documented extraction |
| `document_id` | string | yes | Stable identity used for leakage-safe grouping |
| `chunk_id` | string | recommended | Unique chunk identity within the corpus |
| `source_url` | string | recommended | Original authoritative location |
| `publisher` | string | recommended | Source organization or author |
| `report_year` | int32 | recommended | Publication year for temporal evaluation |
| `published_at` | timestamp/date | optional | More precise temporal cutoff |
| `title` | string | optional | Source title |
| `license` | string | recommended | Content license or usage status |
| `content_hash` | string | recommended | Deduplication and integrity fingerprint |
| `extraction_method` | string | optional | PDF/HTML/CSV extraction provenance |
| `language` | string | optional | Language tag |

Minimum acceptance requires non-null `text` and `document_id`. Before a production run, also require explicit license/permission and reliable provenance. A publisher, report year, or URL may be unavailable, but missingness must be reported rather than silently invented.

Split policy:

1. Deduplicate/near-deduplicate before splitting while retaining an audit manifest.
2. Assign whole `document_id` groups, never individual chunks.
3. Use deterministic hash assignment for ordinary experiments.
4. For freshness research, reserve later publication years for validation/test.
5. Fit tokenizer and preprocessing statistics only on training rows.
6. Store split manifests separately; never mutate raw files.

