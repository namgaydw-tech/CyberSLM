# Contributor guidance

- Treat all non-fixture data as private unless its license and release authorization are recorded.
- Never commit raw reports, production Parquet files, secrets, checkpoints, or experiment binaries.
- Synthetic fixtures must be unmistakably fictional and must not be presented as observations.
- Preserve source identifiers and provenance; group splits by `document_id`.
- Train tokenizers on the training split only.
- Run `make check` (or the PowerShell equivalents in the README) before a pull request.
- Do not start large training jobs automatically. Use the tiny configuration for CI.
- Record dataset, tokenizer, configuration, seed, and code revision for research runs.

