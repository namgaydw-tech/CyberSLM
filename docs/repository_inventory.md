# Repository inventory — 18 November 2026

## Present and verified

- Tracked package under `src/cyberslm`: generic decoder-only Transformer, BPE tokenizer, data loading, training, checkpointing, inference and evaluation utilities.
- Historical phase implementations: Phase 1 RoPE pilot, Phase 2 split evaluation, Phase 3 4M training and Phase 4A–4C provenance/QA pilot.
- Original THEIA standardized event source: 68,027 unique event UUID rows, stored only in ignored local artifact directories.
- Phase 4D and 4E Parquet outputs and notebook cells. Their generator logic was not previously available as reusable scripts.
- Phase 3 4M best checkpoint and recorded evaluation JSON.

## Added in this implementation pass

- Reusable byte tokenizer and answer-only masking dataset.
- Phase 4E diversity audit, Phase 4F deterministic generator, receipts and independent validator.
- CyberSLM-35M public configuration and GPU-ready training launcher.
- Isolation Forest, novelty-mode LOF and graph-feature evidence pipeline.
- Deterministic and majority-answer evaluation framework.
- CLI and Streamlit demonstrations.

## Reproducibility gaps and unresolved items

- Non-fixture source data have no recorded public redistribution authorization and remain excluded from Git.
- The Phase 4F source-session split cannot be source-identity exclusive because one source identity accounts for most 3.11 rows. This overlap is reported, not hidden.
- The existing cross-scenario set is development-exposed and should not be called a pristine final test set.
- No sufficiently large licensed Stage A corpus is recorded for meaningful 35M pretraining.
- No completed CyberSLM-35M training run or evidence-QA model evaluation exists. Only bounded engineering smoke runs were performed.
- Phase 3 4M loss is a next-byte metric on a different objective and is not directly comparable with Phase 4F answer accuracy.

## Git state at takeover

The outer checkout was on `feat/initial-infrastructure` with modified `README.md` and `.gitignore` plus numerous untracked phase directories. A nested `CyberSLM/.git` checkout also existed. No existing work was deleted, reset, committed, pushed or merged.
