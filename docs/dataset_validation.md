# Phase 4F dataset validation

Dataset version: `phase4f_evidence_qa_v2`

The local generated dataset contains 1,494 natural source/time windows: 546 train, 136 validation, 29 cross-scenario and 783 quarantine windows. It accounts for all 68,027 source event UUIDs exactly once.

After weighting, length checks, exact pair deduplication and answer-frequency controls, the retained QA splits contain 388 train, 28 validation and 199 cross-scenario examples. The independent validator recomputed all 615 answers and answer-only masks successfully.

Key checks:

- Source SHA-256 matches the manifest.
- Zero event UUID overlap between evaluation partitions.
- Zero duplicate QA hashes or prompt-answer pairs.
- Zero empty answers and zero sequences above 512 byte tokens.
- Zero answer-frequency-limit violations.
- Zero unsupported positive attack labels.
- Mean window event-type entropy improved from 0.081 to 0.401 bits.
- Single-type windows decreased from 92.40% to 58.90%.

Limitations: source-identity overlap remains because one 3.11 source dominates the corpus; source annotations are not verified verdicts; temporal source sessions are not proven incident boundaries; the cross-scenario set has influenced development.

Generated Parquet, manifests and validation JSON are stored under the ignored local directory `CyberSLM_Phase4_Evidence/runs/phase4f_evidence_qa_v2/` and must not be committed without authorization.
