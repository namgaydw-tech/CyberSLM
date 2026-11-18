# Remaining limitations

1. Phase 4F is computationally consistent but not independently incident-labelled.
2. The QA corpus is only 615 retained examples and remains too small for meaningful 35M training.
3. One source identity dominates the 3.11 scenario, preventing useful source-exclusive train/validation partitions.
4. Cross-scenario data were inspected during development; a new frozen final set is needed.
5. Phase 6 anomaly models lack verified labels and cannot support detection-performance claims.
6. The 35M model has passed smoke tests only. A licensed Stage A corpus, full training budget and frozen evaluation protocol are still required.
7. GPU memory figures are PyTorch peak allocated bytes, not total system/GPU-board consumption.
8. CUDA memory-efficient attention emitted a nondeterminism warning during backward execution; exact bitwise reproducibility is not claimed.
9. Local raw data, Parquet outputs and checkpoints are private/ignored and must not be published without authorization.
