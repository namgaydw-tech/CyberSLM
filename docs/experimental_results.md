# Experimental results — verified engineering scope

## CyberSLM-35M

- Trainable parameters: 35,205,120 (0.586% above 35,000,000).
- Architecture: 10 layers, 512 hidden width, 8 attention heads, 1,600 SwiGLU width, RoPE, RMSNorm and tied 259-token byte embeddings.
- Validated-data one-step GPU smoke: training loss 5.5979; validation loss 4.4751; peak allocated CUDA memory 724,669,952 bytes.
- Controlled fictional-fixture five-step smoke: loss 5.6249 to 3.3037; validation loss 3.2617; peak allocated CUDA memory 723,186,176 bytes.
- Checkpoint reload at step 5 and greedy generation passed.

These are engineering checks, not a trained-model result.

## Baselines on Phase 4F

| Split | Baseline | Exact match | Numeric correctness | Unsupported-claim rate |
| --- | --- | ---: | ---: | ---: |
| Validation (28) | Deterministic recomputation | 1.000 | 1.000 | 0.000 |
| Validation (28) | Majority answer by task | 0.429 | 0.429 | 0.000 |
| Cross-scenario (199) | Deterministic recomputation | 1.000 | 1.000 | 0.000 |
| Cross-scenario (199) | Majority answer by task | 0.246 | 0.467 | 0.000 |

The deterministic baseline is an oracle for calculated tasks and should be expected to outperform a language model.

## Phase 6

Isolation Forest and novelty-mode Local Outlier Factor were fitted only on training-window numerical and graph features. No verified labels were available, so no classification accuracy, precision, recall, F1 or AUROC is reported. Scores are review signals only. Large LOF scores on cross-scenario windows indicate strong distribution shift and possible density degeneracy, not attacks.
