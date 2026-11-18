# CyberSLM Design Thinking coursework progress report

Student: Namgay D Wangchuk (25BTRCL300)
Assigned by: Dr. Sujeet S. Jagtap
Programme: B.Tech CSE (Artificial Intelligence & Machine Learning)
Date: 18 November 2026

## Outcome

The project now contains a real 35.2M-parameter decoder-only CyberSLM implementation, a deterministic and independently validated Phase 4F evidence-QA pipeline, label-free anomaly/graph prototypes, evaluation baselines and runnable demonstrations. Full 35M training was deliberately not claimed because the validated QA corpus is too small and no sufficiently large licensed Stage A corpus is recorded.

## Commands executed

```powershell
$env:PYTHONPATH='src'
python scripts/audit_phase4e.py --source <private-source> --windows <phase4d-manifest> --qa-dir <phase4e-dir> --output <audit.json>
python scripts/build_phase4f.py --source <private-source> --output <phase4f-dir>
python scripts/validate_phase4f.py --source <private-source> --dataset-dir <phase4f-dir>
python scripts/train_35m.py --dataset-dir <phase4f-dir> --check-only
python scripts/train_35m.py --training-config configs/training/cyberslm_35m_smoke.yaml --smoke-steps 5
python scripts/run_phase6_algorithms.py --source <private-source> --windows <phase4f-windows> --output-dir <private-output>
python scripts/evaluate_phase7.py --dataset-dir <phase4f-dir> --output <private-report.json>
python scripts/run_demo.py
```

## Next executable action

Record licensing and authorization for a diverse Stage A text corpus, freeze a new final held-out group, obtain human review of Phase 4F samples, then launch a time-bounded 35M Stage A pilot followed by a separately logged Stage B evidence-conditioned run. Do not tune on the current cross-scenario split.
