# CyberSLM — Evidence-Grounded Cybersecurity Small Language Model

**Research prototype | Scratch-trained decoder-only Transformer | Reproducible data audits | Work in progress**

CyberSLM explores whether a compact, locally trainable language model can read structured cybersecurity evidence, answer constrained questions accurately, and explicitly distinguish **observations**, **computed facts**, **anomaly candidates**, and **verified incident conclusions**. The work progresses from next-byte report modeling toward evidence-conditioned question answering with source-event provenance and auditable evaluation.

> **Research status (17 November 2026):** Phase 3 baseline training and Phase 4A–4E dataset construction are complete to the extent described below. Phase 4D passed its aggregate-answer quality gate; Phase 4E's source-diversity/duplicate-question audit and an independently corroborated incident-level evaluation are pending. **CyberSLM has not been validated as an APT detector, nor has it demonstrated reliable autonomous threat attribution.**

## 1. Research goals and questions

1. Can a ~4M-parameter Transformer learn to answer evidence-conditioned questions on a consumer GPU, rather than simply imitate templated investigative prose?
2. How do source repetition, overlapping entity/graph relationships, and split design affect loss and apparent generalization?
3. Can deterministic evidence checks reduce unsupported numerical or security claims in generated responses?
4. Does a small model generalize to **event-independent, investigation-independent, cross-scenario** evidence once adequate source diversity and defensible evaluation splits exist?
5. How do accuracy, calibration/abstention behavior, compute cost, and rule-based baselines compare? (Planned, not yet established.)

### Scope and non-claims

This is an **experimental LM and data-engineering pipeline**, not a deployed intrusion-detection service. The current supervised QA tasks involve event counts, simple arithmetic, and graph summaries. Neither low language-model loss nor accurate arithmetic proves malicious-activity classification, attack causality, or cybersecurity reasoning. Source annotation `ATTACK_CANDIDATE` is not treated as an independently confirmed APT verdict. Qwen-generated reports are not treated as ground truth merely because they contain a field named `verified_facts`.

## 2. Model design (implemented baseline)

| Component | CyberSLM-4M Phase 3 |
| --- | --- |
| Model | Custom **decoder-only Transformer**, trained from random initialization |
| Parameters | **4,058,496** (reported for the tested configuration) |
| Layers / hidden width / heads | 4 / 288 / 6 |
| Feed-forward network | SwiGLU, intermediate width 768 |
| Positional representation | Rotary Position Embeddings (RoPE) |
| Attention | Causal/masked self-attention |
| Normalization | RMSNorm |
| Embeddings | Tied input/output embeddings |
| Tokenizer | **Byte-level**, 259 IDs: PAD=0, BOS=1, EOS=2, byte IDs=3–258 |
| Phase 3 context | 256 byte tokens |
| Phase 3 optimization | Physical batch 4, gradient accumulation 4, masked padding, CUDA |
| Hardware used | NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB VRAM |

**Tokenizer clarification:** This implementation is byte-level, **not BPE/Unigram tokenization**. The Phase 4C pilot preflight uses a 512-byte-token context and physical batch size 2; it is a distinct experimental configuration, **not evidence that Phase 4E has been trained**.

```text
Original THEIA event rows (UUID, time, event type, source, destination)
              |
        Event lineage / QA audits
              |
  Traceable windows + computed evidence aggregates
              |
       Prompt + question + target answer
              |
   Byte IDs -> RoPE decoder-only Transformer
              |
       Answer-only loss (Phase 4 pilot design)
              |
   Deterministic answer checker / independent evaluation
```

## 3. Data sources and provenance

- **APT-derived Qwen checkpoint reports:** An initial language-modeling corpus of **1,639 distinct report texts** (~2,076,583 UTF-8 byte-level text tokens). This corpus was heavily skewed toward `theia_3.11_success` (~96% of the reports). Generated text and purported verified facts were marked unreviewed; they are not independent labels.
- **Original THEIA event evidence:** `theia_standardized.parquet`, **68,027 event rows with 68,027 unique `event_uuid` values** in the inspected artifact. Columns include `event_type`, `event_time_seconds`, `source_entity`, `destination_entity`, `attack_chain`, `original_label`, and `annotation_status`. Other local processed THEIA Parquet files represent alternate views of overlapping event records, **not extra independent data**.
- **Other candidate data:** CADETS/other source artifacts and a CyberPal paper text sample were investigated, but their use as verified, independent training/test ground truth has **not** been established.

**Sharing restrictions:** Raw/processed event data, privately duplicated datasets, Qwen reports, private research notes, and model checkpoints are intentionally **excluded from this code repository** unless redistribution rights and content review are verified. The source-data repository revision, file SHA-256 receipts and schema descriptions should be retained privately. Public provenance documentation should not imply that private data is publicly reproducible.

## 4. Experimental milestones and observed results

### Phase 1–2 — Infrastructure and small-pilot checks

Developed initial data loading, byte tokenization, Transformer implementation, masked next-token training, local GPU execution, checkpointing and small-scale corpus evaluation. Small report splits were used only to debug infrastructure; early scores are not comparable with later data partitions.

### Phase 3 — Expanded report-language-model baseline (**trained**)

Frozen report split: **1,327 train**, **227 validation**, **57 cross-scenario test**, **28 quarantine** reports. A **300-step** CyberSLM-4M CUDA run completed. Reported outcomes:

| Evaluation dataset | Next-byte loss | Byte-level perplexity |
| --- | ---: | ---: |
| Same-scenario validation | **0.384122** | **1.468324** |
| THEIA 3.3 cross-scenario reports | **1.407338** | **4.085065** |
| External PDF-extracted text | **3.413438** | **30.369479** |

These are **next-byte prediction metrics**, not incident-detection accuracy. Validation likely benefits from repeated report structure and recurring source entities. PDF text extraction can introduce layout/order errors.

**Split audits:** Zero raw exact-text hashes shared across report splits; **1 normalized-text exact match** identified; validation showed strong template similarity. In a dataset-wide identifier audit, **227/227 validation** and **41/57 cross-scenario reports** shared structured identifiers with training reports. The 15 highest-priority cross-scenario pair comparisons had **261 shared-identifier occurrences**, all at `entity_relationships[].source` in the compared records. These identify recurring entities, **not verified duplicate events**. Exact directed `(source, destination)` edge overlap between the Phase 3 training and cross-scenario report summaries was **0**, with 100% relationship extraction coverage in the examined records. Original-event lineage for those specific Qwen batches remains unresolved.

### Phase 4A — Original-event provenance inspection (**performed**)

Inspected source event Parquet schemas and identified THEIA's `event_uuid`, timestamps and directed source/destination relationships. The original event table provides a stronger basis for validating computed facts, but mappings from **individual Qwen checkpoint batches** to their exact underlying event UUID lists have **not** yet been demonstrated.

### Phase 4B–4C — Initial evidence QA and answer-only pilot (**constructed / preflight only**)

Created a small QA pilot: **75 train**, **6 validation**, **69 cross-scenario test** questions, originating from **25 / 2 / 23 distinct source reports** respectively. Answers were derived from **unverified reported aggregates**; `training_approved=False` and human review pending. The Phase 4C `--check-only` preflight succeeded with zero overlength examples and **4,465 / 360 / 4,134 supervised answer-byte targets** (train/validation/test). **The `--check-only` result is not a trained model or a test score.**

### Phase 4D — Raw-event aggregate QA (**built and independently recalculated**)

Created **1,105 non-overlapping UUID windows** spanning **68,027 original events**. Excluding quarantined windows, generated **6,588 QA examples** across 6 tasks:

| Split | Event windows | QA examples | Distinct answers |
| --- | ---: | ---: | ---: |
| Train | 847 | **5,082** | 489 |
| Validation | 211 | **1,266** | 350 |
| Cross-scenario test | 40 | **240** | 179 |
| Quarantine | 7 | Not used for training | — |

**Recorded quality gate:** **6,588/6,588** QA answers matched independent recomputation from indexed original event records; **0 audit errors**; **0 cross-split event-UUID overlaps**. Train–validation did share **8,438 directed graph edges**; train–test shared **0 directed edges**. The split does **not** establish independent attack investigations or confirm source labels. Large portions of the chronological event windows had near-zero time spans, making that QA task overly repetitive.

### Phase 4E — Calculated evidence QA (**generated, under audit**)

Replaced copy-the-total-style targets with computed values: dominant event-type percentage, remaining event count, number of event types, absolute difference of source/destination entity counts, least-common event-type percentage, and the gap between the top two event types.

| Split | QA examples | Source windows | Distinct answers | Task families |
| --- | ---: | ---: | ---: | ---: |
| Train | **4,275** | 847 | 197 | 6 across the split |
| Validation | **1,058** | 211 | 81 | 6 across the split |
| Cross-scenario test | **237** | 40 | 131 | 6 across the split |
| **Total** | **5,570** | **1,098** | — | — |

No overlength examples were skipped by the generator. **Critical limitation:** Only **40/847 training windows** had at least two event types; **807** were single-type. The three most frequent training answers appeared **3,222/4,275 (75.4%)** times. Examples include `EVENT_CONNECT accounts for 100.0% of events.` (**1,610** occurrences), `There are 1 distinct event types.` (**807**), and `0 events are not EVENT_CONNECT.` (**805**). Single-type windows can cause identical prompt/answer pairs to be generated under both *most common* and *least common* task labels. The QA hash includes `qa_type`, so it does not by itself eliminate this duplication.

**Phase 4E has not yet passed a final duplicate/answer-diversity audit, and it has not been used for a confirmed successful 4M training experiment.**

## 5. Algorithms and validation: implemented versus planned

| Method | Status | Purpose / validation |
| --- | --- | --- |
| Byte tokenizer + causal Transformer + RoPE + SwiGLU + RMSNorm | **Implemented** | Local scratch-training baseline; verify parameter count, masking, checkpoints and next-byte loss |
| Gradient accumulation and padding-mask-aware loss | **Implemented** | Fit training on 6 GB GPU; confirm effective supervised token counts |
| SHA-256 source receipts / source-family split manifest | **Implemented for reported experiments** | Track artifact identity and frozen partitions |
| Exact/normalized text, TF-IDF similarity, entity-ID overlap audits | **Performed** | Discover report-template repetition and candidate cross-split dependencies |
| Graph edge sets and UUID-level overlap checks | **Performed** | Quantify relationships shared between evaluation partitions |
| Deterministic recomputation of QA aggregates | **Phase 4D verified; Phase 4E pending final audit** | Use exact counts and event-row provenance as the correctness oracle |
| Answer-only loss masking | **Phase 4C pilot design/preflight** | Exclude input prompt bytes and padding from response supervision |
| Provenance-aware incident/session grouping and embargoed time splits | **Planned** | Reduce evaluation dependence across related events and windows |
| Sample weighting, duplicate-prompt elimination, stronger QA diversity gates | **Planned Phase 4F** | Reduce dominant-answer shortcuts without manufacturing fake source diversity |
| Multi-window temporal/graph comparison tasks | **Planned** | Test evidence integration beyond copying explicit aggregate fields |
| Isolation Forest / Local Outlier Factor on suitable numeric features | **Research candidates, not trained/evaluated here** | Unsupervised anomaly screening; must not be reported as confirmed APT classification |
| Graph models such as GraphSAGE / graph features | **Research candidates, not trained/evaluated here** | Investigate neighborhood/context features on appropriately constructed graphs |
| Confidence calibration, abstention / unsupported-claim scoring | **Planned** | Quantify uncertainty and unsupported statements |

For percentages, the deterministic baseline is `100 * event_type_count / total_events`; for frequency gaps, `largest_count - second_largest_count`; for graph overlap, set intersection on canonical directed edges. These computations are exact baselines and should remain part of any reported model comparison.

## 6. Known limitations and risk controls

1. **Source imbalance:** THEIA 3.11 and `EVENT_CONNECT` heavily dominate portions of the available data. Deduplication alone cannot create authentic variation in incident types.
2. **Target repetition:** Phase 4D durations and Phase 4E arithmetic targets are concentrated on trivial repeated values; Phase 4E generator may produce same prompt/answer under different task labels.
3. **Evaluation dependence:** Zero shared event UUIDs does not mean unrelated investigations. Phase 4D train/validation share 8,438 edges. Scenario-specific held-out evaluation is necessary but not sufficient.
4. **Incomplete original-event attribution:** Although source events are available, there is no established event-UUID-to-Qwen-batch mapping for every generated checkpoint report.
5. **Weak labels:** `ATTACK_CANDIDATE` and generated text are not verified benign/malicious ground truth. No validated detection precision/recall/F1/AUROC is claimed.
6. **Small-model and compute constraints:** 4M parameters, byte-level sequences and a 6 GB GPU restrict representational capacity and context. Scaling to 15M/35M is deferred until dataset validity and baseline generalization improve.
7. **Privacy and redistribution:** Some datasets and artifact paths are private or third-party derived; permission/license checks precede any public data release.
8. **Document-derived text:** Extracted PDF text may contain multi-column ordering artifacts and is only an exploratory language-model generalization probe.

## 7. Next milestones (proposed sequence)

**4E audit (immediate):** Inspect raw event-type distribution by case, single-type window rates, duplicate `(split, window, prompt, answer)` pairs, entity/edge reuse, and answer concentration. Do **not** push a new training result before this audit is finalized.

**4F — Evidence dataset v2 (planned):** Version a corrected generator with no duplicate answer tasks for single-type windows; preserve authentic causal/temporal groups; add reliable multi-event windows where available; introduce numerical, temporal and graph comparisons. Compare the model against deterministic rules and answer-frequency baselines. Require separate test questions and do not tune on the cross-scenario test set.

**4G — Controlled 4M pilot (planned):** Train a fresh scratch model or an explicitly labeled controlled continuation; use answer-only masking; freeze tokenizer/config/seeds/splits; save training receipts, number of effective target tokens, convergence curves, QA exact-match/numerical-error, unseen-window and cross-scenario scores; perform per-task and per-family reporting. Avoid presenting a 300-step next-byte LM run as an evidence-reasoning result.

**4H — Anomaly + graph evidence (research candidate):** Only after feature/label provenance is established, explore Isolation Forest, LOF and graph-neighborhood evidence, with case-independent evaluation and manually assessed alerts. Keep anomaly scores distinct from confirmed attack labels.

**4I — Robustness and reporting (planned):** Multiple seeds, confidence intervals with grouping by investigation, error taxonomy, counterfactual/perturbed evidence checks, calibration and abstention, latency/VRAM, ablations (no evidence / shuffled evidence / altered numbers / different group splits). Clearly label unvalidated findings.

## 8. Local reproduction and project layout

The source code is developed on Windows under:

```text
D:\Design thinking Cybersecurity SLM\
├── .venv/                           # Local only; not in Git
├── CyberSLM_Phase3_Expanded/
│   └── cyberslm_phase3_train.py    # Phase 3 training implementation
├── CyberSLM_Phase4_Evidence/
│   ├── phase4a_provenance.py
│   ├── phase4b_build_qa.py
│   ├── phase4c_train_pilot.py
│   └── runs/                        # Excluded: local data/artifacts
└── data/                            # Excluded: private/restricted source data
```

*Paths are documented from local experiment logs. Individual scripts are only available in GitHub after they are deliberately copied, reviewed and committed.* Generated QA data, large raw event sources and checkpoints are **not** bundled in the public code distribution.

With the project's existing `.venv` activated:

```powershell
cd "D:\Design thinking Cybersecurity SLM"
.\.venv\Scripts\Activate.ps1
python -m jupyter lab
```

Phase 4C **preflight only** (does not train or score test):

```powershell
python .\CyberSLM_Phase4_Evidence\phase4c_train_pilot.py --root "D:\Design thinking Cybersecurity SLM" --pilot-unreviewed --check-only
```

The Phase 4D/4E dataset-generating and auditing routines currently depend on local Jupyter cells and restricted source data; future work is to refactor them into documented, deterministic CLI scripts and test fixtures before claiming full end-to-end public reproducibility.

## 9. Experiment evidence / publication checklist

Before claiming a new result, record: source license and exact revision, input file hashes, Python/PyTorch/CUDA versions, dataset row/window/group counts, configuration, RNG seeds, exclusion criteria, split-group definitions, duplicate/edge-overlap results, training token exposures, held-out metrics per task and scenario, baseline comparisons, uncertainty estimates, limitations, and any human-review protocol.

**Research integrity rule:** Successful script execution (`returncode=0`), low perplexity, or correct arithmetic on self-generated prompts is not evidence of independently verified attack detection.

## 10. Status and attribution

Project maintained by **Namgay D Wangchuk**. This repository documents experimental engineering, local data processing and research validation. Any collaboration, dataset ownership, institutional affiliation or third-party code should be credited in the relevant files once confirmed. **No license is asserted for third-party source data or derived reports.**

**Last research-status update:** 17 November 2026. Subsequent experiments should be added as dated entries with their own artifacts and limitations instead of overwriting earlier results.
