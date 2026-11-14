# Research methodology

## Pretraining

The initial objective is causal next-token prediction from random weights. Configuration files and tokenizer fingerprints make experiments comparable. Candidate 30M/50M/100M labels are starting budgets, not established final sizes. Selection depends on permitted training tokens, hardware, scaling behavior, and controlled ablations.

## Evaluation

Implemented metrics are validation loss, perplexity, parameter count, batch latency, and CUDA peak allocated memory. Future benchmark records support prompt, reference answer, category, evidence URLs, and publication year. QA accuracy, unsupported-claim rate, source-grounding consistency, temporal knowledge, and user-level explanation quality must only be reported when reviewed ground truth and scoring procedures exist.

Planned ablations include model budget, context length, cybersecurity-adapted versus general-purpose tokenization, corpus categories, temporal cutoff, and retrieval. Report seeds and uncertainty; avoid selecting on the test set.

## Future modules

- Temporal grounding: retrieve evidence using an `as_of_year` cutoff and test knowledge freshness.
- User adaptation: render beginner, student, and professional explanations from the same factual evidence; evaluate factual invariance separately from style.
- Tokenization: compare fragmentation, tokens per identifier, fertility, throughput, and unknown rate on CVEs, domains, hashes, logs, and network identifiers.
- Retrieval: implement the `EvidenceRetriever` contract using reviewed, licensed, trusted sources and surface citations.
- Supervised fine-tuning: add only after instruction sources, licenses, safety scope, and leakage controls are approved.

## Current limitations

No real dataset has been integrated. No large model has been trained. The code is single-process and does not yet support DDP/FSDP, memory-mapped token caches, sharded checkpoints, or mature safety evaluation. The fictional fixture establishes plumbing correctness, not cybersecurity capability.

