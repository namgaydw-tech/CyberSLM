# Architecture

## CyberSLM-35M coursework configuration

The verified `configs/model/cyberslm_35m.yaml` configuration uses 10 decoder blocks,
hidden width 512, 8 attention heads, SwiGLU width 1,600, a 259-token UTF-8 byte
vocabulary and a 512-token context. Input/output embeddings are tied. The resulting
trainable parameter count is 35,205,120. RoPE, RMSNorm, causal scaled-dot-product
attention, residual connections, gradient checkpointing and answer-only `-100` loss
masking are implemented locally; no pretrained model weights are downloaded.

The model supports PyTorch SDPA and an eager reference attention path. CUDA smoke tests
used SDPA. CUDA memory-efficient backward reported nondeterministic-kernel behavior, so
seed recording does not imply bitwise-identical GPU results.

The data path is intentionally staged and non-destructive:

```text
immutable source Parquet
  -> schema and quality scan (Arrow record batches)
  -> document-level split manifest
  -> training-only BPE tokenizer
  -> packed next-token blocks
  -> decoder-only Transformer
  -> JSONL metrics + compatible checkpoints
  -> controlled validation and generation
```

`ParquetCorpus` creates a lazy PyArrow dataset and yields bounded record batches. Profiling counts rows, nulls, empty/invalid text, exact duplicate text hashes, documents, missing supported fields, and unexpected fields. It does not rewrite sources.

The model applies token embeddings followed by pre-normalized Transformer blocks. Each block uses rotary position embeddings in causal self-attention, RMSNorm, and a SwiGLU feed-forward network. The projection may share embedding weights. Configuration controls all dimensions; grouped-query attention is supported through `n_kv_heads`.

The training path uses shifted next-token labels, AdamW, warmup/cosine scheduling, gradient accumulation and clipping, optional CUDA autocast, finite loss/gradient checks, JSONL metrics, and checkpoint compatibility validation. Checkpoints contain model, optimizer, scheduler, state, model configuration, and tokenizer fingerprint.

Retrieval and benchmark modules are contracts only. Future implementations must return attributable evidence and use datasets with suitable ground truth.

