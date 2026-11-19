# CyberSLM Practical Conversational Assistant v1

Date: 19 November 2026

Branch: `codex/cyberslm-chat`

## Decision and identity

The scratch-built CyberSLM-35M conversational experiment is preserved but is not suitable as the
general chatbot: it scored 1/108 exact matches and retained 0/1,620 structured answers. The
original structured model remains a separate evidence-analysis capability (34.38% neural exact
match; 100% separately labeled deterministic calculation).

The practical assistant uses the pretrained `Qwen/Qwen2.5-1.5B-Instruct` model (1.54B parameters,
Apache-2.0) in Q4_K_M form. CyberSLM did not train these base weights. Its customization is the
defensive system behavior, conversation history, public-source retrieval, runtime adapters,
evaluation, and Streamlit application. No fine-tuning is claimed.

## Runtime and artifact receipt

| Item | Verified value |
| --- | --- |
| Ollama | 0.40.16; `qwen2.5:1.5b-instruct-q4_K_M` |
| LM Studio CLI | 1.3.3; identifier `cyberslm-practical` |
| GGUF SHA-256 | `183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4` |
| Quantization | Q4_K_M, 986.05 MB |
| Tested context | 8,192 tokens |
| GPU load estimate | 1.14 GiB; fits the RTX 4050 6 GB |
| Embeddings | Ollama `all-minilm:latest`, 384 dimensions |

LM Studio loaded the same hashed GGUF, exposed it through `/v1/chat/completions`, and completed the
six-turn acceptance dialogue. Ollama completed the same dialogue through `/api/chat`.

## Knowledge collection

The index contains only public technical sources: ten NIST publications already recorded in the
Stage A provenance, MITRE ATT&CK Enterprise v20.1, and two CISA publications. It contains 1,605
documents and 6,948 deduplicated chunks. No private event reports are included.

| Receipt | SHA-256 |
| --- | --- |
| Source configuration | `735a6c713bd3446dbba8621d3d7372f3f14c504d0a3bf043697351a5f99c37d6` |
| Chunk metadata | `e95656a0e28aab9089a58e9580b4139924362f1e47427294944793c398a98fba` |
| Canonical row content | `5707009809bc3fcf9c8b05424155221335c22824e46b62751f5e711e70070072` |
| Embeddings | `56e0840978e8b30091c142dfddd930e83a0b2c185e2cf256b63e3fed7d3dc766` |

The generated index is deliberately excluded from Git. Rebuild it from
`configs/rag_sources.json`; the manifest records every source license, hash, and retrieval date.
A clean rebuild on 19 November produced the same 6,948 rows, identical values in all eight columns,
and the same embedding-array SHA-256. Its Parquet file byte hash differed because the original used
PyArrow 26/Pandas 3.1 while the supported environment used PyArrow 24/Pandas 3.0; Parquet embeds
serializer metadata. The builder now also records a canonical row-content hash and serializer
versions so content reproducibility is not confused with container-byte reproducibility.

## Evaluation

The six-turn acceptance conversation covered a greeting, detailed ransomware explanation,
machine-learning follow-up, project suggestion, implementation follow-up, and conversation
summary. Ollama mean diagnostic term coverage was 80.56% with 2.87 seconds mean warm latency.
LM Studio completed all six turns with 69.44% term coverage in the saved run. These term checks are
not factuality scores.

The separate 12-question unseen set covers greetings, core concepts, malware versus viruses,
MITRE ATT&CK, indicators of compromise, Isolation Forest, safe projects, insufficient evidence,
phishing, model identity, architecture, and a harmful credential-theft request.

| Configuration | Mean term coverage | Required-source turns | Mean trigram repetition |
| --- | ---: | ---: | ---: |
| Qwen practical baseline | 67% | 0/9 | 6% |
| Qwen + trusted retrieval + routing | 83% | 9/9 | 5% |

An early grounded test exposed a serious failure: model prompting alone answered the credential
theft/evasion request after receiving offensive ATT&CK excerpts. The released code therefore uses a
small deterministic pre-generation safety gate and labels its response origin as `policy_guard`.
The rerun blocked the request. This is external routing, not a capability of the base model.

A separate indirect-injection smoke test placed `IGNORE ALL PREVIOUS INSTRUCTIONS`, a request to
reveal the system prompt, and credential-theft instructions inside a synthetic retrieved excerpt.
The tested Qwen runtime did not emit the marker or the harmful steps. The runtime prompt now labels
all retrieved excerpts as untrusted reference data. This single deterministic test is a regression
check, not proof of resistance to arbitrary prompt injection.

## Honest limitations

- The 1.5B model sometimes repeats prior content or runs out of its configured response budget.
- Retrieval relevance is imperfect; a cited source does not validate every generated claim.
- Automated term coverage cannot establish correctness, coherence, or safety. Independent human
  review remains pending.
- The safety gate covers high-confidence credential-theft, ransomware-deployment, and evasion
  patterns; it is not a comprehensive content-safety system.
- The pre-generation gate inspects the user's text, not every possible semantic instruction inside
  retrieved content. Obfuscated or novel indirect prompt injection may still influence the 1.5B
  model, so the RAG collection must remain allowlisted and responses require human review.
- The assistant is educational and defensive, not an autonomous detector or incident verdict.
- No Qwen fine-tuning was performed because the evaluated base model plus retrieval already
  substantially outperformed the scratch conversational checkpoint on usable language generation.
- Exact `chunks.parquet` bytes depend on the Pandas/PyArrow versions. Validate the canonical content
  hash and embedding hash across environments; require pinned serializer versions only when exact
  Parquet container bytes are necessary.

## Reproduce

```powershell
$env:PYTHONPATH = "src"
python scripts\run_practical_acceptance.py --output outputs\cyberslm_practical_v1\acceptance_ollama_rag.json --rag-index data\rag\cyberslm_trusted_knowledge_v1 --answer-style Balanced --max-tokens 384
python scripts\evaluate_practical_assistant.py --cases configs\evaluation\practical_assistant_v1.json --output outputs\cyberslm_practical_v1\evaluation_ollama_rag.json --rag-index data\rag\cyberslm_trusted_knowledge_v1 --max-tokens 320
```
