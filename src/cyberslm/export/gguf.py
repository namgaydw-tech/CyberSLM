from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def byte_token_gguf_vocab() -> tuple[list[str], list[float], list[int]]:
    """Return the exact CyberSLM ID order as an SPM byte-fallback vocabulary."""
    # Import locally so the rest of the package does not require the optional GGUF dependency.
    import gguf  # type: ignore[import-not-found]

    tokens = ["<pad>", "<s>", "</s>"]
    tokens.extend(f"<0x{value:02X}>" for value in range(256))
    scores = [0.0, 0.0, 0.0, *([-1000.0] * 256)]
    types = [
        int(gguf.TokenType.CONTROL),
        int(gguf.TokenType.CONTROL),
        int(gguf.TokenType.CONTROL),
        *([int(gguf.TokenType.BYTE)] * 256),
    ]
    return tokens, scores, types


def expected_tensor_mapping(n_layers: int) -> dict[str, str]:
    """Map CyberSLM state-dict names to llama.cpp's canonical Llama GGUF names."""
    mapping = {
        "token_embedding.weight": "token_embd.weight",
        "final_norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    for layer in range(n_layers):
        source = f"layers.{layer}"
        target = f"blk.{layer}"
        mapping.update(
            {
                f"{source}.attention_norm.weight": f"{target}.attn_norm.weight",
                f"{source}.attention.q_proj.weight": f"{target}.attn_q.weight",
                f"{source}.attention.k_proj.weight": f"{target}.attn_k.weight",
                f"{source}.attention.v_proj.weight": f"{target}.attn_v.weight",
                f"{source}.attention.out_proj.weight": f"{target}.attn_output.weight",
                f"{source}.ffn_norm.weight": f"{target}.ffn_norm.weight",
                f"{source}.feed_forward.gate.weight": f"{target}.ffn_gate.weight",
                f"{source}.feed_forward.down.weight": f"{target}.ffn_down.weight",
                f"{source}.feed_forward.up.weight": f"{target}.ffn_up.weight",
            }
        )
    return mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint_for_llama_gguf(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["model_config"]
    state = payload["model"]
    required = expected_tensor_mapping(int(config["n_layers"]))
    missing = sorted(set(required) - set(state))
    unexpected = sorted(set(state) - set(required))
    if missing or unexpected:
        raise ValueError(f"Checkpoint tensor mismatch: missing={missing}, unexpected={unexpected}")
    if int(config["vocab_size"]) != 259:
        raise ValueError("CyberSLM GGUF export requires the verified 259-token byte vocabulary")
    if int(config["d_model"]) % int(config["n_heads"]):
        raise ValueError("d_model must be divisible by n_heads")
    head_dim = int(config["d_model"]) // int(config["n_heads"])
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    if not bool(config["tie_embeddings"]):
        raise ValueError("This exporter currently requires tied input/output embeddings")
    if not torch.equal(state["token_embedding.weight"], state["lm_head.weight"]):
        raise ValueError("Checkpoint claims tied embeddings but tensors differ")
    return {
        "tensor_count": len(required),
        "head_dim": head_dim,
        "direct_qk_export": True,
        "rope_layout": "adjacent_even_odd_pairs",
        "note": (
            "Q/K weights are intentionally not HF-Llama-permuted: CyberSLM already applies "
            "adjacent-pair RoPE, matching llama.cpp's internal normal-RoPE tensor layout."
        ),
    }


def export_checkpoint_to_gguf(
    checkpoint_path: Path,
    output_path: Path,
    *,
    output_dtype: str = "f32",
) -> dict[str, Any]:
    """Export a CyberSLM checkpoint into an experimental Llama-architecture GGUF file.

    This establishes metadata/tensor-format compatibility only. Runtime logit parity must be
    measured with llama.cpp before the artifact can be described as faithful.
    """
    import gguf

    if output_dtype not in {"f32", "f16"}:
        raise ValueError("output_dtype must be f32 or f16")
    payload: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validation = validate_checkpoint_for_llama_gguf(payload)
    config = payload["model_config"]
    state: dict[str, torch.Tensor] = payload["model"]
    mapping = expected_tensor_mapping(int(config["n_layers"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(output_path, "llama")
    writer.add_name("CyberSLM-35M")
    writer.add_description(
        "Experimental GGUF export of the scratch-built CyberSLM decoder-only Transformer"
    )
    writer.add_context_length(int(config["max_seq_len"]))
    writer.add_embedding_length(int(config["d_model"]))
    writer.add_block_count(int(config["n_layers"]))
    writer.add_feed_forward_length(int(config["ffn_hidden_dim"]))
    writer.add_head_count(int(config["n_heads"]))
    writer.add_head_count_kv(int(config["n_kv_heads"]))
    writer.add_rope_dimension_count(validation["head_dim"])
    writer.add_rope_freq_base(float(config["rope_base"]))
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_causal_attention(True)
    writer.add_file_type(
        int(
            gguf.GGMLQuantizationType.F32
            if output_dtype == "f32"
            else gguf.GGMLQuantizationType.F16
        )
    )

    tokens, scores, token_types = byte_token_gguf_vocab()
    writer.add_tokenizer_model("llama")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    writer.add_pad_token_id(0)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)
    writer.add_add_space_prefix(False)
    writer.add_remove_extra_whitespaces(False)

    numpy_dtype = np.float32 if output_dtype == "f32" else np.float16
    for source_name, target_name in mapping.items():
        raw_tensor = state[source_name].detach().cpu().float().numpy()
        # llama.cpp expects 1-D normalization weights in F32 even in an F16 GGUF. Casting
        # them to F16 triggers a mixed-type broadcast stride assertion during warm-up.
        tensor_dtype = np.float32 if raw_tensor.ndim == 1 else numpy_dtype
        tensor = raw_tensor.astype(tensor_dtype)
        writer.add_tensor(target_name, tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    report = {
        "status": "FORMAT_EXPORTED_RUNTIME_PARITY_NOT_YET_VERIFIED",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "gguf": str(output_path.resolve()),
        "gguf_sha256": _sha256(output_path),
        "gguf_bytes": output_path.stat().st_size,
        "output_dtype": output_dtype,
        "model_config": config,
        "tokenizer": {
            "model": "llama_byte_fallback",
            "vocab_size": len(tokens),
            "pad_id": 0,
            "bos_id": 1,
            "eos_id": 2,
            "byte_id_range": [3, 258],
            "add_space_prefix": False,
        },
        "validation": validation,
        "required_next_gate": (
            "Compare native PyTorch and llama.cpp logits/tokenization before Ollama import."
        ),
    }
    report_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
