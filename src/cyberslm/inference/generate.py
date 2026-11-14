from __future__ import annotations

import torch

from cyberslm.models.transformer import CyberSLM
from cyberslm.tokenization.bpe import BPETokenizer


def generate_text(
    model: CyberSLM,
    tokenizer: BPETokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int | None = 50,
    device: str | torch.device = "cpu",
) -> str:
    model.to(device).eval()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_id]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id,
    )
    return tokenizer.decode(output[0].tolist())
