"""CyberSLM Phase 1: byte-level decoder-only RoPE Transformer from scratch.

Pipeline smoke test, NOT a validated cybersecurity model.
Requires: torch, pandas, pyarrow.

Example (PowerShell, virtual environment active):
python cyberslm_rope_pilot.py --model 4m --pilot-unreviewed --steps 30
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# Small, deterministic BYTE vocabulary for pipeline verification.
# PAD=0, BOS=1, EOS=2, bytes 0..255 -> tokens 3..258. This is NOT BPE.
PAD, BOS, EOS, VOCAB_SIZE = 0, 1, 2, 259
DEFAULT_ROOT = Path(r"D:\Design thinking Cybersecurity SLM")
MODEL_SPECS = {
    "1m": {"n_layers": 2, "d_model": 192, "n_heads": 4, "ff_dim": 512},
    "4m": {"n_layers": 4, "d_model": 288, "n_heads": 6, "ff_dim": 768},
}


def encode(text: str) -> list[int]:
    return [BOS] + [b + 3 for b in text.encode("utf-8")] + [EOS]


def decode(tokens: list[int]) -> str:
    bs = bytes([x - 3 for x in tokens if 3 <= x < VOCAB_SIZE])
    return bs.decode("utf-8", errors="replace")


class TextBlocks(Dataset):
    """Independent document sequences, pad labels with -100; never merge documents."""
    def __init__(self, texts: list[str], seq_len: int):
        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.seq_len = seq_len
        for text in texts:
            ids = encode(text)
            for begin in range(0, len(ids) - 1, seq_len):
                segment = ids[begin:begin + seq_len + 1]
                if len(segment) < 2:
                    continue
                xs, ys = segment[:-1], segment[1:]
                x = torch.tensor(xs + [PAD] * (seq_len - len(xs)), dtype=torch.long)
                y = torch.tensor(ys + [-100] * (seq_len - len(ys)), dtype=torch.long)
                self.items.append((x, y))
        if not self.items:
            raise ValueError("No token blocks could be created; verify the text data.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


@dataclass
class Config:
    vocab_size: int = VOCAB_SIZE
    n_layers: int = 4
    d_model: int = 288
    n_heads: int = 6
    ff_dim: int = 768
    max_seq_len: int = 256


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: [batch, heads, sequence, head_dim]; head_dim must be even.
    length, dim = q.shape[-2], q.shape[-1]
    assert dim % 2 == 0
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=q.device).float() / dim))
    positions = torch.arange(length, device=q.device).float()
    angle = torch.outer(positions, inv_freq)
    angle = torch.cat((angle, angle), dim=-1)[None, None, :, :]
    cosine = angle.cos().to(q.dtype)
    sine = angle.sin().to(q.dtype)
    return (q * cosine + rotate_half(q) * sine,
            k * cosine + rotate_half(k) * sine)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0 or (cfg.d_model // cfg.n_heads) % 2:
            raise ValueError("Each attention head dimension must be even.")
        self.heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ff_dim, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ff_dim, bias=False)
        self.down = nn.Linear(cfg.ff_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attention_norm = RMSNorm(cfg.d_model)
        self.attention = CausalSelfAttention(cfg)
        self.ff_norm = RMSNorm(cfg.d_model)
        self.ff = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class CyberSLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        # Weight-tied output head; no separate output matrix.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        if input_ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError("Input exceeds configured maximum context length")
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = F.linear(x, self.token_embedding.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss


def load_source(path: Path, allow_unreviewed: bool):
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}\nGenerate qwen_review_staging.parquet first.")
    df = pd.read_parquet(path)
    if "report_text" not in df.columns:
        raise ValueError("Expected a 'report_text' column in the Parquet input")
    if "training_approved" in df.columns:
        approved = df["training_approved"].fillna(False).eq(True)
        if not allow_unreviewed:
            df = df[approved].copy()
        else:
            print("WARNING: unreviewed model-generated reports included: SMOKE TEST ONLY.")
    elif not allow_unreviewed:
        raise ValueError("No training_approved column. Use --pilot-unreviewed for pipeline test only.")
    else:
        print("WARNING: approval metadata absent; SMOKE TEST ONLY.")
    df = df[df["report_text"].notna()].copy()
    df["report_text"] = df["report_text"].astype(str).str.strip()
    df = df[df["report_text"].str.len() >= 40].copy()
    df["text_hash"] = df["report_text"].map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())
    df = df.drop_duplicates("text_hash").reset_index(drop=True)
    if len(df) < 5:
        raise ValueError(f"Found only {len(df)} eligible unique documents; at least 5 required.")
    return df


@torch.no_grad()
def evaluate(model, loader, device, use_amp, amp_dtype):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        count = (y != -100).sum().item()
        total_loss += loss.item() * count
        total_tokens += count
    model.train()
    return total_loss / max(total_tokens, 1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--model", choices=MODEL_SPECS, default="4m")
    parser.add_argument("--pilot-unreviewed", action="store_true",
                        help="Explicitly allow unverified Qwen text for a SMOKE TEST ONLY")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true",
                        help="Check param count and forward pass without reading data or training")
    args = parser.parse_args(argv)

    if min(args.steps, args.seq_len, args.batch_size, args.grad_accum) <= 0:
        parser.error("steps, seq-len, batch-size and grad-accum must all be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    spec = MODEL_SPECS[args.model]
    cfg = Config(**spec, max_seq_len=args.seq_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CyberSLM(cfg).to(device)
    print("DEVICE:", device, "GPU:" , torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    print("MODEL:", args.model, "EXACT PARAMETERS:", f"{count_params(model):,}")
    print("LAYERS:", cfg.n_layers, "HIDDEN:", cfg.d_model, "HEADS:", cfg.n_heads,
          "FFN:", cfg.ff_dim, "VOCAB:", cfg.vocab_size, "CONTEXT:", cfg.max_seq_len)

    test_x = torch.randint(3, VOCAB_SIZE, (2, min(16, cfg.max_seq_len)), device=device)
    test_y = test_x.clone()
    model.eval()
    with torch.no_grad():
        test_logits, test_loss = model(test_x, test_y)
    print("FORWARD CHECK:", tuple(test_logits.shape), "finite loss:", bool(torch.isfinite(test_loss).item()))
    if args.check_only:
        print("CHECK-ONLY SUCCESS: no training was performed")
        return

    default_folder = args.root / "data" / "processed" / "ismail_apt_pilot"
    source = args.input or (default_folder / "qwen_review_staging.parquet")
    if args.input is None and not source.exists():
        source = default_folder / "qwen_review_candidates.parquet"
    df = load_source(source, args.pilot_unreviewed)
    rng = random.Random(args.seed)
    positions = list(range(len(df)))
    rng.shuffle(positions)
    val_count = max(1, round(len(df) * 0.2))
    val_indices = positions[:val_count]
    train_indices = positions[val_count:]
    # Split by REPORT before batching to avoid same report in train and validation.
    train_texts = df.iloc[train_indices]["report_text"].tolist()
    val_texts = df.iloc[val_indices]["report_text"].tolist()
    assert set(df.iloc[train_indices]["text_hash"]).isdisjoint(set(df.iloc[val_indices]["text_hash"]))
    train_ds = TextBlocks(train_texts, args.seq_len)
    val_ds = TextBlocks(val_texts, args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print("DOCUMENTS: train", len(train_texts), "val", len(val_texts))
    print("BATCH BLOCKS: train", len(train_ds), "val", len(val_ds),
          "effective blocks per step", args.batch_size * args.grad_accum)

    use_amp = device.type == "cuda"
    amp_dtype = (torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported()
                 else torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    train_it = iter(train_loader)
    records = []
    output = args.root / "outputs" / f"cyberslm_rope_{args.model}_smoke"
    output.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(args.grad_accum):
            try:
                x, y = next(train_it)
            except StopIteration:
                train_it = iter(train_loader)
                x, y = next(train_it)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
                scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()
            accumulated += loss.item() / args.grad_accum
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 10 == 0 or step == args.steps:
            v_loss = evaluate(model, val_loader, device, use_amp, amp_dtype)
            entry = {"step": step, "train_loss_last_batches": round(accumulated, 5),
                     "val_loss": round(v_loss, 5),
                     "val_byte_perplexity": round(math.exp(min(v_loss, 20)), 4)}
            records.append(entry)
            print(json.dumps(entry))

    checkpoint = output / "last_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg),
                "model_size": args.model, "steps": args.steps,
                "tokenizer": "utf8_byte_v1_offset3",
                "source_path": str(source), "pilot_unreviewed": args.pilot_unreviewed}, checkpoint)
    (output / "training_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    manifest = {
        "source_path": str(source),
        "source_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "train_document_hashes": df.iloc[train_indices]["text_hash"].tolist(),
        "val_document_hashes": df.iloc[val_indices]["text_hash"].tolist(),
        "model_config": asdict(cfg),
        "parameters": count_params(model),
        "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "effective_batch_blocks": args.batch_size * args.grad_accum,
        "seed": args.seed,
        "status": "PIPELINE_SMOKE_TEST_UNREVIEWED" if args.pilot_unreviewed else "PILOT_TRAINING_APPROVED_TEXT_ONLY",
        "caveat": "Small Qwen-derived reports do not establish generalizable cybersecurity expertise."
    }
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nDONE. SMOKE TEST CHECKPOINT:", checkpoint)
    print("METRICS:", output / "training_metrics.json")
    print("This checkpoint is NOT a validated cybersecurity model.")


if __name__ == "__main__":
    main()
