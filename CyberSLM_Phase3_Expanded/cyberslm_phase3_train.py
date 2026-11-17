"""CyberSLM Phase 3: fresh 4M RoPE LM with previously frozen scenario-aware splits.

This is EXPERIMENTAL TEXT PREDICTION on unreviewed Qwen-generated reports.
Not a verified cybersecurity assistant or an APT detector.

Example:
  python cyberslm_phase3_train.py --pilot-unreviewed --check-only
  python cyberslm_phase3_train.py --pilot-unreviewed --steps 300 --model 4m
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cyberslm_rope_pilot import (
    CyberSLM, Config, MODEL_SPECS, TextBlocks, count_params, evaluate,
)

DEFAULT_ROOT = Path(r"D:\Design thinking Cybersecurity SLM")


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized_digest(value: str) -> str:
    clean = " ".join(value.casefold().split())
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()


def load_split(path: Path, split: str, column: str = "report_text") -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen {split} split: {path}")
    df = pd.read_parquet(path)
    if column not in df:
        raise ValueError(f"{path} missing required column: {column}")
    if df.empty or df[column].isna().any():
        raise ValueError(f"{split}: empty dataset or null text")
    df = df.copy()
    df[column] = df[column].astype(str)
    if df[column].str.strip().str.len().min() < 40:
        raise ValueError(f"{split}: contains empty or <40-character text")
    df["_normalized_hash"] = df[column].map(normalized_digest)
    if df["_normalized_hash"].duplicated().any():
        raise ValueError(f"{split}: repeated normalized text found within split")
    return df


def audit_splits(parts: dict[str, pd.DataFrame]) -> dict:
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if overlap := set(parts[a]["_normalized_hash"]) & set(parts[b]["_normalized_hash"]):
                raise ValueError(f"Normalized text overlap between {a} and {b}: {len(overlap)}")
            if "facts_signature" in parts[a] and "facts_signature" in parts[b]:
                left = set(parts[a]["facts_signature"].dropna().astype(str))
                right = set(parts[b]["facts_signature"].dropna().astype(str))
                if common := left & right:
                    raise ValueError(f"Fact-summary overlap between {a} and {b}: {len(common)}")
            if "source_file" in parts[a] and "source_file" in parts[b]:
                if common := set(parts[a]["source_file"]) & set(parts[b]["source_file"]):
                    raise ValueError(f"File overlap between {a} and {b}: {len(common)}")
    for split in ["train", "validation"]:
        if "investigation_family" not in parts[split]:
            raise ValueError(f"{split}: missing investigation_family field")
        allowed = {"theia_3.11_success"}
        actual = set(parts[split]["investigation_family"].astype(str))
        if actual != allowed:
            raise ValueError(f"Unexpected {split} families: {actual} (expected {allowed})")
    test = parts["cross_scenario_test"]
    if "case_group" not in test or set(test["case_group"]) != {"THEIA_3.3"}:
        raise ValueError("Cross-scenario test must consist solely of THEIA_3.3")
    for part_name, df in parts.items():
        if "review_status" not in df:
            raise ValueError(f"{part_name}: missing review_status provenance field")
    return {
        "document_counts": {key: len(df) for key, df in parts.items()},
        "within_scenario_validation": True,
        "external_scenario_test": "THEIA_3.3",
        "limitations": [
            "File-based scenario labels do not prove independent incidents.",
            "Training and validation belong to the same THEIA_3.11 scenario; event windows may overlap.",
            "Unique fact-summary hashes are not a substitute for source-event lineage checks.",
            "Model-generated Qwen narratives have not been human-approved for factual training.",
            "Held-out next-byte loss does not measure APT classification or explanation faithfulness.",
        ],
    }


def token_audit(frame: pd.DataFrame, column: str, seq_len: int) -> dict:
    """Count UTF-8 BYTE targets, including one EOS target per document.

    The byte tokenizer maps each UTF-8 byte to one token (plus BOS/EOS).
    PAD is never counted as a target, and each document has its own blocks.
    """
    text_bytes = [len(text.encode("utf-8")) for text in frame[column]]
    valid_targets = sum(size + 1 for size in text_bytes)  # EOS; BOS is an input only
    blocks = sum(math.ceil((size + 1) / seq_len) for size in text_bytes)
    slots = blocks * seq_len
    return {
        "documents": len(frame),
        "text_utf8_bytes": sum(text_bytes),
        "valid_next_token_targets": valid_targets,
        "token_blocks": blocks,
        "available_slots": slots,
        "padding_slots": slots - valid_targets,
        "target_slot_occupancy": round(valid_targets / slots, 6) if slots else 0.0,
        "tokenizer": "UTF8 byte: PAD=0, BOS=1, EOS=2, bytes +3; NOT BPE",
    }


def test_report(model, frame, column, seq_len, batch_size, device, use_amp, amp_dtype):
    blocks = TextBlocks(frame[column].tolist(), seq_len)
    loader = DataLoader(blocks, batch_size=batch_size, shuffle=False, num_workers=0)
    loss = float(evaluate(model, loader, device, use_amp, amp_dtype))
    return {
        "loss_nats_per_byte_token": round(loss, 6),
        "byte_token_perplexity": round(math.exp(min(20, loss)), 6),
        "token_blocks": len(blocks),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--splits", type=Path, default=None, help="Frozen Phase 3 split folder")
    p.add_argument("--external-pdf", type=Path, default=None)
    p.add_argument("--skip-external-pdf", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--pilot-unreviewed", action="store_true")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--model", choices=MODEL_SPECS, default="4m")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if min(args.steps, args.seq_len, args.batch_size, args.grad_accum, args.eval_every) < 1:
        p.error("steps, seq-len, batch-size, grad-accum, eval-every must be positive")
    if not args.pilot_unreviewed:
        p.error("This experiment uses unreviewed Qwen outputs. Pass --pilot-unreviewed for a non-deployable SMOKE TEST only.")

    splits_dir = args.splits or (args.root / "data" / "processed" / "ismail_apt_full" / "phase3_splits")
    expected = ["train", "validation", "cross_scenario_test", "quarantine"]
    parts = {name: load_split(splits_dir / f"{name}.parquet", name) for name in expected}
    audit = audit_splits(parts)
    manifest_path = splits_dir / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen split manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("split_counts", {})
    for name, actual in audit["document_counts"].items():
        if declared.get(name) != actual:
            raise ValueError(f"Manifest mismatch {name}: manifest {declared.get(name)}, file {actual}")

    report = {
        **audit,
        "token_audit": {name: token_audit(frame, "report_text", args.seq_len)
                        for name, frame in parts.items()},
        "source_split_manifest": str(manifest_path),
        "source_split_manifest_sha256": digest_file(manifest_path),
        "source_split_file_sha256": {name: digest_file(splits_dir / f"{name}.parquet") for name in expected},
        "training_corpus_bytes": sum(len(s.encode("utf-8")) for s in parts["train"]["report_text"]),
        "model": args.model,
        "context_length": args.seq_len,
        "seed": args.seed,
        "status": "UNREVIEWED_QWEN_CORPUS_ENGINEERING_EXPERIMENT",
    }

    pdf = None
    pdf_path = args.external_pdf or (
        args.root / "outputs" / "cyberslm_rope_4m_multisplit_smoke" / "data_splits" / "external_pdf_test.parquet"
    )
    if not args.skip_external_pdf:
        pdf = load_split(pdf_path, "external_pdf_test", column="text")
        qwen_hashes = set().union(*(set(part["_normalized_hash"]) for part in parts.values()))
        if overlap := qwen_hashes & set(pdf["_normalized_hash"]):
            raise ValueError(f"PDF and Qwen data have {len(overlap)} normalized-text overlaps")
        report["external_pdf_chunks"] = len(pdf)
        report["external_pdf_sha256"] = digest_file(pdf_path)
        report["token_audit"]["external_pdf_test"] = token_audit(pdf, "text", args.seq_len)

    out = args.output or (args.root / "outputs" / f"cyberslm_phase3_{args.model}_expanded")
    out.mkdir(parents=True, exist_ok=True)
    (out / "preflight_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PREFLIGHT OK - splits used without regeneration")
    print("RECORDS:", json.dumps(audit["document_counts"]))
    print("TRAIN TEXT BYTES:", f"{report['training_corpus_bytes']:,}")
    print("TOKEN ACCOUNTING (BYTE TOKENIZER; VALID TARGETS EXCLUDE PADDING)")
    for name, counts in report["token_audit"].items():
        print(f"  {name}: documents={counts['documents']:,} "
              f"targets={counts['valid_next_token_targets']:,} "
              f"blocks={counts['token_blocks']:,} "
              f"occupancy={100*counts['target_slot_occupancy']:.1f}%")
    print("EXTERNAL PDF CHUNKS:", report.get("external_pdf_chunks", "SKIPPED"))
    print("AUDIT:", out / "preflight_audit.json")
    print("WARNING: same THEIA 3.11 case used for training and validation; underlying event leakage not ruled out")
    if args.check_only:
        print("CHECK ONLY SUCCESS - model not trained; test set not scored")
        return
    if (out / "best_validation_checkpoint.pt").exists():
        raise FileExistsError(
            f"Output already has a checkpoint: {out}; use a NEW --output folder for a fresh run."
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    spec = MODEL_SPECS[args.model]
    cfg = Config(**spec, max_seq_len=args.seq_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CyberSLM(cfg).to(device)  # fresh weights, no loading of older runs
    print("DEVICE:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    print("FRESH PARAMETERS:", f"{count_params(model):,}")
    train_blocks = TextBlocks(parts["train"]["report_text"].tolist(), args.seq_len)
    val_blocks = TextBlocks(parts["validation"]["report_text"].tolist(), args.seq_len)
    train_loader = DataLoader(train_blocks, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_blocks, batch_size=args.batch_size, shuffle=False, num_workers=0)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_loss, best_step = float("inf"), 0
    best_path = out / "best_validation_checkpoint.pt"
    train_iterator = iter(train_loader)
    metrics = []
    total_optimizer_target_tokens = 0  # includes repeated passes, never PAD
    print("TOKEN BLOCKS train/val:", len(train_blocks), len(val_blocks))
    print("NOMINAL TOKENS PER STEP:", args.seq_len * args.batch_size * args.grad_accum)
    print("STEPS:", args.steps)
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Gather microbatches on CPU, then normalize accumulation by the
        # TOTAL number of non-padding target tokens. Without this, a short
        # padded block can receive too much gradient weight.
        microbatches = []
        for _ in range(args.grad_accum):
            try:
                x, y = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                x, y = next(train_iterator)
            token_count = int((y != -100).sum().item())
            microbatches.append((x, y, token_count))
        step_tokens = sum(count for _, _, count in microbatches)
        if not step_tokens:
            raise ValueError("No valid target tokens in optimizer step")
        recent = 0.0
        for x, y, token_count in microbatches:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            weight = token_count / step_tokens
            scaler.scale(loss * weight).backward()
            recent += float(loss.item()) * weight
        total_optimizer_target_tokens += step_tokens
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step in (1, args.steps) or step % args.eval_every == 0:
            validation_loss = float(evaluate(model, val_loader, device, use_amp, amp_dtype))
            metric = {"step": step, "train_loss_last_batches": round(recent, 6),
                      "validation_loss": round(validation_loss, 6),
                      "actual_train_target_tokens_seen": total_optimizer_target_tokens}
            metrics.append(metric)
            print("PROGRESS:", json.dumps(metric), flush=True)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_step = step
                torch.save({
                    "model_state_dict": model.state_dict(), "config": asdict(cfg),
                    "model_size": args.model, "steps": step,
                    "tokenizer": "utf8_byte_v1_offset3", "pilot_unreviewed": True,
                    "data_split_manifest_sha256": report["source_split_manifest_sha256"],
                }, best_path)
    (out / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_score = test_report(model, parts["validation"], "report_text", args.seq_len,
                            args.batch_size, device, use_amp, amp_dtype)
    test_score = test_report(model, parts["cross_scenario_test"], "report_text", args.seq_len,
                             args.batch_size, device, use_amp, amp_dtype)
    pdf_score = (test_report(model, pdf, "text", args.seq_len, args.batch_size, device, use_amp, amp_dtype)
                 if pdf is not None else None)
    results = {
        "experiment_type": "unreviewed_qwen_phase3_smoke_test",
        "parameters": count_params(model), "best_checkpoint_step": best_step,
        "validation": val_score,
        "cross_scenario_test": test_score,
        "external_pdf_test": pdf_score,
        "split_counts": audit["document_counts"],
        "token_audit": report["token_audit"],
        "actual_train_target_tokens_seen": total_optimizer_target_tokens,
        "train_loss_gradient_weighting": "non_padding_target_token_weighted",
        "limitations": report["limitations"],
        "no_claim": "NOT APT DETECTION ACCURACY OR VERIFIED CYBERSECURITY REASONING",
    }
    (out / "final_evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("FINAL RESULTS:", json.dumps(results, indent=2), flush=True)
    print("CHECKPOINT:", best_path)
    print("METRICS:", out / "training_metrics.json")


if __name__ == "__main__":
    main()
