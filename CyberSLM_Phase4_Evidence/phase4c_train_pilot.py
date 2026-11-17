"""Phase 4C: answer-only supervised-loss SMOKE TEST, 4M RoPE from scratch.

Inputs are rule-generated QA candidates from UNVERIFIED aggregate statistics.
Do NOT use this checkpoint for security conclusions. No Qwen narratives as targets.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import random
import sys

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT_DEFAULT = Path(r"D:\Design thinking Cybersecurity SLM")

class AnswerOnlyDataset(Dataset):
    """One QA example per block; labels for prompts and PAD are -100."""
    def __init__(self, frame: pd.DataFrame, context: int, pad=0, bos=1, eos=2):
        self.items = []
        self.skipped_long = 0
        self.answer_tokens = 0
        for _, row in frame.iterrows():
            prompt = str(row["prompt"]).encode("utf-8")
            answer = str(row["answer"]).encode("utf-8")
            if not prompt or not answer:
                continue
            if len(prompt) + len(answer) + 1 > context:
                self.skipped_long += 1
                continue
            ids = [bos] + [b+3 for b in prompt] + [b+3 for b in answer] + [eos]
            x = ids[:-1]
            y = [-100] * len(prompt) + [b+3 for b in answer] + [eos]
            if len(y) != len(x):
                raise AssertionError("Mismatch in answer-only token masks")
            self.answer_tokens += len(answer)+1
            n=context-len(x)
            self.items.append((torch.tensor(x+[pad]*n,dtype=torch.long),
                               torch.tensor(y+[-100]*n,dtype=torch.long)))
        if not self.items:
            raise ValueError("No fitting QA examples. Shorten prompts/answers or increase --context.")
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

@torch.no_grad()
def evaluation(model, loader, device, amp):
    model.eval()
    total_loss=0.
    total_targets=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=amp):
            _,loss=model(x,y)
        count=int((y!=-100).sum().item())
        total_loss+=float(loss.item())*count
        total_targets+=count
    model.train()
    return {"loss_nats_per_answer_byte":round(total_loss/max(1,total_targets),6),
            "answer_byte_perplexity":round(math.exp(min(20,total_loss/max(1,total_targets))),6),
            "supervised_answer_targets":total_targets}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ROOT_DEFAULT)
    p.add_argument("--qa-dir",type=Path,default=None)
    p.add_argument("--output",type=Path,default=None)
    p.add_argument("--pilot-unreviewed",action="store_true")
    p.add_argument("--check-only",action="store_true")
    p.add_argument("--steps",type=int,default=100)
    p.add_argument("--context",type=int,default=512)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--grad-accum",type=int,default=4)
    p.add_argument("--eval-every",type=int,default=25)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if not args.pilot_unreviewed:
        p.error("Requires --pilot-unreviewed because source facts are NOT independently verified.")
    if min(args.steps,args.context,args.batch_size,args.grad_accum,args.eval_every)<1:
        p.error("All positive training settings required")
    sys.path.insert(0,str(args.root/"CyberSLM_Phase3_Expanded"))
    try:
        from cyberslm_rope_pilot import CyberSLM, Config, MODEL_SPECS
    except ImportError as e:
        raise ImportError("Missing Phase 3 cyberslm_rope_pilot.py in project folder") from e
    qa=args.qa_dir or (args.root/"CyberSLM_Phase4_Evidence"/"runs"/"phase4b_qa")
    out=args.output or (args.root/"CyberSLM_Phase4_Evidence"/"runs"/"phase4c_4m_pilot")
    out.mkdir(parents=True,exist_ok=True)
    parts={}
    file_hashes={}
    for s in ("train","validation","cross_scenario_test"):
        file=qa/f"{s}_qa.parquet"
        if not file.is_file():
            raise FileNotFoundError(file)
        file_hashes[s]=digest(file)
        parts[s]=pd.read_parquet(file)
        if not {"prompt","answer","source_file","training_approved","qa_sha256"}.issubset(parts[s]):
            raise RuntimeError(f"Missing required evidence QA columns in {s}")
        if parts[s]["training_approved"].fillna(False).any():
            raise RuntimeError("Unexpected approved examples; this pipeline is strictly for unreviewed pilot data.")
    all_hashes=[set(parts[s]["qa_sha256"]) for s in parts]
    for i in range(len(all_hashes)):
        for j in range(i+1,len(all_hashes)):
            if all_hashes[i]&all_hashes[j]:
                raise RuntimeError("QA exact overlap between splits")
    ds={s:AnswerOnlyDataset(parts[s],args.context) for s in parts}
    details={s:{"rows":len(parts[s]),"usable_qa":len(ds[s]),"too_long":ds[s].skipped_long,
                "supervised_answer_byte_targets":ds[s].answer_tokens} for s in ds}
    report={"phase":"4C","experiment":"UNVERIFIED_RULE_BASED_EVIDENCE_QA_PILOT",
            "qa_input_sha256":file_hashes,"datasets":details,
            "context":args.context,"batch_size":args.batch_size,"grad_accum":args.grad_accum,
            "train_objective":"CAUSAL_LANGUAGE_MODEL_ANSWER_ONLY_MASKED_PROMPT",
            "factual_accuracy_validated":False,"security_deployment_allowed":False}
    (out/"preflight.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("PHASE4C PREFLIGHT",json.dumps(details,indent=2))
    print("Input file hashes:",file_hashes)
    if any(ds[s].skipped_long>len(ds[s])*0.2 for s in ds):
        raise RuntimeError("Too many long examples excluded. Increase --context, then rerun preflight.")
    if args.check_only:
        print("CHECK ONLY SUCCESS - no model training or test scoring")
        return
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp=(device.type=="cuda" and torch.cuda.is_bf16_supported())
    spec=MODEL_SPECS["4m"]
    cfg=Config(**spec,max_seq_len=args.context)
    model=CyberSLM(cfg).to(device)
    nparams=sum(p.numel() for p in model.parameters())
    assert nparams==4058496, f"Unexpected parameter count: {nparams}"
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    train_loader=DataLoader(ds["train"],batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(ds["validation"],batch_size=args.batch_size,shuffle=False,num_workers=0)
    test_loader=DataLoader(ds["cross_scenario_test"],batch_size=args.batch_size,shuffle=False,num_workers=0)
    iterator=iter(train_loader)
    history=[]
    best_val=float("inf")
    best_step=None
    checkpoint=out/"best_validation_checkpoint.pt"
    print("DEVICE:",device,"PARAMETERS:",nparams,"AMP BF16:",amp)
    for step in range(1,args.steps+1):
        model.train()
        optim.zero_grad(set_to_none=True)
        total_answer_targets=0
        summed_loss=0.
        for _ in range(args.grad_accum):
            try: x,y=next(iterator)
            except StopIteration:
                iterator=iter(train_loader)
                x,y=next(iterator)
            x,y=x.to(device),y.to(device)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=amp):
                _,loss=model(x,y)
            valid=int((y!=-100).sum().item())
            if not valid: continue
            # Sum gradient numerator; divide once by actual answer targets.
            (loss*valid).backward()
            total_answer_targets+=valid
            summed_loss+=float(loss.item())*valid
        if not total_answer_targets:
            raise RuntimeError("No supervised answer tokens in this step")
        for param in model.parameters():
            if param.grad is not None:
                param.grad.div_(total_answer_targets)
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optim.step()
        if step==1 or step%args.eval_every==0 or step==args.steps:
            val=evaluation(model,val_loader,device,amp)
            record={"step":step,"train_answer_loss":round(summed_loss/total_answer_targets,6),
                    "validation_answer_loss":val["loss_nats_per_answer_byte"],
                    "tokens_in_last_step":total_answer_targets}
            history.append(record)
            print(json.dumps(record),flush=True)
            if val["loss_nats_per_answer_byte"]<best_val:
                best_val=val["loss_nats_per_answer_byte"]
                best_step=step
                torch.save({"model_state_dict":model.state_dict(),"config":asdict(cfg),
                            "selected_step":step,"train_source_hashes":file_hashes,
                            "experiment_type":"UNVERIFIED_RULE_BASED_QA_SMOKE_TEST"},checkpoint)
    (out/"training_metrics.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    selected=torch.load(checkpoint,map_location=device,weights_only=True)
    model.load_state_dict(selected["model_state_dict"])
    val_best=evaluation(model,val_loader,device,amp)
    # Test scored ONCE at end, never for model selection.
    test_final=evaluation(model,test_loader,device,amp)
    summary={**report,"parameters":nparams,"best_validation_checkpoint_step":best_step,
             "best_validation":val_best,"exploratory_cross_scenario_test":test_final,
             "notes":"Loss measures template-based answer byte prediction; NOT evidence factuality or APT accuracy."}
    (out/"final_evaluation.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("COMPLETE: Phase 4C non-deployable SMOKE TEST")
    print(json.dumps({"best_step":best_step,"validation":val_best,"cross_scenario_test":test_final},indent=2))
    print("Saved:",out)

if __name__=="__main__":
    main()
