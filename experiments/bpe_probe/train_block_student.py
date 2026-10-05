from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
SYNTH = HERE.parent / "synthetic"
sys.path.insert(0, str(SYNTH))
from poc import hidden_states, inject_lora, merge_lora_, sample_contexts, seed_all

from probe_anchor_horizon import load_teacher, top_p_sample

HORIZON = 6
ROLLOUTS = 16
RANK = 8
SURPRISAL_BUDGET = 2.5


def select_best_of_n(tokens: torch.Tensor, log_probs: torch.Tensor):
    if tokens.shape != log_probs.shape or tokens.ndim != 3:
        raise ValueError("expected matching [B,N,H] tensors")
    score = log_probs.sum(-1)
    best = score.argmax(1)
    batch = torch.arange(tokens.shape[0], device=tokens.device)
    return tokens[batch, best], log_probs[batch, best]


def target_lengths_from_log_probs(log_probs: torch.Tensor, max_mean_surprisal: float):
    running = -log_probs.cumsum(-1) / torch.arange(
        1, log_probs.shape[1] + 1, device=log_probs.device
    )
    lengths = torch.zeros(log_probs.shape[0], dtype=torch.long, device=log_probs.device)
    for h in range(log_probs.shape[1]):
        safe = running[:, h] <= max_mean_surprisal
        lengths = torch.where(safe & lengths.eq(h), lengths + 1, lengths)
    return lengths


@torch.no_grad()
def sample_scored_rollouts(teacher, contexts, n_rollouts, horizon, generator):
    B = contexts.shape[0]
    ctx = (
        contexts[:, None, :]
        .expand(B, n_rollouts, contexts.shape[1])
        .reshape(B * n_rollouts, contexts.shape[1])
        .clone()
    )
    toks, lps = [], []
    for _ in range(horizon):
        logits = teacher(ctx[:, -teacher.config.block_size:])[0][:, -1]
        token = top_p_sample(logits, generator)
        lp = logits.log_softmax(-1).gather(-1, token[:, None]).squeeze(-1)
        toks.append(token.reshape(B, n_rollouts))
        lps.append(lp.reshape(B, n_rollouts))
        ctx = torch.cat([ctx, token[:, None]], 1)
    return torch.stack(toks, -1), torch.stack(lps, -1)


class BPEBlockStudent(nn.Module):
    def __init__(self, teacher, rank=RANK, horizon=HORIZON):
        super().__init__()
        self.backbone = copy.deepcopy(teacher)
        self.backbone.requires_grad_(False)
        inject_lora(self.backbone, rank)
        self.rank = rank
        self.horizon = horizon
        width = teacher.config.n_embd
        bottleneck = max(16, width // 2)
        self.slot_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(width, bottleneck), nn.GELU(), nn.Linear(bottleneck, width))
            for _ in range(horizon + 1)
        ])
        for head in self.slot_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.eob_head = nn.Linear(width, 1)
        nn.init.zeros_(self.eob_head.weight)
        nn.init.constant_(self.eob_head.bias, -2.0)

    @property
    def eob_id(self):
        return self.backbone.config.vocab_size

    def forward(self, idx):
        h = hidden_states(self.backbone, idx)[:, -1]
        states = torch.stack([h + head(h) for head in self.slot_heads], 1)
        token_logits = self.backbone.lm_head(states)
        eob_logits = self.eob_head(states)
        return torch.cat([token_logits, eob_logits], -1)


def make_augmented_targets(tokens, lengths, eob_id):
    B, H = tokens.shape
    target = torch.full((B, H + 1), -100, dtype=torch.long)
    for b in range(B):
        k = int(lengths[b])
        if k:
            target[b, :k] = tokens[b, :k]
        target[b, k] = eob_id
    return target


@torch.no_grad()
def prepare_bank(split, n, seed):
    teacher = load_teacher()
    seq = torch.tensor(np.load(HERE / "data" / f"{split}.npy").astype(np.int64))
    gctx = torch.Generator().manual_seed(seed)
    gsamp = torch.Generator().manual_seed(seed + 1000)
    x = sample_contexts(seq, n, teacher.config.block_size, gctx)

    # ordinary AR anchor, sampled from the same top-p policy used for distillation
    first_logits = teacher(x)[0][:, -1]
    anchor = top_p_sample(first_logits, gsamp)
    anchored = torch.cat([x[:, 1:], anchor[:, None]], 1)

    all_tokens, all_lps = [], []
    for st in range(0, n, 32):
        tokens, lps = sample_scored_rollouts(
            teacher, anchored[st:st+32], ROLLOUTS, HORIZON, gsamp
        )
        all_tokens.append(tokens)
        all_lps.append(lps)
    tokens = torch.cat(all_tokens)
    lps = torch.cat(all_lps)
    best_tokens, best_lps = select_best_of_n(tokens, lps)
    lengths = target_lengths_from_log_probs(best_lps, SURPRISAL_BUDGET)
    target = make_augmented_targets(best_tokens, lengths, teacher.config.vocab_size)
    return {
        "x": anchored,
        "anchor": anchor,
        "tokens": best_tokens,
        "log_probs": best_lps,
        "lengths": lengths,
        "target": target,
    }


def prepare():
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    rows = []
    for split, n, seed in [("train", 2048, 4101), ("dev", 256, 4102), ("test", 512, 4103)]:
        t0 = time.perf_counter()
        bank = prepare_bank(split, n, seed)
        torch.save(bank, out / f"block_{split}.pt")
        hist = torch.bincount(bank["lengths"], minlength=HORIZON + 1).tolist()
        row = {
            "split": split,
            "n": n,
            "mean_length": bank["lengths"].float().mean().item(),
            "histogram": hist,
            "elapsed_s": time.perf_counter() - t0,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out / "block_data_config.json").write_text(json.dumps({
        "horizon": HORIZON,
        "rollouts": ROLLOUTS,
        "top_p": 0.95,
        "surprisal_budget_nats": SURPRISAL_BUDGET,
        "candidate_selection": "best total teacher log-probability among top-p rollouts",
        "splits": rows,
    }, indent=2))


@torch.no_grad()
def evaluate(model, bank):
    model.eval()
    ce, len_exact, token_acc = [], [], []
    for st in range(0, len(bank["x"]), 128):
        x = bank["x"][st:st+128]
        target = bank["target"][st:st+128]
        tokens = bank["tokens"][st:st+128]
        lengths = bank["lengths"][st:st+128]
        logits = model(x)
        loss = F.cross_entropy(
            logits.flatten(0, 1), target.flatten(), ignore_index=-100, reduction="none"
        ).reshape(len(x), HORIZON + 1)
        mask = target.ne(-100)
        ce.extend(((loss * mask).sum(1) / mask.sum(1)).tolist())

        pred = logits.argmax(-1)
        for i in range(len(x)):
            hits = (pred[i] == model.eob_id).nonzero(as_tuple=False)
            k = min(int(hits[0, 0]) if len(hits) else HORIZON, HORIZON)
            L = int(lengths[i])
            len_exact.append(float(k == L))
            if L:
                token_acc.append(float(pred[i, :L].eq(tokens[i, :L]).float().mean()))
            else:
                token_acc.append(1.0)
    return {
        "ce": float(np.mean(ce)),
        "length_exact": float(np.mean(len_exact)),
        "safe_prefix_token_accuracy": float(np.mean(token_acc)),
    }


def train(steps=1200, seed=4200):
    seed_all(seed, 2)
    teacher = load_teacher()
    model = BPEBlockStudent(teacher)
    train_bank = torch.load(HERE / "results" / "block_train.pt", weights_only=True)
    dev_bank = torch.load(HERE / "results" / "block_dev.pt", weights_only=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-3)
    rng = torch.Generator().manual_seed(seed + 1)
    best = float("inf")
    history = []

    for step in range(1, steps + 1):
        model.train()
        ix = torch.randint(len(train_bank["x"]), (64,), generator=rng)
        logits = model(train_bank["x"][ix])
        target = train_bank["target"][ix]
        seq_loss = F.cross_entropy(
            logits.flatten(0, 1), target.flatten(), ignore_index=-100
        )
        aux = F.cross_entropy(
            logits[:, :HORIZON, :teacher.config.vocab_size].reshape(-1, teacher.config.vocab_size),
            train_bank["tokens"][ix].reshape(-1),
        )
        loss = seq_loss + 0.25 * aux
        lr = 2e-3 * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % 100 == 0 or step == steps:
            metrics = evaluate(model, dev_bank)
            row = {"step": step, "loss": loss.item(), **metrics}
            history.append(row)
            print(json.dumps(row), flush=True)
            if metrics["ce"] < best:
                best = metrics["ce"]
                torch.save({
                    "model": model.state_dict(),
                    "rank": RANK,
                    "horizon": HORIZON,
                    "metrics": metrics,
                }, HERE / "results" / "block_student.pt")

    (HERE / "results" / "block_train_history.json").write_text(json.dumps(history, indent=2))


def load_student(merged=True):
    teacher = load_teacher()
    model = BPEBlockStudent(teacher)
    ck = torch.load(HERE / "results" / "block_student.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    model.eval()
    if merged:
        merge_lora_(model)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "train"])
    ap.add_argument("--steps", type=int, default=1200)
    args = ap.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        train(args.steps)


if __name__ == "__main__":
    main()
