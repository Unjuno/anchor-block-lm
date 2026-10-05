from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
SYNTH = HERE.parent / "synthetic"
sys.path.insert(0, str(SYNTH))

from poc import hidden_states, sample_contexts, seed_all
from probe_anchor_horizon import load_teacher

HORIZON = 6
RANK = 8
SURPRISAL_BUDGET = 2.5


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
def teacher_greedy_macro_targets(teacher, contexts: torch.Tensor, horizon: int = HORIZON):
    anchor_logits = teacher(contexts)[0][:, -1]
    anchor = anchor_logits.argmax(-1)

    ctx = torch.cat([contexts[:, 1:], anchor[:, None]], 1)
    tokens = []
    log_probs = []
    for _ in range(horizon):
        logits = teacher(ctx[:, -teacher.config.block_size:])[0][:, -1]
        tok = logits.argmax(-1)
        lp = logits.log_softmax(-1).gather(-1, tok[:, None]).squeeze(-1)
        tokens.append(tok)
        log_probs.append(lp)
        ctx = torch.cat([ctx[:, 1:], tok[:, None]], 1)

    return anchor, torch.stack(tokens, 1), torch.stack(log_probs, 1)


def make_augmented_targets(tokens: torch.Tensor, lengths: torch.Tensor, eob_id: int):
    B, H = tokens.shape
    target = torch.full((B, H + 1), -100, dtype=torch.long)
    for b in range(B):
        k = int(lengths[b])
        if k:
            target[b, :k] = tokens[b, :k]
        target[b, k] = eob_id
    return target


class LowRankResidual(nn.Module):
    """LoRA-style residual used only by the continuation branch.

    The autoregressive backbone and LM head are never modified, so the
    ordinary next-token path remains bit-identical to the teacher.
    """

    def __init__(self, width: int, rank: int, alpha: float | None = None):
        super().__init__()
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        self.A = nn.Parameter(torch.empty(rank, width))
        self.B = nn.Parameter(torch.zeros(width, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        return F.linear(F.linear(x, self.A), self.B) * (self.alpha / self.rank)


class OnePassAnchorStudent(nn.Module):
    """One backbone pass: exact AR anchor + cheap anchor-conditioned block."""

    def __init__(self, teacher, rank: int = RANK, horizon: int = HORIZON):
        super().__init__()
        self.backbone = copy.deepcopy(teacher)
        self.backbone.requires_grad_(False)
        self.rank = rank
        self.horizon = horizon

        width = teacher.config.n_embd
        self.continuation_lora = LowRankResidual(width, rank, alpha=rank)

        self.anchor_condition = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        nn.init.zeros_(self.anchor_condition[-1].weight)
        nn.init.zeros_(self.anchor_condition[-1].bias)

        bottleneck = max(16, width // 2)
        self.slot_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, width),
            )
            for _ in range(horizon + 1)
        ])
        for head in self.slot_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

        # Confidence-aware stop head. It sees each slot state plus token-head
        # confidence; it cannot alter the AR or token logits themselves.
        self.eob_head = nn.Sequential(
            nn.Linear(width + 3, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )
        nn.init.zeros_(self.eob_head[-1].weight)
        nn.init.constant_(self.eob_head[-1].bias, -2.0)

    @property
    def eob_id(self):
        return self.backbone.config.vocab_size

    def forward(self, idx: torch.Tensor, anchor_override: torch.Tensor | None = None):
        # This path is exactly the frozen teacher backbone.
        h = hidden_states(self.backbone, idx)[:, -1]
        anchor_logits = self.backbone.lm_head(h)

        if anchor_override is None:
            anchor = anchor_logits.argmax(-1)
        else:
            anchor = anchor_override

        anchor_emb = self.backbone.transformer.wte(anchor)
        z = (
            h
            + self.continuation_lora(h)
            + self.anchor_condition(torch.cat([h, anchor_emb], -1))
        )
        states = torch.stack([z + head(z) for head in self.slot_heads], 1)
        token_logits = self.backbone.lm_head(states)

        probs = token_logits.softmax(-1)
        top2 = probs.topk(2, dim=-1).values
        top1 = top2[..., 0]
        margin = top2[..., 0] - top2[..., 1]
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1)
        entropy = entropy / math.log(token_logits.shape[-1])
        conf = torch.stack([top1, margin, entropy], -1)

        eob_logits = self.eob_head(torch.cat([states, conf], -1))
        block_logits = torch.cat([token_logits, eob_logits], -1)
        return anchor_logits, block_logits


@torch.no_grad()
def prepare_bank(split: str, n: int, seed: int):
    teacher = load_teacher()
    seq = torch.tensor(np.load(HERE / "data" / f"{split}.npy").astype(np.int64))
    g = torch.Generator().manual_seed(seed)
    x = sample_contexts(seq, n, teacher.config.block_size, g)
    anchor, tokens, log_probs = teacher_greedy_macro_targets(teacher, x)
    lengths = target_lengths_from_log_probs(log_probs, SURPRISAL_BUDGET)
    target = make_augmented_targets(tokens, lengths, teacher.config.vocab_size)
    return {
        "x": x,
        "anchor": anchor,
        "tokens": tokens,
        "log_probs": log_probs,
        "lengths": lengths,
        "target": target,
    }


def prepare():
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    rows = []
    for split, n, seed in [("train", 4096, 8101), ("dev", 512, 8102), ("test", 1024, 8103)]:
        bank = prepare_bank(split, n, seed)
        torch.save(bank, out / f"onepass_{split}.pt")
        row = {
            "split": split,
            "n": n,
            "mean_continuation_length": bank["lengths"].float().mean().item(),
            "histogram": torch.bincount(
                bank["lengths"], minlength=HORIZON + 1
            ).tolist(),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out / "onepass_data_config.json").write_text(json.dumps({
        "anchor_target": "teacher greedy next token",
        "continuation_target": "teacher greedy rollout after realized anchor",
        "eob_rule": "longest prefix whose running mean teacher surprisal <= budget",
        "surprisal_budget_nats": SURPRISAL_BUDGET,
        "horizon": HORIZON,
        "ar_path": "fully frozen; LoRA-style residual exists only in continuation branch",
        "splits": rows,
    }, indent=2))


@torch.no_grad()
def evaluate(model: OnePassAnchorStudent, bank: dict):
    model.eval()
    anchor_hits = []
    ce = []
    length_exact = []
    token_acc = []
    for st in range(0, len(bank["x"]), 128):
        x = bank["x"][st:st+128]
        anchor = bank["anchor"][st:st+128]
        target = bank["target"][st:st+128]
        tokens = bank["tokens"][st:st+128]
        lengths = bank["lengths"][st:st+128]

        anchor_logits, block_logits = model(x, anchor_override=anchor)
        anchor_hits.extend(anchor_logits.argmax(-1).eq(anchor).float().tolist())

        loss = F.cross_entropy(
            block_logits.flatten(0, 1),
            target.flatten(),
            ignore_index=-100,
            reduction="none",
        ).reshape(len(x), HORIZON + 1)
        mask = target.ne(-100)
        ce.extend(((loss * mask).sum(1) / mask.sum(1)).tolist())

        pred = block_logits.argmax(-1)
        for i in range(len(x)):
            hits = (pred[i] == model.eob_id).nonzero(as_tuple=False)
            k = min(int(hits[0, 0]) if len(hits) else HORIZON, HORIZON)
            L = int(lengths[i])
            length_exact.append(float(k == L))
            token_acc.append(
                float(pred[i, :L].eq(tokens[i, :L]).float().mean())
                if L else 1.0
            )

    return {
        "anchor_accuracy": float(np.mean(anchor_hits)),
        "block_ce": float(np.mean(ce)),
        "length_exact": float(np.mean(length_exact)),
        "safe_prefix_token_accuracy": float(np.mean(token_acc)),
    }


def train(steps: int = 1200, seed: int = 8200):
    seed_all(seed, 2)
    teacher = load_teacher()
    model = OnePassAnchorStudent(teacher)

    train_bank = torch.load(HERE / "results" / "onepass_train.pt", weights_only=True)
    dev_bank = torch.load(HERE / "results" / "onepass_dev.pt", weights_only=True)

    lora_params = list(model.continuation_lora.parameters())
    head_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("continuation_lora.")
    ]

    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": 3e-4, "weight_decay": 0.0},
        {"params": head_params, "lr": 2e-3, "weight_decay": 1e-3},
    ])
    rng = torch.Generator().manual_seed(seed + 1)
    best = float("inf")
    history = []

    for step in range(1, steps + 1):
        model.train()
        ix = torch.randint(len(train_bank["x"]), (64,), generator=rng)
        x = train_bank["x"][ix]
        anchor = train_bank["anchor"][ix]
        target = train_bank["target"][ix]
        tokens = train_bank["tokens"][ix]

        anchor_logits, block_logits = model(x, anchor_override=anchor)

        # Anchor logits are a frozen invariant, not an optimization target.
        seq_loss = F.cross_entropy(
            block_logits.flatten(0, 1), target.flatten(), ignore_index=-100
        )
        aux = F.cross_entropy(
            block_logits[:, :HORIZON, :model.eob_id].reshape(-1, model.eob_id),
            tokens.reshape(-1),
        )
        loss = seq_loss + 0.25 * aux

        scale = 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * step / steps))
        opt.param_groups[0]["lr"] = 3e-4 * scale
        opt.param_groups[1]["lr"] = 2e-3 * scale

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
        opt.step()

        if step % 100 == 0 or step == steps:
            metrics = evaluate(model, dev_bank)
            row = {"step": step, "loss": loss.item(), **metrics}
            history.append(row)
            print(json.dumps(row), flush=True)
            if metrics["anchor_accuracy"] != 1.0:
                raise RuntimeError("frozen AR path changed")
            if metrics["block_ce"] < best:
                best = metrics["block_ce"]
                torch.save({
                    "model": model.state_dict(),
                    "rank": RANK,
                    "horizon": HORIZON,
                    "metrics": metrics,
                }, HERE / "results" / "onepass_student.pt")

    (HERE / "results" / "onepass_train_history.json").write_text(
        json.dumps(history, indent=2)
    )


def load_student(merged: bool = True):
    # No backbone LoRA exists in v2; argument retained for benchmark API.
    teacher = load_teacher()
    model = OnePassAnchorStudent(teacher)
    ck = torch.load(
        HERE / "results" / "onepass_student.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "train"])
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=8200)
    args = ap.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        train(args.steps, args.seed)


if __name__ == "__main__":
    main()
