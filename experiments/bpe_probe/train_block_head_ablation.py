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


class LowRankResidual(nn.Module):
    def __init__(self, width: int, rank: int):
        super().__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.empty(rank, width))
        self.B = nn.Parameter(torch.zeros(width, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        return F.linear(F.linear(x, self.A), self.B)


class BlockContentStudent(nn.Module):
    """Content-only one-pass block predictor.

    The AR path is a frozen copy of the teacher. The experiment varies only the
    continuation head:
      - independent: one MLP residual per future position
      - transformer: learned slot queries coordinated by a small Transformer
    """

    def __init__(
        self,
        teacher,
        rank: int = RANK,
        horizon: int = HORIZON,
        head_kind: str = "independent",
    ):
        super().__init__()
        if head_kind not in ("independent", "transformer"):
            raise ValueError(head_kind)
        self.backbone = copy.deepcopy(teacher)
        self.backbone.requires_grad_(False)
        self.horizon = horizon
        self.head_kind = head_kind
        width = teacher.config.n_embd

        self.low_rank = LowRankResidual(width, rank)
        self.anchor_condition = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        nn.init.zeros_(self.anchor_condition[-1].weight)
        nn.init.zeros_(self.anchor_condition[-1].bias)

        if head_kind == "independent":
            bottleneck = max(16, width // 2)
            self.slot_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(width, bottleneck),
                    nn.GELU(),
                    nn.Linear(bottleneck, width),
                )
                for _ in range(horizon)
            ])
            for head in self.slot_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        else:
            self.slot_embeddings = nn.Parameter(
                torch.randn(horizon, width) * 0.02
            )
            nhead = 4 if width % 4 == 0 else 1
            layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=nhead,
                dim_feedforward=2 * width,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.block_transformer = nn.TransformerEncoder(layer, num_layers=1)
            self.block_norm = nn.LayerNorm(width)

    def forward(
        self,
        idx: torch.Tensor,
        anchor_override: torch.Tensor | None = None,
    ):
        h = hidden_states(self.backbone, idx)[:, -1]
        anchor_logits = self.backbone.lm_head(h)
        anchor = (
            anchor_logits.argmax(-1)
            if anchor_override is None
            else anchor_override
        )
        anchor_emb = self.backbone.transformer.wte(anchor)
        z = (
            h
            + self.low_rank(h)
            + self.anchor_condition(torch.cat([h, anchor_emb], -1))
        )

        if self.head_kind == "independent":
            states = torch.stack(
                [z + head(z) for head in self.slot_heads], 1
            )
        else:
            q = z[:, None, :] + self.slot_embeddings[None, :, :]
            states = self.block_norm(self.block_transformer(q))

        return anchor_logits, self.backbone.lm_head(states)


@torch.no_grad()
def make_bank(split: str, n: int, seed: int):
    teacher = load_teacher()
    seq = torch.tensor(np.load(HERE / "data" / f"{split}.npy").astype(np.int64))
    g = torch.Generator().manual_seed(seed)
    x = sample_contexts(seq, n, teacher.config.block_size, g)

    anchor_logits = teacher(x)[0][:, -1]
    anchor = anchor_logits.argmax(-1)
    ctx = torch.cat([x[:, 1:], anchor[:, None]], 1)
    tokens = []
    for _ in range(HORIZON):
        logits = teacher(ctx)[0][:, -1]
        tok = logits.argmax(-1)
        tokens.append(tok)
        ctx = torch.cat([ctx[:, 1:], tok[:, None]], 1)
    return {"x": x, "anchor": anchor, "tokens": torch.stack(tokens, 1)}


def prepare():
    out = HERE / "results"
    for split, n, seed in [
        ("train", 4096, 13101),
        ("dev", 512, 13102),
        ("test", 1024, 13103),
    ]:
        bank = make_bank(split, n, seed)
        torch.save(bank, out / f"head_ablation_{split}.pt")
        print(json.dumps({"split": split, "n": n}), flush=True)


def consecutive_teacher_acceptance(
    predicted: torch.Tensor,
    teacher_tokens: torch.Tensor,
):
    if predicted.shape != teacher_tokens.shape:
        raise ValueError("shape mismatch")
    alive = torch.ones(predicted.shape[0], dtype=torch.bool)
    lengths = torch.zeros(predicted.shape[0], dtype=torch.long)
    for h in range(predicted.shape[1]):
        alive = alive & predicted[:, h].eq(teacher_tokens[:, h])
        lengths += alive.long()
    return lengths


@torch.no_grad()
def actual_acceptance(teacher, x, anchor, predicted):
    ctx = torch.cat([x[:, 1:], anchor[:, None]], 1)
    teacher_tokens = []
    for h in range(predicted.shape[1]):
        logits = teacher(ctx)[0][:, -1]
        greedy = logits.argmax(-1)
        teacher_tokens.append(greedy)
        ctx = torch.cat([ctx[:, 1:], predicted[:, h:h+1]], 1)
    teacher_tokens = torch.stack(teacher_tokens, 1)
    return consecutive_teacher_acceptance(predicted, teacher_tokens)


@torch.no_grad()
def evaluate(model, teacher, bank):
    model.eval()
    slot_hits = []
    accept = []
    for st in range(0, len(bank["x"]), 128):
        x = bank["x"][st:st+128]
        anchor = bank["anchor"][st:st+128]
        targets = bank["tokens"][st:st+128]
        anchor_logits, block_logits = model(x, anchor_override=anchor)
        if not bool(anchor_logits.argmax(-1).eq(anchor).all()):
            raise RuntimeError("frozen anchor path changed")
        pred = block_logits.argmax(-1)
        slot_hits.append(pred.eq(targets).float())
        accept.append(actual_acceptance(teacher, x, anchor, pred))
    hits = torch.cat(slot_hits)
    lengths = torch.cat(accept)
    return {
        "slot_accuracy": hits.mean(0).tolist(),
        "mean_actual_acceptance": lengths.float().mean().item(),
        "acceptance_histogram": torch.bincount(
            lengths, minlength=HORIZON + 1
        ).tolist(),
        "fraction_accept_at_least_1": lengths.ge(1).float().mean().item(),
        "fraction_accept_at_least_2": lengths.ge(2).float().mean().item(),
        "fraction_accept_at_least_3": lengths.ge(3).float().mean().item(),
    }


def train(kind: str, steps: int = 1000, seed: int = 13200):
    seed_all(seed, 2)
    teacher = load_teacher()
    model = BlockContentStudent(teacher, head_kind=kind)
    train_bank = torch.load(
        HERE / "results" / "head_ablation_train.pt", weights_only=True
    )
    dev_bank = torch.load(
        HERE / "results" / "head_ablation_dev.pt", weights_only=True
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-3)
    rng = torch.Generator().manual_seed(seed + (0 if kind == "independent" else 100))
    best = -1.0
    history = []

    for step in range(1, steps + 1):
        model.train()
        ix = torch.randint(len(train_bank["x"]), (64,), generator=rng)
        x = train_bank["x"][ix]
        anchor = train_bank["anchor"][ix]
        target = train_bank["tokens"][ix]
        _, logits = model(x, anchor_override=anchor)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
        )

        scale = 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * step / steps))
        for group in opt.param_groups:
            group["lr"] = 2e-3 * scale
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % 100 == 0 or step == steps:
            metrics = evaluate(model, teacher, dev_bank)
            row = {"kind": kind, "step": step, "loss": loss.item(), **metrics}
            history.append(row)
            print(json.dumps(row), flush=True)
            if metrics["mean_actual_acceptance"] > best:
                best = metrics["mean_actual_acceptance"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "kind": kind,
                        "metrics": metrics,
                    },
                    HERE / "results" / f"head_ablation_{kind}.pt",
                )

    (HERE / "results" / f"head_ablation_{kind}_history.json").write_text(
        json.dumps(history, indent=2)
    )


def load_model(kind: str):
    teacher = load_teacher()
    model = BlockContentStudent(teacher, head_kind=kind)
    ck = torch.load(
        HERE / "results" / f"head_ablation_{kind}.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(ck["model"])
    model.eval()
    return model


@torch.no_grad()
def compare():
    teacher = load_teacher()
    test = torch.load(
        HERE / "results" / "head_ablation_test.pt", weights_only=True
    )
    result = {}
    for kind in ("independent", "transformer"):
        result[kind] = evaluate(load_model(kind), teacher, test)
    result["delta_transformer_minus_independent"] = {
        "mean_actual_acceptance": (
            result["transformer"]["mean_actual_acceptance"]
            - result["independent"]["mean_actual_acceptance"]
        )
    }
    path = HERE / "results" / "head_ablation_result.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "stage",
        choices=["prepare", "train-independent", "train-transformer", "compare"],
    )
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "train-independent":
        train("independent", args.steps)
    elif args.stage == "train-transformer":
        train("transformer", args.steps)
    else:
        compare()


if __name__ == "__main__":
    main()
