from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
SYNTH = HERE.parent / "synthetic"
sys.path.insert(0, str(SYNTH))
from poc import merge_lora_, sample_contexts, seed_all

from probe_anchor_horizon import load_teacher
from train_block_student import (
    BPEBlockStudent,
    HORIZON,
    RANK,
    ROLLOUTS,
    SURPRISAL_BUDGET,
    load_student,
    make_augmented_targets,
    sample_scored_rollouts,
    select_best_of_n,
    target_lengths_from_log_probs,
)
from benchmark_block_student import decode_variable


def targets_from_rollouts(tokens: torch.Tensor, log_probs: torch.Tensor, max_mean_surprisal: float):
    best_tokens, best_log_probs = select_best_of_n(tokens, log_probs)
    lengths = target_lengths_from_log_probs(best_log_probs, max_mean_surprisal)
    return best_tokens, lengths


@torch.no_grad()
def collect_student_contexts(
    starts: int = 256,
    cycles: int = 4,
    eob_bias: float = -1.0,
    seed: int = 6100,
):
    """Collect states actually visited by the current free-running student.

    Each saved context is immediately after a normal greedy AR anchor, i.e. the
    state from which the block student is invoked at inference.
    """
    teacher = load_teacher()
    student = load_student(merged=True)
    seq = torch.tensor(np.load(HERE / "data" / "train.npy").astype(np.int64))
    g = torch.Generator().manual_seed(seed)
    prefixes = sample_contexts(seq, starts, teacher.config.block_size, g)

    contexts = []
    for i in range(starts):
        out = prefixes[i:i+1].clone()
        for _ in range(cycles):
            # ordinary AR anchor
            logits = teacher(out[:, -teacher.config.block_size:])[0][:, -1]
            anchor = logits.argmax(-1)
            out = torch.cat([out, anchor[:, None]], 1)
            contexts.append(out[:, -teacher.config.block_size:].clone()[0])

            block_logits = student(out[:, -teacher.config.block_size:])
            tokens, length = decode_variable(
                block_logits, student.eob_id, HORIZON, eob_bias=eob_bias
            )
            if length:
                out = torch.cat([out, tokens[:length].view(1, -1)], 1)
    return torch.stack(contexts)


@torch.no_grad()
def relabel_contexts(contexts: torch.Tensor, seed: int = 6200):
    teacher = load_teacher()
    generator = torch.Generator().manual_seed(seed)
    all_tokens = []
    all_lps = []
    for st in range(0, len(contexts), 32):
        tokens, lps = sample_scored_rollouts(
            teacher,
            contexts[st:st+32],
            ROLLOUTS,
            HORIZON,
            generator,
        )
        all_tokens.append(tokens)
        all_lps.append(lps)
    roll_tokens = torch.cat(all_tokens)
    roll_lps = torch.cat(all_lps)
    best_tokens, lengths = targets_from_rollouts(
        roll_tokens, roll_lps, SURPRISAL_BUDGET
    )
    target = make_augmented_targets(
        best_tokens, lengths, teacher.config.vocab_size
    )
    return {
        "x": contexts,
        "tokens": best_tokens,
        "lengths": lengths,
        "target": target,
    }


def prepare(starts=256, cycles=4, eob_bias=-1.0):
    out = HERE / "results"
    contexts = collect_student_contexts(starts, cycles, eob_bias)
    bank = relabel_contexts(contexts)
    torch.save(bank, out / "block_onpolicy_train.pt")
    info = {
        "contexts": len(contexts),
        "starts": starts,
        "cycles": cycles,
        "rollout_eob_bias": eob_bias,
        "mean_relabel_length": bank["lengths"].float().mean().item(),
        "histogram": torch.bincount(
            bank["lengths"], minlength=HORIZON + 1
        ).tolist(),
    }
    (out / "block_onpolicy_config.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


@torch.no_grad()
def evaluate(model, bank):
    model.eval()
    ce = []
    exact = []
    token_acc = []
    for st in range(0, len(bank["x"]), 128):
        x = bank["x"][st:st+128]
        target = bank["target"][st:st+128]
        tokens = bank["tokens"][st:st+128]
        lengths = bank["lengths"][st:st+128]
        logits = model(x)
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            target.flatten(),
            ignore_index=-100,
            reduction="none",
        ).reshape(len(x), HORIZON + 1)
        mask = target.ne(-100)
        ce.extend(((loss * mask).sum(1) / mask.sum(1)).tolist())

        pred = logits.argmax(-1)
        for i in range(len(x)):
            hits = (pred[i] == model.eob_id).nonzero(as_tuple=False)
            k = min(int(hits[0, 0]) if len(hits) else HORIZON, HORIZON)
            L = int(lengths[i])
            exact.append(float(k == L))
            token_acc.append(
                float(pred[i, :L].eq(tokens[i, :L]).float().mean())
                if L else 1.0
            )
    return {
        "ce": float(np.mean(ce)),
        "length_exact": float(np.mean(exact)),
        "safe_prefix_token_accuracy": float(np.mean(token_acc)),
    }


def finetune(steps=400, seed=6300):
    seed_all(seed, 2)
    # Continue from the teacher-forced checkpoint, preserving frozen base weights.
    model = load_student(merged=False)
    original = torch.load(HERE / "results" / "block_train.pt", weights_only=True)
    onpolicy = torch.load(
        HERE / "results" / "block_onpolicy_train.pt", weights_only=True
    )
    dev = torch.load(HERE / "results" / "block_dev.pt", weights_only=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=8e-4, weight_decay=1e-3)
    rng = torch.Generator().manual_seed(seed + 1)

    best = float("inf")
    history = []
    for step in range(1, steps + 1):
        model.train()
        n = 32
        ix = torch.randint(len(original["x"]), (n,), generator=rng)
        jx = torch.randint(len(onpolicy["x"]), (n,), generator=rng)
        x = torch.cat([original["x"][ix], onpolicy["x"][jx]], 0)
        target = torch.cat(
            [original["target"][ix], onpolicy["target"][jx]], 0
        )
        tokens = torch.cat(
            [original["tokens"][ix], onpolicy["tokens"][jx]], 0
        )

        logits = model(x)
        seq_loss = F.cross_entropy(
            logits.flatten(0, 1), target.flatten(), ignore_index=-100
        )
        aux = F.cross_entropy(
            logits[:, :HORIZON, :model.eob_id].reshape(-1, model.eob_id),
            tokens.reshape(-1),
        )
        loss = seq_loss + 0.25 * aux

        lr = 8e-4 * (
            0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * step / steps))
        )
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % 50 == 0 or step == steps:
            metrics = evaluate(model, dev)
            row = {"step": step, "loss": loss.item(), **metrics}
            history.append(row)
            print(json.dumps(row), flush=True)
            if metrics["ce"] < best:
                best = metrics["ce"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "rank": RANK,
                        "horizon": HORIZON,
                        "metrics": metrics,
                    },
                    HERE / "results" / "block_student_onpolicy.pt",
                )
    (HERE / "results" / "block_onpolicy_history.json").write_text(
        json.dumps(history, indent=2)
    )


def load_onpolicy_student(merged=True):
    teacher = load_teacher()
    model = BPEBlockStudent(teacher)
    ck = torch.load(
        HERE / "results" / "block_student_onpolicy.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(ck["model"])
    model.eval()
    if merged:
        merge_lora_(model)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "train"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--starts", type=int, default=256)
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--eob-bias", type=float, default=-1.0)
    args = ap.parse_args()
    if args.stage == "prepare":
        prepare(args.starts, args.cycles, args.eob_bias)
    else:
        finetune(args.steps)


if __name__ == "__main__":
    main()
