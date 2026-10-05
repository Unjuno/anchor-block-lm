from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
SYNTH = HERE.parent / "synthetic"
sys.path.insert(0, str(SYNTH))

from poc import sample_contexts, seed_all
from probe_anchor_horizon import load_teacher
from train_onepass_anchor_student import (
    HORIZON,
    OnePassAnchorStudent,
    load_student,
    make_augmented_targets,
)


def consecutive_acceptance_lengths(
    predicted: torch.Tensor,
    teacher_greedy: torch.Tensor,
):
    if predicted.shape != teacher_greedy.shape:
        raise ValueError("shape mismatch")
    lengths = torch.zeros(predicted.shape[0], dtype=torch.long)
    alive = torch.ones(predicted.shape[0], dtype=torch.bool)
    for h in range(predicted.shape[1]):
        alive = alive & predicted[:, h].eq(teacher_greedy[:, h])
        lengths += alive.long()
    return lengths


@torch.no_grad()
def label_contexts(contexts: torch.Tensor):
    teacher = load_teacher()
    student = load_student(merged=False)

    anchor_logits, block_logits = student(contexts)
    anchor = anchor_logits.argmax(-1)
    predicted = block_logits[:, :HORIZON, :student.eob_id].argmax(-1)

    ctx = torch.cat([contexts[:, 1:], anchor[:, None]], 1)
    teacher_tokens = []
    for h in range(HORIZON):
        logits = teacher(ctx[:, -teacher.config.block_size:])[0][:, -1]
        greedy = logits.argmax(-1)
        teacher_tokens.append(greedy)
        # Score the state actually induced by the student's predicted block.
        ctx = torch.cat([ctx[:, 1:], predicted[:, h:h+1]], 1)

    teacher_greedy = torch.stack(teacher_tokens, 1)
    lengths = consecutive_acceptance_lengths(predicted, teacher_greedy)
    target = make_augmented_targets(predicted, lengths, student.eob_id)
    return {
        "x": contexts,
        "anchor": anchor,
        "predicted": predicted,
        "teacher_greedy": teacher_greedy,
        "lengths": lengths,
        "target": target,
    }


@torch.no_grad()
def collect_fixed1_onpolicy_contexts(
    split: str,
    starts: int,
    cycles: int,
    seed: int,
):
    teacher = load_teacher()
    student = load_student(merged=False)
    seq = torch.tensor(np.load(HERE / "data" / f"{split}.npy").astype(np.int64))
    g = torch.Generator().manual_seed(seed)
    base = sample_contexts(seq, starts, teacher.config.block_size, g)

    contexts = []
    for i in range(starts):
        out = base[i:i+1].clone()
        for _ in range(cycles):
            contexts.append(out[:, -teacher.config.block_size:].clone()[0])
            anchor_logits, block_logits = student(
                out[:, -teacher.config.block_size:]
            )
            anchor = anchor_logits.argmax(-1)
            cont = block_logits[:, 0, :student.eob_id].argmax(-1)
            out = torch.cat([out, anchor[:, None], cont[:, None]], 1)
    return torch.stack(contexts)


def prepare():
    teacher = load_teacher()
    out = HERE / "results"
    rows = []

    for split, random_n, starts, cycles, seed in [
        ("train", 3072, 256, 4, 10101),
        ("dev", 512, 64, 2, 10102),
    ]:
        seq = torch.tensor(np.load(HERE / "data" / f"{split}.npy").astype(np.int64))
        g = torch.Generator().manual_seed(seed)
        random_contexts = sample_contexts(
            seq, random_n, teacher.config.block_size, g
        )
        onpolicy_contexts = collect_fixed1_onpolicy_contexts(
            split, starts, cycles, seed + 1000
        )
        contexts = torch.cat([random_contexts, onpolicy_contexts], 0)
        bank = label_contexts(contexts)
        torch.save(bank, out / f"onepass_calibration_{split}.pt")

        row = {
            "split": split,
            "random_contexts": random_n,
            "onpolicy_contexts": len(onpolicy_contexts),
            "total": len(contexts),
            "mean_accepted_continuation": bank["lengths"].float().mean().item(),
            "histogram": torch.bincount(
                bank["lengths"], minlength=HORIZON + 1
            ).tolist(),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    (out / "onepass_calibration_config.json").write_text(
        json.dumps({
            "label": (
                "longest consecutive prefix where student's parallel "
                "continuation token equals teacher greedy token under the "
                "student-induced prefix"
            ),
            "content_model_frozen_during_calibration": True,
            "splits": rows,
        }, indent=2)
    )


@torch.no_grad()
def evaluate(model, bank):
    model.eval()
    exact = []
    mae = []
    for st in range(0, len(bank["x"]), 128):
        x = bank["x"][st:st+128]
        true = bank["lengths"][st:st+128]
        _, logits = model(x)
        pred = logits.argmax(-1)
        lengths = []
        for i in range(len(x)):
            hits = (pred[i] == model.eob_id).nonzero(as_tuple=False)
            k = min(int(hits[0, 0]) if len(hits) else HORIZON, HORIZON)
            lengths.append(k)
        lengths = torch.tensor(lengths)
        exact.extend(lengths.eq(true).float().tolist())
        mae.extend((lengths - true).abs().float().tolist())
    return {
        "length_exact": float(np.mean(exact)),
        "length_mae": float(np.mean(mae)),
    }


def train(steps: int = 500, seed: int = 10200):
    seed_all(seed, 2)
    model = load_student(merged=False)

    # Freeze the entire content model. Only the stop/confidence head is
    # calibrated against actual student acceptance.
    model.requires_grad_(False)
    model.eob_head.requires_grad_(True)

    train_bank = torch.load(
        HERE / "results" / "onepass_calibration_train.pt",
        weights_only=True,
    )
    dev_bank = torch.load(
        HERE / "results" / "onepass_calibration_dev.pt",
        weights_only=True,
    )

    opt = torch.optim.AdamW(
        model.eob_head.parameters(), lr=2e-3, weight_decay=1e-3
    )
    rng = torch.Generator().manual_seed(seed + 1)
    best = float("inf")
    history = []

    for step in range(1, steps + 1):
        model.train()
        ix = torch.randint(len(train_bank["x"]), (128,), generator=rng)
        x = train_bank["x"][ix]
        target = train_bank["target"][ix]

        _, logits = model(x)
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            target.flatten(),
            ignore_index=-100,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == steps:
            metrics = evaluate(model, dev_bank)
            row = {"step": step, "loss": loss.item(), **metrics}
            history.append(row)
            print(json.dumps(row), flush=True)
            score = metrics["length_mae"]
            if score < best:
                best = score
                torch.save(
                    {
                        "model": model.state_dict(),
                        "metrics": metrics,
                    },
                    HERE / "results" / "onepass_student_calibrated.pt",
                )

    (HERE / "results" / "onepass_calibration_history.json").write_text(
        json.dumps(history, indent=2)
    )


def load_calibrated_student():
    teacher = load_teacher()
    model = OnePassAnchorStudent(teacher)
    ck = torch.load(
        HERE / "results" / "onepass_student_calibrated.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "train"])
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        train(args.steps)


if __name__ == "__main__":
    main()
