from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SYNTH = HERE.parent / "synthetic"
sys.path.insert(0, str(SYNTH))
from poc import frozen_backbone_equal

from probe_anchor_horizon import load_teacher, sample_test_contexts
from train_block_student import HORIZON, load_student


def decode_variable(logits: torch.Tensor, eob_id: int, max_tokens: int, eob_bias: float):
    work = logits.clone()
    work[..., eob_id] += float(eob_bias)
    pred = work.argmax(-1)[0]
    limit = min(max_tokens, logits.shape[1] - 1)
    length = limit
    for i in range(limit + 1):
        if int(pred[i]) == eob_id:
            length = min(i, limit)
            break
    token_logits = logits[0, :limit, :eob_id]
    tokens = token_logits.argmax(-1)
    return tokens, length


def decode_fixed(logits: torch.Tensor, eob_id: int, k: int):
    if k < 1 or k > logits.shape[1] - 1:
        raise ValueError("invalid k")
    return logits[0, :k, :eob_id].argmax(-1)


@torch.no_grad()
def teacher_score(teacher, prefix: torch.Tensor, tokens: torch.Tensor):
    ctx = prefix.clone()
    matches = []
    nlls = []
    for tok in tokens:
        logits = teacher(ctx[:, -teacher.config.block_size:])[0][:, -1]
        t = tok.view(1)
        matches.append(float(logits.argmax(-1).eq(t).item()))
        nll = -logits.log_softmax(-1).gather(-1, t[:, None]).squeeze(-1)
        nlls.append(float(nll.item()))
        ctx = torch.cat([ctx, t[:, None]], 1)
    return matches, nlls


@torch.no_grad()
def generate(teacher, student, prefix, count, mode, value):
    out = prefix.clone()
    calls = 0
    while out.shape[1] - prefix.shape[1] < count:
        # ordinary AR anchor
        logits = teacher(out[:, -teacher.config.block_size:])[0][:, -1]
        anchor = logits.argmax(-1)
        out = torch.cat([out, anchor[:, None]], 1)
        calls += 1
        remaining = count - (out.shape[1] - prefix.shape[1])
        if remaining <= 0:
            break

        block_logits = student(out[:, -student.backbone.config.block_size:])
        if mode == "fixed":
            toks = decode_fixed(block_logits, student.eob_id, int(value))
            take = min(len(toks), remaining)
        elif mode == "variable":
            toks, length = decode_variable(
                block_logits, student.eob_id, HORIZON, float(value)
            )
            take = min(length, remaining)
        else:
            raise ValueError(mode)

        if take:
            out = torch.cat([out, toks[:take].view(1, -1)], 1)
        calls += 1
    return out[:, -count:], calls


@torch.no_grad()
def evaluate_policy(teacher, student, prefixes, count, mode, value):
    generated = []
    calls = []
    all_match = []
    all_nll = []
    for i in range(len(prefixes)):
        tokens, n_calls = generate(teacher, student, prefixes[i:i+1], count, mode, value)
        generated.append(tokens[0])
        calls.append(n_calls)
        matches, nlls = teacher_score(teacher, prefixes[i:i+1], tokens[0])
        all_match.extend(matches)
        all_nll.extend(nlls)
    return {
        "mode": mode,
        "value": value,
        "tokens_per_backbone_call": float(len(prefixes) * count / np.sum(calls)),
        "local_teacher_agreement": float(np.mean(all_match)),
        "local_teacher_nll": float(np.mean(all_nll)),
    }


def load_contexts(split: str, n: int, seed: int):
    teacher = load_teacher()
    seq = torch.tensor(np.load(HERE / "data" / f"{split}.npy").astype(np.int64))
    g = torch.Generator().manual_seed(seed)
    ix = torch.randint(0, len(seq) - teacher.config.block_size - 1, (n,), generator=g)
    return seq[ix[:, None] + torch.arange(teacher.config.block_size)[None, :]]


def choose_bias(dev_rows, minimum_speed=1.35):
    eligible = [r for r in dev_rows if r["tokens_per_backbone_call"] >= minimum_speed]
    if not eligible:
        return max(dev_rows, key=lambda r: r["tokens_per_backbone_call"])
    return max(eligible, key=lambda r: (r["local_teacher_agreement"], -r["local_teacher_nll"]))


@torch.no_grad()
def run(dev_prompts=48, test_prompts=96, count=48):
    teacher = load_teacher()
    unmerged = load_student(merged=False)
    frozen_ok = frozen_backbone_equal(teacher, unmerged)
    student = load_student(merged=True)

    dev = load_contexts("dev", dev_prompts, 5101)
    test = load_contexts("test", test_prompts, 5102)

    fixed_dev = [
        evaluate_policy(teacher, student, dev, count, "fixed", k)
        for k in (1, 2, 3, 4)
    ]
    variable_dev = [
        evaluate_policy(teacher, student, dev, count, "variable", bias)
        for bias in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    ]
    chosen = choose_bias(variable_dev)

    fixed_test = [
        evaluate_policy(teacher, student, test, count, "fixed", k)
        for k in (1, 2, 3, 4)
    ]
    variable_test = evaluate_policy(
        teacher, student, test, count, "variable", chosen["value"]
    )

    result = {
        "conditions": {
            "dev_prompts": dev_prompts,
            "test_prompts": test_prompts,
            "generated_tokens_per_prompt": count,
            "anchor": "teacher greedy",
            "student_base_frozen": frozen_ok,
            "variable_bias_selection": "best dev teacher agreement subject to >=1.35 tokens/call",
        },
        "dev_fixed": fixed_dev,
        "dev_variable_sweep": variable_dev,
        "selected_variable_bias": chosen["value"],
        "test_fixed": fixed_test,
        "test_variable": variable_test,
    }
    path = HERE / "results" / "block_benchmark.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-prompts", type=int, default=48)
    ap.add_argument("--test-prompts", type=int, default=96)
    ap.add_argument("--count", type=int, default=48)
    args = ap.parse_args()
    run(args.dev_prompts, args.test_prompts, args.count)


if __name__ == "__main__":
    main()
