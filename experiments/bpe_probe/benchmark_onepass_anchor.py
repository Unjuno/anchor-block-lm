from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from probe_anchor_horizon import load_teacher
from benchmark_block_student import load_contexts, teacher_score
from train_onepass_anchor_student import HORIZON, load_student
from poc import frozen_backbone_equal


def decode_macro(
    anchor_logits: torch.Tensor,
    block_logits: torch.Tensor,
    eob_id: int,
    max_continuation: int,
    eob_bias: float,
):
    anchor = anchor_logits.argmax(-1)[0]
    work = block_logits.clone()
    work[..., eob_id] += float(eob_bias)
    pred = work.argmax(-1)[0]

    length = min(max_continuation, block_logits.shape[1] - 1)
    for i in range(length + 1):
        if int(pred[i]) == eob_id:
            length = min(i, length)
            break

    continuation = block_logits[0, :length, :eob_id].argmax(-1)
    return torch.cat([anchor.view(1), continuation])


def decode_macro_fixed(
    anchor_logits: torch.Tensor,
    block_logits: torch.Tensor,
    eob_id: int,
    k: int,
):
    if k < 0 or k > block_logits.shape[1] - 1:
        raise ValueError("invalid k")
    anchor = anchor_logits.argmax(-1)[0]
    continuation = block_logits[0, :k, :eob_id].argmax(-1)
    return torch.cat([anchor.view(1), continuation])


@torch.no_grad()
def generate(student, prefix: torch.Tensor, count: int, mode: str, value: float):
    out = prefix.clone()
    calls = 0
    while out.shape[1] - prefix.shape[1] < count:
        anchor_logits, block_logits = student(
            out[:, -student.backbone.config.block_size:]
        )
        if mode == "fixed":
            emitted = decode_macro_fixed(
                anchor_logits, block_logits, student.eob_id, int(value)
            )
        elif mode == "variable":
            emitted = decode_macro(
                anchor_logits,
                block_logits,
                student.eob_id,
                HORIZON,
                float(value),
            )
        else:
            raise ValueError(mode)

        remaining = count - (out.shape[1] - prefix.shape[1])
        emitted = emitted[:remaining]
        out = torch.cat([out, emitted.view(1, -1)], 1)
        calls += 1
    return out[:, -count:], calls


@torch.no_grad()
def evaluate_policy(teacher, student, prefixes, count, mode, value):
    calls = []
    all_match = []
    all_nll = []
    for i in range(len(prefixes)):
        tokens, n_calls = generate(
            student, prefixes[i:i+1], count, mode, value
        )
        calls.append(n_calls)
        matches, nlls = teacher_score(
            teacher, prefixes[i:i+1], tokens[0]
        )
        all_match.extend(matches)
        all_nll.extend(nlls)
    return {
        "mode": mode,
        "value": value,
        "tokens_per_backbone_call": float(len(prefixes) * count / np.sum(calls)),
        "local_teacher_agreement": float(np.mean(all_match)),
        "local_teacher_nll": float(np.mean(all_nll)),
    }


def choose_bias(dev_rows, minimum_speed=1.35):
    eligible = [
        r for r in dev_rows
        if r["tokens_per_backbone_call"] >= minimum_speed
    ]
    if not eligible:
        return max(dev_rows, key=lambda r: r["tokens_per_backbone_call"])
    return max(
        eligible,
        key=lambda r: (
            r["local_teacher_agreement"],
            -r["local_teacher_nll"],
        ),
    )


@torch.no_grad()
def run(dev_prompts=64, test_prompts=128, count=64):
    teacher = load_teacher()
    unmerged = load_student(merged=False)
    frozen_ok = frozen_backbone_equal(teacher, unmerged)
    student = load_student(merged=True)

    dev = load_contexts("dev", dev_prompts, 9101)
    test = load_contexts("test", test_prompts, 9102)

    fixed_dev = [
        evaluate_policy(teacher, student, dev, count, "fixed", k)
        for k in (0, 1, 2, 3)
    ]
    variable_dev = [
        evaluate_policy(teacher, student, dev, count, "variable", bias)
        for bias in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    ]
    chosen = choose_bias(variable_dev)

    fixed_test = [
        evaluate_policy(teacher, student, test, count, "fixed", k)
        for k in (0, 1, 2, 3)
    ]
    variable_test = evaluate_policy(
        teacher, student, test, count, "variable", chosen["value"]
    )

    result = {
        "conditions": {
            "dev_prompts": dev_prompts,
            "test_prompts": test_prompts,
            "generated_tokens_per_prompt": count,
            "architecture": (
                "one backbone pass -> ordinary anchor token -> "
                "cheap anchor-conditioned continuation block"
            ),
            "student_base_frozen": frozen_ok,
            "variable_bias_selection": (
                "best dev teacher agreement subject to >=1.35 tokens/call"
            ),
        },
        "dev_fixed": fixed_dev,
        "dev_variable_sweep": variable_dev,
        "selected_variable_bias": chosen["value"],
        "test_fixed": fixed_test,
        "test_variable": variable_test,
    }
    path = Path(__file__).resolve().parent / "results" / "onepass_benchmark.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-prompts", type=int, default=64)
    ap.add_argument("--test-prompts", type=int, default=128)
    ap.add_argument("--count", type=int, default=64)
    args = ap.parse_args()
    run(args.dev_prompts, args.test_prompts, args.count)


if __name__ == "__main__":
    main()
