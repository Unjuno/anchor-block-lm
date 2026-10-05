from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_onepass_anchor import (
    choose_bias,
    evaluate_policy,
    generate,
)
from benchmark_block_student import load_contexts, teacher_score
from calibrate_onepass_eob import load_calibrated_student
from probe_anchor_horizon import load_teacher
from poc import frozen_backbone_equal


@torch.no_grad()
def evaluate_detailed(teacher, student, prefixes, count, mode, value):
    calls = []
    prompt_agreement = []
    prompt_nll = []
    for i in range(len(prefixes)):
        tokens, n_calls = generate(
            student, prefixes[i:i+1], count, mode, value
        )
        matches, nlls = teacher_score(
            teacher, prefixes[i:i+1], tokens[0]
        )
        calls.append(n_calls)
        prompt_agreement.append(float(np.mean(matches)))
        prompt_nll.append(float(np.mean(nlls)))
    return {
        "mode": mode,
        "value": value,
        "tokens_per_backbone_call": float(
            len(prefixes) * count / np.sum(calls)
        ),
        "local_teacher_agreement": float(np.mean(prompt_agreement)),
        "local_teacher_nll": float(np.mean(prompt_nll)),
        "prompt_agreement": prompt_agreement,
        "prompt_nll": prompt_nll,
    }


@torch.no_grad()
def generate_random01(student, prefix, count, p_one, generator):
    out = prefix.clone()
    calls = 0
    while out.shape[1] - prefix.shape[1] < count:
        anchor_logits, block_logits = student(
            out[:, -student.backbone.config.block_size:]
        )
        anchor = anchor_logits.argmax(-1)
        emitted = [anchor]
        if (
            out.shape[1] - prefix.shape[1] + 1 < count
            and torch.rand((), generator=generator).item() < p_one
        ):
            continuation = block_logits[:, 0, :student.eob_id].argmax(-1)
            emitted.append(continuation)
        out = torch.cat(
            [out] + [token.view(1, 1) for token in emitted], dim=1
        )
        calls += 1
    return out[:, -count:], calls


@torch.no_grad()
def evaluate_random01(
    teacher, student, prefixes, count, p_one, seed
):
    generator = torch.Generator().manual_seed(seed)
    calls = []
    prompt_agreement = []
    prompt_nll = []
    for i in range(len(prefixes)):
        tokens, n_calls = generate_random01(
            student, prefixes[i:i+1], count, p_one, generator
        )
        matches, nlls = teacher_score(
            teacher, prefixes[i:i+1], tokens[0]
        )
        calls.append(n_calls)
        prompt_agreement.append(float(np.mean(matches)))
        prompt_nll.append(float(np.mean(nlls)))
    return {
        "mode": "random01",
        "p_one": float(p_one),
        "tokens_per_backbone_call": float(
            len(prefixes) * count / np.sum(calls)
        ),
        "local_teacher_agreement": float(np.mean(prompt_agreement)),
        "local_teacher_nll": float(np.mean(prompt_nll)),
        "prompt_agreement": prompt_agreement,
        "prompt_nll": prompt_nll,
    }


def choose_speed_matched_random(
    teacher, student, dev, count, target_speed
):
    center = float(np.clip(target_speed - 1.0, 0.0, 1.0))
    candidates = sorted({
        float(np.clip(center + delta, 0.0, 1.0))
        for delta in (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)
    })
    rows = [
        evaluate_random01(
            teacher, student, dev, count, p, 12100 + j
        )
        for j, p in enumerate(candidates)
    ]
    chosen = min(
        rows,
        key=lambda row: abs(
            row["tokens_per_backbone_call"] - target_speed
        ),
    )
    return rows, chosen


def bootstrap_paired_difference(a, b, samples=5000, seed=12200):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("paired arrays must have equal length")
    diff = a - b
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        ix = rng.integers(0, len(diff), len(diff))
        means[i] = diff[ix].mean()
    return {
        "mean_difference": float(diff.mean()),
        "ci95": [
            float(x) for x in np.quantile(means, [0.025, 0.975])
        ],
        "fraction_positive_prompts": float((diff > 0).mean()),
    }


def strip_prompt_arrays(row):
    return {
        k: v for k, v in row.items()
        if k not in ("prompt_agreement", "prompt_nll")
    }


def run(dev_prompts=64, test_prompts=128, count=64):
    teacher = load_teacher()
    student = load_calibrated_student()
    frozen_ok = frozen_backbone_equal(teacher, student)

    dev = load_contexts("dev", dev_prompts, 11101)
    test = load_contexts("test", test_prompts, 11102)

    fixed_dev = [
        evaluate_policy(teacher, student, dev, count, "fixed", k)
        for k in (0, 1, 2, 3)
    ]
    variable_dev = [
        evaluate_policy(teacher, student, dev, count, "variable", bias)
        for bias in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    ]
    chosen = choose_bias(variable_dev, minimum_speed=1.10)

    random_dev_sweep, random_dev = choose_speed_matched_random(
        teacher,
        student,
        dev,
        count,
        chosen["tokens_per_backbone_call"],
    )

    fixed_test = [
        evaluate_detailed(
            teacher, student, test, count, "fixed", k
        )
        for k in (0, 1, 2, 3)
    ]
    variable_test = evaluate_detailed(
        teacher, student, test, count, "variable", chosen["value"]
    )
    random_test = evaluate_random01(
        teacher,
        student,
        test,
        count,
        random_dev["p_one"],
        seed=12345,
    )

    if fixed_test[0]["local_teacher_agreement"] != 1.0:
        raise RuntimeError(
            "AR fallback is not exactly teacher-equivalent: "
            f"{fixed_test[0]}"
        )

    paired = bootstrap_paired_difference(
        variable_test["prompt_agreement"],
        random_test["prompt_agreement"],
    )

    result = {
        "conditions": {
            "dev_prompts": dev_prompts,
            "test_prompts": test_prompts,
            "generated_tokens_per_prompt": count,
            "architecture": (
                "one frozen backbone pass -> exact AR anchor -> "
                "continuation-branch low-rank adapter + confidence EOB"
            ),
            "student_base_frozen": frozen_ok,
            "eob_calibration": (
                "actual student acceptance under teacher greedy"
            ),
            "variable_bias_selection": (
                "best dev teacher agreement subject to >=1.10 tokens/call"
            ),
            "random_baseline": (
                "randomly emit zero or one continuation token per macro-step; "
                "probability selected on dev to match variable tokens/call"
            ),
        },
        "dev_fixed": fixed_dev,
        "dev_variable_sweep": variable_dev,
        "selected_variable_bias": chosen["value"],
        "dev_random01_sweep": [
            strip_prompt_arrays(row) for row in random_dev_sweep
        ],
        "selected_random01_p": random_dev["p_one"],
        "test_fixed": [
            strip_prompt_arrays(row) for row in fixed_test
        ],
        "test_variable": strip_prompt_arrays(variable_test),
        "test_random01_speed_matched": strip_prompt_arrays(random_test),
        "paired_variable_minus_random_agreement": paired,
    }

    path = (
        Path(__file__).resolve().parent
        / "results"
        / "onepass_calibrated_benchmark.json"
    )
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
