from __future__ import annotations

import argparse
import json

from benchmark_block_student import (
    choose_bias,
    evaluate_policy,
    load_contexts,
)
from probe_anchor_horizon import load_teacher
from onpolicy_block_refresh import load_onpolicy_student
from poc import frozen_backbone_equal


def run(dev_prompts=48, test_prompts=96, count=48):
    teacher = load_teacher()
    unmerged = load_onpolicy_student(merged=False)
    frozen_ok = frozen_backbone_equal(teacher, unmerged)
    student = load_onpolicy_student(merged=True)

    dev = load_contexts("dev", dev_prompts, 7101)
    test = load_contexts("test", test_prompts, 7102)

    fixed_dev = [
        evaluate_policy(teacher, student, dev, count, "fixed", k)
        for k in (1, 2, 3, 4)
    ]
    variable_dev = [
        evaluate_policy(teacher, student, dev, count, "variable", bias)
        for bias in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    ]
    chosen = choose_bias(variable_dev, minimum_speed=1.35)

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
            "student": "after on-policy self-distillation refresh",
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
    from pathlib import Path
    path = Path(__file__).resolve().parent / "results" / "block_onpolicy_benchmark.json"
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
