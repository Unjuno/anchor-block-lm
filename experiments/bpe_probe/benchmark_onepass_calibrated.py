from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_onepass_anchor import (
    choose_bias,
    evaluate_policy,
)
from benchmark_block_student import load_contexts
from calibrate_onepass_eob import load_calibrated_student
from probe_anchor_horizon import load_teacher
from poc import frozen_backbone_equal


def run(dev_prompts=32, test_prompts=64, count=48):
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
    chosen = choose_bias(variable_dev, minimum_speed=1.35)

    fixed_test = [
        evaluate_policy(teacher, student, test, count, "fixed", k)
        for k in (0, 1, 2, 3)
    ]
    variable_test = evaluate_policy(
        teacher, student, test, count, "variable", chosen["value"]
    )

    if fixed_test[0]["local_teacher_agreement"] != 1.0:
        raise RuntimeError(
            "AR fallback is not exactly teacher-equivalent: "
            f"{fixed_test[0]}"
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
            "eob_calibration": "actual student acceptance under teacher greedy",
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
    ap.add_argument("--dev-prompts", type=int, default=32)
    ap.add_argument("--test-prompts", type=int, default=64)
    ap.add_argument("--count", type=int, default=48)
    args = ap.parse_args()
    run(args.dev_prompts, args.test_prompts, args.count)


if __name__ == "__main__":
    main()
