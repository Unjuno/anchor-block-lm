from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_block_student import load_contexts
from benchmark_onepass_calibrated import (
    bootstrap_paired_difference,
    evaluate_detailed,
    evaluate_random01,
    strip_prompt_arrays,
)
from calibrate_onepass_eob import load_calibrated_student
from probe_anchor_horizon import load_teacher
from poc import frozen_backbone_equal

# Frozen from the earlier 16-dev / 32-test diagnostic run.
# These are NOT selected on the expanded test.
PRESELECTED_EOB_BIAS = 2.0
PRESELECTED_RANDOM_P = 0.1327433628318584


def run(test_prompts=64, count=48):
    teacher = load_teacher()
    student = load_calibrated_student()
    frozen_ok = frozen_backbone_equal(teacher, student)
    test = load_contexts("test", test_prompts, 11102)

    ar = evaluate_detailed(
        teacher, student, test, count, "fixed", 0
    )
    fixed1 = evaluate_detailed(
        teacher, student, test, count, "fixed", 1
    )
    variable = evaluate_detailed(
        teacher,
        student,
        test,
        count,
        "variable",
        PRESELECTED_EOB_BIAS,
    )
    random01 = evaluate_random01(
        teacher,
        student,
        test,
        count,
        PRESELECTED_RANDOM_P,
        seed=12345,
    )

    if ar["local_teacher_agreement"] != 1.0:
        raise RuntimeError("AR fallback invariant failed")

    paired = bootstrap_paired_difference(
        variable["prompt_agreement"],
        random01["prompt_agreement"],
        samples=10000,
        seed=14100,
    )

    result = {
        "conditions": {
            "test_prompts": test_prompts,
            "generated_tokens_per_prompt": count,
            "selection": (
                "EOB bias and random gating probability frozen from the "
                "earlier small dev diagnostic; no tuning on this expanded test"
            ),
            "preselected_eob_bias": PRESELECTED_EOB_BIAS,
            "preselected_random_p": PRESELECTED_RANDOM_P,
            "student_base_frozen": frozen_ok,
        },
        "ar": strip_prompt_arrays(ar),
        "fixed1": strip_prompt_arrays(fixed1),
        "variable_eob": strip_prompt_arrays(variable),
        "random01": strip_prompt_arrays(random01),
        "paired_variable_minus_random_agreement": paired,
    }

    path = (
        Path(__file__).resolve().parent
        / "results"
        / "onepass_preregistered_benchmark.json"
    )
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-prompts", type=int, default=64)
    ap.add_argument("--count", type=int, default=48)
    args = ap.parse_args()
    run(args.test_prompts, args.count)


if __name__ == "__main__":
    main()
