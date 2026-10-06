"""Iterative dynamic-boundary experiment.

Each round:
1) recollect actor-visited states using the current continuation model + k policy,
2) improve continuation LoRA by unfiltered teacher KD on those states,
3) improve the length-policy LoRA with a KL-constrained reward,
4) reevaluate the same fixed probe states.

The evaluator and exact AR anchor path are frozen throughout.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TWO = HERE.parent / "two_model_rl"
ADAPT = HERE.parent / "adaptive_k_rl"
FIXED = HERE.parent / "fixed_k_gate"
for p in (TWO, ADAPT, FIXED):
    sys.path.insert(0, str(p))

import fixed_k as fk
from adaptive_k import AdaptiveKGate
from evaluate_saved_policy import prompt_cluster_interval
from run_two_model import (
    deployment_copy,
    evaluate,
    load_saved,
    measure_costs,
    reconstruct_splits,
    time_generation,
)
from two_model import policy_batch
from dynamic_boundary import (
    KLDualController,
    collect_on_policy_states,
    enable_content_lora,
    probe_policy,
    train_round,
)


def save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def state_bytes_hash(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sample_contexts(seq: torch.Tensor, n: int, window: int, seed: int):
    if n < 1 or len(seq) <= window:
        raise ValueError("invalid context sample request")
    g = torch.Generator().manual_seed(seed)
    starts = torch.randperm(len(seq) - window, generator=g)[:n]
    return seq[starts[:, None] + torch.arange(window)], starts


def initial_selected_risk_target(prefix_risks: torch.Tensor, k: torch.Tensor) -> float:
    if prefix_risks.ndim != 2 or k.shape != (len(prefix_risks),):
        raise ValueError("risk/k shapes do not match")
    selected = prefix_risks.gather(-1, (k.long() - 1)[:, None]).squeeze(-1)
    return float(selected.mean())


@torch.no_grad()
def evaluate_fixed_probe(
    teacher,
    student,
    gate,
    probe_contexts: torch.Tensor,
    *,
    seed: int,
    samples: int,
):
    bank = policy_batch(
        teacher,
        student,
        probe_contexts,
        torch.Generator().manual_seed(seed),
        samples=samples,
    )
    return probe_policy(gate, bank["features"], bank["risk"])


def train_dynamic_loop(
    teacher,
    student,
    gate,
    train_seq: torch.Tensor,
    probe_contexts: torch.Tensor,
    *,
    rounds: int,
    content_steps: int,
    policy_steps: int,
    state_contexts: int,
    cycles: int,
    batch_size: int,
    samples: int,
    costs: torch.Tensor,
    initial_dual: float,
    dual_lr: float,
    kl_target: float,
    seed: int,
):
    if min(
        rounds, content_steps, policy_steps, state_contexts,
        cycles, batch_size, samples
    ) < 1:
        raise ValueError("training sizes must be positive")

    # Keep the policy LoRA-only. The caller may already have a LoRA continuation
    # model; a plain FixedKStudent is also supported by unit tests.
    gate.eval().requires_grad_(False)
    for name, p in gate.named_parameters():
        if name.endswith((".A", ".B")):
            p.requires_grad_(True)

    dual = KLDualController(initial=initial_dual, lr=dual_lr, target=kl_target)
    probe_seed = seed + 90_000
    probe_history = [
        {
            "round": 0,
            "dual_lambda": dual.value,
            **evaluate_fixed_probe(
                teacher, student, gate, probe_contexts,
                seed=probe_seed, samples=max(samples, 8),
            ),
        }
    ]
    visited_hashes = []
    round_history = []
    window = teacher.config.block_size
    starts_rng = torch.Generator().manual_seed(seed + 11)

    for r in range(rounds):
        starts = torch.randint(
            len(train_seq) - window,
            (state_contexts,),
            generator=starts_rng,
        )
        prefixes = train_seq[starts[:, None] + torch.arange(window)]
        states = collect_on_policy_states(
            student,
            gate,
            prefixes,
            cycles=cycles,
            seed=seed + 1_000 + r,
        )
        visited_hashes.append(state_bytes_hash(states))
        row = train_round(
            teacher,
            student,
            gate,
            states,
            content_steps=content_steps,
            policy_steps=policy_steps,
            batch_size=batch_size,
            samples=samples,
            costs=costs,
            dual=dual,
            seed=seed + 10_000 + r,
        )
        round_history.append(
            {
                "round": r + 1,
                "training_context_starts": starts.tolist(),
                "visited_state_sha256": visited_hashes[-1],
                **row,
            }
        )
        probe_history.append(
            {
                "round": r + 1,
                "dual_lambda": dual.value,
                **evaluate_fixed_probe(
                    teacher,
                    student,
                    gate,
                    probe_contexts,
                    seed=probe_seed,
                    samples=max(samples, 8),
                ),
            }
        )

    return {
        "probe_history": probe_history,
        "round_history": round_history,
        "visited_state_hashes": visited_hashes,
        "final_dual_lambda": dual.value,
        "kl_target": kl_target,
    }


def run(args):
    if min(
        args.rounds, args.content_steps, args.policy_steps,
        args.state_contexts, args.cycles, args.batch_size,
        args.samples, args.constraint_contexts, args.dev_probe_contexts,
        args.eval_prompts, args.eval_draws, args.count
    ) < 1:
        raise ValueError("all experiment sizes must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    data, meta = reconstruct_splits(args.artifact_root, args.out)
    teacher, archived_student, archived_gate, old = load_saved(
        args.artifact_root / "outputs" / f"seed-{args.seed}"
    )
    teacher_hash = fk.state_hash(teacher)

    student = enable_content_lora(copy.deepcopy(archived_student), rank=args.content_rank)
    gate = copy.deepcopy(archived_gate).eval().requires_grad_(False)
    for name, p in gate.named_parameters():
        if name.endswith((".A", ".B")):
            p.requires_grad_(True)

    # Fixed train-only probe defines the quality budget. Dev never tunes epsilon.
    constraint_x, constraint_starts = sample_contexts(
        data["train"], args.constraint_contexts, teacher.config.block_size,
        args.seed + 100,
    )
    constraint_bank = policy_batch(
        teacher, student, constraint_x,
        torch.Generator().manual_seed(args.seed + 101),
        samples=max(args.samples, 16),
    )
    with torch.no_grad():
        initial_k = gate(constraint_bank["features"]).argmax(-1) + 1
    kl_target = initial_selected_risk_target(constraint_bank["risk"], initial_k)

    dev_probe, dev_starts = sample_contexts(
        data["dev"], args.dev_probe_contexts, teacher.config.block_size,
        args.seed + 200,
    )

    cost_x, _ = sample_contexts(
        data["train"], min(32, args.constraint_contexts),
        teacher.config.block_size, args.seed + 300,
    )
    cost_student = deployment_copy(archived_student)
    cost_gate = deployment_copy(archived_gate)
    costs, cost_info = measure_costs(cost_student, cost_gate, cost_x)

    initial_student = copy.deepcopy(student).eval().requires_grad_(False)
    initial_gate = copy.deepcopy(gate).eval().requires_grad_(False)

    training = train_dynamic_loop(
        teacher,
        student,
        gate,
        data["train"],
        dev_probe,
        rounds=args.rounds,
        content_steps=args.content_steps,
        policy_steps=args.policy_steps,
        state_contexts=args.state_contexts,
        cycles=args.cycles,
        batch_size=args.batch_size,
        samples=args.samples,
        costs=costs,
        initial_dual=args.initial_dual,
        dual_lr=args.dual_lr,
        kl_target=kl_target,
        seed=args.seed + 1_000,
    )

    if fk.state_hash(teacher) != teacher_hash:
        raise RuntimeError("frozen evaluator changed")
    if fk.state_hash(student.backbone) != teacher_hash:
        raise RuntimeError("exact AR backbone changed")

    eval_x, eval_starts = sample_contexts(
        data["dev"], args.eval_prompts, teacher.config.block_size,
        args.seed + 400,
    )
    evaluations = {}
    raw = {}
    for name, s, g in (
        ("initial", initial_student, initial_gate),
        ("final", student.eval().requires_grad_(False), gate.eval().requires_grad_(False)),
    ):
        metrics, arrays = evaluate(
            teacher, s, g, eval_x,
            draws=args.eval_draws, count=args.count,
            seed=args.seed + 5_000,
        )
        ds = deployment_copy(s)
        dg = deployment_copy(g)
        metrics["timing"] = time_generation(
            ds, dg, eval_x[:min(8, len(eval_x))],
            count=args.count, seed=args.seed + 6_000, repeats=5,
        )
        evaluations[name] = metrics
        raw[name] = arrays

    ar_student = deployment_copy(archived_student)
    ar_metrics, ar_arrays = evaluate(
        teacher, ar_student, None, eval_x,
        draws=args.eval_draws, count=args.count,
        seed=args.seed + 5_000,
    )
    ar_metrics["timing"] = time_generation(
        ar_student, None, eval_x[:min(8, len(eval_x))],
        count=args.count, seed=args.seed + 6_000, repeats=5,
    )
    evaluations["ar"] = ar_metrics
    raw["ar"] = ar_arrays

    ar_time = evaluations["ar"]["timing"]["median_seconds"]
    for row in evaluations.values():
        row["speed_vs_pure_ar"] = ar_time / row["timing"]["median_seconds"]

    diff = (raw["final"]["log_ratio"] - raw["initial"]["log_ratio"]) / args.count
    result = {
        "experiment_version": "dynamic-boundary-constrained-rl-v1",
        "seed": args.seed,
        "dataset": meta["dataset"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cpu": next(
                (line.split(":", 1)[1].strip()
                 for line in Path("/proc/cpuinfo").read_text().splitlines()
                 if line.startswith("model name")),
                "unknown",
            ),
            "threads": 1,
            "dtype": "float32",
            "kv_cache": False,
        },
        "protocol": {
            "max_horizon": fk.BLOCK_SIZE,
            "commit_actions": [1, 2, 3, 4],
            "content_update": "teacher-sample KD only on current actor-visited states",
            "policy_update": "categorical LoRA REINFORCE with adaptive KL Lagrange multiplier",
            "confidence_is_reward": False,
            "kl_target_source": "fixed train-only initial-policy probe",
            "kl_target": kl_target,
            "test_split_used": False,
            "rounds": args.rounds,
            "content_steps_per_round": args.content_steps,
            "policy_steps_per_round": args.policy_steps,
            "state_contexts_per_round": args.state_contexts,
            "on_policy_cycles": args.cycles,
            "samples_per_teacher_estimate": args.samples,
            "normalized_action_costs": costs.tolist(),
        },
        "invariants": {
            "evaluator_unchanged": fk.state_hash(teacher) == teacher_hash,
            "ar_backbone_unchanged": fk.state_hash(student.backbone) == teacher_hash,
            "content_changed": fk.state_hash(student) != fk.state_hash(initial_student),
            "gate_changed": fk.state_hash(gate) != fk.state_hash(initial_gate),
        },
        "constraint_probe_starts": constraint_starts.tolist(),
        "dev_probe_starts": dev_starts.tolist(),
        "evaluation_starts": eval_starts.tolist(),
        "training": training,
        "evaluation": evaluations,
        "final_minus_initial_sequence_kl_nats_per_token": {
            "mean": float(diff.mean()),
            "prompt_cluster_95ci": prompt_cluster_interval(
                diff, args.seed + 7_000
            ),
        },
        "initial_cost_calibration": cost_info,
        "duration_seconds": time.perf_counter() - started,
        "limitations": [
            "tiny nanoGPT and one corpus",
            "dev-only development experiment; test split remains untouched",
            "KL budget is estimated by Monte Carlo teacher scoring",
            "policy RL is contextual-bandit style, not long-horizon PPO",
            "CPU timing without KV cache is mechanism evidence, not production latency",
        ],
    }

    save(args.out / "result.json", result)
    torch.save(
        {
            "student": student.state_dict(),
            "gate": gate.state_dict(),
            "teacher_config": vars(teacher.config),
        },
        args.out / "final_checkpoint.pt",
    )
    np.savez_compressed(
        args.out / "evaluation_raw.npz",
        initial_log_ratio=raw["initial"]["log_ratio"],
        final_log_ratio=raw["final"]["log_ratio"],
        ar_log_ratio=raw["ar"]["log_ratio"],
        eval_starts=eval_starts.numpy(),
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--content-steps", type=int, default=64)
    ap.add_argument("--policy-steps", type=int, default=64)
    ap.add_argument("--state-contexts", type=int, default=32)
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--constraint-contexts", type=int, default=64)
    ap.add_argument("--dev-probe-contexts", type=int, default=64)
    ap.add_argument("--eval-prompts", type=int, default=16)
    ap.add_argument("--eval-draws", type=int, default=4)
    ap.add_argument("--count", type=int, default=48)
    ap.add_argument("--content-rank", type=int, default=8)
    ap.add_argument("--initial-dual", type=float, default=1.0)
    ap.add_argument("--dual-lr", type=float, default=0.1)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
