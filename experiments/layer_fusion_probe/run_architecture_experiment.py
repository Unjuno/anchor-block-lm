"""Three-seed distillation ablation for frozen-backbone all-layer fusion.

Both conditions start from the exact same archived distilled student and receive
the exact same teacher-sampled training bank and minibatch order.  The control
continues training the existing final-layer continuation head.  The treatment
adds per-layer low-rank projections plus a small fusion MLP.  Evaluation uses
only the dev split; test remains unused by this architecture decision.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TWO_MODEL = HERE.parent / "two_model_rl"
FIXED = HERE.parent / "fixed_k_gate"
sys.path.insert(0, str(TWO_MODEL))
sys.path.insert(0, str(FIXED))

import fixed_k as fk
from run_two_model import reconstruct_splits, load_saved
from run_fusion_ablation import (
    build_shared_distill_bank,
    forward_kl_from_bank,
    initialize_baseline_from_source,
    initialize_fusion_from_source,
    paired_forward_kl_summary,
    train_from_shared_bank,
)


def sample_contexts(seq: torch.Tensor, n: int, window: int, seed: int):
    if min(n, window) < 1:
        raise ValueError("n and window must be positive")
    available = len(seq) - window + 1
    if available < n:
        raise ValueError("not enough unique context starts")
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randperm(available, generator=generator)[:n]
    contexts = seq[starts[:, None] + torch.arange(window)[None, :]]
    return contexts.long(), starts


def trainable_parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def seed_experiment(seed: int):
    """Seed newly initialized fusion weights as well as all global RNGs."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))


def run(args):
    if min(
        args.steps,
        args.batch_size,
        args.train_contexts,
        args.dev_contexts,
        args.train_samples,
        args.dev_samples,
        args.fusion_rank,
        args.fusion_hidden,
    ) < 1:
        raise ValueError("all experiment sizes must be positive")

    seed_experiment(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    data, metadata = reconstruct_splits(args.artifact_root, args.out / "source-cache")
    teacher, source, _, old = load_saved(
        args.artifact_root / "outputs" / f"seed-{args.seed}"
    )
    teacher_hash = fk.state_hash(teacher)
    source_hash = fk.state_hash(source)
    source_backbone_hash = fk.state_hash(source.backbone)
    if source_backbone_hash != teacher_hash:
        raise RuntimeError("archived source backbone does not match evaluator")

    train_x, train_starts = sample_contexts(
        data["train"],
        args.train_contexts,
        teacher.config.block_size,
        args.seed + 100,
    )
    dev_x, dev_starts = sample_contexts(
        data["dev"],
        args.dev_contexts,
        teacher.config.block_size,
        args.seed + 200,
    )

    train_bank = build_shared_distill_bank(
        teacher,
        train_x,
        samples=args.train_samples,
        seed=args.seed + 300,
    )
    dev_bank = build_shared_distill_bank(
        teacher,
        dev_x,
        samples=args.dev_samples,
        seed=args.seed + 400,
    )

    source_scores = forward_kl_from_bank({"source": source}, dev_bank)["source"]

    baseline = initialize_baseline_from_source(source)
    fusion = initialize_fusion_from_source(
        teacher,
        source,
        rank=args.fusion_rank,
        fusion_hidden=args.fusion_hidden,
    )

    initial_pair = forward_kl_from_bank(
        {"baseline": baseline, "fusion": fusion},
        dev_bank,
    )
    torch.testing.assert_close(
        initial_pair["baseline"],
        initial_pair["fusion"],
        rtol=0,
        atol=0,
    )

    baseline_params = trainable_parameter_count(baseline)
    fusion_params = trainable_parameter_count(fusion)
    baseline_history = train_from_shared_bank(
        baseline,
        train_bank,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed + 500,
        lr=args.lr,
    )
    fusion_history = train_from_shared_bank(
        fusion,
        train_bank,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed + 500,
        lr=args.lr,
    )

    if fk.state_hash(teacher) != teacher_hash:
        raise RuntimeError("fixed evaluator changed")
    if fk.state_hash(baseline.backbone) != teacher_hash:
        raise RuntimeError("baseline AR backbone changed")
    if fk.state_hash(fusion.backbone) != teacher_hash:
        raise RuntimeError("fusion AR backbone changed")

    scores = forward_kl_from_bank(
        {
            "source": source,
            "baseline": baseline,
            "fusion": fusion,
        },
        dev_bank,
    )
    paired = paired_forward_kl_summary(
        scores["baseline"],
        scores["fusion"],
        seed=args.seed + 600,
        bootstrap_samples=args.bootstrap_samples,
    )
    source_per_context = scores["source"].double().mean(1).cpu().numpy()
    baseline_per_context = scores["baseline"].double().mean(1).cpu().numpy()
    fusion_per_context = scores["fusion"].double().mean(1).cpu().numpy()

    result = {
        "experiment_version": "all-layer-lowrank-fusion-distill-v1",
        "seed": args.seed,
        "dataset": metadata["dataset"],
        "data_preprocessing_version": metadata["preprocessing_version"],
        "protocol": {
            "source_training_run": 37324518978,
            "fusion_initialization_seed": args.seed,
            "source_student_hash": source_hash,
            "evaluator_hash": teacher_hash,
            "train_split_only_for_updates": True,
            "dev_split_only_for_architecture_evaluation": True,
            "test_split_used": False,
            "same_teacher_samples_both_conditions": True,
            "same_minibatch_order_both_conditions": True,
            "teacher_sampling": "temperature=1, top-p=1, unfiltered full three-token tail",
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "train_contexts": args.train_contexts,
            "train_samples_per_context": args.train_samples,
            "dev_contexts": args.dev_contexts,
            "dev_samples_per_context": args.dev_samples,
            "fusion_rank_per_layer": args.fusion_rank,
            "fusion_hidden": args.fusion_hidden,
        },
        "conditions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "threads": 1,
            "dtype": "float32",
            "teacher_config": vars(teacher.config),
        },
        "parameter_counts": {
            "source_total": sum(p.numel() for p in source.parameters()),
            "baseline_trainable": baseline_params,
            "fusion_trainable": fusion_params,
            "fusion_extra_trainable": fusion_params - baseline_params,
        },
        "invariants": {
            "source_backbone_matches_evaluator": source_backbone_hash == teacher_hash,
            "evaluator_unchanged": fk.state_hash(teacher) == teacher_hash,
            "baseline_backbone_unchanged": fk.state_hash(baseline.backbone) == teacher_hash,
            "fusion_backbone_unchanged": fk.state_hash(fusion.backbone) == teacher_hash,
            "initial_baseline_fusion_scores_exactly_equal": True,
        },
        "dev_forward_kl_nats_per_three_token_tail": {
            "source_mean": float(source_per_context.mean()),
            "baseline_mean": float(baseline_per_context.mean()),
            "fusion_mean": float(fusion_per_context.mean()),
            "fusion_minus_baseline": paired["fusion_minus_baseline_mean"],
            "fusion_minus_baseline_95ci": paired["fusion_minus_baseline_95ci"],
        },
        "dev_forward_kl_nats_per_token": {
            "source_mean": float(source_per_context.mean() / fk.TAIL),
            "baseline_mean": float(baseline_per_context.mean() / fk.TAIL),
            "fusion_mean": float(fusion_per_context.mean() / fk.TAIL),
            "fusion_minus_baseline": float(
                paired["fusion_minus_baseline_mean"] / fk.TAIL
            ),
            "fusion_minus_baseline_95ci": [
                float(x / fk.TAIL)
                for x in paired["fusion_minus_baseline_95ci"]
            ],
        },
        "training_history": {
            "baseline": baseline_history,
            "fusion": fusion_history,
        },
        "context_starts": {
            "train": train_starts.tolist(),
            "dev": dev_starts.tolist(),
        },
        "source_previous_result_reference": {
            "previous_full_horizon_forward_kl": old.get("full_horizon_forward_kl"),
        },
        "duration_seconds": time.perf_counter() - started,
        "limitations": [
            "one tiny nanoGPT family and one public-domain play",
            "fusion has more trainable parameters than the continued-baseline control",
            "this stage tests distillation fidelity only; no RL update or latency claim is made",
            "dev is an architecture diagnostic, not a fresh cross-dataset confirmation",
        ],
    }

    _save(args.out / "result.json", result)
    torch.save(
        {
            "baseline": baseline.state_dict(),
            "fusion": fusion.state_dict(),
            "teacher_config": vars(teacher.config),
        },
        args.out / "models.pt",
    )
    np.savez_compressed(
        args.out / "dev_scores.npz",
        source=scores["source"].cpu().numpy(),
        baseline=scores["baseline"].cpu().numpy(),
        fusion=scores["fusion"].cpu().numpy(),
        dev_starts=dev_starts.cpu().numpy(),
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-contexts", type=int, default=1536)
    ap.add_argument("--train-samples", type=int, default=4)
    ap.add_argument("--dev-contexts", type=int, default=128)
    ap.add_argument("--dev-samples", type=int, default=32)
    ap.add_argument("--fusion-rank", type=int, default=8)
    ap.add_argument("--fusion-hidden", type=int, default=64)
    ap.add_argument("--bootstrap-samples", type=int, default=4000)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
