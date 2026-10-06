from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
TWO = EXP.parent / "two_model_rl"
ADAPT = EXP.parent / "adaptive_k_rl"
FIXED = EXP.parent / "fixed_k_gate"
for p in (EXP, TWO, ADAPT, FIXED):
    sys.path.insert(0, str(p))

import fixed_k as fk
from adaptive_k import AdaptiveKGate


def tiny_teacher():
    torch.manual_seed(801)
    return fk.GPT(fk.GPTConfig(
        block_size=16,
        vocab_size=13,
        n_layer=1,
        n_head=1,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )).eval()


def load_runner():
    spec = importlib.util.find_spec("run_dynamic_experiment")
    assert spec is not None, "run_dynamic_experiment has not been implemented yet"
    return importlib.import_module("run_dynamic_experiment")


def test_training_loop_records_fixed_probe_before_and_after_each_round():
    runner = load_runner()
    teacher = tiny_teacher()
    student = fk.FixedKStudent(teacher, rank=2, components=2)
    gate_dim = 2 * teacher.config.n_embd + 2 * fk.TAIL + 1
    gate = AdaptiveKGate(gate_dim, max_k=4, rank=2)
    seq = torch.randint(0, teacher.config.vocab_size, (400,))
    probe = torch.randint(
        0, teacher.config.vocab_size, (6, teacher.config.block_size)
    )

    result = runner.train_dynamic_loop(
        teacher,
        student,
        gate,
        seq,
        probe,
        rounds=2,
        content_steps=2,
        policy_steps=2,
        state_contexts=4,
        cycles=2,
        batch_size=4,
        samples=2,
        costs=torch.ones(4),
        initial_dual=0.5,
        dual_lr=0.1,
        kl_target=0.5,
        seed=901,
    )

    assert len(result["probe_history"]) == 3
    assert [row["round"] for row in result["probe_history"]] == [0, 1, 2]
    assert len(result["visited_state_hashes"]) == 2
    assert all(set(row["k_histogram"]) == {"1", "2", "3", "4"}
               for row in result["probe_history"])
    assert fk.state_hash(student.backbone) == fk.state_hash(teacher)


def test_constraint_target_can_be_derived_from_initial_training_probe_only():
    runner = load_runner()
    risks = torch.tensor([
        [0.0, .1, .4, .8],
        [0.0, .2, .5, .9],
    ])
    k = torch.tensor([2, 3])
    target = runner.initial_selected_risk_target(risks, k)
    assert target == torch.tensor([.1, .5]).mean().item()
