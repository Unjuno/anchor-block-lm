from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
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
    torch.manual_seed(71)
    return fk.GPT(fk.GPTConfig(
        block_size=16,
        vocab_size=17,
        n_layer=1,
        n_head=1,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )).eval()


def load_feature():
    spec = importlib.util.find_spec("dynamic_boundary")
    assert spec is not None, "dynamic_boundary feature has not been implemented yet"
    return importlib.import_module("dynamic_boundary")


def test_dual_controller_increases_penalty_above_kl_budget_and_decreases_below():
    db = load_feature()
    dual = db.KLDualController(initial=0.5, lr=0.2, target=0.3)
    high = dual.update(torch.tensor([0.6, 0.5]))
    assert high > 0.5
    low = dual.update(torch.tensor([0.0, 0.1]))
    assert 0.0 <= low < high


def test_constrained_rewards_reward_progress_but_penalize_teacher_risk():
    db = load_feature()
    risks = torch.tensor([[0.0, 0.1, 0.4, 0.8]])
    costs = torch.tensor([1.0, 1.15, 1.20, 1.25])
    r = db.constrained_rewards(risks, costs, dual_lambda=2.0)
    expected = torch.tensor([[-1.0, -0.35, 0.0, 0.15]])
    torch.testing.assert_close(r, expected)


def test_probe_tracks_dynamic_k_histogram_and_selected_teacher_risk():
    db = load_feature()
    features = torch.zeros(4, 3)
    risk = torch.tensor([
        [0.0, 0.1, 0.3, 0.8],
        [0.0, 0.2, 0.5, 1.0],
        [0.0, 0.1, 0.2, 0.4],
        [0.0, 0.4, 0.7, 1.2],
    ])

    class Scripted(torch.nn.Module):
        def forward(self, x):
            logits = torch.full((4, 4), -20.0)
            for i, k in enumerate((1, 2, 3, 4)):
                logits[i, k - 1] = 20.0
            return logits

    row = db.probe_policy(Scripted(), features, risk)
    assert row["k_histogram"] == {"1": 1, "2": 1, "3": 1, "4": 1}
    assert row["mean_k"] == pytest.approx(2.5)
    assert row["selected_k"] == [1, 2, 3, 4]
    assert row["mean_selected_teacher_risk"] == pytest.approx((0 + .2 + .2 + 1.2) / 4)


def test_round_recollection_changes_state_bank_when_actor_policy_changes():
    db = load_feature()
    teacher = tiny_teacher()
    student = fk.FixedKStudent(teacher, rank=2, components=2).eval().requires_grad_(False)
    gate = AdaptiveKGate(2 * teacher.config.n_embd + 2 * fk.TAIL + 1, max_k=4, rank=2).eval()
    prefixes = torch.randint(0, teacher.config.vocab_size, (3, teacher.config.block_size))

    with torch.no_grad():
        for p in gate.parameters():
            if p.ndim:
                p.zero_()
        # Force k=1 first.
        gate.net[-1].base.bias.zero_()
        gate.net[-1].base.bias[0] = 10.0
    s1 = db.collect_on_policy_states(student, gate, prefixes, cycles=3, seed=101)

    with torch.no_grad():
        gate.net[-1].base.bias.zero_()
        gate.net[-1].base.bias[3] = 10.0
    s2 = db.collect_on_policy_states(student, gate, prefixes, cycles=3, seed=101)

    assert s1.shape == s2.shape
    assert not torch.equal(s1, s2)


def test_alternating_round_updates_content_and_gate_but_never_backbone():
    db = load_feature()
    teacher = tiny_teacher()
    student = fk.FixedKStudent(teacher, rank=2, components=2)
    student = db.enable_content_lora(student, rank=2)
    gate_dim = 2 * teacher.config.n_embd + 2 * fk.TAIL + 1
    gate = AdaptiveKGate(gate_dim, max_k=4, rank=2)
    for n, p in gate.named_parameters():
        p.requires_grad_(n.endswith((".A", ".B")))
    prefixes = torch.randint(0, teacher.config.vocab_size, (8, teacher.config.block_size))

    before_backbone = fk.state_hash(student.backbone)
    before_student = fk.state_hash(student)
    before_gate = fk.state_hash(gate)
    dual = db.KLDualController(initial=0.5, lr=0.1, target=0.4)
    costs = torch.ones(4)

    row = db.train_round(
        teacher,
        student,
        gate,
        prefixes,
        content_steps=3,
        policy_steps=3,
        batch_size=4,
        samples=2,
        costs=costs,
        dual=dual,
        seed=203,
    )

    assert fk.state_hash(student.backbone) == before_backbone
    assert fk.state_hash(student) != before_student
    assert fk.state_hash(gate) != before_gate
    assert set(row["k_histogram"]) == {"1", "2", "3", "4"}
    assert row["dual_lambda"] >= 0
    assert torch.isfinite(torch.tensor(row["selected_constrained_reward_mean"]))
