from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
FIXED = EXPERIMENT.parent / "fixed_k_gate"
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(FIXED))

import fixed_k as fk
from layer_fusion import LayerFusionStudent


def tiny_teacher():
    torch.manual_seed(909)
    return fk.GPT(fk.GPTConfig(
        block_size=16,
        vocab_size=17,
        n_layer=2,
        n_head=1,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )).eval()


def load_feature():
    spec = importlib.util.find_spec("objective_ablation")
    assert spec is not None, "objective_ablation has not been implemented yet"
    return importlib.import_module("objective_ablation")


def test_leave_one_out_baseline_excludes_current_sample():
    oa = load_feature()
    x = torch.tensor([[1.0, 3.0, 8.0]])
    actual = oa.leave_one_out(x)
    expected = torch.tensor([[5.5, 4.5, 2.0]])
    torch.testing.assert_close(actual, expected)


def test_fixed_kd_bank_uses_full_unfiltered_teacher_tail():
    oa = load_feature()
    teacher = tiny_teacher()
    x = torch.randint(0, teacher.config.vocab_size, (6, teacher.config.block_size))
    bank = oa.build_kd_bank(teacher, x, samples=4, seed=31)
    assert bank["contexts"].shape == x.shape
    assert bank["anchor"].shape == (6,)
    assert bank["tail"].shape == (6, 4, fk.TAIL)
    assert bank["teacher_logp"].shape == (6, 4)


def test_kd_only_step_changes_actor_but_not_teacher_or_backbone():
    oa = load_feature()
    teacher = tiny_teacher()
    model = LayerFusionStudent(teacher, rank=4, fusion_hidden=24)
    x = torch.randint(0, teacher.config.vocab_size, (20, teacher.config.block_size))
    bank = oa.build_kd_bank(teacher, x, samples=3, seed=41)
    before_teacher = fk.state_hash(teacher)
    before_backbone = fk.state_hash(model.backbone)
    before_actor = fk.state_hash(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    metrics = oa.objective_step(
        teacher,
        model,
        optimizer,
        bank,
        torch.arange(12),
        torch.zeros(12, dtype=torch.long),
        q_generator=torch.Generator().manual_seed(42),
        rl_weight=0.0,
        q_samples=4,
    )
    assert metrics["kd_loss"] > 0
    assert metrics["rl_pg_loss"] == pytest.approx(0.0)
    assert fk.state_hash(teacher) == before_teacher
    assert fk.state_hash(model.backbone) == before_backbone
    assert fk.state_hash(model) != before_actor


def test_rl_kd_step_reports_sampled_reverse_kl_and_keeps_backbone_frozen():
    oa = load_feature()
    teacher = tiny_teacher()
    model = LayerFusionStudent(teacher, rank=4, fusion_hidden=24)
    x = torch.randint(0, teacher.config.vocab_size, (20, teacher.config.block_size))
    bank = oa.build_kd_bank(teacher, x, samples=3, seed=51)
    before_backbone = fk.state_hash(model.backbone)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    metrics = oa.objective_step(
        teacher,
        model,
        optimizer,
        bank,
        torch.arange(12),
        torch.ones(12, dtype=torch.long),
        q_generator=torch.Generator().manual_seed(52),
        rl_weight=1.0,
        q_samples=4,
    )
    assert torch.isfinite(torch.tensor(metrics["sampled_reverse_kl_nats"]))
    assert torch.isfinite(torch.tensor(metrics["rl_pg_loss"]))
    assert fk.state_hash(model.backbone) == before_backbone


def test_objective_conflict_requires_reverse_improvement_and_forward_degradation():
    oa = load_feature()
    summary = oa.classify_objective_conflict(
        kd_forward=4.0,
        rl_forward=4.2,
        kd_reverse=0.35,
        rl_reverse=0.25,
    )
    assert summary["reverse_kl_improved"] is True
    assert summary["forward_kl_worsened"] is True
    assert summary["proxy_conflict"] is True

    no_conflict = oa.classify_objective_conflict(
        kd_forward=4.0,
        rl_forward=3.9,
        kd_reverse=0.35,
        rl_reverse=0.25,
    )
    assert no_conflict["proxy_conflict"] is False
