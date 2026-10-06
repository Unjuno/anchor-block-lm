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
    torch.manual_seed(222)
    return fk.GPT(fk.GPTConfig(
        block_size=16,
        vocab_size=23,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )).eval()


def load_runner():
    spec = importlib.util.find_spec("run_fusion_ablation")
    assert spec is not None, "run_fusion_ablation has not been implemented yet"
    return importlib.import_module("run_fusion_ablation")


def test_fusion_initialization_reproduces_source_student_before_training():
    runner = load_runner()
    teacher = tiny_teacher()
    source = fk.FixedKStudent(teacher).eval()
    fusion = runner.initialize_fusion_from_source(teacher, source, rank=4, fusion_hidden=24)
    x = torch.randint(0, teacher.config.vocab_size, (7, teacher.config.block_size))
    h, anchor_logits = source.encode(x)
    anchor = anchor_logits.argmax(-1)
    source_w, source_l = source.tail(h, anchor)
    taps, fusion_anchor = fusion.encode_layers(x)
    fusion_w, fusion_l = fusion.tail_from_layers(taps, anchor)

    torch.testing.assert_close(fusion_anchor, anchor_logits, rtol=0, atol=0)
    torch.testing.assert_close(fusion_w, source_w, rtol=0, atol=0)
    torch.testing.assert_close(fusion_l, source_l, rtol=0, atol=0)


def test_shared_bank_contains_unfiltered_full_horizon_teacher_samples():
    runner = load_runner()
    teacher = tiny_teacher()
    x = torch.randint(0, teacher.config.vocab_size, (9, teacher.config.block_size))
    bank = runner.build_shared_distill_bank(
        teacher,
        x,
        samples=5,
        seed=303,
    )
    assert bank["contexts"].shape == x.shape
    assert bank["anchor"].shape == (9,)
    assert bank["tail"].shape == (9, 5, fk.TAIL)
    assert bank["teacher_logp"].shape == (9, 5)


def test_training_from_shared_bank_keeps_backbone_frozen_for_both_conditions():
    runner = load_runner()
    teacher = tiny_teacher()
    source = fk.FixedKStudent(teacher).eval()
    baseline = runner.initialize_baseline_from_source(source)
    fusion = runner.initialize_fusion_from_source(
        teacher, source, rank=4, fusion_hidden=24
    )
    x = torch.randint(0, teacher.config.vocab_size, (32, teacher.config.block_size))
    bank = runner.build_shared_distill_bank(teacher, x, samples=3, seed=404)
    before_teacher = fk.state_hash(teacher)
    before_base_backbone = fk.state_hash(baseline.backbone)
    before_fusion_backbone = fk.state_hash(fusion.backbone)

    runner.train_from_shared_bank(baseline, bank, steps=4, batch_size=16, seed=405)
    runner.train_from_shared_bank(fusion, bank, steps=4, batch_size=16, seed=406)

    assert fk.state_hash(teacher) == before_teacher
    assert fk.state_hash(baseline.backbone) == before_base_backbone
    assert fk.state_hash(fusion.backbone) == before_fusion_backbone


def test_forward_kl_uses_identical_teacher_samples_for_all_models():
    runner = load_runner()
    teacher = tiny_teacher()
    source = fk.FixedKStudent(teacher).eval()
    fusion = runner.initialize_fusion_from_source(teacher, source, rank=4, fusion_hidden=24)
    x = torch.randint(0, teacher.config.vocab_size, (6, teacher.config.block_size))
    bank = runner.build_shared_distill_bank(teacher, x, samples=4, seed=505)
    scores = runner.forward_kl_from_bank(
        {"source": source, "fusion": fusion},
        bank,
    )
    torch.testing.assert_close(scores["source"], scores["fusion"], rtol=0, atol=0)

def test_paired_forward_kl_summary_reports_negative_difference_when_fusion_is_better():
    runner = load_runner()
    baseline = torch.tensor([
        [1.0, 1.2, 0.8],
        [2.0, 2.2, 1.8],
        [3.0, 3.2, 2.8],
    ])
    fusion = baseline - 0.25
    summary = runner.paired_forward_kl_summary(
        baseline,
        fusion,
        seed=606,
        bootstrap_samples=1000,
    )
    assert summary["baseline_mean_nats_per_tail"] == pytest.approx(2.0)
    assert summary["fusion_mean_nats_per_tail"] == pytest.approx(1.75)
    assert summary["fusion_minus_baseline_mean"] == pytest.approx(-0.25)
    assert summary["fusion_minus_baseline_95ci"] == pytest.approx([-0.25, -0.25])
