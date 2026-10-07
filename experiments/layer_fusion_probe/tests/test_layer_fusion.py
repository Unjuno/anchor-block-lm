from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
FIXED = EXPERIMENT.parent / "fixed_k_gate"
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(FIXED))

import fixed_k as fk


def tiny_teacher(n_layer: int = 2):
    torch.manual_seed(123)
    return fk.GPT(fk.GPTConfig(
        block_size=16,
        vocab_size=19,
        n_layer=n_layer,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )).eval()


def load_feature():
    spec = importlib.util.find_spec("layer_fusion")
    assert spec is not None, "layer_fusion feature has not been implemented yet"
    return importlib.import_module("layer_fusion")


def test_layer_fusion_reads_every_transformer_layer_and_preserves_anchor_exactly():
    lf = load_feature()
    teacher = tiny_teacher(n_layer=3)
    model = lf.LayerFusionStudent(teacher, rank=4, fusion_hidden=24).eval()
    x = torch.randint(0, teacher.config.vocab_size, (5, teacher.config.block_size))

    taps, anchor_logits = model.encode_layers(x)
    expected = teacher(x)[0][:, -1]
    torch.testing.assert_close(anchor_logits, expected, rtol=0, atol=0)
    assert taps.shape == (5, teacher.config.n_layer, teacher.config.n_embd)
    assert len(model.layer_down) == teacher.config.n_layer

    weights, logits = model.tail_from_layers(taps, anchor_logits.argmax(-1))
    assert weights.shape == (5, model.components)
    assert logits.shape == (
        5,
        model.components,
        fk.TAIL,
        teacher.config.vocab_size,
    )


def test_only_fusion_and_continuation_branch_are_trainable():
    lf = load_feature()
    model = lf.LayerFusionStudent(tiny_teacher(), rank=4, fusion_hidden=24)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all(not name.startswith("backbone.") for name in trainable)
    assert any(name.startswith("layer_down.") for name in trainable)
    assert any(name.startswith("fusion_mlp.") for name in trainable)


def test_distillation_step_changes_fusion_branch_but_not_frozen_backbone():
    lf = load_feature()
    teacher = tiny_teacher()
    model = lf.LayerFusionStudent(teacher, rank=4, fusion_hidden=24)
    x = torch.randint(0, teacher.config.vocab_size, (32, teacher.config.block_size))
    before_backbone = fk.state_hash(model.backbone)
    before_model = fk.state_hash(model)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-3,
    )
    generator = torch.Generator().manual_seed(77)

    metrics = lf.distill_step(
        teacher,
        model,
        optimizer,
        x,
        generator,
        teacher_samples=4,
    )

    assert fk.state_hash(model.backbone) == before_backbone
    assert fk.state_hash(model) != before_model
    assert metrics["loss"] > 0
    assert metrics["teacher_nll"] > 0


def test_teacher_forward_kl_evaluator_accepts_same_teacher_samples_for_both_models():
    lf = load_feature()
    teacher = tiny_teacher()
    baseline = fk.FixedKStudent(teacher).eval()
    fusion = lf.LayerFusionStudent(teacher, rank=4, fusion_hidden=24).eval()
    x = torch.randint(0, teacher.config.vocab_size, (8, teacher.config.block_size))
    generator = torch.Generator().manual_seed(91)

    result = lf.shared_teacher_forward_kl(
        teacher,
        {"baseline": baseline, "fusion": fusion},
        x,
        generator,
        samples=3,
    )

    assert set(result) == {"baseline", "fusion"}
    assert result["baseline"].shape == (8, 3)
    assert result["fusion"].shape == (8, 3)
    assert torch.isfinite(result["baseline"]).all()
    assert torch.isfinite(result["fusion"]).all()
