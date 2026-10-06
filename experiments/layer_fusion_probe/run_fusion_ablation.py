"""Shared-bank distillation utilities for the all-layer fusion ablation."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FIXED = HERE.parent / "fixed_k_gate"
sys.path.insert(0, str(FIXED))

import fixed_k as fk
from layer_fusion import LayerFusionStudent, _teacher_tail


def _enable_continuation_params(model):
    model.requires_grad_(True)
    model.backbone.requires_grad_(False)
    model.train()
    model.backbone.eval()
    return model


def initialize_baseline_from_source(source):
    return _enable_continuation_params(copy.deepcopy(source))


def initialize_fusion_from_source(
    teacher,
    source,
    rank: int = 8,
    fusion_hidden: int = 64,
):
    model = LayerFusionStudent(
        teacher,
        rank=rank,
        fusion_hidden=fusion_hidden,
        components=source.components,
        base_rank=source.low_rank.A.shape[0],
    )
    missing, unexpected = model.load_state_dict(source.state_dict(), strict=False)
    allowed = {
        *(f"layer_down.{i}.weight" for i in range(teacher.config.n_layer)),
        "fusion_mlp.0.weight",
        "fusion_mlp.0.bias",
        "fusion_mlp.2.weight",
        "fusion_mlp.2.bias",
    }
    if unexpected:
        raise ValueError(f"Unexpected source parameters: {unexpected}")
    if set(missing) != allowed:
        raise ValueError(f"Unexpected missing fusion parameters: {missing}")
    return _enable_continuation_params(model)


@torch.no_grad()
def build_shared_distill_bank(
    teacher,
    contexts: torch.Tensor,
    samples: int,
    seed: int,
):
    if samples < 1:
        raise ValueError("samples must be positive")
    generator = torch.Generator().manual_seed(seed)
    anchor_logits = teacher(contexts)[0][:, -1]
    anchor = torch.multinomial(
        anchor_logits.softmax(-1), 1, generator=generator
    ).squeeze(-1)
    tails = []
    logps = []
    for _ in range(samples):
        tail, logp = _teacher_tail(
            teacher,
            contexts,
            anchor,
            generator,
        )
        tails.append(tail)
        logps.append(logp)
    return {
        "contexts": contexts.clone(),
        "anchor": anchor,
        "tail": torch.stack(tails, dim=1),
        "teacher_logp": torch.stack(logps, dim=1),
    }


def _tail_logits(model, contexts, anchor):
    if isinstance(model, LayerFusionStudent):
        taps, _ = model.encode_layers(contexts)
        return model.tail_from_layers(taps, anchor)
    h, _ = model.encode(contexts)
    return model.tail(h, anchor)


def train_from_shared_bank(
    model,
    bank,
    steps: int,
    batch_size: int,
    seed: int,
    lr: float = 2e-3,
):
    if min(steps, batch_size) < 1:
        raise ValueError("steps and batch_size must be positive")
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("model has no trainable continuation parameters")
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)
    rng = torch.Generator().manual_seed(seed)
    contexts = bank["contexts"]
    anchor = bank["anchor"]
    targets = bank["tail"]
    n_samples = targets.shape[1]
    history = []

    for step in range(1, steps + 1):
        ix = torch.randint(
            len(contexts),
            (min(batch_size, len(contexts)),),
            generator=rng,
        )
        sample_ix = torch.randint(
            n_samples,
            (len(ix),),
            generator=rng,
        )
        weights, logits = _tail_logits(model, contexts[ix], anchor[ix])
        target = targets[ix, sample_ix]
        loss = -fk.joint_log_prob(weights, logits, target).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            history.append({"step": step, "nll": float(loss.detach())})
    model.eval()
    return history


@torch.no_grad()
def forward_kl_from_bank(models: dict[str, torch.nn.Module], bank):
    contexts = bank["contexts"]
    anchor = bank["anchor"]
    target = bank["tail"]
    teacher_logp = bank["teacher_logp"]
    result = {}
    samples = target.shape[1]

    for name, model in models.items():
        weights, logits = _tail_logits(model, contexts, anchor)
        w = weights[:, None].expand(-1, samples, -1).reshape(
            len(contexts) * samples, -1
        )
        l = logits[:, None].expand(-1, samples, -1, -1, -1).reshape(
            len(contexts) * samples,
            logits.shape[1],
            logits.shape[2],
            logits.shape[3],
        )
        y = target.reshape(len(contexts) * samples, fk.TAIL)
        logq = fk.joint_log_prob(w, l, y).reshape(len(contexts), samples)
        result[name] = teacher_logp - logq
    return result


def paired_forward_kl_summary(
    baseline: torch.Tensor,
    fusion: torch.Tensor,
    seed: int,
    bootstrap_samples: int = 4000,
):
    if baseline.shape != fusion.shape or baseline.ndim != 2:
        raise ValueError("Expected matching [contexts,samples] tensors")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    base_per_context = baseline.double().mean(1).cpu().numpy()
    fusion_per_context = fusion.double().mean(1).cpu().numpy()
    diff = fusion_per_context - base_per_context
    rng = np.random.default_rng(seed)
    draws = diff[
        rng.integers(0, len(diff), size=(bootstrap_samples, len(diff)))
    ].mean(1)
    return {
        "baseline_mean_nats_per_tail": float(base_per_context.mean()),
        "fusion_mean_nats_per_tail": float(fusion_per_context.mean()),
        "fusion_minus_baseline_mean": float(diff.mean()),
        "fusion_minus_baseline_95ci": [
            float(x) for x in np.quantile(draws, [0.025, 0.975])
        ],
    }
