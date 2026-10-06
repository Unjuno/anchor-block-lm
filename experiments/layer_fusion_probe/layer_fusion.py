"""Frozen-backbone all-layer low-rank fusion for continuation distillation.

The ordinary AR logits are computed from the unchanged teacher path.  A
separate continuation branch reads the last-position representation from every
Transformer block, compresses each to a low-rank vector, fuses them with a
small MLP, and adds the result only to the continuation representation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

FIXED = Path(__file__).resolve().parents[1] / "fixed_k_gate"
sys.path.insert(0, str(FIXED))
import fixed_k as fk


class LayerFusionStudent(fk.FixedKStudent):
    def __init__(
        self,
        teacher,
        rank: int = 8,
        fusion_hidden: int = 64,
        components: int = 4,
    ):
        super().__init__(teacher, rank=rank, components=components)
        width = teacher.config.n_embd
        self.fusion_rank = rank
        self.layer_down = nn.ModuleList([
            nn.Linear(width, rank, bias=False)
            for _ in range(teacher.config.n_layer)
        ])
        self.fusion_mlp = nn.Sequential(
            nn.Linear(teacher.config.n_layer * rank, fusion_hidden),
            nn.GELU(),
            nn.Linear(fusion_hidden, width),
        )
        # Start as the existing final-layer student; fusion earns its effect
        # through distillation rather than perturbing the initial actor.
        nn.init.zeros_(self.fusion_mlp[-1].weight)
        nn.init.zeros_(self.fusion_mlp[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def encode_layers(self, idx: torch.Tensor):
        if idx.ndim != 2 or not 1 <= idx.shape[1] <= self.backbone.config.block_size:
            raise ValueError("Context must fit backbone window")
        tr = self.backbone.transformer
        pos = torch.arange(idx.shape[1], device=idx.device)
        h = tr.drop(tr.wte(idx) + tr.wpe(pos))
        taps = []
        for block in tr.h:
            h = block(h)
            taps.append(h[:, -1])
        stacked = torch.stack(taps, dim=1)
        final = tr.ln_f(h)[:, -1]
        anchor_logits = self.backbone.lm_head(final)
        return stacked, anchor_logits

    def fused_hidden(self, taps: torch.Tensor):
        if taps.ndim != 3 or taps.shape[1] != len(self.layer_down):
            raise ValueError("Expected [B,n_layer,width] taps")
        projected = [
            down(F.layer_norm(taps[:, i], (taps.shape[-1],)))
            for i, down in enumerate(self.layer_down)
        ]
        fused = self.fusion_mlp(torch.cat(projected, dim=-1))
        # Reconstruct the exact final hidden from the final block tap before
        # adding a continuation-only residual.
        final = self.backbone.transformer.ln_f(taps[:, -1])
        return final + fused

    def tail_from_layers(self, taps: torch.Tensor, anchor: torch.Tensor):
        return super().tail(self.fused_hidden(taps), anchor)

    @torch.no_grad()
    def features_from_layers(self, taps, anchor, weights, logits):
        h = self.fused_hidden(taps)
        return super().features(h, anchor, weights, logits)


@torch.no_grad()
def _teacher_tail(teacher, contexts, anchor, generator, tokens=None):
    ctx = torch.cat([contexts, anchor[:, None]], dim=1)
    ys = []
    logp = torch.zeros(len(ctx), dtype=teacher.lm_head.weight.dtype)
    for j in range(fk.TAIL):
        logits = teacher(ctx[:, -teacher.config.block_size:])[0][:, -1]
        y = (
            torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)
            if tokens is None else tokens[:, j]
        )
        logp = logp + logits.log_softmax(-1).gather(-1, y[:, None]).squeeze(-1)
        ys.append(y)
        ctx = torch.cat([ctx, y[:, None]], dim=1)
    return torch.stack(ys, dim=1), logp


def _student_tail(model, contexts, anchor):
    if isinstance(model, LayerFusionStudent):
        taps, anchor_logits = model.encode_layers(contexts)
        return anchor_logits, model.tail_from_layers(taps, anchor)
    h, anchor_logits = model.encode(contexts)
    return anchor_logits, model.tail(h, anchor)


def distill_step(
    teacher,
    model: LayerFusionStudent,
    optimizer,
    contexts: torch.Tensor,
    generator: torch.Generator,
    teacher_samples: int = 4,
):
    if teacher_samples < 1:
        raise ValueError("teacher_samples must be positive")
    taps, anchor_logits = model.encode_layers(contexts)
    anchor = torch.multinomial(
        anchor_logits.softmax(-1), 1, generator=generator
    ).squeeze(-1)
    weights, logits = model.tail_from_layers(taps, anchor)

    targets = []
    for _ in range(teacher_samples):
        target, _ = _teacher_tail(teacher, contexts, anchor, generator)
        targets.append(target)
    target = torch.stack(targets, dim=1)
    w = weights[:, None].expand(-1, teacher_samples, -1).reshape(
        len(contexts) * teacher_samples, -1
    )
    l = logits[:, None].expand(-1, teacher_samples, -1, -1, -1).reshape(
        len(contexts) * teacher_samples,
        logits.shape[1],
        logits.shape[2],
        logits.shape[3],
    )
    y = target.reshape(len(contexts) * teacher_samples, fk.TAIL)
    nll = -fk.joint_log_prob(w, l, y).mean()

    optimizer.zero_grad(set_to_none=True)
    nll.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    return {"loss": float(nll.detach()), "teacher_nll": float(nll.detach())}


@torch.no_grad()
def shared_teacher_forward_kl(
    teacher,
    models: dict[str, nn.Module],
    contexts: torch.Tensor,
    generator: torch.Generator,
    samples: int = 16,
):
    if samples < 1:
        raise ValueError("samples must be positive")
    teacher_logits = teacher(contexts)[0][:, -1]
    anchor = torch.multinomial(
        teacher_logits.softmax(-1), 1, generator=generator
    ).squeeze(-1)

    targets = []
    teacher_logps = []
    for _ in range(samples):
        y, lp = _teacher_tail(teacher, contexts, anchor, generator)
        targets.append(y)
        teacher_logps.append(lp)
    target = torch.stack(targets, dim=1)
    teacher_logp = torch.stack(teacher_logps, dim=1)

    result = {}
    for name, model in models.items():
        _, (weights, logits) = _student_tail(model, contexts, anchor)
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
