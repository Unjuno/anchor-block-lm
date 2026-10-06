"""Dynamic-boundary training: content KD + constrained adaptive-length RL.

The frozen evaluator/backbone defines quality. Continuation LoRA is updated only
with unfiltered teacher distillation on actor-visited states. A separate LoRA
policy chooses k in {1..H}; its Lagrange multiplier adapts to a teacher-risk
budget. Confidence features are observations, never rewards.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TWO = HERE.parent / "two_model_rl"
ADAPT = HERE.parent / "adaptive_k_rl"
FIXED = HERE.parent / "fixed_k_gate"
for p in (TWO, ADAPT, FIXED):
    sys.path.insert(0, str(p))

import fixed_k as fk
from adaptive_k import policy_step
from two_model import (
    collect_states as _collect_states,
    enable_content_lora,
    policy_batch,
    teacher_tail,
)

H = fk.BLOCK_SIZE


class KLDualController:
    def __init__(self, initial: float, lr: float, target: float):
        if initial < 0 or lr <= 0 or target < 0:
            raise ValueError("invalid dual-controller parameters")
        self.value = float(initial)
        self.lr = float(lr)
        self.target = float(target)

    def update(self, selected_risk: torch.Tensor) -> float:
        if selected_risk.numel() == 0 or not torch.isfinite(selected_risk).all():
            raise ValueError("selected risk must be finite and nonempty")
        violation = float(selected_risk.float().mean()) - self.target
        self.value = max(0.0, self.value + self.lr * violation)
        return self.value


def constrained_rewards(
    prefix_risks: torch.Tensor,
    normalized_costs: torch.Tensor,
    dual_lambda: float,
) -> torch.Tensor:
    if prefix_risks.ndim != 2:
        raise ValueError("prefix_risks must be [B,K]")
    if normalized_costs.shape != (prefix_risks.shape[1],):
        raise ValueError("cost vector must match actions")
    if dual_lambda < 0:
        raise ValueError("dual lambda must be nonnegative")
    extra = torch.arange(
        prefix_risks.shape[1],
        dtype=prefix_risks.dtype,
        device=prefix_risks.device,
    )
    return extra[None, :] - normalized_costs[None, :] - float(dual_lambda) * prefix_risks


@torch.no_grad()
def probe_policy(gate, features: torch.Tensor, prefix_risks: torch.Tensor):
    logits = gate(features)
    if logits.shape != prefix_risks.shape:
        raise ValueError("gate logits and risk matrix must match")
    k = logits.argmax(-1) + 1
    selected = prefix_risks.gather(-1, (k - 1)[:, None]).squeeze(-1)
    return {
        "k_histogram": {
            str(i): int(k.eq(i).sum()) for i in range(1, prefix_risks.shape[1] + 1)
        },
        "mean_k": float(k.float().mean()),
        "mean_selected_teacher_risk": float(selected.mean()),
        "selected_teacher_risk": selected.detach().cpu().tolist(),
    }


@torch.no_grad()
def collect_on_policy_states(student, gate, prefixes, cycles: int, seed: int):
    return _collect_states(student, gate, prefixes, cycles=cycles, seed=seed)


def _content_kd_step(
    teacher,
    student,
    optimizer,
    contexts: torch.Tensor,
    generator: torch.Generator,
    samples: int,
):
    if samples < 1:
        raise ValueError("samples must be positive")
    h, anchor_logits = student.encode(contexts)
    anchor = torch.multinomial(
        anchor_logits.softmax(-1), 1, generator=generator
    ).squeeze(-1)
    weights, logits = student.tail(h, anchor)

    targets = []
    for _ in range(samples):
        y, _ = teacher_tail(teacher, contexts, anchor, generator)
        targets.append(y)
    target = torch.stack(targets, dim=1)

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
    loss = -fk.joint_log_prob(w, l, y).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    params = [p for p in student.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    return float(loss.detach())


def train_round(
    teacher,
    student,
    gate,
    states: torch.Tensor,
    *,
    content_steps: int,
    policy_steps: int,
    batch_size: int,
    samples: int,
    costs: torch.Tensor,
    dual: KLDualController,
    seed: int,
    content_lr: float = 3e-4,
    gate_lr: float = 1e-3,
):
    if min(content_steps, policy_steps, batch_size, samples) < 1:
        raise ValueError("positive training sizes required")
    if len(states) < 1:
        raise ValueError("state bank is empty")

    content_params = [p for p in student.parameters() if p.requires_grad]
    gate_params = [p for p in gate.parameters() if p.requires_grad]
    if not content_params or not gate_params:
        raise ValueError("student and gate need trainable LoRA parameters")

    opt_content = torch.optim.Adam(content_params, lr=content_lr)
    opt_gate = torch.optim.Adam(gate_params, lr=gate_lr)
    rng = torch.Generator().manual_seed(seed)
    content_losses = []

    for _ in range(content_steps):
        ix = torch.randint(
            len(states),
            (min(batch_size, len(states)),),
            generator=rng,
        )
        content_losses.append(
            _content_kd_step(
                teacher,
                student,
                opt_content,
                states[ix],
                rng,
                samples=samples,
            )
        )

    policy_losses = []
    selected_risks = []
    for _ in range(policy_steps):
        ix = torch.randint(
            len(states),
            (min(batch_size, len(states)),),
            generator=rng,
        )
        bank = policy_batch(
            teacher,
            student,
            states[ix],
            rng,
            samples=samples,
        )
        rewards = constrained_rewards(bank["risk"], costs, dual.value)
        policy_losses.append(
            policy_step(gate, opt_gate, bank["features"], rewards)
        )
        with torch.no_grad():
            action = gate(bank["features"]).argmax(-1)
            risk = bank["risk"].gather(-1, action[:, None]).squeeze(-1)
            selected_risks.append(risk)
            dual.update(risk)

    with torch.no_grad():
        probe_bank = policy_batch(
            teacher,
            student,
            states,
            torch.Generator().manual_seed(seed + 100_000),
            samples=max(samples, 4),
        )
        probe = probe_policy(gate, probe_bank["features"], probe_bank["risk"])

    return {
        **probe,
        "dual_lambda": dual.value,
        "content_nll_mean": float(sum(content_losses) / len(content_losses)),
        "policy_loss_mean": float(sum(policy_losses) / len(policy_losses)),
        "training_selected_teacher_risk_mean": float(
            torch.cat(selected_risks).mean()
        ),
    }


__all__ = [
    "H",
    "KLDualController",
    "collect_on_policy_states",
    "constrained_rewards",
    "enable_content_lora",
    "probe_policy",
    "train_round",
]
