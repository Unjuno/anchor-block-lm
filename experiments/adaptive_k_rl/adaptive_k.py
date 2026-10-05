from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch import nn

FIXED_DIR = Path(__file__).resolve().parents[1] / 'fixed_k_gate'
sys.path.insert(0, str(FIXED_DIR))
from fixed_k import BLOCK_SIZE, LoRALinear

MAX_K = BLOCK_SIZE


def prefix_joint_log_prob(weights: torch.Tensor, logits: torch.Tensor,
                          tokens: torch.Tensor, tail_len: int) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError('logits must be [B,M,J,V]')
    if tokens.ndim != 2 or tokens.shape[0] != logits.shape[0]:
        raise ValueError('tokens must be [B,J]')
    if not 0 <= tail_len <= logits.shape[2]:
        raise ValueError('tail_len out of range')
    if tail_len == 0:
        return torch.zeros(logits.shape[0], device=logits.device, dtype=logits.dtype)
    chosen = logits[:, :, :tail_len].log_softmax(-1).gather(
        -1,
        tokens[:, None, :tail_len, None].expand(-1, logits.shape[1], -1, 1),
    ).squeeze(-1).sum(-1)
    return torch.logsumexp(weights.log_softmax(-1) + chosen, -1)


class AdaptiveKGate(nn.Module):
    def __init__(self, dim: int, max_k: int = MAX_K, rank: int = 4):
        super().__init__()
        if max_k < 1:
            raise ValueError('max_k must be positive')
        self.max_k = max_k
        self.net = nn.Sequential(
            LoRALinear(dim, 32, rank),
            nn.Tanh(),
            LoRALinear(32, max_k, rank),
        )
        self.register_buffer('center', torch.zeros(dim))
        self.register_buffer('scale', torch.ones(dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net((features - self.center) / self.scale)


def commit_lengths(action_logits: torch.Tensor) -> torch.Tensor:
    if action_logits.ndim != 2:
        raise ValueError('action logits must be [B,K]')
    return action_logits.argmax(-1) + 1


def action_rewards(prefix_risks: torch.Tensor, penalty: float) -> torch.Tensor:
    if prefix_risks.ndim != 2:
        raise ValueError('prefix_risks must be [B,K]')
    extra = torch.arange(prefix_risks.shape[1], device=prefix_risks.device,
                         dtype=prefix_risks.dtype)
    return extra[None, :] - float(penalty) * prefix_risks


def policy_step(gate: AdaptiveKGate, optimizer, features: torch.Tensor,
                rewards: torch.Tensor) -> float:
    logits = gate(features)
    if rewards.shape != logits.shape:
        raise ValueError('rewards must match gate logits')
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    chosen = rewards.gather(-1, action[:, None]).squeeze(-1)
    baseline = (dist.probs.detach() * rewards).sum(-1)
    loss = -(dist.log_prob(action) * (chosen - baseline).detach()).mean()
    loss = loss - 0.01 * dist.entropy().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    params = [p for p in gate.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    return float(loss.detach())


def emitted_length(k: int, remaining: int, max_k: int = MAX_K) -> int:
    if remaining < 1:
        raise ValueError('remaining must be positive')
    if not 1 <= int(k) <= max_k:
        raise ValueError('k out of range')
    return min(int(k), int(remaining))


def validate_prefix_risks(risks: torch.Tensor, max_k: int = MAX_K) -> torch.Tensor:
    if risks.ndim != 3 or risks.shape[-1] != max_k:
        raise ValueError('risks must be [B,S,K]')
    if not torch.allclose(risks[..., 0], torch.zeros_like(risks[..., 0])):
        raise ValueError('k=1 exact-AR risk column must be zero')
    return risks
