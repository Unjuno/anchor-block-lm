from __future__ import annotations
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
FIXED_DIR = HERE.parent / 'fixed_k_gate'
sys.path.insert(0, str(FIXED_DIR))

from fixed_k import teacher_probs, joint_sample
from adaptive_k import (
    MAX_K, AdaptiveKGate, action_rewards, policy_step,
    prefix_joint_log_prob, commit_lengths, emitted_length,
)


@torch.no_grad()
def prefix_risk_bank(teacher, student, x: torch.Tensor, samples: int, seed: int):
    if samples < 1:
        raise ValueError('samples must be positive')
    g = torch.Generator().manual_seed(seed)
    h, anchor_logits = student.encode(x)
    anchor = torch.multinomial(teacher_probs(anchor_logits), 1, generator=g).squeeze(-1)
    weights, logits = student.tail(h, anchor)
    features = student.features(h, anchor, weights, logits)

    mixture = weights.softmax(-1)
    marginal = (mixture[:, :, None, None] * logits.softmax(-1)).sum(1)
    uncertainty = -(marginal * marginal.clamp_min(1e-30).log()).sum((-1, -2))

    ww = weights.repeat_interleave(samples, 0)
    ll = logits.repeat_interleave(samples, 0)
    tail = joint_sample(ww, ll, g)

    base = x.repeat_interleave(samples, 0)
    aa = anchor.repeat_interleave(samples)
    ctx = torch.cat([base, aa[:, None]], 1)
    cumulative_logp = torch.zeros(len(ctx), dtype=logits.dtype)
    risk_columns = [torch.zeros(len(ctx), dtype=logits.dtype)]
    for j in range(1, MAX_K):
        teacher_logits = teacher(ctx[:, -teacher.config.block_size:])[0][:, -1]
        tok = tail[:, j - 1]
        cumulative_logp += teacher_logits.log_softmax(-1).gather(-1, tok[:, None]).squeeze(-1)
        logq = prefix_joint_log_prob(ww, ll, tail, j)
        risk_columns.append(logq - cumulative_logp)
        ctx = torch.cat([ctx, tok[:, None]], 1)

    risk = torch.stack(risk_columns, -1).reshape(len(x), samples, MAX_K)
    return {
        'features': features,
        'risk_samples': risk,
        'uncertainty': uncertainty,
        'anchor': anchor,
    }


@torch.no_grad()
def generate(student, gate, prefix: torch.Tensor, count: int, seed: int):
    if count < 0:
        raise ValueError('count must be nonnegative')
    g = torch.Generator().manual_seed(seed)
    out = prefix.clone()
    calls = 0
    requested, emitted = [], []
    while out.shape[1] - prefix.shape[1] < count:
        h, anchor_logits = student.encode(out[:, -student.backbone.config.block_size:])
        anchor = torch.multinomial(teacher_probs(anchor_logits), 1, generator=g).squeeze(-1)
        weights, logits = student.tail(h, anchor)
        features = student.features(h, anchor, weights, logits)
        k = int(commit_lengths(gate(features))[0])
        remaining = count - (out.shape[1] - prefix.shape[1])
        actual = emitted_length(k, remaining, max_k=MAX_K)
        tokens = anchor[:, None]
        if actual > 1:
            tail = joint_sample(weights, logits, g)
            tokens = torch.cat([tokens, tail[:, :actual - 1]], 1)
        out = torch.cat([out, tokens], 1)
        calls += 1
        requested.append(k)
        emitted.append(actual)
    return out[:, -count:], calls, requested, emitted


def train_gate(bank, steps: int, seed: int):
    if steps < 1:
        raise ValueError('steps must be positive')
    features = bank['features']
    risks = bank['risk_samples']
    if risks.ndim != 3 or risks.shape[-1] != MAX_K:
        raise ValueError('risk_samples must be [N,S,K]')
    gate = AdaptiveKGate(features.shape[-1], max_k=MAX_K, rank=4)
    gate.center.copy_(features.mean(0))
    gate.scale.copy_(features.std(0).clamp_min(.05))
    full_expected = risks[..., -1].mean(1)
    q = float(full_expected.quantile(.25).clamp_min(.25))
    penalty = (MAX_K - 1) / q
    opt = torch.optim.Adam([p for p in gate.parameters() if p.requires_grad], lr=.003)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        ix = torch.randint(len(features), (min(128, len(features)),), generator=g)
        sample_ix = torch.randint(risks.shape[1], (len(ix),), generator=g)
        sampled_risk = risks[ix, sample_ix]
        rewards = action_rewards(sampled_risk, penalty)
        policy_step(gate, opt, features[ix], rewards)
    gate.eval().requires_grad_(False)
    return gate, penalty


@torch.no_grad()
def summarize_policy(bank, gate, penalty: float):
    logits = gate(bank['features'])
    k = commit_lengths(logits)
    expected_risk = bank['risk_samples'].mean(1)
    selected_risk = expected_risk.gather(-1, (k - 1)[:, None]).squeeze(-1)
    selected_reward = (k - 1).to(selected_risk.dtype) - float(penalty) * selected_risk
    hist = {str(i): int(k.eq(i).sum()) for i in range(1, MAX_K + 1)}
    by_k = {}
    for i in range(1, MAX_K + 1):
        mask = k.eq(i)
        if mask.any():
            by_k[str(i)] = float(bank['uncertainty'][mask].mean())
    return {
        'k_histogram': hist,
        'mean_k': float(k.float().mean()),
        'mean_selected_reverse_kl_nats': float(selected_risk.mean()),
        'reverse_kl_per_emitted_token': float(selected_risk.sum() / k.sum().clamp_min(1)),
        'mean_reward': float(selected_reward.mean()),
        'mean_uncertainty_by_k': by_k,
    }


@torch.no_grad()
def generate_ar(student, prefix: torch.Tensor, count: int, seed: int):
    if count < 0:
        raise ValueError('count must be nonnegative')
    g = torch.Generator().manual_seed(seed)
    out = prefix.clone()
    calls = 0
    while out.shape[1] - prefix.shape[1] < count:
        _, anchor_logits = student.encode(
            out[:, -student.backbone.config.block_size:]
        )
        anchor = torch.multinomial(
            teacher_probs(anchor_logits), 1, generator=g
        ).squeeze(-1)
        out = torch.cat([out, anchor[:, None]], 1)
        calls += 1
    return out[:, -count:], calls
