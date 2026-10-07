"""Fixed four-token joint student and a separate binary LoRA gate.

BLOCK_SIZE includes one exact teacher-distribution anchor plus three parallel
continuation slots. No EOB, variable horizon, reranking, or rejection verifier.
The shared mixture component models tail dependence; it is still an approximation.
"""
from __future__ import annotations
import copy
import hashlib
import math
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM = Path(__file__).resolve().parents[1] / 'synthetic' / 'third_party' / 'nanogpt'
sys.path.insert(0, str(UPSTREAM))
from model import GPT, GPTConfig

BLOCK_SIZE = 4
TAIL = BLOCK_SIZE - 1


def state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def teacher_probs(logits: torch.Tensor) -> torch.Tensor:
    """Temperature 1, untruncated teacher: equivalent to top-p=1, not 0.95."""
    return logits.softmax(-1)


def joint_log_prob(weights, logits, tokens):
    """Normalized mixture of products, not product of marginal probabilities."""
    if logits.ndim != 4 or tokens.shape != (logits.shape[0], logits.shape[2]):
        raise ValueError('Expected weights [B,M], logits [B,M,J,V], tokens [B,J]')
    chosen = logits.log_softmax(-1).gather(
        -1, tokens[:, None, :, None].expand(-1, logits.shape[1], -1, 1)
    ).squeeze(-1).sum(-1)
    return torch.logsumexp(weights.log_softmax(-1) + chosen, -1)


@torch.no_grad()
def joint_sample(weights, logits, generator):
    component = torch.multinomial(weights.softmax(-1), 1, generator=generator).squeeze(-1)
    selected = logits[torch.arange(len(logits)), component].softmax(-1)
    return torch.multinomial(selected.reshape(-1, selected.shape[-1]), 1,
                             generator=generator).reshape(len(logits), logits.shape[2])


class LowRankResidual(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, width))
        self.B = nn.Parameter(torch.zeros(width, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(F.linear(x, self.A), self.B)


class FixedKStudent(nn.Module):
    def __init__(self, teacher, rank=8, components=4):
        super().__init__()
        self.backbone = copy.deepcopy(teacher).eval().requires_grad_(False)
        self.components = components
        width = teacher.config.n_embd
        self.low_rank = LowRankResidual(width, rank)
        self.condition = nn.Sequential(nn.Linear(2*width, width), nn.GELU(), nn.Linear(width, width))
        self.slot = nn.Parameter(torch.randn(TAIL, width) * .02)
        self.component = nn.Parameter(torch.randn(components, width) * .02)
        self.decoder = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))
        self.mix = nn.Linear(width, components)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def encode(self, idx):
        if idx.ndim != 2 or not 1 <= idx.shape[1] <= self.backbone.config.block_size:
            raise ValueError('Context must fit backbone window')
        tr = self.backbone.transformer
        pos = torch.arange(idx.shape[1], device=idx.device)
        h = tr.drop(tr.wte(idx) + tr.wpe(pos))
        for block in tr.h: h = block(h)
        h = tr.ln_f(h)[:, -1]
        return h, self.backbone.lm_head(h)

    def tail(self, hidden, anchor):
        emb = self.backbone.transformer.wte(anchor)
        z = hidden + self.low_rank(hidden) + self.condition(torch.cat([hidden, emb], -1))
        s = z[:,None,None,:] + self.component[None,:,None,:] + self.slot[None,None,:,:]
        states = s + self.decoder(s)
        return self.mix(z), self.backbone.lm_head(states)

    @torch.no_grad()
    def features(self, h, anchor, weights, logits):
        probs = (weights.softmax(-1)[:,:,None,None] * logits.softmax(-1)).sum(1)
        entropy = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
        mixture = weights.softmax(-1)
        me = -(mixture * mixture.clamp_min(1e-30).log()).sum(-1,keepdim=True)
        # Only context, realized anchor, and distributions; NO sampled tail.
        return torch.cat([h, self.backbone.transformer.wte(anchor),
                          probs.max(-1).values, entropy, me], -1)


class LoRALinear(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super().__init__()
        self.base = nn.Linear(input_dim, output_dim).requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, input_dim))
        self.B = nn.Parameter(torch.zeros(output_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B)


class LoRAGate(nn.Module):
    def __init__(self, dim, rank=4):
        super().__init__()
        self.net = nn.Sequential(LoRALinear(dim,32,rank), nn.Tanh(), LoRALinear(32,1,1))
        self.register_buffer('center', torch.zeros(dim))
        self.register_buffer('scale', torch.ones(dim))

    def forward(self, features):
        return self.net((features-self.center)/self.scale).squeeze(-1)


def block_reward(log_ratio, penalty):
    return (BLOCK_SIZE-1) - penalty*log_ratio


def policy_step(gate, optimizer, features, block_rewards):
    distribution = torch.distributions.Bernoulli(logits=gate(features))
    action = distribution.sample()
    reward = action*block_rewards
    # State-dependent baseline is detached and independent of sampled action.
    baseline = distribution.probs.detach()*block_rewards
    loss = -(distribution.log_prob(action)*(reward-baseline)).mean() - .01*distribution.entropy().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in gate.parameters() if p.requires_grad], 1.)
    optimizer.step()
    return float(loss.detach())


def emitted_length(commit_block, remaining):
    if remaining < 1: raise ValueError('No generation budget left')
    return BLOCK_SIZE if commit_block and remaining >= BLOCK_SIZE else 1


def matched_mask(scores, count):
    if scores.ndim!=1 or not 0<=count<=len(scores): raise ValueError('Invalid allocation')
    out=torch.zeros(len(scores),dtype=torch.bool)
    order=torch.argsort(scores,descending=True,stable=True)
    out[order[:count]]=True
    return out
