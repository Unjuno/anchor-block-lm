from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
NANOGPT = HERE.parent / "synthetic" / "third_party" / "nanogpt"
sys.path.insert(0, str(NANOGPT))
from model import GPT, GPTConfig

TOP_P = 0.95
TEMPERATURE = 1.0


def top_p_sample(logits: torch.Tensor, generator: torch.Generator, top_p: float = TOP_P):
    probs = (logits / TEMPERATURE).softmax(-1)
    sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(-1)
    remove = (cumulative - sorted_probs) >= top_p
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(-1, keepdim=True)
    selected = torch.multinomial(sorted_probs, 1, generator=generator)
    return sorted_ids.gather(-1, selected).squeeze(-1)


def modal_prefix_path(rollouts: torch.Tensor):
    """Return modal token path and joint sample mass surviving along that exact path."""
    if rollouts.ndim != 3:
        raise ValueError("rollouts must have shape [B,N,H]")
    B, N, H = rollouts.shape
    modal = torch.empty((B, H), dtype=rollouts.dtype, device=rollouts.device)
    mass = torch.zeros((B, H), dtype=torch.float32, device=rollouts.device)
    for b in range(B):
        alive = torch.ones(N, dtype=torch.bool, device=rollouts.device)
        for h in range(H):
            values = rollouts[b, alive, h]
            if values.numel() == 0:
                break
            uniq, counts = torch.unique(values, sorted=True, return_counts=True)
            token = uniq[counts.argmax()]
            modal[b, h] = token
            alive = alive & rollouts[b, :, h].eq(token)
            mass[b, h] = alive.float().mean()
    return modal, mass


def safe_lengths(mass: torch.Tensor, threshold: float):
    lengths = torch.zeros(mass.shape[0], dtype=torch.long, device=mass.device)
    for h in range(mass.shape[1]):
        still_safe = mass[:, h] >= threshold
        # Contiguity is required: once unsafe, later recovery does not extend the block.
        lengths = torch.where(still_safe & lengths.eq(h), lengths + 1, lengths)
    return lengths


@torch.no_grad()
def sample_rollouts(
    teacher: GPT,
    contexts: torch.Tensor,
    n_rollouts: int,
    horizon: int,
    generator: torch.Generator,
):
    B = contexts.shape[0]
    ctx = (
        contexts[:, None, :]
        .expand(B, n_rollouts, contexts.shape[1])
        .reshape(B * n_rollouts, contexts.shape[1])
        .clone()
    )
    generated = []
    for _ in range(horizon):
        logits = teacher(ctx[:, -teacher.config.block_size:])[0][:, -1]
        token = top_p_sample(logits, generator)
        generated.append(token.reshape(B, n_rollouts))
        ctx = torch.cat([ctx, token[:, None]], dim=1)
    return torch.stack(generated, dim=-1)


def load_teacher():
    saved = torch.load(HERE / "results" / "teacher.pt", map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**saved["config"]))
    model.load_state_dict(saved["model"])
    model.eval()
    model.requires_grad_(False)
    return model


def sample_test_contexts(n: int, context: int, seed: int):
    seq = torch.tensor(np.load(HERE / "data" / "test.npy").astype(np.int64))
    g = torch.Generator().manual_seed(seed)
    ix = torch.randint(0, len(seq) - context - 1, (n,), generator=g)
    return seq[ix[:, None] + torch.arange(context)[None, :]]


def bootstrap_mean_diff(diff: np.ndarray, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    vals = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        ix = rng.integers(0, len(diff), len(diff))
        vals[i] = diff[ix].mean()
    return [float(x) for x in np.quantile(vals, [0.025, 0.975])]


@torch.no_grad()
def run_probe(
    contexts: int = 256,
    rollouts: int = 16,
    horizon: int = 6,
    chunk_size: int = 16,
    seed: int = 1234,
):
    teacher = load_teacher()
    x = sample_test_contexts(contexts, teacher.config.block_size, seed)

    direct_mass_parts = []
    anchor_mass_parts = []
    anchor_tokens = []

    direct_gen = torch.Generator().manual_seed(seed + 1)
    anchor_token_gen = torch.Generator().manual_seed(seed + 2)
    anchor_rollout_gen = torch.Generator().manual_seed(seed + 3)

    for st in range(0, contexts, chunk_size):
        batch = x[st:st + chunk_size]

        direct_rollouts = sample_rollouts(
            teacher, batch, rollouts, horizon, direct_gen
        )
        _, direct_mass = modal_prefix_path(direct_rollouts)
        direct_mass_parts.append(direct_mass)

        logits = teacher(batch)[0][:, -1]
        anchor = top_p_sample(logits, anchor_token_gen)
        anchor_tokens.append(anchor)
        anchored = torch.cat([batch[:, 1:], anchor[:, None]], dim=1)

        conditioned_rollouts = sample_rollouts(
            teacher, anchored, rollouts, horizon, anchor_rollout_gen
        )
        _, anchor_mass = modal_prefix_path(conditioned_rollouts)
        anchor_mass_parts.append(anchor_mass)

    direct_mass = torch.cat(direct_mass_parts)
    anchor_mass = torch.cat(anchor_mass_parts)
    anchor_tokens = torch.cat(anchor_tokens)

    thresholds = [0.60, 0.70, 0.80, 0.90, 0.95]
    summaries = {}
    for threshold in thresholds:
        direct = safe_lengths(direct_mass, threshold)
        anchored = safe_lengths(anchor_mass, threshold)
        diff = (anchored - direct).cpu().numpy().astype(np.float64)
        summaries[str(threshold)] = {
            "direct_mean_safe_length": direct.float().mean().item(),
            "anchor_mean_safe_continuation_length": anchored.float().mean().item(),
            "mean_difference_anchor_minus_direct": float(diff.mean()),
            "paired_bootstrap_95ci": bootstrap_mean_diff(diff, 2000, seed + int(threshold * 100)),
            "fraction_anchor_longer": float((diff > 0).mean()),
            "fraction_equal": float((diff == 0).mean()),
            "direct_histogram_0_to_H": torch.bincount(direct, minlength=horizon + 1).tolist(),
            "anchor_histogram_0_to_H": torch.bincount(anchored, minlength=horizon + 1).tolist(),
        }

    result = {
        "conditions": {
            "dataset": "Romeo and Juliet / Project Gutenberg",
            "tokenizer": "train-only byte-level BPE",
            "contexts": contexts,
            "context_tokens": teacher.config.block_size,
            "rollouts_per_context": rollouts,
            "horizon": horizon,
            "top_p": TOP_P,
            "temperature": TEMPERATURE,
            "anchor_policy": "one top-p teacher sample, then condition all continuation rollouts on it",
            "test_split_used_for_probe_only": True,
        },
        "teacher_checkpoint": json.loads((HERE / "results" / "teacher_metrics.json").read_text()),
        "summaries": summaries,
    }

    out = HERE / "results" / "predictability_probe.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, default=256)
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--chunk-size", type=int, default=16)
    args = ap.parse_args()
    run_probe(args.contexts, args.rollouts, args.horizon, args.chunk_size)


if __name__ == "__main__":
    main()
