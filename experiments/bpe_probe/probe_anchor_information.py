from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from probe_anchor_horizon import HERE, TOP_P, load_teacher, sample_test_contexts, top_p_sample


def entropy_nats(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * probs.clamp_min(1e-12).log()).sum(-1)


def information_from_conditionals(conditional_probs: torch.Tensor):
    """
    conditional_probs: [B, A, V], where A indexes sampled anchor tokens.

    The equally weighted Monte-Carlo mixture estimates p(X_{t+2}|context)
    under the same top-p anchor sampling policy. The average entropy of each
    conditional estimates E[H(X_{t+2}|context, anchor)].
    """
    if conditional_probs.ndim != 3:
        raise ValueError("expected [B,A,V]")
    mixture = conditional_probs.mean(1)
    h_mix = entropy_nats(mixture)
    h_cond = entropy_nats(conditional_probs).mean(1)
    top1_mix = mixture.max(-1).values
    top1_cond = conditional_probs.max(-1).values.mean(1)
    return {
        "mixture_entropy_nats": h_mix,
        "conditional_entropy_nats": h_cond,
        "mi_nats": h_mix - h_cond,
        "mi_bits": (h_mix - h_cond) / np.log(2.0),
        "mixture_top1": top1_mix,
        "conditional_top1": top1_cond,
        "conditional_top1_gain": top1_cond - top1_mix,
    }


@torch.no_grad()
def sample_anchor_conditionals(
    teacher,
    contexts: torch.Tensor,
    anchors_per_context: int,
    generator: torch.Generator,
):
    B = contexts.shape[0]
    first_logits = teacher(contexts)[0][:, -1]
    expanded_logits = (
        first_logits[:, None, :]
        .expand(B, anchors_per_context, first_logits.shape[-1])
        .reshape(B * anchors_per_context, first_logits.shape[-1])
    )
    anchors = top_p_sample(expanded_logits, generator, top_p=TOP_P).reshape(
        B, anchors_per_context
    )

    repeated = (
        contexts[:, None, :]
        .expand(B, anchors_per_context, contexts.shape[1])
        .reshape(B * anchors_per_context, contexts.shape[1])
    )
    conditioned = torch.cat(
        [repeated[:, 1:], anchors.reshape(-1, 1)],
        dim=1,
    )
    next_logits = teacher(conditioned)[0][:, -1]
    next_probs = next_logits.softmax(-1).reshape(
        B, anchors_per_context, next_logits.shape[-1]
    )
    return anchors, next_probs


def bootstrap_mean(values: np.ndarray, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    out = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        ix = rng.integers(0, len(values), len(values))
        out[i] = values[ix].mean()
    return [float(x) for x in np.quantile(out, [0.025, 0.975])]


@torch.no_grad()
def run_probe(
    contexts: int = 256,
    anchors_per_context: int = 32,
    chunk_size: int = 16,
    seed: int = 3234,
):
    teacher = load_teacher()
    x = sample_test_contexts(contexts, teacher.config.block_size, seed)
    generator = torch.Generator().manual_seed(seed + 1)

    all_stats = {
        "mixture_entropy_nats": [],
        "conditional_entropy_nats": [],
        "mi_nats": [],
        "mi_bits": [],
        "mixture_top1": [],
        "conditional_top1": [],
        "conditional_top1_gain": [],
    }
    duplicate_rates = []

    for st in range(0, contexts, chunk_size):
        batch = x[st:st + chunk_size]
        anchors, conditional = sample_anchor_conditionals(
            teacher, batch, anchors_per_context, generator
        )
        stats = information_from_conditionals(conditional)
        for key, val in stats.items():
            all_stats[key].append(val.cpu())

        for row in anchors:
            duplicate_rates.append(1.0 - len(torch.unique(row)) / float(len(row)))

    joined = {key: torch.cat(parts) for key, parts in all_stats.items()}

    mi = joined["mi_nats"].numpy()
    gain = joined["conditional_top1_gain"].numpy()
    result = {
        "conditions": {
            "contexts": contexts,
            "context_tokens": teacher.config.block_size,
            "anchors_per_context": anchors_per_context,
            "anchor_sampling": f"top-p={TOP_P}",
            "metric_target": "same future position X_(t+2), before vs after observing X_(t+1)",
            "test_split_used_for_probe_only": True,
        },
        "means": {
            key: value.float().mean().item()
            for key, value in joined.items()
        },
        "paired_bootstrap_95ci": {
            "mi_nats": bootstrap_mean(mi, 4000, seed + 10),
            "conditional_top1_gain": bootstrap_mean(gain, 4000, seed + 20),
        },
        "fractions": {
            "contexts_positive_mi": float((mi > 1e-8).mean()),
            "contexts_top1_improved": float((gain > 1e-8).mean()),
            "mean_duplicate_anchor_rate": float(np.mean(duplicate_rates)),
        },
    }

    path = HERE / "results" / "anchor_information_probe.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, default=256)
    ap.add_argument("--anchors-per-context", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=16)
    args = ap.parse_args()
    run_probe(args.contexts, args.anchors_per_context, args.chunk_size)


if __name__ == "__main__":
    main()
