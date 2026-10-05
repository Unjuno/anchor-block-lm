from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from probe_anchor_horizon import (
    HERE,
    TOP_P,
    load_teacher,
    sample_test_contexts,
    top_p_sample,
    bootstrap_mean_diff,
)


def contiguous_min_probability_lengths(probabilities: torch.Tensor, threshold: float):
    """Longest prefix whose every token probability is >= threshold."""
    lengths = torch.zeros(probabilities.shape[0], dtype=torch.long, device=probabilities.device)
    for h in range(probabilities.shape[1]):
        safe = probabilities[:, h] >= threshold
        lengths = torch.where(safe & lengths.eq(h), lengths + 1, lengths)
    return lengths


def contiguous_mean_surprisal_lengths(log_probs: torch.Tensor, max_mean_surprisal: float):
    """Longest prefix whose running mean surprisal remains under the budget."""
    running = -log_probs.cumsum(-1) / torch.arange(
        1, log_probs.shape[1] + 1, device=log_probs.device
    )
    lengths = torch.zeros(log_probs.shape[0], dtype=torch.long, device=log_probs.device)
    for h in range(log_probs.shape[1]):
        safe = running[:, h] <= max_mean_surprisal
        lengths = torch.where(safe & lengths.eq(h), lengths + 1, lengths)
    return lengths


@torch.no_grad()
def sample_scored_rollouts(
    teacher,
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
    tokens = []
    log_probs = []
    for _ in range(horizon):
        logits = teacher(ctx[:, -teacher.config.block_size:])[0][:, -1]
        token = top_p_sample(logits, generator, top_p=TOP_P)
        lp = logits.log_softmax(-1).gather(-1, token[:, None]).squeeze(-1)
        tokens.append(token.reshape(B, n_rollouts))
        log_probs.append(lp.reshape(B, n_rollouts))
        ctx = torch.cat([ctx, token[:, None]], dim=1)
    return torch.stack(tokens, -1), torch.stack(log_probs, -1)


def select_best_of_n(tokens: torch.Tensor, log_probs: torch.Tensor):
    score = log_probs.sum(-1)
    best = score.argmax(1)
    batch = torch.arange(tokens.shape[0], device=tokens.device)
    return tokens[batch, best], log_probs[batch, best], score[batch, best]


def summarize_lengths(direct: torch.Tensor, anchored: torch.Tensor, seed: int):
    diff = (anchored - direct).cpu().numpy().astype(np.float64)
    horizon = direct.max().item() if direct.numel() else 0
    size = max(int(horizon), int(anchored.max().item() if anchored.numel() else 0)) + 1
    return {
        "direct_mean_length": direct.float().mean().item(),
        "anchor_mean_length": anchored.float().mean().item(),
        "mean_difference_anchor_minus_direct": float(diff.mean()),
        "paired_bootstrap_95ci": bootstrap_mean_diff(diff, 2000, seed),
        "fraction_anchor_longer": float((diff > 0).mean()),
        "fraction_equal": float((diff == 0).mean()),
        "direct_histogram": torch.bincount(direct, minlength=size).tolist(),
        "anchor_histogram": torch.bincount(anchored, minlength=size).tolist(),
    }


@torch.no_grad()
def run_probe(
    contexts: int = 256,
    rollouts: int = 16,
    horizon: int = 6,
    chunk_size: int = 16,
    seed: int = 2234,
):
    teacher = load_teacher()
    x = sample_test_contexts(contexts, teacher.config.block_size, seed)

    direct_lp_parts = []
    anchor_lp_parts = []
    direct_score_parts = []
    anchor_score_parts = []

    direct_gen = torch.Generator().manual_seed(seed + 1)
    anchor_token_gen = torch.Generator().manual_seed(seed + 2)
    anchor_gen = torch.Generator().manual_seed(seed + 3)

    for st in range(0, contexts, chunk_size):
        batch = x[st:st + chunk_size]

        dtok, dlp = sample_scored_rollouts(
            teacher, batch, rollouts, horizon, direct_gen
        )
        _, best_dlp, best_dscore = select_best_of_n(dtok, dlp)
        direct_lp_parts.append(best_dlp)
        direct_score_parts.append(best_dscore)

        logits = teacher(batch)[0][:, -1]
        anchor = top_p_sample(logits, anchor_token_gen)
        anchored = torch.cat([batch[:, 1:], anchor[:, None]], dim=1)

        atok, alp = sample_scored_rollouts(
            teacher, anchored, rollouts, horizon, anchor_gen
        )
        _, best_alp, best_ascore = select_best_of_n(atok, alp)
        anchor_lp_parts.append(best_alp)
        anchor_score_parts.append(best_ascore)

    direct_lp = torch.cat(direct_lp_parts)
    anchor_lp = torch.cat(anchor_lp_parts)
    direct_probs = direct_lp.exp()
    anchor_probs = anchor_lp.exp()

    probability_thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
    probability_results = {}
    for threshold in probability_thresholds:
        dlen = contiguous_min_probability_lengths(direct_probs, threshold)
        alen = contiguous_min_probability_lengths(anchor_probs, threshold)
        probability_results[str(threshold)] = summarize_lengths(
            dlen, alen, seed + int(threshold * 1000)
        )

    surprisal_budgets = [1.0, 1.5, 2.0, 2.5, 3.0]
    surprisal_results = {}
    for budget in surprisal_budgets:
        dlen = contiguous_mean_surprisal_lengths(direct_lp, budget)
        alen = contiguous_mean_surprisal_lengths(anchor_lp, budget)
        surprisal_results[str(budget)] = summarize_lengths(
            dlen, alen, seed + int(budget * 100)
        )

    result = {
        "conditions": {
            "contexts": contexts,
            "rollouts_per_context": rollouts,
            "horizon": horizon,
            "top_p": TOP_P,
            "candidate_selection": "highest teacher sequence log-probability among fixed-length top-p samples",
            "anchor_policy": "one top-p teacher token before continuation candidate search",
        },
        "mean_teacher_probability_by_slot": {
            "direct": direct_probs.mean(0).tolist(),
            "anchor": anchor_probs.mean(0).tolist(),
        },
        "mean_best_sequence_log_probability": {
            "direct": torch.cat(direct_score_parts).mean().item(),
            "anchor": torch.cat(anchor_score_parts).mean().item(),
        },
        "minimum_token_probability_thresholds": probability_results,
        "running_mean_surprisal_budgets_nats": surprisal_results,
    }
    path = HERE / "results" / "bestofn_block_probe.json"
    path.write_text(json.dumps(result, indent=2))
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
