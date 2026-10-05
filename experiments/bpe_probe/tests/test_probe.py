from __future__ import annotations

import pytest
import torch

from prepare_data import split_text
from probe_anchor_horizon import modal_prefix_path, safe_lengths


def test_split_text_is_disjoint_and_lossless():
    text = "line-1\nline-2\nline-3\nline-4\nline-5\n"
    train, dev, test = split_text(text, train_fraction=0.6, dev_fraction=0.2)
    assert train + dev + test == text
    assert len(train) > len(dev) > 0
    assert len(test) > 0


def test_modal_prefix_mass_tracks_joint_survival_not_slot_majority():
    # Slot-wise majorities would misleadingly suggest [1,2,7].
    rollouts = torch.tensor([[
        [1, 2, 7],
        [1, 2, 8],
        [1, 3, 7],
        [4, 2, 7],
    ]])
    modal, mass = modal_prefix_path(rollouts)
    assert modal[0].tolist() == [1, 2, 7]
    assert mass[0].tolist() == [0.75, 0.50, 0.25]


def test_safe_lengths_stop_at_first_joint_mass_failure():
    mass = torch.tensor([
        [1.00, 0.90, 0.81, 0.79, 0.95],
        [0.79, 0.95, 0.95, 0.95, 0.95],
    ])
    assert safe_lengths(mass, 0.80).tolist() == [3, 0]


def test_contiguous_min_probability_length_stops_at_first_weak_token():
    from probe_bestofn_blocks import contiguous_min_probability_lengths
    probs = torch.tensor([
        [0.8, 0.6, 0.19, 0.9],
        [0.1, 0.9, 0.9, 0.9],
    ])
    assert contiguous_min_probability_lengths(probs, 0.2).tolist() == [2, 0]


def test_anchor_information_is_mutual_information_of_mixture():
    from probe_anchor_information import information_from_conditionals
    conditional = torch.tensor([[
        [1.0, 0.0],
        [0.0, 1.0],
    ]])
    stats = information_from_conditionals(conditional)
    assert stats["mi_nats"][0].item() == pytest.approx(0.693147, rel=1e-5)
    assert stats["mixture_entropy_nats"][0].item() == pytest.approx(0.693147, rel=1e-5)
    assert stats["conditional_entropy_nats"][0].item() == pytest.approx(0.0)
    assert stats["conditional_top1_gain"][0].item() == pytest.approx(0.5)


def test_bestofn_selects_highest_sequence_log_probability():
    from train_block_student import select_best_of_n
    tokens = torch.tensor([[
        [1, 2, 3],
        [4, 5, 6],
    ]])
    log_probs = torch.tensor([[
        [-1.0, -1.0, -1.0],
        [-0.2, -0.2, -0.2],
    ]])
    best_tokens, best_log_probs = select_best_of_n(tokens, log_probs)
    assert best_tokens.tolist() == [[4, 5, 6]]
    assert best_log_probs.tolist() == [[-0.2, -0.2, -0.2]]


def test_surprisal_budget_stops_at_first_running_mean_failure():
    from train_block_student import target_lengths_from_log_probs
    log_probs = torch.tensor([
        [-1.0, -2.0, -6.0, -0.1],
        [-3.0, -0.1, -0.1, -0.1],
    ])
    assert target_lengths_from_log_probs(log_probs, 2.0).tolist() == [2, 0]
