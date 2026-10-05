from __future__ import annotations

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
