from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
sys.path.insert(0, str(EXPERIMENT))


def load_experiment():
    spec = importlib.util.find_spec("run_architecture_experiment")
    assert spec is not None, "run_architecture_experiment is not implemented yet"
    return importlib.import_module("run_architecture_experiment")


def test_sample_contexts_is_deterministic_unique_and_in_range():
    exp = load_experiment()
    seq = torch.arange(500)
    a, starts_a = exp.sample_contexts(seq, n=12, window=16, seed=717)
    b, starts_b = exp.sample_contexts(seq, n=12, window=16, seed=717)
    assert torch.equal(a, b)
    assert torch.equal(starts_a, starts_b)
    assert len(torch.unique(starts_a)) == 12
    assert int(starts_a.min()) >= 0
    assert int(starts_a.max()) <= len(seq) - 16
    assert a.shape == (12, 16)


def test_trainable_parameter_count_excludes_frozen_backbone():
    exp = load_experiment()

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = torch.nn.Linear(3, 4).requires_grad_(False)
            self.trainable = torch.nn.Linear(4, 2)

    toy = Toy()
    assert exp.trainable_parameter_count(toy) == sum(
        p.numel() for p in toy.trainable.parameters()
    )


def test_experiment_seeding_controls_new_fusion_parameter_initialization():
    import torch
    import numpy as np
    import run_architecture_experiment as exp
    assert hasattr(exp, 'seed_experiment'), 'new fusion weights need explicit reproducible seeding'
    exp.seed_experiment(48017)
    a=torch.randn(11); b=np.random.random(4)
    torch.randn(71); np.random.random(38)
    exp.seed_experiment(48017)
    torch.testing.assert_close(torch.randn(11),a,rtol=0,atol=0)
    np.testing.assert_array_equal(np.random.random(4),b)
