import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'fixed_k_gate'))

import pytest
import torch
import fixed_k as fk
import adaptive_k as ak


def tiny():
    torch.manual_seed(7)
    return fk.GPT(fk.GPTConfig(
        block_size=16, vocab_size=7, n_layer=1, n_head=1,
        n_embd=16, dropout=0, bias=True
    )).eval()


def test_prefix_joint_log_prob_matches_full_joint_at_max_prefix():
    weights = torch.randn(5, 3)
    logits = torch.randn(5, 3, 3, 7)
    tokens = torch.randint(7, (5, 3))
    expected = fk.joint_log_prob(weights, logits, tokens)
    actual = ak.prefix_joint_log_prob(weights, logits, tokens, 3)
    torch.testing.assert_close(actual, expected)


def test_prefix_joint_log_prob_zero_for_empty_prefix():
    weights = torch.randn(4, 2)
    logits = torch.randn(4, 2, 3, 5)
    tokens = torch.randint(5, (4, 3))
    out = ak.prefix_joint_log_prob(weights, logits, tokens, 0)
    torch.testing.assert_close(out, torch.zeros(4))


def test_adaptive_gate_has_one_action_per_commit_length():
    gate = ak.AdaptiveKGate(11, max_k=4, rank=2)
    logits = gate(torch.randn(6, 11))
    assert logits.shape == (6, 4)
    assert ak.commit_lengths(logits).tolist() == [
        int(x) for x in (logits.argmax(-1) + 1)
    ]
    assert all(
        n.endswith(('.A', '.B'))
        for n, p in gate.named_parameters() if p.requires_grad
    )


def test_action_rewards_use_speed_gain_and_prefix_risk_without_changing_horizon():
    risks = torch.tensor([[0.0, 0.2, 0.5, 1.0]])
    rewards = ak.action_rewards(risks, penalty=2.0)
    expected = torch.tensor([[0.0, 0.6, 1.0, 1.0]])
    torch.testing.assert_close(rewards, expected)
    assert rewards.shape[-1] == 4


def test_policy_step_updates_gate_but_not_frozen_generator():
    student = fk.FixedKStudent(
        tiny(), rank=2, components=2
    ).eval().requires_grad_(False)
    before = fk.state_hash(student)
    gate = ak.AdaptiveKGate(9, max_k=4, rank=2)
    gate_before = fk.state_hash(gate)
    opt = torch.optim.Adam(
        [p for p in gate.parameters() if p.requires_grad], lr=.05
    )
    features = torch.randn(64, 9)
    risks = torch.rand(64, 4)
    risks[:, 0] = 0
    rewards = ak.action_rewards(risks, penalty=1.5)
    for _ in range(4):
        ak.policy_step(gate, opt, features, rewards)
    assert fk.state_hash(student) == before
    assert fk.state_hash(gate) != gate_before
    assert all(p.grad is None for p in student.parameters())


def test_emitted_prefix_length_is_dynamic_between_one_and_max_horizon():
    assert ak.emitted_length(1, 20) == 1
    assert ak.emitted_length(2, 20) == 2
    assert ak.emitted_length(3, 20) == 3
    assert ak.emitted_length(4, 20) == 4
    assert ak.emitted_length(4, 2) == 2
    with pytest.raises(ValueError):
        ak.emitted_length(0, 20)
    with pytest.raises(ValueError):
        ak.emitted_length(5, 20)


def test_prefix_risk_tensor_has_zero_ar_column_and_all_horizons():
    q = torch.tensor([
        [[0.0, 0.1, 0.4, 0.9], [0.0, -0.2, 0.3, 0.8]]
    ])
    out = ak.validate_prefix_risks(q, max_k=4)
    assert out.shape == (1, 2, 4)
    torch.testing.assert_close(out[..., 0], torch.zeros(1, 2))


def test_prefix_risk_bank_covers_every_commit_length_and_k1_is_exact_ar():
    from run_adaptive_k import prefix_risk_bank
    teacher = tiny()
    student = fk.FixedKStudent(
        teacher, rank=2, components=2
    ).eval().requires_grad_(False)
    x = torch.randint(7, (3, 16))
    bank = prefix_risk_bank(teacher, student, x, samples=5, seed=11)
    assert bank['risk_samples'].shape == (3, 5, 4)
    torch.testing.assert_close(
        bank['risk_samples'][..., 0], torch.zeros(3, 5), atol=1e-6, rtol=0
    )
    assert bank['features'].shape[0] == 3
    assert bank['uncertainty'].shape == (3,)


class AlwaysK(torch.nn.Module):
    def __init__(self, k, max_k=4):
        super().__init__()
        self.k = k
        self.max_k = max_k

    def forward(self, x):
        out = torch.full((len(x), self.max_k), -20.0)
        out[:, self.k - 1] = 20.0
        return out


def test_generate_uses_dynamic_k_without_teacher_verification():
    from run_adaptive_k import generate
    teacher = tiny()
    student = fk.FixedKStudent(
        teacher, rank=2, components=2
    ).eval().requires_grad_(False)
    prefix = torch.zeros(1, 16, dtype=torch.long)
    for k in (1, 2, 3, 4):
        out, calls, requested, emitted = generate(
            student, AlwaysK(k), prefix, count=8, seed=31+k
        )
        assert out.shape == (1, 8)
        assert set(requested) == {k}
        assert all(1 <= z <= 4 for z in emitted)
        assert calls == len(requested)


def test_summarize_policy_uses_gate_selected_k_and_reports_entropy_by_k():
    from run_adaptive_k import summarize_policy
    bank = {
        'features': torch.zeros(4, 2),
        'risk_samples': torch.tensor([
            [[0., .1, .2, .3]],
            [[0., .2, .4, .6]],
            [[0., .3, .6, .9]],
            [[0., .4, .8, 1.2]],
        ]),
        'uncertainty': torch.tensor([4., 3., 2., 1.]),
    }

    class Scripted(torch.nn.Module):
        def forward(self, x):
            out = torch.full((4, 4), -10.0)
            for i, k in enumerate((1, 2, 3, 4)):
                out[i, k - 1] = 10.0
            return out

    result = summarize_policy(bank, Scripted(), penalty=2.0)
    assert result['k_histogram'] == {'1': 1, '2': 1, '3': 1, '4': 1}
    assert result['mean_k'] == pytest.approx(2.5)
    assert result['mean_selected_reverse_kl_nats'] == pytest.approx(
        (0 + .2 + .6 + 1.2) / 4
    )
    assert result['mean_uncertainty_by_k']['1'] == pytest.approx(4.0)
    assert result['mean_uncertainty_by_k']['4'] == pytest.approx(1.0)


def test_train_gate_returns_positive_penalty_and_does_not_need_selected_targets():
    from run_adaptive_k import train_gate
    torch.manual_seed(3)
    bank = {
        'features': torch.randn(64, 6),
        'risk_samples': torch.rand(64, 8, 4),
        'uncertainty': torch.rand(64),
    }
    bank['risk_samples'][..., 0] = 0
    gate, penalty = train_gate(bank, steps=20, seed=41)
    assert penalty > 0
    assert gate(torch.randn(3, 6)).shape == (3, 4)
