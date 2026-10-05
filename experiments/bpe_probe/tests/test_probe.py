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
    torch.testing.assert_close(best_log_probs, torch.full((1, 3), -0.2))


def test_surprisal_budget_stops_at_first_running_mean_failure():
    from train_block_student import target_lengths_from_log_probs
    log_probs = torch.tensor([
        [-1.0, -2.0, -6.0, -0.1],
        [-3.0, -0.1, -0.1, -0.1],
    ])
    assert target_lengths_from_log_probs(log_probs, 2.0).tolist() == [2, 0]


def test_decode_variable_block_stops_at_eob_and_fixed_ignores_it():
    from benchmark_block_student import decode_variable, decode_fixed
    vocab = 5
    eob = vocab
    logits = torch.full((1, 4, vocab + 1), -10.0)
    logits[0, 0, 1] = 5.0
    logits[0, 1, 2] = 5.0
    logits[0, 2, eob] = 6.0
    logits[0, 3, 3] = 5.0
    tokens, length = decode_variable(logits, eob, max_tokens=3, eob_bias=0.0)
    assert length == 2
    assert tokens[:2].tolist() == [1, 2]
    fixed = decode_fixed(logits, eob, k=3)
    assert fixed.tolist() == [1, 2, 0]


def test_onpolicy_relabel_uses_bestofn_and_surprisal_budget():
    from onpolicy_block_refresh import targets_from_rollouts
    tokens = torch.tensor([[
        [1, 2, 3],
        [4, 5, 6],
    ]])
    log_probs = torch.tensor([[
        [-1.0, -1.0, -6.0],
        [-0.2, -0.2, -5.0],
    ]])
    best_tokens, lengths = targets_from_rollouts(tokens, log_probs, max_mean_surprisal=1.0)
    assert best_tokens.tolist() == [[4, 5, 6]]
    assert lengths.tolist() == [2]


def test_onepass_student_conditions_block_on_anchor_and_has_expected_shapes():
    from train_onepass_anchor_student import OnePassAnchorStudent
    from poc import GPT, GPTConfig
    teacher = GPT(GPTConfig(
        block_size=16, vocab_size=17, n_layer=1, n_head=1, n_embd=16,
        dropout=0.0, bias=True,
    )).eval()
    model = OnePassAnchorStudent(teacher, rank=2, horizon=3).eval()
    x = torch.randint(0, teacher.config.vocab_size, (2, teacher.config.block_size))
    anchor = torch.tensor([1, 2])
    anchor_logits, block_logits = model(x, anchor_override=anchor)
    assert anchor_logits.shape == (2, teacher.config.vocab_size)
    assert block_logits.shape == (2, 4, teacher.config.vocab_size + 1)


def test_onepass_decode_immediate_eob_still_emits_anchor():
    from benchmark_onepass_anchor import decode_macro
    vocab = 7
    anchor_logits = torch.full((1, vocab), -10.0)
    anchor_logits[0, 3] = 5.0
    block_logits = torch.full((1, 4, vocab + 1), -10.0)
    block_logits[0, 0, vocab] = 8.0
    emitted = decode_macro(anchor_logits, block_logits, eob_id=vocab, max_continuation=3, eob_bias=0.0)
    assert emitted.tolist() == [3]


def test_onepass_fixed_k_emits_anchor_plus_k_continuations():
    from benchmark_onepass_anchor import decode_macro_fixed
    vocab = 7
    anchor_logits = torch.full((1, vocab), -10.0)
    anchor_logits[0, 3] = 5.0
    block_logits = torch.full((1, 4, vocab + 1), -10.0)
    block_logits[0, 0, 1] = 5.0
    block_logits[0, 1, 2] = 5.0
    emitted = decode_macro_fixed(anchor_logits, block_logits, eob_id=vocab, k=2)
    assert emitted.tolist() == [3, 1, 2]


def test_onepass_anchor_path_is_exactly_frozen_teacher():
    from train_onepass_anchor_student import OnePassAnchorStudent
    from poc import GPT, GPTConfig
    teacher = GPT(GPTConfig(
        block_size=16, vocab_size=17, n_layer=1, n_head=1, n_embd=16,
        dropout=0.0, bias=True,
    )).eval()
    student = OnePassAnchorStudent(teacher, rank=2, horizon=3).eval()
    x = torch.randint(0, 17, (3, 16))
    expected = teacher(x)[0][:, -1]
    actual, _ = student(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Continuation parameters may change, but the AR path must remain identical.
    with torch.no_grad():
        for name, p in student.named_parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.1)
    actual2, _ = student(x)
    torch.testing.assert_close(actual2, expected, rtol=0, atol=0)


def test_consecutive_acceptance_length_stops_on_first_student_error():
    from calibrate_onepass_eob import consecutive_acceptance_lengths
    predicted = torch.tensor([
        [1, 2, 9, 4],
        [5, 6, 7, 8],
    ])
    teacher_greedy = torch.tensor([
        [1, 2, 3, 4],
        [0, 6, 7, 8],
    ])
    assert consecutive_acceptance_lengths(predicted, teacher_greedy).tolist() == [2, 0]
