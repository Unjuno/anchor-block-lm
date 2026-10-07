"""Adaptive commit-length RL on a fixed maximum-horizon distilled student.

The generator always learns the same H=4 teacher horizon on all contexts.
After distillation the generator is frozen. Only a categorical LoRA policy
chooses how many of the already-predicted prefix tokens to commit: k=1..4.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FIXED_DIR = HERE.parent / 'fixed_k_gate'
sys.path.insert(0, str(FIXED_DIR))

import fixed_k as fk
import run_experiment as base
from adaptive_k import MAX_K, commit_lengths
from run_adaptive_k import prefix_risk_bank, train_gate, summarize_policy, generate, generate_ar

PROTOCOL_VERSION = 'adaptive-k4-categorical-lora-v1'


def save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def histogram(values, max_k=MAX_K):
    return {str(k): int(sum(v == k for v in values)) for k in range(1, max_k + 1)}


@torch.no_grad()
def online_diagnostic(student, gate, prefixes, seed, count=48, repeats=3):
    timing = []
    representative = None
    for repeat in range(repeats):
        calls = 0
        requested, emitted = [], []
        outputs = []
        start = time.perf_counter()
        for i in range(len(prefixes)):
            out, n, req, em = generate(
                student, gate, prefixes[i:i+1], count=count,
                seed=seed + 1000 * repeat + i,
            )
            calls += n
            requested += req
            emitted += em
            outputs.append(out[0])
        timing.append(time.perf_counter() - start)
        if representative is None:
            representative = {
                'calls': calls,
                'requested_k_histogram': histogram(requested),
                'emitted_length_histogram': histogram(emitted),
                'mean_requested_k': float(np.mean(requested)),
                'tokens_per_backbone_call': len(prefixes) * count / calls,
                'output_sha256': hashlib.sha256(
                    torch.stack(outputs).cpu().numpy().tobytes()
                ).hexdigest(),
            }
    representative['generation_seconds_median'] = float(np.median(timing))
    representative['generation_seconds_range'] = [float(min(timing)), float(max(timing))]
    return representative


@torch.no_grad()
def ar_timing(student, prefixes, seed, count=48, repeats=3):
    timing = []
    representative = None
    for repeat in range(repeats):
        calls = 0
        outputs = []
        start = time.perf_counter()
        for i in range(len(prefixes)):
            out, n = generate_ar(
                student, prefixes[i:i+1], count=count,
                seed=seed + 1000 * repeat + i,
            )
            calls += n
            outputs.append(out[0])
        timing.append(time.perf_counter() - start)
        if representative is None:
            representative = {
                'calls': calls,
                'requested_k_histogram': {'1': calls, '2': 0, '3': 0, '4': 0},
                'emitted_length_histogram': {'1': calls, '2': 0, '3': 0, '4': 0},
                'mean_requested_k': 1.0,
                'tokens_per_backbone_call': len(prefixes) * count / calls,
                'output_sha256': hashlib.sha256(
                    torch.stack(outputs).cpu().numpy().tobytes()
                ).hexdigest(),
            }
    representative['generation_seconds_median'] = float(np.median(timing))
    representative['generation_seconds_range'] = [
        float(min(timing)), float(max(timing))
    ]
    return representative


def run(args):
    base.seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    data, meta = base.smoke_data() if args.smoke else base.load_data(args.data)
    save(args.out / 'dataset_metadata.json', meta)
    save(args.out / 'protocol.json', {
        'version': PROTOCOL_VERSION,
        'max_horizon_H': MAX_K,
        'actions': list(range(1, MAX_K + 1)),
        'distillation': 'all contexts, full teacher categorical, temperature=1, top_p=1',
        'no_context_selection_for_student': True,
        'generator_frozen_during_rl': True,
        'rl_objective': '(k-1) - lambda * sampled prefix log(q/p)',
        'policy': 'categorical LoRA contextual-bandit REINFORCE',
        'inference_teacher_verification': False,
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    })

    teacher = base.train_teacher(data, meta['actual_vocab_size'], args.teacher_steps, args.seed, args.out)
    base.seed_all(args.seed + 10)
    student = fk.FixedKStudent(teacher)
    base.train_content(teacher, student, data, args, args.out)
    student.eval().requires_grad_(False)
    student_before = fk.state_hash(student)

    train_x, _ = base.contexts(data['train'], args.train_contexts, args.seed + 101, 64)
    train_bank = prefix_risk_bank(teacher, student, train_x, samples=8, seed=args.seed + 201)
    gate, penalty = train_gate(train_bank, steps=args.gate_steps, seed=args.seed + 301)
    student_after = fk.state_hash(student)
    if student_before != student_after:
        raise RuntimeError('distilled generator changed during RL')

    test_x, test_ix = base.contexts(data['test'], 64, args.seed + 401, 64, True)
    test_bank = prefix_risk_bank(teacher, student, test_x, samples=64, seed=args.seed + 501)
    policy = summarize_policy(test_bank, gate, penalty)

    with torch.no_grad():
        selected_k = commit_lengths(gate(test_bank['features']))
    dynamic_k_observed = int(torch.unique(selected_k).numel()) > 1

    online = online_diagnostic(student, gate, test_x[:8], args.seed + 601,
                               count=args.generation_tokens, repeats=3)
    ar = ar_timing(student, test_x[:8], args.seed + 701,
                   count=args.generation_tokens, repeats=3)
    online['latency_ratio_vs_ar'] = ar['generation_seconds_median'] / online['generation_seconds_median']
    online['call_reduction_vs_ar'] = 1.0 - online['calls'] / ar['calls']

    test_distill = base.distill_bank(teacher, student, test_x, 16, args.seed + 801)
    full_horizon_forward_kl = base.forward_kl(student, test_distill)

    cpu = 'unknown'
    if Path('/proc/cpuinfo').exists():
        cpu = next((line.split(':', 1)[1].strip()
                    for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')), 'unknown')

    result = {
        'protocol_version': PROTOCOL_VERSION,
        'seed': args.seed,
        'dataset': meta['dataset'],
        'conditions': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'numpy': np.__version__,
            'device': 'cpu',
            'cpu': cpu,
            'threads': 1,
            'dtype': 'float32',
            'kv_cache': False,
            'clock': 'not pinned',
            'teacher_config': vars(teacher.config),
            'teacher_parameters': sum(p.numel() for p in teacher.parameters()),
        },
        'invariants': {
            'max_horizon_H': MAX_K,
            'available_k': list(range(1, MAX_K + 1)),
            'no_eob': True,
            'no_student_target_selection_by_k': True,
            'student_frozen_during_rl': student_before == student_after,
            'student_hash_before_rl': student_before,
            'student_hash_after_rl': student_after,
            'base_matches_teacher': fk.state_hash(student.backbone) == fk.state_hash(teacher),
            'trainable_gate_parameters': [n for n, p in gate.named_parameters() if n.endswith(('.A', '.B'))],
        },
        'train_reward_penalty_lambda': penalty,
        'full_horizon_forward_kl': full_horizon_forward_kl,
        'heldout_common_state_policy': policy,
        'dynamic_k_observed': dynamic_k_observed,
        'online_adaptive': online,
        'online_ar': ar,
        'test_context_starts': test_ix.tolist(),
        'duration_seconds': time.perf_counter() - start,
        'limitations': [
            'H=4 student joint is still a finite-mixture approximation of the teacher',
            'selected-prefix reverse KL is Monte Carlo estimated on one small corpus',
            'REINFORCE here is a contextual bandit, not long-horizon sequence RL',
            'CPU timing is not a production GPU/KV-cache benchmark',
            'AR anchor is exact, but committed continuation prefixes may distort the teacher distribution',
        ],
    }
    save(args.out / 'result.json', result)
    torch.save({'student': student.state_dict(), 'gate': gate.state_dict()}, args.out / 'checkpoint.pt')
    np.savez_compressed(
        args.out / 'test_policy_bank.npz',
        risk_samples=test_bank['risk_samples'].numpy(),
        uncertainty=test_bank['uncertainty'].numpy(),
        selected_k=selected_k.numpy(),
        context_starts=test_ix.numpy(),
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=HERE / 'data')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=48017)
    ap.add_argument('--teacher-steps', type=int, default=600)
    ap.add_argument('--content-steps', type=int, default=1200)
    ap.add_argument('--gate-steps', type=int, default=800)
    ap.add_argument('--train-contexts', type=int, default=1024)
    ap.add_argument('--generation-tokens', type=int, default=48)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if min(args.teacher_steps, args.content_steps, args.gate_steps,
           args.train_contexts, args.generation_tokens) < 1:
        ap.error('all sizes must be positive')
    run(args)


if __name__ == '__main__':
    main()
