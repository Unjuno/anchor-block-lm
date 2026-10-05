from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from anchor_block_experiment import gen_anchor, load_student as load_fixed, teacher_score_block
from eob_experiment import H, decode_eob_logits
from onpolicy_eob_refresh import load_onpolicy
from poc import ROOT, dump_json, frozen_backbone_equal, load_teacher, seed_all


@torch.no_grad()
def gen_eob_onpolicy(teacher, model, prefix, count, eob_bias=0.0):
    out = prefix.clone()
    calls = 0
    block_lengths = []
    while out.shape[1] - prefix.shape[1] < count:
        logits = teacher(out[:, -teacher.config.block_size:])[0][:, -1]
        anchor = logits.argmax(-1)
        out = torch.cat([out, anchor[:, None]], 1)
        calls += 1
        rem = count - (out.shape[1] - prefix.shape[1])
        if rem <= 0:
            break
        block_logits, _ = model(out[:, -model.backbone.config.block_size:])
        toks, k = decode_eob_logits(block_logits, model.eob_id, H, eob_bias=eob_bias)
        take = min(int(k[0]), rem)
        if take:
            out = torch.cat([out, toks[:, :take]], 1)
        calls += 1
        block_lengths.append(take)
    return out, calls, block_lengths


@torch.no_grad()
def evaluate(n_prompts=64, count=64, bootstrap=2000, seed=990):
    seed_all(seed, 1)
    teacher = load_teacher(ROOT / 'results' / 'teacher.pt')
    fixed = load_fixed('anchor', merged=True)
    eob = load_onpolicy(merged=True)
    bank = torch.load(ROOT / 'results_anchor' / 'direct_test.pt', weights_only=True)
    prefixes = bank['x'][:n_prompts]

    rows = []
    agreement_vectors = {}
    methods = {
        'fixed_cont3': lambda p: (*gen_anchor(teacher, fixed, p, count, 3), []),
        'eob_onpolicy_bias0': lambda p: gen_eob_onpolicy(teacher, eob, p, count, 0.0),
        'eob_onpolicy_bias2': lambda p: gen_eob_onpolicy(teacher, eob, p, count, 2.0),
    }

    for name, fn in methods.items():
        outs, calls, block_lengths = [], [], []
        for i in range(n_prompts):
            out, n_calls, lens = fn(prefixes[i:i+1])
            outs.append(out[0, -count:])
            calls.append(n_calls)
            block_lengths.extend(lens)
        outs = torch.stack(outs)
        _, local_agree, local_nll = teacher_score_block(teacher, prefixes, outs)
        agreement_vectors[name] = local_agree.float().mean(1).cpu().numpy()
        rows.append({
            'method': name,
            'chars_per_call': float(n_prompts * count / np.sum(calls)),
            'local_teacher_agreement': local_agree.float().mean().item(),
            'local_teacher_nll': local_nll.mean().item(),
            'mean_block_continuation_length': float(np.mean(block_lengths)) if block_lengths else 0.0,
        })

    diff = agreement_vectors['eob_onpolicy_bias0'] - agreement_vectors['fixed_cont3']
    rng = np.random.default_rng(seed + 1)
    boots = np.empty(bootstrap)
    for i in range(bootstrap):
        ix = rng.integers(0, len(diff), len(diff))
        boots[i] = diff[ix].mean()

    result = {
        'conditions': {
            'prompts': n_prompts,
            'chars': count,
            'test_untouched_by_training': True,
            'device': 'cpu',
            'dtype': 'float32',
            'batch': 1,
            'kv_cache': False,
        },
        'frozen_base_equal': frozen_backbone_equal(teacher, eob),
        'rows': rows,
        'paired_agreement_diff_eob0_minus_fixed3': float(diff.mean()),
        'paired_bootstrap_95ci': [float(x) for x in np.quantile(boots, [0.025, 0.975])],
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', type=int, default=64)
    ap.add_argument('--chars', type=int, default=64)
    ap.add_argument('--bootstrap', type=int, default=2000)
    args = ap.parse_args()
    result = evaluate(args.prompts, args.chars, args.bootstrap)
    path = ROOT / 'results_eob' / 'published_benchmark.json'
    dump_json(path, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
