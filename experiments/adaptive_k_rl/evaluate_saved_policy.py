"""Offline rollout scoring of existing adaptive-k checkpoints; never retrains.

Length decisions are deterministic given prefix and the sampled anchor. They
precede sampling the continuation, so the recorded segmentation determines the
sequence likelihood. The score is log Q(sequence) - log P_teacher(sequence).
Individual samples may be negative; their Q-sampled average estimates reverse
KL. Teacher replay is evaluation work, not inference or timing work.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import numpy as np
import torch


@torch.no_grad()
def score_sequence(teacher, student, prefix: torch.Tensor,
                   tokens: torch.Tensor, lengths: list[int]) -> dict[str, float]:
    if tokens.ndim != 1 or prefix.ndim != 2 or prefix.shape[0] != 1:
        raise ValueError('Expected prefix [1,T] and tokens [L]')
    if any(not isinstance(n, int) or not 1 <= n <= 4 for n in lengths) or sum(lengths) != len(tokens):
        raise ValueError('Invalid emitted segmentation')
    if not len(tokens):
        return {'student_logq':0., 'teacher_logp':0., 'log_ratio_nats':0.}
    logq = 0.
    pos = 0
    context = prefix.clone()
    for size in lengths:
        block = tokens[pos:pos+size]
        h, anchor_logits = student.encode(context[:, -student.backbone.config.block_size:])
        anchor = block[:1]
        logq += float(anchor_logits.log_softmax(-1)[0, anchor[0]])
        if size > 1:
            weights, logits = student.tail(h, anchor)
            # Independently implemented marginal of the normalized joint mixture.
            selected = logits[0, :, :size-1].log_softmax(-1)
            index = block[1:][None, :, None].expand(logits.shape[1], -1, 1)
            component_logp = selected.gather(-1, index).squeeze(-1).sum(-1)
            logq += float(torch.logsumexp(weights[0].log_softmax(-1) + component_logp, 0))
        context = torch.cat([context, block[None]], 1)
        pos += size
    logp = 0.
    context = prefix.clone()
    for token in tokens:
        logits = teacher(context[:, -teacher.config.block_size:])[0][:, -1]
        logp += float(logits.log_softmax(-1)[0, token])
        context = torch.cat([context, token.reshape(1,1)], 1)
    return {'student_logq':logq, 'teacher_logp':logp, 'log_ratio_nats':logq-logp}


def prompt_cluster_interval(values: np.ndarray, seed: int, samples: int = 4000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError('Expected finite [prompts, samples] values')
    per_prompt = values.mean(1)
    rng = np.random.default_rng(seed)
    draws = per_prompt[rng.integers(len(per_prompt), size=(samples,len(per_prompt)))].mean(1)
    return np.quantile(draws,[.025,.975]).tolist()


def reconstruct_test(artifact: Path, out: Path) -> torch.Tensor:
    """Reuse the archived tokenizer; do not train a different tokenizer."""
    from tokenizers import Tokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bpe_probe'))
    from prepare_data import load_source, strip_gutenberg_wrapper, split_text
    meta = json.loads((artifact/'data/metadata.json').read_text())
    sha = lambda b: hashlib.sha256(b).hexdigest()
    tokenizer_path = artifact/'data/tokenizer.json'
    if sha(tokenizer_path.read_bytes()) != meta['tokenizer_sha256']:
        raise ValueError('Archived tokenizer hash mismatch')
    raw = load_source(out/'source-cache',meta['source_url'],meta['raw_sha256'])
    text = strip_gutenberg_wrapper(raw.decode('utf-8-sig'))
    if sha(text.encode()) != meta['clean_text_sha256']:
        raise ValueError('Body hash mismatch')
    part = split_text(text)[2]
    if sha(part.encode()) != meta['splits']['test']['text_sha256']:
        raise ValueError('Test text hash mismatch')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    ids = np.asarray(tokenizer.encode(part).ids,dtype=np.uint16)
    serialized = io.BytesIO()
    np.save(serialized,ids)
    if sha(serialized.getvalue()) != meta['splits']['test']['array_sha256']:
        raise ValueError('Test token-array hash mismatch')
    return torch.from_numpy(ids.astype(np.int64))


def run(artifact: Path, out: Path, prompts: int, draws: int, count: int):
    if min(prompts,draws,count) < 1:
        raise ValueError('All sizes must be positive')
    out.mkdir(parents=True,exist_ok=False)
    here = Path(__file__).resolve().parent
    sys.path.insert(0,str(here))
    sys.path.insert(0,str(here.parent/'fixed_k_gate'))
    import fixed_k as fk
    from adaptive_k import AdaptiveKGate
    from run_adaptive_k import generate, generate_ar
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seq = reconstruct_test(artifact,out)
    result = {'evaluation_version':'saved-adaptive-rollout-v1',
              'training_run_id':37324518978,
              'training_commit':'c059008244c10cdcd952c88c6bab097151e42696',
              'protocol':{'prompts':prompts,'draws_per_prompt':draws,'tokens_per_draw':count,
                          'selection':'first saved test offsets; no tuning or training',
                          'holdout_status':'post-hoc diagnostic on previously used held-out corpus',
                          'metric':'mean generated-sequence log Q/P; not human quality',
                          'interval':'prompt-cluster bootstrap, not training-seed uncertainty',
                          'teacher_scoring_excluded_from_inference':True},
              'environment':{'python':platform.python_version(),'torch':torch.__version__,
                             'numpy':np.__version__,'threads':1,'dtype':'float32','device':'cpu'},
              'seeds':[]}
    for directory in sorted((artifact/'outputs').glob('seed-*')):
        old = json.loads((directory/'result.json').read_text())
        seed = old['seed']
        ck = torch.load(directory/'teacher.pt',map_location='cpu',weights_only=True)
        teacher = fk.GPT(fk.GPTConfig(**ck['config'])).eval().requires_grad_(False)
        teacher.load_state_dict(ck['model'])
        saved = torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
        student = fk.FixedKStudent(teacher).eval().requires_grad_(False)
        student.load_state_dict(saved['student'])
        gate = AdaptiveKGate(saved['gate']['center'].numel()).eval().requires_grad_(False)
        gate.load_state_dict(saved['gate'])
        before = fk.state_hash(student)
        before_gate = fk.state_hash(gate)
        if before != old['invariants']['student_hash_after_rl']:
            raise ValueError('Student hash mismatch')
        if fk.state_hash(teacher) != fk.state_hash(student.backbone):
            raise ValueError('Frozen AR backbone mismatch')
        offsets = old['test_context_starts'][:prompts]
        if len(offsets)!=prompts:
            raise ValueError('Not enough archived contexts')
        prefix = seq[torch.tensor(offsets)[:,None]+torch.arange(teacher.config.block_size)]
        scores = np.zeros((2,prompts,draws,3))
        calls = np.zeros((2,prompts,draws),dtype=np.int64)
        arrays = np.zeros((2,prompts,draws,count),dtype=np.int64)
        hist = {str(k):0 for k in range(1,5)}
        requested_hist = dict(hist)
        for i in range(prompts):
            for j in range(draws):
                rng_seed = 950000 + seed*100 + i*draws + j
                tokens,n,requested,lengths = generate(student,gate,prefix[i:i+1],count,rng_seed)
                calls[0,i,j]=n
                for length in lengths: hist[str(length)]+=1
                for length in requested: requested_hist[str(length)]+=1
                ar_tokens,ar_calls = generate_ar(student,prefix[i:i+1],count,rng_seed)
                calls[1,i,j]=ar_calls
                for a,(generated,segments) in enumerate(((tokens,lengths),(ar_tokens,[1]*count))):
                    scored=score_sequence(teacher,student,prefix[i:i+1],generated[0],segments)
                    scores[a,i,j]=[scored['log_ratio_nats'],scored['student_logq'],scored['teacher_logp']]
                    arrays[a,i,j]=generated[0].numpy()
        if not np.allclose(scores[1,:,:,0],0.,atol=1e-4):
            raise RuntimeError('Exact AR score is not zero')
        if fk.state_hash(student)!=before or fk.state_hash(gate)!=before_gate:
            raise RuntimeError('Evaluation changed a model')
        metrics = {'seed':seed,'generator_hash':before,'gate_hash':before_gate,
                   'models_unchanged':True,'context_starts':offsets,
                   'requested_k_histogram':requested_hist,'emitted_k_histogram':hist,
                   'adaptive':{'tokens_per_call':float(prompts*draws*count/calls[0].sum()),
                               'mean_sequence_log_ratio_nats':float(scores[0,:,:,0].mean()),
                               'log_ratio_nats_per_token':float(scores[0,:,:,0].mean()/count),
                               'prompt_cluster_95ci_nats_per_token':prompt_cluster_interval(scores[0,:,:,0]/count,seed),
                               'teacher_nll_nats_per_token':float(-scores[0,:,:,2].mean()/count)},
                   'ar':{'tokens_per_call':1.,'max_absolute_log_ratio_nats':float(np.abs(scores[1,:,:,0]).max()),
                         'teacher_nll_nats_per_token':float(-scores[1,:,:,2].mean()/count)},
                   'original_pure_ar_timing':{'adaptive_seconds_median':old['online_adaptive']['generation_seconds_median'],
                                              'ar_seconds_median':old['online_ar']['generation_seconds_median'],
                                              'speed_ratio':old['online_adaptive']['latency_ratio_vs_ar']}}
        np.savez_compressed(out/f'rollout-seed-{seed}.npz',scores=scores,calls=calls,tokens=arrays,offsets=offsets)
        result['seeds'].append(metrics)
        print(json.dumps(metrics),flush=True)
    (out/'rollout_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifact-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--prompts',type=int,default=16)
    ap.add_argument('--draws',type=int,default=8)
    ap.add_argument('--count',type=int,default=48)
    args=ap.parse_args()
    run(args.artifact_root,args.out,args.prompts,args.draws,args.count)
