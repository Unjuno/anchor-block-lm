"""Audit teacher-to-student distribution coverage at shared contexts.

This scores already-trained actors only. No test-time training or policy tuning.
All actors receive exactly the same teacher-sampled anchors and full tail targets.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from run_two_model import (reconstruct_splits,load_saved,save,fk,prompt_cluster_interval)
from two_model import teacher_tail,enable_content_lora,deployment_copy
import copy

@torch.no_grad()
def score_contents(teacher,actors,contexts,samples=16,seed=89001):
    if samples<1:raise ValueError('positive samples required')
    g=torch.Generator().manual_seed(seed)
    x=contexts.repeat_interleave(samples,0)
    al=teacher(x)[0][:,-1]
    anchor=torch.multinomial(al.softmax(-1),1,generator=g).squeeze(-1)
    target,logp=teacher_tail(teacher,x,anchor,g)
    raw={'contexts':contexts.numpy(),'anchor':anchor.numpy(),'teacher_tokens':target.numpy()}
    results={}
    for name,actor in actors.items():
        h,_=actor.encode(x);w,l=actor.tail(h,anchor)
        ratios=(logp-fk.joint_log_prob(w,l,target)).reshape(len(contexts),samples).numpy()
        raw[name]=ratios
        results[name]={'forward_kl_nats_per_tail':float(ratios.mean()),
                       'tail_tokens':3,'prompt_cluster_95ci':prompt_cluster_interval(ratios,seed)}
    return results,raw

def run(args):
    torch.set_num_threads(1);args.out.mkdir(parents=True,exist_ok=False)
    data,meta=reconstruct_splits(args.artifact_root,args.out)
    teacher,initial,_,old=load_saved(args.artifact_root/'outputs'/f'seed-{args.seed}')
    before=fk.state_hash(teacher);actors={'initial':deployment_copy(initial)}
    for name in ('gate_only','joint'):
        ck=torch.load(args.trained_root/f'{name}.pt',map_location='cpu',weights_only=True)
        actor=enable_content_lora(copy.deepcopy(initial));actor.load_state_dict(ck['student'])
        actors[name]=deployment_copy(actor)
    offsets=old['test_context_starts'][:16]
    ctx=data['test'][torch.tensor(offsets)[:,None]+torch.arange(teacher.config.block_size)]
    metrics,raw=score_contents(teacher,actors,ctx,samples=32,seed=args.seed+18000)
    delta=raw['joint']-raw['gate_only']
    result={'seed':args.seed,'evaluation':'shared-context full-tail teacher-sampled forward KL',
            'prompts':16,'samples_per_prompt':32,'tail_tokens':3,'teacher_unchanged':fk.state_hash(teacher)==before,
            'results':metrics,'joint_minus_gate_only_mean':float(delta.mean()),
            'joint_minus_gate_only_prompt_cluster_95ci':prompt_cluster_interval(delta,args.seed+19000),
            'same_teacher_samples_all_actors':True,'trained_models_not_updated':True}
    np.savez_compressed(args.out/'content_fidelity.npz',**raw)
    save(args.out/'content_fidelity.json',result);print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifact-root',type=Path,required=True)
    ap.add_argument('--trained-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,required=True)
    run(ap.parse_args())
