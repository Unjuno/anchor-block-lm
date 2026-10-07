"""Resume archived dynamic-boundary actors with aligned categorical dual updates.

No new teacher, large model, test-set tuning, length forcing, or inference-time
verification. Training and exclusive evaluation are separate command phases.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from consistent_dual import H, Trainer, collect, generate, probe
import fixed_k as fk
from adaptive_k import AdaptiveKGate
from two_model import enable_content_lora, deployment_copy
from run_two_model import load_saved, reconstruct_splits
from evaluate_saved_policy import score_sequence, prompt_cluster_interval


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transitions(before,after):
    if len(before)!=len(after) or not len(before):raise ValueError('matching nonempty actions required')
    a=np.asarray(before);b=np.asarray(after)
    if ((a<1)|(a>H)|(b<1)|(b>H)).any():raise ValueError('invalid length')
    matrix=np.zeros((H,H),dtype=int);np.add.at(matrix,(a-1,b-1),1)
    return {'matrix':matrix.tolist(),'increased':int((b>a).sum()),'decreased':int((b<a).sum()),
            'unchanged':int((b==a).sum()),'changed_fraction':float((a!=b).mean())}


def resume_settings(old):
    p=old['protocol'];target=p['kl_target']
    if not math.isfinite(target) or target<0:raise ValueError('invalid archived budget')
    return {'target':target,'initial_dual':old['training']['final_dual_lambda'],
            'costs':p['normalized_action_costs'],'dev_starts':old['dev_probe_starts'],
            'eval_starts':old['evaluation_starts']}


def prepare(root):
    directory=root/'prepared';directory.mkdir(exist_ok=True)
    if not (directory/'source-cache').exists():
        shutil.copytree(root/'previous/seed-48017/source-cache',directory/'source-cache')
    data,meta=reconstruct_splits(root/'original',directory)
    # Original loader verifies all archive hashes. Only train/dev are retained.
    torch.save({'train':data['train'],'dev':data['dev']},directory/'train_dev.pt')
    save(directory/'metadata.json',meta)


def load(root,seed,path=None):
    teacher,original,_,_=load_saved(root/'original/outputs'/f'seed-{seed}')
    ck=torch.load(path or root/'previous'/f'seed-{seed}'/'final_checkpoint.pt',weights_only=True,map_location='cpu')
    student=enable_content_lora(original,rank=8)
    student.load_state_dict(ck['student'])
    gate=AdaptiveKGate(ck['gate']['center'].numel())
    gate.load_state_dict(ck['gate']);gate.eval()
    return teacher,student,gate


def snapshot(s,g): return {'student':s.state_dict(),'gate':g.state_dict()}


def train(root,seed):
    torch.set_num_threads(1);torch.manual_seed(seed+720000);np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    out=root/'new'/f'seed-{seed}';out.mkdir(parents=True,exist_ok=False)
    old=json.loads((root/'previous'/f'seed-{seed}'/'result.json').read_text());cfg=resume_settings(old)
    teacher,s,g=load(root,seed);th=fk.state_hash(teacher)
    initial_s=fk.state_hash(s);initial_g=fk.state_hash(g)
    frozen={n:p.clone() for n,p in s.state_dict().items() if not n.endswith(('.A','.B'))}
    torch.save(snapshot(s,g),out/'initial.pt')
    data=torch.load(root/'prepared/train_dev.pt',weights_only=True)
    window=teacher.config.block_size;ar=torch.arange(window)
    dev=data['dev'][torch.tensor(cfg['dev_starts'])[:,None]+ar]
    costs=torch.tensor(cfg['costs']);ref=cfg['initial_dual']
    tr=Trainer(teacher,s,g,costs,cfg['target'],ref,.1,seed+810000)
    history=[];logs=[];anchors=None
    def fixed_probe(r,phase):
        nonlocal anchors
        row,raw=probe(teacher,s,g,dev,seed+820000,32,costs,ref)
        if anchors is None:anchors=raw['anchors'].copy()
        assert np.array_equal(anchors,raw['anchors']), 'probe anchor changed'
        row.update({'round':r,'phase':phase,'dual_lambda':tr.dual.value})
        history.append(row);np.savez_compressed(out/f'probe_{r}_{phase}.npz',**raw)
        print(json.dumps({'seed':seed,'round':r,'phase':phase,'expected_k':row['mean_k_expected'],
                          'greedy_k':row['mean_k_greedy'],'risk':row['expected_risk'],'lambda':tr.dual.value}),flush=True)
    fixed_probe(0,'initial');rng=torch.Generator().manual_seed(seed+830000)
    started=time.perf_counter()
    for r in range(1,5):
        offsets=torch.randint(len(data['train'])-window+1,(32,),generator=rng)
        prefixes=data['train'][offsets[:,None]+ar]
        batch=collect(s,g,prefixes,4,seed+840000+r)
        cm=tr.content(batch['states'],64,32,4)
        fixed_probe(r,'content')
        current=collect(s,g,prefixes,4,seed+850000+r)
        pm=tr.policy(current['states'],128,32,4)
        fixed_probe(r,'policy')
        logs.append({'round':r,'starts':offsets.tolist(),'content':cm,'policy':pm,
                     'content_states_sha256':hashlib.sha256(batch['states'].numpy().tobytes()).hexdigest(),
                     'policy_states_sha256':hashlib.sha256(current['states'].numpy().tobytes()).hexdigest(),
                     'visited_k_histogram':np.bincount(current['lengths'],minlength=H+1)[1:].tolist()})
        assert fk.state_hash(teacher)==th==fk.state_hash(s.backbone)
    offsets=torch.randint(len(data['train'])-window+1,(32,),generator=rng)
    fresh=collect(s,g,data['train'][offsets[:,None]+ar],4,seed+860000)
    train_row,train_raw=probe(teacher,s,g,fresh['states'],seed+870000,32,costs,ref)
    np.savez_compressed(out/'train_final_probe.npz',**train_raw)
    invariants={'teacher_unchanged':fk.state_hash(teacher)==th,'backbone_unchanged':fk.state_hash(s.backbone)==th,
                'content_changed':fk.state_hash(s)!=initial_s,'policy_changed':fk.state_hash(g)!=initial_g,
                'non_lora_unchanged':all(torch.equal(s.state_dict()[n],p) for n,p in frozen.items()),
                'content_optimizer_steps':sorted({int(x['step']) for x in tr.opt_c.state.values()}),
                'policy_optimizer_steps':sorted({int(x['step']) for x in tr.opt_g.state.values()})}
    assert all(invariants[n] for n in ('teacher_unchanged','backbone_unchanged','content_changed','policy_changed','non_lora_unchanged'))
    torch.save(snapshot(s,g),out/'final.pt')
    torch.save({'content':tr.opt_c.state_dict(),'policy':tr.opt_g.state_dict(),'dual':tr.dual.value},out/'optimizer.pt')
    save(out/'training.json',{'seed':seed,'resumed_checkpoint_sha256':sha(root/'previous'/f'seed-{seed}'/'final_checkpoint.pt'),
        'settings':cfg,'history':history,'round_logs':logs,'train_final':train_row,'invariants':invariants,
        'greedy_boundaries':transitions(history[0]['greedy_k'],history[-1]['greedy_k']),
        'seconds':time.perf_counter()-started})


def rollout_summary(a,count,mode,seed):
    r=a['ratios']/count
    return {'tokens_per_call':float(r.size*count/a['calls'].sum()),
            'score_kind':('augmented_trace_reverse_kl_bound' if mode=='sampled' else 'token_sequence_reverse_kl'),
            'mean_log_ratio_nats_per_token':float(r.mean()),'prompt_cluster_95ci':prompt_cluster_interval(r,seed),
            'teacher_nll_nats_per_token':float(a['nll'].mean()/count)}


@torch.no_grad()
def evaluate(root,seed):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
    out=root/'new'/f'seed-{seed}';info=json.loads((out/'training.json').read_text())
    teacher,initial,ig=load(root,seed,out/'initial.pt');_,final,fg=load(root,seed,out/'final.pt')
    initial=deployment_copy(initial);ig=deployment_copy(ig);final=deployment_copy(final);fg=deployment_copy(fg)
    data=torch.load(root/'prepared/train_dev.pt',weights_only=True)
    x=data['dev'][torch.tensor(info['settings']['eval_starts'])[:,None]+torch.arange(teacher.config.block_size)]
    count=48;draws=4
    methods={'ar':(initial,None,'greedy'),'initial_sampled':(initial,ig,'sampled'),
             'final_sampled':(final,fg,'sampled'),'initial_greedy':(initial,ig,'greedy'),'final_greedy':(final,fg,'greedy')}
    raw={};result={};original_hashes={n:(fk.state_hash(s),fk.state_hash(g) if g else None) for n,(s,g,_) in methods.items()}
    for name,(s,g,mode) in methods.items():
        a={'ratios':np.zeros((len(x),draws)),'calls':np.zeros((len(x),draws),dtype=int),
           'nll':np.zeros((len(x),draws)),'tokens':np.zeros((len(x),draws,count),dtype=np.int64)}
        hist=np.zeros(H,dtype=int)
        for i in range(len(x)):
            for j in range(draws):
                y,ls=generate(s,g,x[i:i+1],count,seed+880000+i*draws+j,greedy=mode=='greedy')
                score=score_sequence(teacher,s,x[i:i+1],y[0],ls)
                a['ratios'][i,j]=score['log_ratio_nats'];a['nll'][i,j]=-score['teacher_logp'];a['calls'][i,j]=len(ls);a['tokens'][i,j]=y[0].numpy()
                hist+=np.bincount(ls,minlength=H+1)[1:]
        result[name]=rollout_summary(a,count,mode,seed+890000);result[name]['emitted_histogram']=hist.tolist()
        raw[name]=a;np.savez_compressed(out/f'{name}.npz',**a)
        print(json.dumps({'seed':seed,'method':name,**result[name]}),flush=True)
    assert abs(raw['ar']['ratios']).max()<1e-4
    # Timing excludes all teacher replay, warms up, alternates order, and replays
    # identical seeds per method. Run this phase WITHOUT concurrent training.
    times={n:[] for n in methods};trace={n:[] for n in methods}
    for s,g,mode in methods.values():generate(s,g,x[:1],8,seed+900000,greedy=mode=='greedy')
    for rep in range(5):
        for name in np.random.default_rng(seed+rep).permutation(list(methods)):
            s,g,mode=methods[name];ys=[];calls=0;start=time.perf_counter()
            for i in range(min(8,len(x))):
                y,ls=generate(s,g,x[i:i+1],count,seed+900000+i,greedy=mode=='greedy');ys.append(y);calls+=len(ls)
            times[name].append(time.perf_counter()-start)
            trace[name].append((hashlib.sha256(torch.cat(ys).numpy().tobytes()).hexdigest(),calls))
    for n,(s,g,_) in methods.items():
        assert len(set(trace[n]))==1
        assert original_hashes[n]==(fk.state_hash(s),fk.state_hash(g) if g else None)
        result[n]['timing']={'median_s':float(np.median(times[n])),'range_s':[min(times[n]),max(times[n])],
                              'trials_s':times[n],'trace_identical':True,'calls':trace[n][0][1]}
        result[n]['speed_vs_ar']=float(np.median(times['ar'])/np.median(times[n]))
    diff={}
    for mode in ('greedy','sampled'):
        a=(raw[f'final_{mode}']['ratios']-raw[f'initial_{mode}']['ratios'])/count
        diff[mode]={'mean':float(a.mean()),'prompt_cluster_95ci':prompt_cluster_interval(a,seed+910000)}
    cpu=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'unknown')
    save(out/'evaluation.json',{'seed':seed,'methods':result,'final_minus_initial':diff,
        'environment':{'cpu':cpu,'threads':1,'batch_decode':1,'dtype':'float32','clock':'not pinned','kv_cache':False,
                       'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                       'eval_prompts':len(x),'draws':draws,'tokens':count,'timing_repeats':5,'timing_prompts':min(8,len(x))}})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=('prepare','train','evaluate'))
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--seed',type=int,default=48017)
    args=ap.parse_args()
    if args.phase=='prepare':prepare(args.root)
    elif args.phase=='train':train(args.root,args.seed)
    else:evaluate(args.root,args.seed)

if __name__=='__main__': main()
