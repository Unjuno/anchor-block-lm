"""Paired tiny-nanoGPT experiment: fixed evaluator, joint LoRA actor update.

No architecture/length change. H=4; deployed k is chosen before sampling tail.
Run gate-only and joint on-policy training from identical archived checkpoints.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'fixed_k_gate'))
sys.path.insert(0,str(HERE.parent/'adaptive_k_rl'))
import fixed_k as fk
from adaptive_k import AdaptiveKGate, policy_step
from evaluate_saved_policy import score_sequence,prompt_cluster_interval
from two_model import (H,enable_content_lora,content_step,policy_batch,time_rewards,
                       deployment_copy,generate,collect_states)

def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def train_arm(teacher,initial_student,initial_gate,train_seq,arm,penalty,costs,
              rounds=4,updates=64,batch=32,seed=10):
    if arm not in ('gate_only','joint'):raise ValueError(arm)
    torch.manual_seed(seed)
    student=enable_content_lora(copy.deepcopy(initial_student),rank=8)
    if arm=='gate_only':student.requires_grad_(False)
    gate=copy.deepcopy(initial_gate).eval().requires_grad_(False)
    for n,p in gate.named_parameters():
        if n.endswith(('.A','.B')):p.requires_grad_(True)
    content_names=[n for n,p in student.named_parameters() if p.requires_grad]
    gate_names=[n for n,p in gate.named_parameters() if p.requires_grad]
    assert all(n.endswith(('.A','.B')) for n in content_names+gate_names)
    optg=torch.optim.Adam([p for p in gate.parameters() if p.requires_grad],lr=.001)
    optc=(torch.optim.Adam([p for p in student.parameters() if p.requires_grad],lr=.0003)
          if arm=='joint' else None)
    th=fk.state_hash(teacher); sh=fk.state_hash(student);gh=fk.state_hash(gate)
    rng=torch.Generator().manual_seed(seed+1);history=[];window=teacher.config.block_size
    for r in range(rounds):
        offsets=torch.randint(len(train_seq)-window,(32,),generator=rng)
        prefixes=train_seq[offsets[:,None]+torch.arange(window)]
        states=collect_states(student,gate,prefixes,cycles=4,seed=seed+100+r)
        totals=[]
        for step in range(updates):
            ix=torch.randint(len(states),(batch,),generator=rng);x=states[ix]
            cm=content_step(teacher,student,optc,x,rng,samples=4,kd_weight=.5) if optc else {}
            bank=policy_batch(teacher,student,x,rng,samples=4)
            rewards=time_rewards(bank['risk'],costs,penalty)
            gl=policy_step(gate,optg,bank['features'],rewards)
            totals.append({'policy_loss':gl,**cm})
        row={'round':r,'training_offsets':offsets.tolist(),
             'visited_context_sha256':sha_bytes(states.numpy().tobytes()),
             'loss_means':{k:float(np.mean([t[k] for t in totals])) for k in totals[0]}}
        history.append(row)
        print(json.dumps({'arm':arm,'round':r,'loss_means':row['loss_means']}),flush=True)
    info={'teacher_unchanged':fk.state_hash(teacher)==th,
          'anchor_unchanged':fk.state_hash(student.backbone)==th,
          'content_changed':fk.state_hash(student)!=sh,'gate_changed':fk.state_hash(gate)!=gh,
          'content_hash_before':sh,'content_hash_after':fk.state_hash(student),
          'gate_hash_before':gh,'gate_hash_after':fk.state_hash(gate),
          'trainable_content_names':content_names,'trainable_gate_names':gate_names,
          'history':history}
    assert info['teacher_unchanged'] and info['anchor_unchanged']
    assert info['content_changed']==(arm=='joint')
    student.eval().requires_grad_(False);gate.eval().requires_grad_(False)
    return student,gate,info

@torch.no_grad()
def measure_costs(student,gate,prefixes,repeats=3):
    """Initial microbench costs; a frozen reward proxy, not final speed evidence."""
    g=torch.Generator().manual_seed(3199);measurements={k:[] for k in range(H+1)}
    for repeat in range(repeats):
        for k in np.random.default_rng(3200+repeat).permutation(H+1):
            start=time.perf_counter()
            for i in range(len(prefixes)):
                ctx=prefixes[i:i+1];h,al=student.encode(ctx)
                a=torch.multinomial(al.softmax(-1),1,generator=g).squeeze(-1)
                ys=a[:,None]
                if k>0:
                    w,l=student.tail(h,a);gate(student.features(h,a,w,l))
                    if k>1:ys=torch.cat([ys,fk.joint_sample(w,l,g)[:,:k-1]],1)
                torch.cat([ctx,ys],1)
            measurements[int(k)].append((time.perf_counter()-start)/len(prefixes))
    medians={k:float(np.median(v)) for k,v in measurements.items()}
    return torch.tensor([medians[k]/medians[0] for k in range(1,H+1)]),{
        'pure_ar_seconds_per_step':medians[0],
        'adaptive_seconds_per_step':{str(k):medians[k] for k in range(1,H+1)},
        'trials_seconds_per_step':measurements,'purpose':'fixed training reward proxy only'}

@torch.no_grad()
def evaluate(teacher,student,gate,prefixes,draws=4,count=48,seed=3400):
    shape=(len(prefixes),draws);ratios=np.zeros(shape);logps=np.zeros(shape)
    calls=np.zeros(shape,dtype=np.int64);tokens_all=np.zeros((*shape,count),dtype=np.int64)
    hist={str(k):0 for k in range(1,H+1)};before=fk.state_hash(student)
    for i in range(len(prefixes)):
        for j in range(draws):
            tokens,lengths=generate(student,gate,prefixes[i:i+1],count,seed+i*draws+j)
            score=score_sequence(teacher,student,prefixes[i:i+1],tokens[0],lengths)
            ratios[i,j]=score['log_ratio_nats'];logps[i,j]=score['teacher_logp']
            tokens_all[i,j]=tokens[0].numpy();calls[i,j]=len(lengths)
            for k in lengths:hist[str(k)]+=1
    if fk.state_hash(student)!=before:raise RuntimeError('evaluation altered model')
    if gate is None and np.abs(ratios).max()>1e-4:raise RuntimeError('AR log ratio not zero')
    metrics={'tokens_per_call':float(tokens_all.size/calls.sum()),'emitted_k_histogram':hist,
             'sequence_kl_nats_per_token':float(ratios.mean()/count),
             'prompt_cluster_95ci':prompt_cluster_interval(ratios/count,seed),
             'teacher_nll_nats_per_token':float(-logps.mean()/count)}
    return metrics,{'tokens':tokens_all,'log_ratio':ratios,'teacher_logp':logps,'calls':calls}

@torch.no_grad()
def time_generation(student,gate,prefixes,count=48,seed=3500,repeats=5):
    # Warmup excluded. Replays use identical RNG seeds/traces per method.
    generate(student,gate,prefixes[:1],min(8,count),seed)
    times=[];hashes=[];counts=[]
    for _ in range(repeats):
        ys=[];calls=0;start=time.perf_counter()
        for i in range(len(prefixes)):
            y,ls=generate(student,gate,prefixes[i:i+1],count,seed+i)
            ys.append(y[0]);calls+=len(ls)
        times.append(time.perf_counter()-start);counts.append(calls)
        hashes.append(sha_bytes(torch.stack(ys).numpy().tobytes()))
    assert len(set(hashes))==1 and len(set(counts))==1
    return {'tokens':len(prefixes)*count,'calls':counts[0],
            'median_seconds':float(np.median(times)),
            'range_seconds':[float(min(times)),float(max(times))],
            'times_seconds':times,'outputs_identical_across_repeats':True,'trace_hash':hashes[0]}

def reconstruct_splits(artifact,out):
    from tokenizers import Tokenizer
    sys.path.insert(0,str(HERE.parent/'bpe_probe'))
    from prepare_data import load_source,strip_gutenberg_wrapper,split_text
    meta=json.loads((artifact/'data/metadata.json').read_text())
    if meta.get('preprocessing_version')!='gutenberg-body-v2':raise ValueError('unaudited data')
    tokenizer_path=artifact/'data/tokenizer.json'
    if sha_bytes(tokenizer_path.read_bytes())!=meta['tokenizer_sha256']:raise ValueError('tokenizer changed')
    raw=load_source(out/'source-cache',meta['source_url'],meta['raw_sha256'])
    text=strip_gutenberg_wrapper(raw.decode('utf-8-sig'))
    if sha_bytes(text.encode())!=meta['clean_text_sha256']:raise ValueError('body changed')
    tokenizer=Tokenizer.from_file(str(tokenizer_path));result={}
    for split,part in zip(('train','dev','test'),split_text(text)):
        if sha_bytes(part.encode())!=meta['splits'][split]['text_sha256']:raise ValueError('split changed')
        ids=np.asarray(tokenizer.encode(part).ids,dtype=np.uint16);b=io.BytesIO();np.save(b,ids)
        if sha_bytes(b.getvalue())!=meta['splits'][split]['array_sha256']:raise ValueError('token ids changed')
        result[split]=torch.from_numpy(ids.astype(np.int64))
    return result,meta

def load_saved(directory):
    old=json.loads((directory/'result.json').read_text())
    ck=torch.load(directory/'teacher.pt',map_location='cpu',weights_only=True)
    teacher=fk.GPT(fk.GPTConfig(**ck['config'])).eval().requires_grad_(False)
    teacher.load_state_dict(ck['model'])
    saved=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
    student=fk.FixedKStudent(teacher).eval().requires_grad_(False);student.load_state_dict(saved['student'])
    gate=AdaptiveKGate(saved['gate']['center'].numel()).eval().requires_grad_(False);gate.load_state_dict(saved['gate'])
    if fk.state_hash(student)!=old['invariants']['student_hash_after_rl']:raise ValueError('student archive mismatch')
    if fk.state_hash(student.backbone)!=fk.state_hash(teacher):raise ValueError('teacher archive mismatch')
    return teacher,student,gate,old

def run(args):
    if min(args.rounds,args.updates,args.prompts,args.draws,args.count)<1:raise ValueError('positive sizes required')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();data,meta=reconstruct_splits(args.artifact_root,args.out)
    directory=args.artifact_root/'outputs'/f'seed-{args.seed}'
    teacher,initial,gate,old=load_saved(directory);th=fk.state_hash(teacher)
    test_offsets=old['test_context_starts'][:args.prompts]
    if len(test_offsets)!=args.prompts:raise ValueError('not enough saved held-out prompts')
    test=data['test'][torch.tensor(test_offsets)[:,None]+torch.arange(teacher.config.block_size)]
    train_prefix=data['train'][torch.arange(32)[:,None]*64+torch.arange(64)]
    di=deployment_copy(initial);dg=deployment_copy(gate)
    costs,cost_info=measure_costs(di,dg,train_prefix)
    penalty=old['train_reward_penalty_lambda']
    save(args.out/'protocol.json',{'version':'two-model-online-lora-v1','seed':args.seed,
         'maximum_horizon':H,'variable_commit_lengths':[1,2,3,4],'args':vars(args)|{'artifact_root':str(args.artifact_root),'out':str(args.out)},
         'evaluator_frozen':True,'anchor_frozen':True,'teacher_sampling_top_p':1.,
         'generator_objective':'full-tail reverse-KL REINFORCE + 0.5 teacher-sample NLL on actor-visited training states',
         'gate_objective':'k - measured_initial_action_cost/pure_AR_cost - fixed_lambda * prefix_KL',
         'penalty_from_previous_train_only_run':penalty,'normalized_initial_costs':costs.tolist(),
         'selection':'fixed step count, no dev/test hyperparameter or checkpoint selection',
         'test_status':'exploratory reuse of archived held-out contexts, not new confirmatory data'})
    trained={'ar':(di,None),'initial':(di,dg)};invariants={}
    for arm in ('gate_only','joint'):
        actor,policy,info=train_arm(teacher,initial,gate,data['train'],arm,penalty,costs,
                                  rounds=args.rounds,updates=args.updates,seed=args.seed+2000)
        invariants[arm]=info
        torch.save({'student':actor.state_dict(),'gate':policy.state_dict(),'source_teacher_config':vars(teacher.config)},args.out/f'{arm}.pt')
        dd=deployment_copy(actor);gg=deployment_copy(policy)
        h,al=actor.encode(train_prefix);anchors=al.argmax(-1)
        for x,y in zip(actor.tail(h,anchors),dd.tail(h,anchors)):torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-5)
        w,l=actor.tail(h,anchors);features=actor.features(h,anchors,w,l)
        torch.testing.assert_close(policy(features),gg(features),rtol=2e-5,atol=2e-5)
        trained[arm]=(dd,gg)
    results={};arrays={}
    # No learning uses evaluation results. Evaluation teacher work is outside timing.
    for name,(actor,policy) in trained.items():
        metrics,raw=evaluate(teacher,actor,policy,test,args.draws,args.count,args.seed+9000)
        metrics['timing']=time_generation(actor,policy,test[:8],args.count,args.seed+10000,repeats=5)
        results[name]=metrics;arrays[name]=raw
        np.savez_compressed(args.out/f'{name}_rollouts.npz',**raw,context_starts=test_offsets)
        print(json.dumps({'method':name,**metrics}),flush=True)
    for name,row in results.items():row['speed_vs_pure_ar']=results['ar']['timing']['median_seconds']/row['timing']['median_seconds']
    diff=(arrays['joint']['log_ratio']-arrays['gate_only']['log_ratio'])/args.count
    compared={'joint_minus_gate_only_kl_nats_per_token':float(diff.mean()),
              'prompt_cluster_95ci':prompt_cluster_interval(diff,args.seed+11000),
              'same_prompt_independent_sample_difference_not_same_sequence':True,
              'joint_minus_gate_tokens_per_call':results['joint']['tokens_per_call']-results['gate_only']['tokens_per_call']}
    assert fk.state_hash(teacher)==th
    result={'seed':args.seed,'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
            'cpu':next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'unknown'),
            'threads':1,'dtype':'float32','batch_decode':1,'clock':'not pinned','kv_cache':False},
            'evaluator_hash':th,'source_student_hash':fk.state_hash(initial),'teacher_unchanged':True,
            'data_manifest':meta,'initial_reward_cost_calibration':cost_info,'invariants':invariants,
            'evaluation':results,'joint_vs_gate_only':compared,'duration_seconds':time.perf_counter()-start,
            'limitations':['one tiny teacher family and one play','no human semantic score',
             'on-policy distillation plus contextual-bandit RL, not full sequence-return PPO',
             'measured initial step costs are a fixed reward proxy, not online latency measurements',
             'different emitted lengths imply different throughput; no same-quality domination is assumed']}
    save(args.out/'result.json',result)
    print(json.dumps({'seed':args.seed,'joint_vs_gate_only':compared}),flush=True)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifact-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=48017);ap.add_argument('--rounds',type=int,default=4)
    ap.add_argument('--updates',type=int,default=64);ap.add_argument('--prompts',type=int,default=16)
    ap.add_argument('--draws',type=int,default=4);ap.add_argument('--count',type=int,default=48)
    run(ap.parse_args())
