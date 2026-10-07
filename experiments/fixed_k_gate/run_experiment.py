"""Bounded nanoGPT fixed-K / binary-LoRA-gate experiment. CPU only.

Main estimand: conditional reverse KL on a common held-out decision bank, at
EXACTLY the same number of selected full blocks. This rank-allocation diagnostic
is not an online latency claim. Separately record unforced online gate behavior.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from fixed_k import (BLOCK_SIZE, TAIL, GPT, GPTConfig, FixedKStudent, LoRAGate,
                     state_hash, teacher_probs, joint_sample, joint_log_prob,
                     block_reward, policy_step, emitted_length, matched_mask)

PROTOCOL_VERSION='fixed4-joint-mixture-binary-lora-v1'
COVERAGE=.125


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path):
    meta=json.loads((path/'metadata.json').read_text())
    if meta.get('preprocessing_version')!='gutenberg-body-v2':
        raise ValueError('Only corrected body-v2 data accepted; use --smoke for synthetic')
    if digest(path/'tokenizer.json') != meta['tokenizer_sha256']:
        raise ValueError('Tokenizer hash mismatch')
    for split in ('train','dev','test'):
        if digest(path/f'{split}.npy')!=meta['splits'][split]['array_sha256']:
            raise ValueError(f'{split} hash mismatch')
    return {s:torch.from_numpy(np.load(path/f'{s}.npy').astype(np.int64))
            for s in ('train','dev','test')},meta


def smoke_data():
    data={}
    for name,n,seed in [('train',4096,37001),('dev',1024,37002),('test',1024,37003)]:
        rng=np.random.default_rng(seed); seq=[]
        for _ in range(n):
            a=int(rng.integers(1,9));b=int(rng.integers(9,13))
            seq.extend([0,a,16+a,24+a,32+a,15,b,40+b,44+b,63])
        data[name]=torch.tensor(seq)
    return data,{'dataset':'synthetic-smoke-grammar','actual_vocab_size':64,
                 'warning':'NOT natural language; software/controlled-mechanism smoke only'}


def contexts(seq,n,seed,window,disjoint=False):
    g=torch.Generator().manual_seed(seed)
    if disjoint:
        starts=torch.arange(0,len(seq)-window,window)
        if len(starts)<n: raise ValueError(f'Need {n} disjoint contexts, have {len(starts)}')
        ix=starts[torch.randperm(len(starts),generator=g)[:n]]
    else: ix=torch.randint(len(seq)-window-1,(n,),generator=g)
    return seq[ix[:,None]+torch.arange(window)],ix


@torch.no_grad()
def loss_on(teacher,seq,seed,batches=4):
    teacher.eval();x,ix=contexts(seq,32*batches,seed,teacher.config.block_size)
    y=seq[ix[:,None]+torch.arange(1,teacher.config.block_size+1)].contiguous()
    return float(np.mean([teacher(x[i:i+32],y[i:i+32])[1].item() for i in range(0,len(x),32)]))


def train_teacher(data,vocab,steps,seed,out):
    seed_all(seed)
    cfg=GPTConfig(block_size=64,vocab_size=vocab,n_layer=2,n_head=4,n_embd=64,dropout=0.,bias=True)
    teacher=GPT(cfg)
    opt=torch.optim.AdamW(teacher.parameters(),lr=.003,weight_decay=.01)
    g=torch.Generator().manual_seed(seed+1);seq=data['train'];best=float('inf');history=[]
    for step in range(1,steps+1):
        teacher.train();ix=torch.randint(len(seq)-65,(32,),generator=g)
        x=seq[ix[:,None]+torch.arange(64)]
        y=seq[ix[:,None]+torch.arange(1,65)].contiguous()
        opt.param_groups[0]['lr']=.003*(.15+.85*.5*(1+math.cos(math.pi*step/steps)))
        loss=teacher(x,y)[1];opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.);opt.step()
        if step%100==0 or step==steps:
            dev=loss_on(teacher,data['dev'],44001)
            row={'stage':'teacher','step':step,'dev_nll':dev};print(json.dumps(row),flush=True);history.append(row)
            if dev<best:
                best=dev;torch.save({'model':teacher.state_dict(),'config':vars(cfg)},out/'teacher.pt')
    ck=torch.load(out/'teacher.pt',weights_only=True);teacher.load_state_dict(ck['model'])
    teacher.eval().requires_grad_(False)
    save(out/'teacher_history.json',history)
    return teacher


@torch.no_grad()
def distill_bank(teacher,student,x,samples,seed):
    g=torch.Generator().manual_seed(seed)
    result={'h':[],'anchor':[],'tokens':[],'teacher_log_tail':[]}
    for st in range(0,len(x),16):
        base=x[st:st+16];h,al=student.encode(base)
        ctx=base.repeat_interleave(samples,0)
        anchor=torch.multinomial(teacher_probs(al).repeat_interleave(samples,0),1,generator=g).squeeze(-1)
        ctx=torch.cat([ctx,anchor[:,None]],1)
        ys=[]; logp=torch.zeros(len(ctx))
        for _ in range(TAIL):
            logits=teacher(ctx[:,-64:])[0][:,-1]
            tok=torch.multinomial(teacher_probs(logits),1,generator=g).squeeze(-1)
            logp+=logits.log_softmax(-1).gather(-1,tok[:,None]).squeeze(-1)
            ys.append(tok);ctx=torch.cat([ctx,tok[:,None]],1)
        result['h'].append(h.repeat_interleave(samples,0));result['anchor'].append(anchor)
        result['tokens'].append(torch.stack(ys,1));result['teacher_log_tail'].append(logp)
    return {k:torch.cat(v) for k,v in result.items()}


@torch.no_grad()
def forward_kl(student,bank):
    values=[]
    for st in range(0,len(bank['h']),128):
        w,l=student.tail(bank['h'][st:st+128],bank['anchor'][st:st+128])
        logq=joint_log_prob(w,l,bank['tokens'][st:st+128])
        values.append(bank['teacher_log_tail'][st:st+128]-logq)
    v=torch.cat(values)
    return {'mean_nats_per_tail':v.mean().item(),'mc_se':v.std().item()/math.sqrt(len(v))}


def train_content(teacher,student,data,args,out):
    x,_=contexts(data['train'],args.train_contexts,45001,64)
    dx,_=contexts(data['dev'],64,45002,64,True)
    bank=distill_bank(teacher,student,x,8,args.seed+100)
    dev=distill_bank(teacher,student,dx,16,args.seed+101)
    g=torch.Generator().manual_seed(args.seed+102)
    trainables=[p for p in student.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainables,lr=.002,weight_decay=.001)
    best=float('inf');hist=[]
    for step in range(1,args.content_steps+1):
        ix=torch.randint(len(bank['h']),(128,),generator=g)
        w,l=student.tail(bank['h'][ix],bank['anchor'][ix])
        loss=-joint_log_prob(w,l,bank['tokens'][ix]).mean()
        opt.param_groups[0]['lr']=.002*(.2+.8*.5*(1+math.cos(math.pi*step/args.content_steps)))
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(trainables,1.);opt.step()
        if step%200==0 or step==args.content_steps:
            metrics=forward_kl(student,dev)
            row={'stage':'content','step':step,'loss':loss.item(),'dev_forward_kl':metrics}
            print(json.dumps(row),flush=True);hist.append(row)
            if metrics['mean_nats_per_tail']<best:
                best=metrics['mean_nats_per_tail'];torch.save(student.state_dict(),out/'student.pt')
    student.load_state_dict(torch.load(out/'student.pt',weights_only=True))
    student.eval().requires_grad_(False)
    save(out/'content_history.json',hist)


@torch.no_grad()
def risk_bank(teacher,student,x,samples,seed):
    """Unbiased MC log q/p at student-drawn tails; teacher used only offline."""
    g=torch.Generator().manual_seed(seed)
    feats=[];ratios=[];conf=[];anchors=[]
    for st in range(0,len(x),16):
        xx=x[st:st+16];h,al=student.encode(xx)
        anchor=torch.multinomial(teacher_probs(al),1,generator=g).squeeze(-1)
        w,l=student.tail(h,anchor)
        feature=student.features(h,anchor,w,l)
        ww=w.repeat_interleave(samples,0);ll=l.repeat_interleave(samples,0)
        tail=joint_sample(ww,ll,g)
        logq=joint_log_prob(ww,ll,tail)
        ctx=torch.cat([xx,anchor[:,None]],1).repeat_interleave(samples,0)
        logp=torch.zeros(len(ctx))
        for j in range(TAIL):
            tl=teacher(ctx[:,-64:])[0][:,-1]
            logp+=tl.log_softmax(-1).gather(-1,tail[:,j,None]).squeeze(-1)
            ctx=torch.cat([ctx,tail[:,j,None]],1)
        # A simple unlabeled confidence baseline; sum of marginal entropies.
        marginal=(w.softmax(-1)[:,:,None,None]*l.softmax(-1)).sum(1)
        score=(marginal*marginal.clamp_min(1e-30).log()).sum((-1,-2))
        feats.append(feature);conf.append(score);anchors.append(anchor)
        ratios.append((logq-logp).reshape(len(xx),samples))
    return {'features':torch.cat(feats),'risk_samples':torch.cat(ratios),
            'confidence':torch.cat(conf),'anchor':torch.cat(anchors)}


def train_gate(bank,steps,seed,out):
    seed_all(seed)
    gate=LoRAGate(bank['features'].shape[-1])
    gate.center.copy_(bank['features'].mean(0));gate.scale.copy_(bank['features'].std(0).clamp_min(.05))
    risks=bank['risk_samples']
    # Set exchange rate from TRAIN only. No dev/test error labels tune reward.
    train_quantile=max(float(risks.mean(1).quantile(.25)),.25)
    penalty=(BLOCK_SIZE-1)/train_quantile
    opt=torch.optim.Adam([p for p in gate.parameters() if p.requires_grad],lr=.003)
    generator=torch.Generator().manual_seed(seed+1);hist=[]
    for step in range(1,steps+1):
        ix=torch.randint(len(risks),(128,),generator=generator)
        j=torch.randint(risks.shape[1],(128,),generator=generator)
        r=block_reward(risks[ix,j],penalty)
        loss=policy_step(gate,opt,bank['features'][ix],r)
        if step%200==0 or step==steps:
            row={'stage':'gate','step':step,'loss':loss};hist.append(row);print(json.dumps(row),flush=True)
    gate.eval().requires_grad_(False)
    torch.save(gate.state_dict(),out/'gate.pt');save(out/'gate_history.json',hist)
    return gate,penalty


def matched_diagnostic(bank,scores,seed,boot=1000):
    """Transductive rank-allocation diagnostic; never label it online routing."""
    risk=bank['risk_samples'].numpy();n=len(risk);b=max(1,round(n*COVERAGE))
    learned=matched_mask(scores,b).numpy()
    confidence=matched_mask(bank['confidence'],b).numpy()
    mean=risk.mean(1)
    rng=np.random.default_rng(seed)
    random_masks=[]
    for _ in range(32):
        m=np.zeros(n,dtype=bool);m[rng.choice(n,b,replace=False)]=True;random_masks.append(m)
    def stats(mask):
        return {'blocks':int(mask.sum()),'calls':n,'emitted_tokens':n+TAIL*b,
                'tokens_per_call':(n+TAIL*b)/n,
                'mean_selected_reverse_kl_nats':float(mean[mask].mean()),
                'total_kl_per_emitted_token':float(mean[mask].sum()/(n+TAIL*b))}
    # Random control expectation over all exact-b subsets. Each subset has b blocks.
    random_expected=float(mean.mean())
    bs=[]
    for _ in range(boot):
        ix=rng.integers(n,size=n)
        mc=risk[ix][np.arange(n)[:,None],rng.integers(risk.shape[1],size=risk.shape)].mean(1)
        sm=matched_mask(scores[ix],b).numpy()
        cm=matched_mask(bank['confidence'][ix],b).numpy()
        bs.append([float(mc[sm].mean()-mc.mean()),float(mc[sm].mean()-mc[cm].mean())])
    intervals=np.quantile(np.asarray(bs),[.025,.975],axis=0).T.tolist()
    return {'type':'offline common-state rank allocation; not deployed online quota',
            'contexts':n,'blocks_each':b,'coverage':b/n,
            'learned':stats(learned),'confidence':stats(confidence),
            'random_expected':{'blocks_each':b,'tokens_per_call':(n+TAIL*b)/n,
                               'mean_selected_reverse_kl_nats':random_expected},
            'random_32_allocations_risk_range':[float(min(mean[m].mean() for m in random_masks)),
                                                float(max(mean[m].mean() for m in random_masks))],
            'learned_minus_uniform_kl':float(mean[learned].mean()-random_expected),
            'learned_minus_confidence_kl':float(mean[learned].mean()-mean[confidence].mean()),
            'nested_bootstrap_95ci_learned_minus_uniform':intervals[0],
            'nested_bootstrap_95ci_learned_minus_confidence':intervals[1],
            'mechanism_pass_vs_uniform':intervals[0][1]<0,
            'incremental_pass_vs_confidence':intervals[1][1]<0}


@torch.no_grad()
def generate(student,gate,prefix,count,kind,threshold,seed):
    g=torch.Generator().manual_seed(seed);out=prefix.clone();calls=0;lengths=[]
    while out.shape[1]-prefix.shape[1]<count:
        h,al=student.encode(out[:,-64:])
        anchor=torch.multinomial(teacher_probs(al),1,generator=g).squeeze(-1)
        remaining=count-(out.shape[1]-prefix.shape[1])
        if kind=='ar': commit=False
        else:
            w,l=student.tail(h,anchor)
            features=student.features(h,anchor,w,l)
            if kind=='rl': commit=bool((gate(features)>threshold).item())
            elif kind=='block': commit=True
            elif kind=='random': commit=bool(torch.rand((),generator=g)<COVERAGE)
            else: raise ValueError(kind)
        size=emitted_length(commit,remaining)
        # Gate decision occurs BEFORE any candidate continuation is sampled.
        emitted=anchor[:,None]
        if size==BLOCK_SIZE: emitted=torch.cat([emitted,joint_sample(w,l,g)],1)
        out=torch.cat([out,emitted],1);calls+=1;lengths.append(size)
    return out[:,-count:],calls,lengths


def online_diagnostic(student,gate,prefixes,threshold,seed,count=48):
    rows=[]
    for kind in ('ar','block','random','rl'):
        runs=[];calls=[];lens=[];outputs=[]
        for repeat in range(3):
            elapsed=0.;calls=[];lens=[];outputs=[]
            for i in range(len(prefixes)):
                t=time.perf_counter()
                tokens,n,ls=generate(student,gate,prefixes[i:i+1],count,kind,threshold,seed+100*i)
                elapsed+=time.perf_counter()-t
                calls.append(n);lens+=ls;outputs.append(tokens[0])
            runs.append(elapsed)
        assert set(lens)<= {1,BLOCK_SIZE}
        rows.append({'policy':kind,'tokens_per_call':len(prefixes)*count/sum(calls),
                     'calls':sum(calls),'ar_calls':lens.count(1),'full_block_calls':lens.count(BLOCK_SIZE),
                     'generation_seconds_median':float(np.median(runs)),
                     'generation_seconds_range':[min(runs),max(runs)],
                     'lengths_seen':sorted(set(lens)),
                     'output_sha256':hashlib.sha256(torch.stack(outputs).numpy().tobytes()).hexdigest()})
    ar_time=rows[0]['generation_seconds_median']
    for row in rows: row['latency_ratio_vs_ar']=ar_time/row['generation_seconds_median']
    return rows


def run(args):
    seed_all(args.seed);args.out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    data,meta=smoke_data() if args.smoke else load_data(args.data)
    save(args.out/'dataset_metadata.json',meta)
    save(args.out/'protocol.json',{'version':PROTOCOL_VERSION,'block_size_including_anchor':BLOCK_SIZE,
         'coverage_primary':COVERAGE,'sampling':'full teacher categorical, temperature=1, top_p=1',
         'no_selection_of_content_targets':True,'gate_training':'REINFORCE contextual bandit; not long-horizon RL',
         'gate_observation':'prefix hidden, realized anchor, distribution features; not sampled candidate',
         'success':'upper nested-bootstrap CI < 0 for learned-minus-uniform conditional KL at matched blocks',
         'test_protocol':'fresh corrected-corpus disjoint context windows, no test tuning',
         'scope':'approximate joint distribution; no proof of exact teacher preservation',
         'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}})
    teacher=train_teacher(data,meta['actual_vocab_size'],args.teacher_steps,args.seed,args.out)
    seed_all(args.seed+10);student=FixedKStudent(teacher)
    train_content(teacher,student,data,args,args.out)
    frozen_before=state_hash(student)
    gate_x,_=contexts(data['train'],args.train_contexts,46001,64)
    gate_bank=risk_bank(teacher,student,gate_x,8,args.seed+201)
    gate,penalty=train_gate(gate_bank,args.gate_steps,args.seed+300,args.out)
    frozen_after=state_hash(student)
    assert frozen_before==frozen_after
    dev_x,dev_ix=contexts(data['dev'],64,46002,64,True)
    dev_bank=risk_bank(teacher,student,dev_x,16,args.seed+202)
    with torch.no_grad(): threshold=float(torch.quantile(gate(dev_bank['features']),1-COVERAGE))
    # Only now touch test contexts and outcomes.
    test_x,test_ix=contexts(data['test'],64,46003,64,True)
    test_bank=risk_bank(teacher,student,test_x,64,args.seed+203)
    with torch.no_grad(): test_scores=gate(test_bank['features'])
    matched=matched_diagnostic(test_bank,test_scores,args.seed+400)
    online=online_diagnostic(student,gate,test_x[:8],threshold,args.seed+500)
    test_distill=distill_bank(teacher,student,test_x,16,args.seed+204)
    test_fk=forward_kl(student,test_distill)
    cpu='unknown'
    if Path('/proc/cpuinfo').exists():
        cpu=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'unknown')
    result={'protocol_version':PROTOCOL_VERSION,'seed':args.seed,'dataset':meta['dataset'],
            'conditions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                          'device':'cpu','cpu':cpu,'threads':1,'dtype':'float32','kv_cache':False,
                          'clock':'not pinned','decode_batch':1,'train_batch_teacher':32,
                          'timing_prompts':8,'timing_tokens_per_prompt':48,'timing_repeats':3,
                          'teacher_config':vars(teacher.config),'teacher_parameters':sum(p.numel() for p in teacher.parameters()),
                          'fresh_output_dir':str(args.out)},
            'invariants':{'K':BLOCK_SIZE,'no_eob':True,'student_frozen_during_rl':frozen_before==frozen_after,
                          'student_hash_before_rl':frozen_before,'student_hash_after_rl':frozen_after,
                          'base_matches_teacher':state_hash(student.backbone)==state_hash(teacher),
                          'trainable_gate_parameter_names':[n for n,p in gate.named_parameters() if n.endswith(('.A','.B'))]},
            'train_selected_reward_penalty':penalty,'dev_selected_gate_logit_threshold':threshold,
            'test_forward_kl':test_fk,'matched_budget':matched,'online_unmatched':online,
            'duration_seconds':time.perf_counter()-start,
            'limitations':['3-token continuation joint is finite-mixture approximation, not exact teacher',
                          'equal-budget bank diagnostic is offline rank allocation, not online quota enforcement',
                          'confidence intervals do not include independent training variation',
                          'natural-language test, when used, is one play not independent corpus validation',
                          'online sampling distribution not certified; no wall-clock extrapolation']}
    save(args.out/'result.json',result)
    np.savez_compressed(args.out/'test_counterfactuals.npz',risk_samples=test_bank['risk_samples'].numpy(),
                        gate_scores=test_scores.numpy(),confidence_scores=test_bank['confidence'].numpy(),
                        context_starts=test_ix.numpy(),dev_context_starts=dev_ix.numpy())
    save(args.out/'file_hashes.json',{p.name:digest(p) for p in args.out.iterdir() if p.is_file()})
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=Path(__file__).parent/'data')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=47017)
    ap.add_argument('--teacher-steps',type=int,default=600)
    ap.add_argument('--content-steps',type=int,default=1200)
    ap.add_argument('--gate-steps',type=int,default=600)
    ap.add_argument('--train-contexts',type=int,default=1024)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if min(args.teacher_steps,args.content_steps,args.gate_steps,args.train_contexts)<1: ap.error('All sizes must be positive')
    run(args)

if __name__=='__main__': main()
