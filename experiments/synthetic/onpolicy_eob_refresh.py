from __future__ import annotations
import json, math, time
import torch
from torch.nn import functional as F
from poc import ROOT, load_teacher, seed_all, dump_json, merge_lora_
from eob_experiment import (OUT, H, ROLLOUTS, THRESHOLD, EOBBlockStudent,
    load_student, decode_eob_logits, sample_rollouts, modal_prefix_path,
    make_eob_targets, evaluate)
    
@torch.no_grad()
def collect_anchor_contexts(split: str, starts: int, cycles: int, bias: float, seed: int):
    seed_all(seed,1)
    teacher=load_teacher(ROOT/'results'/'teacher.pt')
    model=load_student(merged=True)
    base=torch.load(ROOT/'results_anchor'/f'direct_{split}.pt',weights_only=True)['x'][:starts]
    contexts=[]
    for i in range(starts):
        out=base[i:i+1].clone()
        for _ in range(cycles):
            logits=teacher(out[:,-teacher.config.block_size:])[0][:,-1]
            anchor=logits.argmax(-1)
            out=torch.cat([out,anchor[:,None]],1)
            contexts.append(out[:,-teacher.config.block_size:].clone()[0])
            blogits,_=model(out[:,-teacher.config.block_size:])
            toks,k=decode_eob_logits(blogits,model.eob_id,H,eob_bias=bias)
            take=int(k[0])
            if take:
                out=torch.cat([out,toks[:,:take]],1)
    return torch.stack(contexts)

@torch.no_grad()
def label_contexts(x: torch.Tensor, seed: int):
    teacher=load_teacher(ROOT/'results'/'teacher.pt')
    g=torch.Generator().manual_seed(seed)
    rolls=[]
    for st in range(0,len(x),32):
        rolls.append(sample_rollouts(teacher,x[st:st+32],ROLLOUTS,H,g))
    roll=torch.cat(rolls)
    modal,mass=modal_prefix_path(roll)
    target,lengths=make_eob_targets(modal,mass,THRESHOLD,teacher.config.vocab_size)
    return {'x':x,'modal':modal,'mass':mass,'target':target,'lengths':lengths}

def prepare():
    configs=[('train',128,4,1.0,951),('dev',32,4,1.0,952)]
    rows=[]
    for split,starts,cycles,bias,seed in configs:
        t=time.perf_counter();x=collect_anchor_contexts(split,starts,cycles,bias,seed)
        bank=label_contexts(x,seed+1000);torch.save(bank,OUT/f'onpolicy_{split}.pt')
        hist=torch.bincount(bank['lengths'],minlength=H+1).tolist()
        row={'split':split,'contexts':len(x),'hist':hist,'mean_length':bank['lengths'].float().mean().item(),'elapsed_s':time.perf_counter()-t}
        rows.append(row);print(json.dumps(row),flush=True)
    dump_json(OUT/'onpolicy_data_config.json',rows)

def finetune(steps=300):
    seed_all(960,1)
    model=load_student(merged=False)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=.001,weight_decay=.001)
    orig=torch.load(OUT/'train.pt',weights_only=True);op=torch.load(OUT/'onpolicy_train.pt',weights_only=True)
    dev=torch.load(OUT/'dev.pt',weights_only=True);opdev=torch.load(OUT/'onpolicy_dev.pt',weights_only=True)
    rng=torch.Generator().manual_seed(961);hist=[];best=float('inf')
    for step in range(1,steps+1):
        model.train();n=32
        ix=torch.randint(len(orig['x']),(n,),generator=rng);jx=torch.randint(len(op['x']),(n,),generator=rng)
        x=torch.cat([orig['x'][ix],op['x'][jx]]); y=torch.cat([orig['target'][ix],op['target'][jx]])
        logits,_=model(x);loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100)
        lr=.001*(.2+.8*.5*(1+math.cos(math.pi*step/steps)))
        for g in opt.param_groups:g['lr']=lr
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()
        if step%50==0 or step==steps:
            a=evaluate(model,dev);b=evaluate(model,opdev);score=a['ce']+b['ce']
            row={'step':step,'loss':loss.item(),'original_dev':a,'onpolicy_dev':b};hist.append(row);print(json.dumps(row),flush=True)
            if score<best:
                best=score;torch.save({'model':model.state_dict(),'rank':model.rank,'horizon':model.horizon,'metrics':row},OUT/'best_onpolicy.pt')
    dump_json(OUT/'onpolicy_train_history.json',hist)

def load_onpolicy(merged=True):
    teacher=load_teacher(ROOT/'results'/'teacher.pt');m=EOBBlockStudent(teacher,rank=8,horizon=H)
    ck=torch.load(OUT/'best_onpolicy.pt',map_location='cpu',weights_only=True);m.load_state_dict(ck['model']);m.eval()
    if merged:merge_lora_(m)
    return m

if __name__=='__main__':
    import sys
    stage=sys.argv[1]
    if stage=='prepare':prepare()
    elif stage=='train':finetune(int(sys.argv[2]) if len(sys.argv)>2 else 300)
