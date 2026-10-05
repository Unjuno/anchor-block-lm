from __future__ import annotations
import argparse, json, math, time
import numpy as np
import torch
from torch.nn import functional as F
from poc import ROOT, BlockStudent, load_teacher, sample_contexts, seed_all, dump_json, merge_lora_

OUT = ROOT / 'results_anchor'
OUT.mkdir(exist_ok=True)
TOP_P = 0.95
TEMP = 1.0
H = 5
RANK = 8


def top_p_sample(logits: torch.Tensor, top_p: float=TOP_P, temperature: float=TEMP, gen=None):
    logits = logits / temperature
    probs = logits.softmax(-1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cum = sorted_probs.cumsum(-1)
    remove = (cum - sorted_probs) >= top_p
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(-1, keepdim=True)
    pick = torch.multinomial(sorted_probs, 1, generator=gen)
    return sorted_idx.gather(-1, pick).squeeze(-1)


@torch.no_grad()
def sample_teacher_trajectory(teacher, prefix, steps=H, gen=None):
    ys=[]; ctx=prefix.clone(); probs=[]
    for _ in range(steps):
        logits=teacher(ctx[:, -teacher.config.block_size:])[0][:,-1]
        probs.append(logits.softmax(-1))
        tok=top_p_sample(logits, gen=gen)
        ys.append(tok)
        ctx=torch.cat([ctx,tok[:,None]],1)
    return torch.stack(ys,1), torch.stack(probs,1)


def make_bank(split: str, n: int, seed: int):
    seed_all(seed,1)
    teacher=load_teacher(ROOT/'results'/'teacher.pt')
    data=torch.tensor(np.load(ROOT/'data'/f'{split}.npy').astype(np.int64))
    gctx=torch.Generator().manual_seed(seed+1000)
    gsamp=torch.Generator().manual_seed(seed+2000)
    x=sample_contexts(data,n,64,gctx)
    trajectories=[]; tprobs=[]
    for j in range(0,n,128):
        yy,pp=sample_teacher_trajectory(teacher,x[j:j+128],H,gsamp)
        trajectories.append(yy);tprobs.append(pp)
    y=torch.cat(trajectories); p=torch.cat(tprobs)
    xa=torch.cat([x[:,1:],y[:,:1]],1)
    direct={'x':x,'y':y[:,:4],'first_probs':p[:,0]}
    anchor={'x':xa,'y':y[:,1:5],'first_probs':p[:,1], 'anchor':y[:,0], 'original_x':x}
    torch.save(direct,OUT/f'direct_{split}.pt')
    torch.save(anchor,OUT/f'anchor_{split}.pt')
    return {'split':split,'n':n}


def prepare():
    started=time.perf_counter(); rows=[]
    for split,n,seed in [('train',12288,701),('dev',1024,702),('test',2048,703)]:
        rows.append(make_bank(split,n,seed))
        print(json.dumps({'stage':'prepared',**rows[-1],'elapsed_s':time.perf_counter()-started}),flush=True)
    dump_json(OUT/'data_config.json',{'top_p':TOP_P,'temperature':TEMP,'trajectory_horizon':H,'banks':rows})


@torch.no_grad()
def evaluate(model,bank,batch=128):
    model.eval(); ce=[]; hits=[]; first_kl=[]
    for st in range(0,len(bank['x']),batch):
        x=bank['x'][st:st+batch]; y=bank['y'][st:st+batch]
        logits,_=model(x)
        ce.append(F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').reshape(len(x),4))
        hits.append(logits.argmax(-1).eq(y))
        first_kl.append(F.kl_div(logits[:,0].log_softmax(-1),bank['first_probs'][st:st+batch],reduction='none').sum(-1))
    ce=torch.cat(ce); hits=torch.cat(hits); first_kl=torch.cat(first_kl)
    return {'sample_ce_by_slot':ce.mean(0).tolist(),'sample_accuracy_by_slot':hits.float().mean(0).tolist(),
            'sample_block4_exact':hits.all(1).float().mean().item(),'first_slot_teacher_kl':first_kl.mean().item()}


def train(kind: str,total=1800):
    assert kind in ('direct','anchor')
    seed_all(710,1)
    teacher=load_teacher(ROOT/'results'/'teacher.pt')
    student=BlockStudent(teacher,rank=RANK)
    params=[p for p in student.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=.003,weight_decay=.001)
    bank=torch.load(OUT/f'{kind}_train.pt',weights_only=True)
    dev=torch.load(OUT/f'{kind}_dev.pt',weights_only=True)
    rng=torch.Generator().manual_seed(720)
    best=float('inf'); started=time.perf_counter(); hist=[]
    for step in range(1,total+1):
        student.train();ix=torch.randint(len(bank['x']),(64,),generator=rng)
        logits,_=student(bank['x'][ix]);y=bank['y'][ix];p=bank['first_probs'][ix]
        first_kl=F.kl_div(logits[:,0].log_softmax(-1),p,reduction='batchmean')
        future_ce=F.cross_entropy(logits[:,1:].reshape(-1,logits.shape[-1]),y[:,1:].reshape(-1))
        first_sample_ce=F.cross_entropy(logits[:,0],y[:,0])
        loss=future_ce+5.0*first_kl+0.1*first_sample_ce
        lr=.003*(.15+.85*.5*(1+math.cos(math.pi*step/total)))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()
        if step%200==0 or step==total:
            m=evaluate(student,dev);row={'kind':kind,'step':step,'loss':loss.item(),'elapsed_s':time.perf_counter()-started,**m}
            hist.append(row);print(json.dumps(row),flush=True)
            score=sum(m['sample_ce_by_slot'][1:])
            if score<best:
                best=score;torch.save({'model':student.state_dict(),'rank':RANK,'step':step,'metrics':m},OUT/f'{kind}_best.pt')
    dump_json(OUT/f'{kind}_train_history.json',hist)


def load_student(kind: str, merged=False):
    teacher=load_teacher(ROOT/'results'/'teacher.pt')
    s=BlockStudent(teacher,rank=RANK)
    ck=torch.load(OUT/f'{kind}_best.pt',map_location='cpu',weights_only=True);s.load_state_dict(ck['model']);s.eval()
    if merged: merge_lora_(s)
    return s


@torch.no_grad()
def teacher_score_block(teacher,prefix,tokens):
    gaps=[];agree=[];nll=[];ctx=prefix.clone()
    for j in range(tokens.shape[1]):
        logits=teacher(ctx[:,-teacher.config.block_size:])[0][:,-1]
        tok=tokens[:,j]
        chosen=logits.gather(-1,tok[:,None]).squeeze(-1)
        gaps.append(logits.max(-1).values-chosen)
        agree.append(logits.argmax(-1).eq(tok))
        nll.append(-logits.log_softmax(-1).gather(-1,tok[:,None]).squeeze(-1))
        ctx=torch.cat([ctx,tok[:,None]],1)
    return torch.stack(gaps,1),torch.stack(agree,1),torch.stack(nll,1)


@torch.no_grad()
def gen_ar(teacher,prefix,count):
    out=prefix.clone();calls=0
    while out.shape[1]-prefix.shape[1]<count:
        logits=teacher(out[:,-64:])[0][:,-1];tok=logits.argmax(-1);out=torch.cat([out,tok[:,None]],1);calls+=1
    return out,calls


@torch.no_grad()
def gen_anchor(teacher,model,prefix,count,k):
    out=prefix.clone();calls=0
    while out.shape[1]-prefix.shape[1]<count:
        logits=teacher(out[:,-64:])[0][:,-1];tok=logits.argmax(-1);out=torch.cat([out,tok[:,None]],1);calls+=1
        rem=count-(out.shape[1]-prefix.shape[1])
        if rem<=0: break
        logits,_=model(out[:,-64:]);take=min(k,rem);out=torch.cat([out,logits[:,:take].argmax(-1)],1);calls+=1
    return out,calls


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','train-direct','train-anchor'])
    ap.add_argument('--steps',type=int,default=1800);args=ap.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='train-direct':train('direct',args.steps)
    elif args.stage=='train-anchor':train('anchor',args.steps)

if __name__=='__main__':main()
