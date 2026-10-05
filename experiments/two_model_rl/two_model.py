"""Fixed evaluator + trainable continuation LoRA; no inference-time teacher.

Content learning combines on-policy reverse-KL REINFORCE and unfiltered
full-horizon teacher-sample NLL. The actor's exact AR anchor is never updated.
"""
from __future__ import annotations
import copy
import math
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fixed_k_gate'))
import fixed_k as fk
H = fk.BLOCK_SIZE

class LoRAProjection(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8):
        super().__init__()
        if rank < 1: raise ValueError('rank must be positive')
        self.base = copy.deepcopy(base).requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B)

def enable_content_lora(student, rank: int = 8):
    """Freeze old head matrices; update only existing/new A/B tensors."""
    student.requires_grad_(False)
    for parent, key in [(student.condition,'0'),(student.condition,'2'),
                        (student.decoder,'0'),(student.decoder,'2'),(student,'mix')]:
        old = getattr(parent,key)
        if not isinstance(old,nn.Linear): raise ValueError('already adapted or unexpected head')
        setattr(parent,key,LoRAProjection(old,rank))
    student.low_rank.requires_grad_(True)
    student.eval()
    return student

def reinforce_terms(logq, logp, baseline=None):
    if logq.shape != logp.shape: raise ValueError('log probabilities must match')
    delta = logq - logp
    if baseline is not None: delta = delta - baseline
    return logq * delta.detach()

def leave_one_out(values):
    if values.ndim != 2: raise ValueError('expected [contexts,samples]')
    if values.shape[1] < 2: return torch.zeros_like(values)
    return (values.sum(1,keepdim=True)-values)/(values.shape[1]-1)

def time_rewards(risks, normalized_costs, penalty):
    if risks.ndim != 2 or normalized_costs.shape != (risks.shape[1],):
        raise ValueError('cost vector must match the available lengths')
    if not torch.isfinite(normalized_costs).all() or bool((normalized_costs <= 0).any()):
        raise ValueError('costs must be positive finite')
    return torch.arange(1,risks.shape[1]+1,device=risks.device)-normalized_costs-penalty*risks

@torch.no_grad()
def teacher_tail(teacher, contexts, anchor, generator, tokens=None):
    """Score or sample ALL three continuation slots, with the fixed evaluator."""
    ctx = torch.cat([contexts,anchor[:,None]],1)
    ys=[];lp=torch.zeros(len(ctx),device=ctx.device)
    for j in range(H-1):
        logits=teacher(ctx[:,-teacher.config.block_size:])[0][:,-1]
        y=(torch.multinomial(logits.softmax(-1),1,generator=generator).squeeze(-1)
           if tokens is None else tokens[:,j])
        lp+=logits.log_softmax(-1).gather(-1,y[:,None]).squeeze(-1)
        ys.append(y);ctx=torch.cat([ctx,y[:,None]],1)
    return torch.stack(ys,1),lp

def content_step(teacher,student,optimizer,contexts,generator,samples=4,kd_weight=.5):
    if samples<1:raise ValueError('positive samples required')
    h,al=student.encode(contexts)
    anchor=torch.multinomial(al.softmax(-1),1,generator=generator).squeeze(-1)
    w,l=student.tail(h,anchor)
    ww=w.repeat_interleave(samples,0);ll=l.repeat_interleave(samples,0)
    generated=fk.joint_sample(ww.detach(),ll.detach(),generator)
    logq=fk.joint_log_prob(ww,ll,generated).reshape(len(h),samples)
    _,lp=teacher_tail(teacher,contexts.repeat_interleave(samples,0),anchor.repeat_interleave(samples),generator,generated)
    logp=lp.reshape_as(logq)
    delta=(logq-logp).detach()
    pg=reinforce_terms(logq,logp,leave_one_out(delta)).mean()
    target,_=teacher_tail(teacher,contexts,anchor,generator)
    kd=-fk.joint_log_prob(w,l,target).mean()
    loss=pg+kd_weight*kd
    optimizer.zero_grad(set_to_none=True);loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],1.)
    optimizer.step()
    return {'reverse_kl_sample_mean':float(delta.mean()),'teacher_nll':float(kd.detach()),'loss':float(loss.detach())}

@torch.no_grad()
def policy_batch(teacher,student,contexts,generator,samples=4):
    if samples<1: raise ValueError('positive samples required')
    h,al=student.encode(contexts)
    anchor=torch.multinomial(al.softmax(-1),1,generator=generator).squeeze(-1)
    w,l=student.tail(h,anchor); features=student.features(h,anchor,w,l)
    ww=w.repeat_interleave(samples,0);ll=l.repeat_interleave(samples,0)
    y=fk.joint_sample(ww,ll,generator)
    ctx=torch.cat([contexts,anchor[:,None]],1).repeat_interleave(samples,0)
    lp=torch.zeros(len(ctx));risks=[torch.zeros(len(ctx))]
    for j in range(1,H):
        logits=teacher(ctx[:,-teacher.config.block_size:])[0][:,-1]
        lp+=logits.log_softmax(-1).gather(-1,y[:,j-1,None]).squeeze(-1)
        lq=fk.joint_log_prob(ww,ll[:,:,:j,:],y[:,:j])
        risks.append(lq-lp);ctx=torch.cat([ctx,y[:,j-1,None]],1)
    r=torch.stack(risks,-1).reshape(len(h),samples,H)
    return {'features':features,'risk':r.mean(1),'risk_samples':r,'anchor':anchor}

@torch.no_grad()
def deployment_copy(model):
    model=copy.deepcopy(model).eval().requires_grad_(False)
    def merge(parent):
        for name,child in list(parent.named_children()):
            if isinstance(child,(LoRAProjection,fk.LoRALinear)):
                base=copy.deepcopy(child.base);base.weight.add_(child.B@child.A)
                setattr(parent,name,base.requires_grad_(False))
            elif isinstance(child,fk.LowRankResidual):
                base=nn.Linear(child.A.shape[1],child.B.shape[0],bias=False)
                base.weight.copy_(child.B@child.A)
                setattr(parent,name,base.requires_grad_(False))
            else:merge(child)
    merge(model)
    return model

@torch.no_grad()
def generate(student,gate,prefix,count,seed):
    if count<0 or prefix.ndim!=2 or len(prefix)!=1:raise ValueError('batch=1; nonnegative count')
    if count==0:return prefix[:,:0].clone(),[]
    g=torch.Generator().manual_seed(seed);out=prefix.clone();lengths=[]
    while out.shape[1]-prefix.shape[1]<count:
        h,al=student.encode(out[:,-student.backbone.config.block_size:])
        anchor=torch.multinomial(al.softmax(-1),1,generator=g).squeeze(-1)
        if gate is None:k=1
        else:
            w,l=student.tail(h,anchor)
            k=int(gate(student.features(h,anchor,w,l)).argmax(-1)[0])+1
        take=min(k,count-(out.shape[1]-prefix.shape[1]));tokens=anchor[:,None]
        if take>1:tokens=torch.cat([tokens,fk.joint_sample(w,l,g)[:,:take-1]],1)
        out=torch.cat([out,tokens],1);lengths.append(take)
    return out[:,-count:],lengths

@torch.no_grad()
def collect_states(student,gate,prefixes,cycles,seed):
    if cycles<1:raise ValueError('cycles must be positive')
    g=torch.Generator().manual_seed(seed);ctx=prefixes.clone();states=[];window=ctx.shape[1]
    for _ in range(cycles):
        states.append(ctx.clone());h,al=student.encode(ctx)
        a=torch.multinomial(al.softmax(-1),1,generator=g).squeeze(-1)
        if gate is None:ctx=torch.cat([ctx,a[:,None]],1)[:,-window:];continue
        w,l=student.tail(h,a);ks=gate(student.features(h,a,w,l)).argmax(-1)+1
        ys=fk.joint_sample(w,l,g)
        ctx=torch.stack([torch.cat([ctx[i],a[i:i+1],ys[i,:int(ks[i])-1]])[-window:]
                         for i in range(len(ctx))])
    return torch.cat(states)
