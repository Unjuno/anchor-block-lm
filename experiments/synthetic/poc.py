"""Small, reproducible LoRA multi-token experiment using the uploaded nanoGPT.

Inference has no teacher verification. Deterministic greedy imitation only.
The optional reward-based length learner is a contextual bandit, not full RL.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'third_party' / 'nanogpt'))
from model import GPT, GPTConfig

LENGTHS = (1, 2, 4)


def seed_all(seed: int, threads: int = 1) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


class LoRALinear(nn.Module):
    """Train only A/B; a merged copy removes adapter matmuls at inference."""
    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None):
        super().__init__()
        if rank < 1: raise ValueError('rank must be positive')
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * (self.alpha / self.rank)

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        result = copy.deepcopy(self.base)
        result.weight.add_((self.B @ self.A) * (self.alpha / self.rank))
        result.requires_grad_(False)
        return result


def inject_lora(gpt: GPT, rank: int) -> None:
    for block in gpt.transformer.h:
        for parent, name in [(block.attn,'c_attn'),(block.attn,'c_proj'),
                             (block.mlp,'c_fc'),(block.mlp,'c_proj')]:
            setattr(parent, name, LoRALinear(getattr(parent,name), rank, alpha=rank))


def merge_lora_(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear): setattr(module,name,child.merged())
        else: merge_lora_(child)


def hidden_states(gpt: GPT, idx: torch.Tensor) -> torch.Tensor:
    """Exact uploaded nanoGPT backbone, exposing the final hidden sequence."""
    if idx.ndim != 2 or idx.shape[1] > gpt.config.block_size:
        raise ValueError('idx must be B,T and fit the context window')
    pos=torch.arange(idx.shape[1],device=idx.device)
    h=gpt.transformer.drop(gpt.transformer.wte(idx)+gpt.transformer.wpe(pos))
    for block in gpt.transformer.h: h=block(h)
    return gpt.transformer.ln_f(h)


class BlockStudent(nn.Module):
    """One backbone call; slot one is the LM head; later slots use small MLPs."""
    def __init__(self, teacher: GPT, rank: int = 8):
        super().__init__()
        self.backbone=copy.deepcopy(teacher)
        self.backbone.requires_grad_(False)
        inject_lora(self.backbone,rank)
        self.rank=rank
        width=teacher.config.n_embd
        bottleneck=max(8,width//2)
        self.future_heads=nn.ModuleList([
            nn.Sequential(nn.Linear(width,bottleneck),nn.GELU(),nn.Linear(bottleneck,width))
            for _ in range(3)
        ])
        for head in self.future_heads:
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        h=hidden_states(self.backbone,idx)[:,-1]
        states=torch.stack([h]+[h+head(h) for head in self.future_heads],dim=1)
        logits=self.backbone.lm_head(states)
        return logits,h


class LengthController(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(feature_dim,48),nn.Tanh(),nn.Linear(48,3))

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.net(x)

    def enable_lora(self,rank:int=4)->None:
        self.requires_grad_(False)
        for i in (0,2): self.net[i]=LoRALinear(self.net[i],rank,alpha=rank)


def controller_features(logits:torch.Tensor, hidden:torch.Tensor)->torch.Tensor:
    probs=logits.softmax(-1)
    top=probs.topk(2,dim=-1).values
    entropy=-(probs*probs.clamp_min(1e-9).log()).sum(-1)/math.log(logits.shape[-1])
    return torch.cat([hidden,top[:,:,0],top[:,:,0]-top[:,:,1],entropy],dim=-1)


def action_rewards(regrets:torch.Tensor,call_cost:float)->torch.Tensor:
    return torch.stack([-regrets[:,:k].mean(-1)-call_cost/k for k in LENGTHS],dim=-1)


@torch.no_grad()
def teacher_targets(teacher:GPT,prefix:torch.Tensor,count:int=4)->torch.Tensor:
    ys=[]; context=prefix.clone()
    for _ in range(count):
        pred=teacher(context[:,-teacher.config.block_size:])[0][:,-1].argmax(-1)
        ys.append(pred);context=torch.cat([context,pred[:,None]],dim=1)
    return torch.stack(ys,dim=1)


@torch.no_grad()
def teacher_regrets(teacher:GPT,prefix:torch.Tensor,tokens:torch.Tensor):
    gaps=[];matches=[];nll=[]
    for j in range(tokens.shape[1]):
        ctx=torch.cat([prefix,tokens[:,:j]],dim=1)[:,-teacher.config.block_size:]
        logits=teacher(ctx)[0][:,-1]
        chosen=logits.gather(-1,tokens[:,j,None]).squeeze(-1)
        gaps.append(logits.max(-1).values-chosen)
        matches.append(logits.argmax(-1).eq(tokens[:,j]))
        nll.append(-logits.log_softmax(-1).gather(-1,tokens[:,j,None]).squeeze(-1))
    return torch.stack(gaps,1),torch.stack(matches,1),torch.stack(nll,1)


@torch.no_grad()
def generate_student(model:BlockStudent,prefix:torch.Tensor,count:int,
                     fixed_k:int|None=None,controller:LengthController|None=None,
                     random_probs:torch.Tensor|None=None,confidence_threshold:float|None=None):
    if prefix.shape[0]!=1: raise ValueError('Generation benchmark intentionally uses batch=1')
    if count<0: raise ValueError('count must be non-negative')
    if fixed_k is not None and fixed_k not in LENGTHS: raise ValueError('fixed_k must be 1,2,4')
    out=prefix.clone();lengths=[];calls=0
    while out.shape[1]-prefix.shape[1]<count:
        logits,h=model(out[:,-model.backbone.config.block_size:])
        if fixed_k is not None: k=fixed_k
        elif controller is not None:
            k=LENGTHS[controller(controller_features(logits,h)).argmax(-1).item()]
        elif random_probs is not None:
            k=LENGTHS[torch.multinomial(random_probs,1).item()]
        elif confidence_threshold is not None:
            survival=logits.softmax(-1).max(-1).values.cumprod(-1)[0]
            k=1
            for proposal in (2,4):
                if survival[proposal-1].item()>=confidence_threshold:k=proposal
        else: raise ValueError('A generation policy must be supplied')
        k=min(k,count-(out.shape[1]-prefix.shape[1]))
        out=torch.cat([out,logits[:,:k].argmax(-1)],dim=1)
        calls+=1;lengths.append(k)
    return out,dict(calls=calls,committed=count,lengths=lengths)


@torch.no_grad()
def generate_teacher(teacher:GPT,prefix:torch.Tensor,count:int):
    out=prefix.clone()
    for _ in range(count):
        pred=teacher(out[:,-teacher.config.block_size:])[0][:,-1].argmax(-1)
        out=torch.cat([out,pred[:,None]],dim=1)
    return out,dict(calls=count,committed=count,lengths=[1]*count)


def frozen_backbone_equal(teacher:GPT,student:BlockStudent)->bool:
    sd=student.backbone.state_dict()
    normalized={k.replace('.base.','.'):v for k,v in sd.items() if not k.endswith(('.A','.B'))}
    original=teacher.state_dict()
    return normalized.keys()==original.keys() and all(torch.equal(v,normalized[k]) for k,v in original.items())


def load_teacher(path:Path)->GPT:
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=GPT(GPTConfig(**saved['config']))
    model.load_state_dict(saved['model']);model.eval();model.requires_grad_(False)
    return model


def sample_contexts(data:torch.Tensor,n:int,context:int,rng:torch.Generator)->torch.Tensor:
    starts=torch.randint(0,len(data)-context,(n,),generator=rng)
    return data[starts[:,None]+torch.arange(context)[None,:]].long()


def make_synthetic_data(directory:Path)->dict:
    """Whole-record, ID-disjoint splits; deliberately controlled, not natural text."""
    directory.mkdir(parents=True,exist_ok=True)
    words=dict(names=['alice','bruno','carol','david','elena','felix','grace','henry'],
               colors=['red','blue','green','yellow','white','black'],
               objects=['book','cup','key','box','lamp','coin'],
               places=['garden','kitchen','library','station','office','market'])
    chars=None;manifest={}
    for split,n,offset,seed in [('train',6000,0,100),('dev',600,10000,200),('test',600,20000,300)]:
        rng=random.Random(seed);records=[]
        for i in range(n):
            name=rng.choice(words['names']);color=rng.choice(words['colors'])
            obj=rng.choice(words['objects']);place=rng.choice(words['places']);quantity=rng.randrange(1,10)
            template=rng.randrange(4)
            if template==0: content=f'{name} found a {color} {obj} in the {place}.'
            elif template==1: content=f'in the {place}, {name} counted {quantity} {color} {obj}s.'
            elif template==2: content=f'the {color} {obj} belongs to {name}. it is in the {place}.'
            else: content=f'{name} went to the {place}. the {obj} was {color}.'
            records.append(f'record {offset+i:05d}: {content}\n')
        text=''.join(records);(directory/f'{split}.txt').write_text(text)
        manifest[split]={'records':n,'id_first':offset,'id_last':offset+n-1,'chars':len(text),
                         'sha256':hashlib.sha256(text.encode()).hexdigest(),'generator_seed':seed}
    chars=sorted(set((directory/'train.txt').read_text()))
    for split in ('train','dev','test'):
        text=(directory/f'{split}.txt').read_text()
        if set(text)-set(chars):raise ValueError('Unseen validation/test characters')
        ids=np.array([chars.index(ch) for ch in text],dtype=np.uint16)
        np.save(directory/f'{split}.npy',ids)
    metadata={'dataset':'synthetic_record_grammar_v1','NOT_TINY_SHAKESPEARE':True,
              'tokens':'characters','chars':chars,'vocab_size':len(chars),'splits':manifest}
    dump_json(directory/'metadata.json',metadata)
    return metadata
