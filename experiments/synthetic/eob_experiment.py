from __future__ import annotations

import argparse
import copy
import json
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from poc import ROOT, inject_lora, hidden_states, load_teacher, merge_lora_, seed_all, dump_json
from anchor_block_experiment import top_p_sample, teacher_score_block

OUT = ROOT / 'results_eob'
OUT.mkdir(exist_ok=True)
TOP_P = 0.95
TEMP = 1.0
RANK = 8
H = 6
ROLLOUTS = 16
THRESHOLD = 0.80


class EOBBlockStudent(nn.Module):
    def __init__(self, teacher, rank: int = RANK, horizon: int = H):
        super().__init__()
        self.backbone = copy.deepcopy(teacher)
        self.backbone.requires_grad_(False)
        inject_lora(self.backbone, rank)
        self.rank = rank
        self.horizon = horizon
        width = teacher.config.n_embd
        bottleneck = max(8, width // 2)
        self.slot_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(width, bottleneck), nn.GELU(), nn.Linear(bottleneck, width))
            for _ in range(horizon)
        ])
        for head in self.slot_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.eob_head = nn.Linear(width, 1)
        nn.init.zeros_(self.eob_head.weight)
        nn.init.constant_(self.eob_head.bias, -2.0)

    @property
    def eob_id(self) -> int:
        return self.backbone.config.vocab_size

    def forward(self, idx: torch.Tensor):
        h = hidden_states(self.backbone, idx)[:, -1]
        states = torch.stack([h] + [h + head(h) for head in self.slot_heads], dim=1)
        token_logits = self.backbone.lm_head(states)
        eob_logits = self.eob_head(states)
        return torch.cat([token_logits, eob_logits], dim=-1), h


def decode_eob_logits(logits: torch.Tensor, eob_id: int, max_tokens: int, eob_bias: float = 0.0):
    work = logits.clone()
    work[..., eob_id] += float(eob_bias)
    pred = work.argmax(-1)
    bsz = pred.shape[0]
    lengths = torch.full((bsz,), max_tokens, dtype=torch.long, device=pred.device)
    for i in range(bsz):
        hit = (pred[i, :max_tokens + 1] == eob_id).nonzero(as_tuple=False)
        if len(hit):
            lengths[i] = min(int(hit[0, 0]), max_tokens)
    return pred[:, :max_tokens], lengths


def modal_prefix_path(rollouts: torch.Tensor):
    if rollouts.ndim != 3:
        raise ValueError('rollouts must be [B,N,H]')
    B, N, Hh = rollouts.shape
    modal = torch.empty((B, Hh), dtype=rollouts.dtype, device=rollouts.device)
    mass = torch.zeros((B, Hh), dtype=torch.float32, device=rollouts.device)
    for b in range(B):
        mask = torch.ones(N, dtype=torch.bool, device=rollouts.device)
        for h in range(Hh):
            vals = rollouts[b, mask, h]
            if vals.numel() == 0:
                modal[b, h:] = 0
                break
            uniq, counts = torch.unique(vals, sorted=True, return_counts=True)
            tok = uniq[counts.argmax()]
            modal[b, h] = tok
            mask = mask & (rollouts[b, :, h] == tok)
            mass[b, h] = mask.float().mean()
    return modal, mass


def make_eob_targets(modal: torch.Tensor, mass: torch.Tensor, threshold: float, eob_id: int):
    if modal.shape != mass.shape:
        raise ValueError('modal/mass shapes must match')
    B, Hh = modal.shape
    target = torch.full((B, Hh + 1), -100, dtype=torch.long, device=modal.device)
    lengths = torch.zeros(B, dtype=torch.long, device=modal.device)
    for b in range(B):
        k = 0
        for h in range(Hh):
            if mass[b, h].item() >= threshold:
                k = h + 1
            else:
                break
        lengths[b] = k
        if k:
            target[b, :k] = modal[b, :k]
        target[b, k] = eob_id
    return target, lengths


@torch.no_grad()
def sample_rollouts(teacher, x: torch.Tensor, n_rollouts: int = ROLLOUTS, horizon: int = H,
                    generator: torch.Generator | None = None):
    B = x.shape[0]
    ctx = x[:, None, :].expand(B, n_rollouts, x.shape[1]).reshape(B * n_rollouts, x.shape[1]).clone()
    ys = []
    for _ in range(horizon):
        logits = teacher(ctx[:, -teacher.config.block_size:])[0][:, -1]
        tok = top_p_sample(logits, top_p=TOP_P, temperature=TEMP, gen=generator)
        ys.append(tok.reshape(B, n_rollouts))
        ctx = torch.cat([ctx, tok[:, None]], dim=1)
    return torch.stack(ys, dim=-1)


def prepare(n_train=4096, n_dev=512, n_test=1024):
    teacher = load_teacher(ROOT / 'results' / 'teacher.pt')
    source = ROOT / 'results_anchor'
    settings = [('train', n_train, 901), ('dev', n_dev, 902), ('test', n_test, 903)]
    stats = []
    for split, n, seed in settings:
        bank = torch.load(source / f'anchor_{split}.pt', weights_only=True)
        x = bank['x'][:n].clone()
        g = torch.Generator().manual_seed(seed)
        chunks = []
        started = time.perf_counter()
        for st in range(0, n, 32):
            chunks.append(sample_rollouts(teacher, x[st:st+32], ROLLOUTS, H, g))
        roll = torch.cat(chunks, dim=0)
        modal, mass = modal_prefix_path(roll)
        target, lengths = make_eob_targets(modal, mass, THRESHOLD, teacher.config.vocab_size)
        saved = {'x': x, 'modal': modal, 'mass': mass, 'target': target, 'lengths': lengths}
        torch.save(saved, OUT / f'{split}.pt')
        hist = torch.bincount(lengths, minlength=H + 1).tolist()
        row = {'split': split, 'n': n, 'length_histogram_0_to_H': hist,
               'mean_target_length': lengths.float().mean().item(),
               'elapsed_s': time.perf_counter() - started}
        stats.append(row)
        print(json.dumps(row), flush=True)
    dump_json(OUT / 'data_config.json', {
        'top_p': TOP_P, 'temperature': TEMP, 'n_rollouts': ROLLOUTS,
        'horizon': H, 'prefix_mass_threshold': THRESHOLD, 'splits': stats,
        'label_definition': 'modal joint-prefix mass from top-p rollouts; EOB after longest contiguous mass >= threshold'
    })


@torch.no_grad()
def evaluate(model: EOBBlockStudent, bank: dict, batch=128):
    model.eval()
    losses=[]; pred_lengths=[]; true_lengths=[]; token_hits=[]; exact=[]
    for st in range(0, len(bank['x']), batch):
        x=bank['x'][st:st+batch]; target=bank['target'][st:st+batch]
        logits,_=model(x)
        loss=F.cross_entropy(logits.flatten(0,1),target.flatten(),ignore_index=-100,reduction='none').reshape(len(x),H+1)
        mask=target.ne(-100)
        losses.append((loss*mask).sum(1)/mask.sum(1))
        toks,k=decode_eob_logits(logits,model.eob_id,H)
        pred_lengths.append(k); true_lengths.append(bank['lengths'][st:st+batch])
        local=[]
        for i in range(len(x)):
            L=int(bank['lengths'][st+i])
            if L: local.append(toks[i,:L].eq(bank['modal'][st+i,:L]).float().mean())
            else: local.append(torch.tensor(1.0))
            seq_ok=(int(k[i])==L) and (L==0 or bool(toks[i,:L].eq(bank['modal'][st+i,:L]).all()))
            exact.append(float(seq_ok))
        token_hits.extend([float(z) for z in local])
    pl=torch.cat(pred_lengths); tl=torch.cat(true_lengths)
    return {
        'ce': torch.cat(losses).mean().item(),
        'mean_pred_length': pl.float().mean().item(),
        'mean_target_length': tl.float().mean().item(),
        'length_mae': (pl-tl).abs().float().mean().item(),
        'length_exact': pl.eq(tl).float().mean().item(),
        'safe_prefix_token_accuracy': float(np.mean(token_hits)),
        'block_plus_eob_exact': float(np.mean(exact)),
        'pred_length_histogram_0_to_H': torch.bincount(pl,minlength=H+1).tolist(),
    }


def train(total=1200):
    seed_all(910,1)
    teacher=load_teacher(ROOT/'results'/'teacher.pt')
    model=EOBBlockStudent(teacher,rank=RANK,horizon=H)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=.003,weight_decay=.001)
    bank=torch.load(OUT/'train.pt',weights_only=True); dev=torch.load(OUT/'dev.pt',weights_only=True)
    rng=torch.Generator().manual_seed(911)
    best=float('inf'); hist=[]; started=time.perf_counter()
    for step in range(1,total+1):
        model.train(); ix=torch.randint(len(bank['x']),(64,),generator=rng)
        logits,_=model(bank['x'][ix]); target=bank['target'][ix]
        loss=F.cross_entropy(logits.flatten(0,1),target.flatten(),ignore_index=-100)
        lr=.003*(.15+.85*.5*(1+math.cos(math.pi*step/total)))
        for group in opt.param_groups: group['lr']=lr
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()
        if step%200==0 or step==total:
            m=evaluate(model,dev); row={'step':step,'train_loss':loss.item(),'elapsed_s':time.perf_counter()-started,**m}
            hist.append(row); print(json.dumps(row),flush=True)
            if m['ce']<best:
                best=m['ce'];torch.save({'model':model.state_dict(),'rank':RANK,'horizon':H,'metrics':m},OUT/'best.pt')
    dump_json(OUT/'train_history.json',hist)


def load_student(merged=False):
    teacher=load_teacher(ROOT/'results'/'teacher.pt')
    model=EOBBlockStudent(teacher,rank=RANK,horizon=H)
    ck=torch.load(OUT/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model']);model.eval()
    if merged: merge_lora_(model)
    return model


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','train']);ap.add_argument('--steps',type=int,default=1200);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    else:train(a.steps)

if __name__=='__main__':main()
