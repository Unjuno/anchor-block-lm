"""Consistent stochastic length policy, quality dual, and fresh state collection.

The transformer/AR path is frozen. Content uses full-horizon teacher KD; only
continuation and policy LoRA tensors change. A sampled trace score is NOT the
marginal token-sequence KL when lengths are random. See the experiment README.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for name in ('fixed_k_gate', 'adaptive_k_rl', 'two_model_rl'):
    sys.path.insert(0, str(ROOT/name))
import fixed_k as fk
from two_model import policy_batch, teacher_tail

H = fk.BLOCK_SIZE


def expected_risk(probs, risks):
    if probs.ndim != 2 or probs.shape != risks.shape:
        raise ValueError('expected matching [contexts,actions] tensors')
    if not torch.isfinite(probs).all() or not torch.isfinite(risks).all():
        raise ValueError('nonfinite probabilities or risks')
    if (probs < 0).any() or not torch.allclose(probs.sum(-1), torch.ones(len(probs), device=probs.device), atol=1e-5):
        raise ValueError('action probabilities must be normalized')
    return (probs * risks).sum(-1)


class Dual:
    def __init__(self, initial, lr, target):
        if not all(math.isfinite(float(v)) for v in (initial, lr, target)) or initial < 0 or lr <= 0 or target < 0:
            raise ValueError('invalid dual parameters')
        self.value, self.lr, self.target = float(initial), float(lr), float(target)

    def step(self, risk):
        if not risk.numel() or not torch.isfinite(risk).all():
            raise ValueError('risk must be finite and nonempty')
        self.value = max(0., self.value + self.lr * (float(risk.mean()) - self.target))
        return self.value


def policy_update(gate, optimizer, features, risks, costs, dual, generator):
    if costs.shape != (risks.shape[-1],) or not torch.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError('positive finite action costs required')
    logits = gate(features.detach())
    probs = logits.softmax(-1)
    action = torch.multinomial(probs.detach(), 1, generator=generator)
    risk = risks.detach()
    extra = torch.arange(risk.shape[-1], dtype=risk.dtype, device=risk.device)
    rewards = extra[None] - costs[None] - dual.value * risk
    chosen = rewards.gather(-1, action).squeeze(-1)
    control = (probs.detach() * rewards).sum(-1)
    log_probs = logits.log_softmax(-1)
    entropy = -(probs * log_probs).sum(-1)
    loss = -(log_probs.gather(-1, action).squeeze(-1) * (chosen - control)).mean() - .01 * entropy.mean()
    expected = expected_risk(probs.detach(), risk)
    row = {'expected_risk': float(expected.mean()),
           'sampled_risk': float(risk.gather(-1, action).mean()),
           'sampled_reward': float(chosen.mean()),
           'expected_reward': float(control.mean()),
           'lambda_before': dual.value,
           'sampled_k_histogram': torch.bincount(action.flatten(), minlength=H).tolist(),
           'loss': float(loss.detach())}
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in gate.parameters() if p.requires_grad], 1.)
    optimizer.step()
    # Both gradients use the SAME pre-update categorical policy, not argmax.
    dual.step(expected)
    row['lambda_after'] = dual.value
    return row


@torch.no_grad()
def collect(student, gate, prefixes, cycles, seed):
    if cycles < 1 or prefixes.ndim != 2 or not len(prefixes):
        raise ValueError('positive cycles and nonempty [B,T] prefixes required')
    rng = torch.Generator().manual_seed(seed)
    ctx = prefixes.clone(); window = ctx.shape[1]; states = []; lengths = []
    for _ in range(cycles):
        states.append(ctx.clone())
        h, al = student.encode(ctx)
        anchor = torch.multinomial(al.softmax(-1), 1, generator=rng).squeeze(-1)
        w, l = student.tail(h, anchor)
        probs = gate(student.features(h, anchor, w, l)).softmax(-1)
        ks = torch.multinomial(probs, 1, generator=rng).squeeze(-1) + 1
        tails = fk.joint_sample(w, l, rng)
        ctx = torch.stack([torch.cat([ctx[i], anchor[i:i+1], tails[i,:int(ks[i])-1]])[-window:] for i in range(len(ctx))])
        lengths.extend(ks.tolist())
    return {'states': torch.cat(states), 'lengths': lengths}


class Trainer:
    """Construct once per continuation; preserve Adam state across rounds."""
    def __init__(self, teacher, student, gate, costs, target, initial, dual_lr, seed):
        self.teacher, self.student, self.gate = teacher, student, gate
        self.teacher.eval().requires_grad_(False)
        self.student.eval()
        for n,p in gate.named_parameters():
            p.requires_grad_(n.endswith(('.A','.B')))
        self.cp = [p for p in student.parameters() if p.requires_grad]
        self.gp = [p for p in gate.parameters() if p.requires_grad]
        if not self.cp or not self.gp or any(not n.endswith(('.A','.B')) for n,p in student.named_parameters() if p.requires_grad):
            raise ValueError('nonempty LoRA-only actor parameters required')
        self.opt_c = torch.optim.Adam(self.cp, lr=.0003)
        self.opt_g = torch.optim.Adam(self.gp, lr=.001)
        self.dual = Dual(initial, dual_lr, target)
        self.costs = costs
        self.rng = torch.Generator().manual_seed(seed)

    def content(self, states, steps, batch=32, samples=4):
        if min(len(states),steps,batch,samples) < 1:
            raise ValueError('positive sizes required')
        losses = []
        for _ in range(steps):
            ix = torch.randint(len(states), (min(batch,len(states)),), generator=self.rng)
            x = states[ix]; h, al = self.student.encode(x)
            anchor = torch.multinomial(al.softmax(-1), 1, generator=self.rng).squeeze(-1)
            w,l = self.student.tail(h,anchor)
            y,_ = teacher_tail(self.teacher, x.repeat_interleave(samples,0), anchor.repeat_interleave(samples), self.rng)
            loss = -fk.joint_log_prob(w.repeat_interleave(samples,0),l.repeat_interleave(samples,0),y).mean()
            self.opt_c.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cp,1.); self.opt_c.step()
            losses.append(float(loss.detach()))
        return {'teacher_sample_nll': sum(losses)/len(losses)}

    def policy(self, states, steps, batch=32, samples=4):
        if min(len(states),steps,batch,samples) < 1:
            raise ValueError('positive sizes required')
        rows=[]
        for _ in range(steps):
            ix=torch.randint(len(states),(min(batch,len(states)),),generator=self.rng)
            b=policy_batch(self.teacher,self.student,states[ix],self.rng,samples=samples)
            rows.append(policy_update(self.gate,self.opt_g,b['features'],b['risk'],self.costs,self.dual,self.rng))
        return rows

    def round(self, states, content_steps, policy_steps, batch=32, samples=4):
        return {'content':self.content(states,content_steps,batch,samples),
                'policy':self.policy(states,policy_steps,batch,samples)}


@torch.no_grad()
def probe(teacher, student, gate, contexts, seed, samples, costs, reference_lambda):
    """Independent teacher tail draws; fixed contexts, anchor RNG across checkpoints."""
    if samples < 1 or not len(contexts): raise ValueError('empty probe')
    rng=torch.Generator().manual_seed(seed)
    all_probs=[]; all_risks=[]; all_anchors=[]
    for start in range(0,len(contexts),16):
        b=policy_batch(teacher,student,contexts[start:start+16],rng,samples=samples)
        all_probs.append(gate(b['features']).softmax(-1)); all_risks.append(b['risk_samples']);all_anchors.append(b['anchor'])
    probs=torch.cat(all_probs); risks=torch.cat(all_risks); means=risks.mean(1)
    er=expected_risk(probs,means); ks=probs.argmax(-1)+1
    gr=means.gather(-1,(ks-1)[:,None]).squeeze(-1)
    count=torch.arange(1,H+1,dtype=probs.dtype)
    expected_k=(probs*count).sum(-1); cost=(probs*costs).sum(-1)
    reward=expected_k-1-cost-float(reference_lambda)*er
    return {'mean_k_expected':float(expected_k.mean()),'mean_k_greedy':float(ks.float().mean()),
            'greedy_k':ks.tolist(),'greedy_histogram':torch.bincount(ks-1,minlength=H).tolist(),
            'expected_risk':float(er.mean()),'greedy_risk':float(gr.mean()),
            'expected_cost':float(cost.mean()),'reference_reward':float(reward.mean()),
            'reference_lambda':float(reference_lambda)}, {
            'probabilities':probs.cpu().numpy(),'risk_samples':risks.cpu().numpy(),
            'anchors':torch.cat(all_anchors).cpu().numpy(),'greedy_k':ks.cpu().numpy()}


@torch.no_grad()
def generate(student, gate, prefix, count, seed, greedy=False):
    if count < 0 or prefix.ndim!=2 or len(prefix)!=1: raise ValueError('batch1 nonnegative count required')
    if count==0:return prefix[:,:0].clone(),[]
    rng=torch.Generator().manual_seed(seed); out=prefix.clone(); lengths=[]
    while out.shape[1]-prefix.shape[1]<count:
        h,al=student.encode(out[:,-student.backbone.config.block_size:])
        anchor=torch.multinomial(al.softmax(-1),1,generator=rng).squeeze(-1)
        k=1
        if gate is not None:
            w,l=student.tail(h,anchor); logits=gate(student.features(h,anchor,w,l))
            k=(int(logits.argmax(-1)[0])+1 if greedy else int(torch.multinomial(logits.softmax(-1),1,generator=rng)[0,0])+1)
        take=min(k,count-(out.shape[1]-prefix.shape[1])); tokens=anchor[:,None]
        if take>1:tokens=torch.cat([tokens,fk.joint_sample(w,l,rng)[:,:take-1]],1)
        out=torch.cat([out,tokens],1);lengths.append(take)
    return out[:,-count:],lengths
