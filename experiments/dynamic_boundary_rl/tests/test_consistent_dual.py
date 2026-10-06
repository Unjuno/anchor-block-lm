import importlib.util
import sys
from pathlib import Path
import pytest
import torch

EXP = Path(__file__).resolve().parents[1]
for path in (EXP, EXP.parent/'two_model_rl', EXP.parent/'adaptive_k_rl', EXP.parent/'fixed_k_gate'):
    sys.path.insert(0, str(path))
import fixed_k as fk
from adaptive_k import AdaptiveKGate
from two_model import enable_content_lora


def mod():
    assert importlib.util.find_spec('consistent_dual') is not None, 'consistent-dual implementation is missing'
    import consistent_dual
    return consistent_dual


def models():
    torch.manual_seed(19)
    t=fk.GPT(fk.GPTConfig(block_size=8,vocab_size=7,n_layer=1,n_head=1,n_embd=8,dropout=0)).eval().requires_grad_(False)
    s=enable_content_lora(fk.FixedKStudent(t,rank=2,components=2),rank=2)
    g=AdaptiveKGate(2*8+2*fk.TAIL+1,rank=2)
    return t,s,g


def test_dual_uses_full_action_distribution_not_argmax():
    m=mod()
    probs=torch.tensor([[.6,.4,0.,0.]])
    risks=torch.tensor([[0.,2.,3.,4.]])
    assert m.expected_risk(probs,risks).item()==pytest.approx(.8)
    d=m.Dual(.5,.1,.3)
    d.step(m.expected_risk(probs,risks))
    assert d.value==pytest.approx(.55)


def test_dual_projection_and_finite_checks():
    m=mod();d=m.Dual(.01,1.,.3)
    assert d.step(torch.zeros(3))==0.
    for value in (float('nan'),float('inf')):
        with pytest.raises(ValueError): d.step(torch.tensor([value]))
    with pytest.raises(ValueError): m.Dual(1.,.1,-.1)


def test_policy_step_logs_actual_sample_and_expected_risk():
    m=mod();_,_,g=models()
    for n,p in g.named_parameters(): p.requires_grad_(n.endswith(('.A','.B')))
    opt=torch.optim.Adam([p for p in g.parameters() if p.requires_grad],lr=.001)
    x=torch.randn(12,23);risks=torch.tensor([[0.,.2,.4,.8]]).repeat(12,1)
    probs=g(x).softmax(-1).detach();costs=torch.ones(4);d=m.Dual(1.,.1,.3)
    before=m.expected_risk(probs,risks).mean().item()
    r=m.policy_update(g,opt,x,risks,costs,d,torch.Generator().manual_seed(6))
    assert r['expected_risk']==pytest.approx(before)
    assert d.value==pytest.approx(max(0,1+.1*(before-.3)))
    assert sum(r['sampled_k_histogram'])==12


def test_optimizer_state_survives_round_boundary_and_only_lora_changes():
    m=mod();t,s,g=models();x=torch.randint(7,(8,8));th=fk.state_hash(t)
    fixed={n:p.detach().clone() for n,p in s.named_parameters() if not p.requires_grad}
    tr=m.Trainer(t,s,g,torch.ones(4),.4,1.,.1,99)
    tr.round(x,2,2,batch=4,samples=2)
    tr.round(x,2,2,batch=4,samples=2)
    assert {int(v['step']) for v in tr.opt_c.state.values()}=={4}
    assert {int(v['step']) for v in tr.opt_g.state.values()}=={4}
    assert fk.state_hash(t)==th==fk.state_hash(s.backbone)
    assert all(torch.equal(p,fixed[n]) for n,p in s.named_parameters() if n in fixed)
    assert all(n.endswith(('.A','.B')) for n,p in s.named_parameters() if p.requires_grad)


def test_sampled_rollout_is_reproducible_and_uses_each_action():
    m=mod();_,s,g=models();x=torch.randint(7,(2,8))
    class Uniform(torch.nn.Module):
        def forward(self,x): return torch.zeros(len(x),4)
    one=m.collect(s,Uniform(),x,40,66)
    two=m.collect(s,Uniform(),x,40,66)
    assert torch.equal(one['states'],two['states'])
    assert one['lengths']==two['lengths']
    assert set(one['lengths'])=={1,2,3,4}
    assert len(one['states'])==80


def test_gate_none_bypasses_tail_and_matches_ar_scores():
    m=mod();t,s,_=models();x=torch.randint(7,(1,8))
    def forbidden(*args): raise AssertionError('AR must bypass continuation')
    s.tail=forbidden
    y,ks=m.generate(s,None,x,7,4)
    assert y.shape==(1,7) and ks==[1]*7
    assert m.generate(s,None,x,0,4)[0].shape==(1,0)


def test_fixed_probe_records_contextwise_actions_and_probabilities():
    m=mod();t,s,g=models();x=torch.randint(7,(3,8))
    r,raw=m.probe(t,s,g,x,33,8,torch.ones(4),1.)
    assert len(r['greedy_k'])==3
    assert raw['risk_samples'].shape==(3,8,4)
    assert raw['probabilities'].shape==(3,4)
    assert abs(raw['risk_samples'][...,0]).max()==0
    assert r['reference_reward']==pytest.approx(r['mean_k_expected']-1-r['expected_cost']-r['expected_risk'])
