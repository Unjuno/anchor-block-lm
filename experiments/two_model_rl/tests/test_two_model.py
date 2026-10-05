import importlib.util
import sys
from pathlib import Path
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'fixed_k_gate'))
import fixed_k as fk

def api():
    assert importlib.util.find_spec('two_model') is not None, 'two-model implementation is missing'
    import two_model
    return two_model

def tiny():
    torch.manual_seed(18)
    teacher = fk.GPT(fk.GPTConfig(block_size=16, vocab_size=7, n_layer=1,
        n_head=1, n_embd=16, dropout=0., bias=True)).eval().requires_grad_(False)
    student = fk.FixedKStudent(teacher, rank=2, components=2).eval().requires_grad_(False)
    return teacher, student

def test_zero_initialized_continuation_lora_preserves_saved_joint():
    m=api(); teacher, student=tiny(); x=torch.randint(7,(3,16)); a=torch.tensor([1,2,3])
    h,_=student.encode(x); before=student.tail(h,a)
    m.enable_content_lora(student, rank=2)
    after=student.tail(h,a)
    for b,z in zip(before,after): torch.testing.assert_close(b,z,atol=0,rtol=0)
    assert len([p for p in student.parameters() if p.requires_grad])>2
    assert all(n.endswith(('.A','.B')) for n,p in student.named_parameters() if p.requires_grad)

def test_teacher_and_anchor_stay_frozen_when_content_lora_updates():
    m=api(); teacher,student=tiny(); x=torch.randint(7,(8,16)); th=fk.state_hash(teacher)
    m.enable_content_lora(student,rank=2); frozen={n:p.clone() for n,p in student.named_parameters() if not p.requires_grad}
    before=fk.state_hash(student); opt=torch.optim.Adam([p for p in student.parameters() if p.requires_grad],lr=.01)
    metrics=m.content_step(teacher,student,opt,x,torch.Generator().manual_seed(20),samples=3)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert fk.state_hash(teacher)==th and fk.state_hash(student)!=before
    assert all(p.grad is None for p in teacher.parameters())
    for n,p in student.named_parameters():
        if n in frozen: torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)
    torch.testing.assert_close(student.encode(x)[1],teacher(x)[0][:,-1],atol=0,rtol=0)

def test_reinforce_gradient_equals_exact_categorical_reverse_kl():
    m=api(); logits=torch.tensor([.3,-.2,.6],dtype=torch.float64,requires_grad=True)
    logp=torch.tensor([.15,.25,.60],dtype=torch.float64).log(); logq=logits.log_softmax(0)
    exact=(logq.exp()*(logq-logp)).sum(); grad=torch.autograd.grad(exact,logits,retain_graph=True)[0]
    surrogate=(logq.exp().detach()*m.reinforce_terms(logq,logp)).sum()
    actual=torch.autograd.grad(surrogate,logits)[0]
    torch.testing.assert_close(actual,grad,atol=1e-12,rtol=1e-12)

def test_identical_distribution_has_zero_reverse_kl_score_gradient():
    m=api(); logits=torch.randn(4,requires_grad=True); lp=logits.log_softmax(0)
    m.reinforce_terms(lp,lp.detach()).sum().backward()
    torch.testing.assert_close(logits.grad,torch.zeros(4),atol=0,rtol=0)

def test_leave_one_out_baseline_excludes_own_sample():
    m=api(); values=torch.tensor([[1.,4.,7.],[3.,6.,9.]])
    b=m.leave_one_out(values); torch.testing.assert_close(b,torch.tensor([[5.5,4.,2.5],[7.5,6.,4.5]]))
    changed=values.clone(); changed[0,0]+=100
    assert m.leave_one_out(changed)[0,0]==b[0,0]

def test_time_reward_counts_head_cost_not_only_token_count():
    m=api(); risks=torch.tensor([[0.,.1,.3,.5]]); costs=torch.tensor([1.4,1.5,1.5,1.5])
    torch.testing.assert_close(m.time_rewards(risks,costs,2.),torch.tensor([[-.4,.3,.9,1.5]]),atol=1e-6,rtol=0)
    with pytest.raises(ValueError): m.time_rewards(risks,torch.ones(3),2.)

def test_prefix_risks_zero_ar_and_uniform_identical_models():
    m=api(); teacher,student=tiny()
    with torch.no_grad():
        for p in teacher.parameters(): p.zero_()
        student.backbone.load_state_dict(teacher.state_dict())
    x=torch.zeros(4,16,dtype=torch.long)
    bank=m.policy_batch(teacher,student,x,torch.Generator().manual_seed(99),samples=3)
    assert bank['risk'].shape==(4,4)
    torch.testing.assert_close(bank['risk'],torch.zeros(4,4),atol=2e-6,rtol=0)
    assert bank['features'].grad_fn is None

def test_deployment_merging_keeps_logits_and_does_not_mutate_training_model():
    m=api();teacher,student=tiny();m.enable_content_lora(student,rank=2)
    with torch.no_grad():
        for n,p in student.named_parameters():
            if n.endswith('.B') and p.requires_grad:p.normal_(std=.02)
    before=fk.state_hash(student); deploy=m.deployment_copy(student)
    x=torch.randint(7,(3,16));h,_=student.encode(x);a=torch.tensor([0,1,2])
    for b,z in zip(student.tail(h,a),deploy.tail(h,a)):torch.testing.assert_close(b,z,atol=1e-5,rtol=1e-5)
    assert fk.state_hash(student)==before
    assert not any(isinstance(c,m.LoRAProjection) for c in deploy.modules())

def test_eval_generator_emits_all_lengths_and_zero_budget_is_empty():
    m=api();teacher,student=tiny();x=torch.zeros(1,16,dtype=torch.long)
    class Constant(torch.nn.Module):
        def __init__(self,k):super().__init__();self.k=k
        def forward(self,f):
            out=torch.full((len(f),4),-30.);out[:,self.k-1]=30.;return out
    for k in (1,2,3,4):
        tokens,lengths=m.generate(student,Constant(k),x,7,31)
        assert tokens.shape==(1,7) and sum(lengths)==7 and lengths[0]==k
    tokens,lengths=m.generate(student,Constant(4),x,0,31)
    assert tokens.shape==(1,0) and lengths==[]

def test_pure_ar_never_calls_continuation_head():
    m=api();teacher,student=tiny();x=torch.zeros(1,16,dtype=torch.long)
    def forbidden(*args,**kwargs):raise AssertionError('AR invoked the block branch')
    student.tail=forbidden
    tokens,lengths=m.generate(student,None,x,7,31)
    assert tokens.shape==(1,7) and lengths==[1]*7

def test_onpolicy_contexts_are_visited_before_anchor_and_exact_window():
    m=api();teacher,student=tiny();x=torch.zeros(2,16,dtype=torch.long)
    states=m.collect_states(student,None,x,cycles=3,seed=6)
    assert states.shape==(6,16)
    torch.testing.assert_close(states[:2],x,atol=0,rtol=0)

def test_sampling_teacher_likelihoods_preserves_full_horizon():
    m=api();teacher,student=tiny();x=torch.zeros(3,16,dtype=torch.long)
    tokens,logp=m.teacher_tail(teacher,x,torch.tensor([0,1,2]),torch.Generator().manual_seed(12))
    assert tokens.shape==(3,3) and logp.shape==(3,)
    ctx=torch.cat([x,torch.tensor([[0],[1],[2]])],1);expected=torch.zeros(3)
    for j in range(3):
        logits=teacher(ctx[:,-16:])[0][:,-1];expected+=logits.log_softmax(-1).gather(-1,tokens[:,j,None]).squeeze(-1)
        ctx=torch.cat([ctx,tokens[:,j,None]],1)
    torch.testing.assert_close(logp,expected)
