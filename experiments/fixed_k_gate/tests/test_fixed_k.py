import itertools
import pytest
import torch
import fixed_k as f


def tiny():
    torch.manual_seed(99)
    return f.GPT(f.GPTConfig(block_size=16, vocab_size=7, n_layer=1, n_head=1, n_embd=16, dropout=0, bias=True)).eval()


def test_mixture_is_normalized_joint_not_product_of_marginals():
    weights = torch.zeros(1, 2)
    logits = torch.tensor([[[[7., -7.], [7., -7.]], [[-7., 7.], [-7., 7.]]]])
    seqs = torch.tensor(list(itertools.product(range(2), repeat=2)))
    logp = f.joint_log_prob(weights.expand(4, -1), logits.expand(4, -1, -1, -1), seqs)
    assert logp.exp().sum().item() == pytest.approx(1., abs=1e-6)
    assert logp[0].exp().item() > .49
    assert logp[1].exp().item() < .001


def test_sampler_uses_one_shared_component_for_whole_block():
    weights = torch.zeros(1000, 2)
    logits = torch.tensor([[[[9., -9.]] * 3, [[-9., 9.]] * 3]]).expand(1000,-1,-1,-1)
    y = f.joint_sample(weights, logits, torch.Generator().manual_seed(3))
    assert y.shape == (1000,3)
    assert torch.equal(y[:,0],y[:,2])
    assert .4 < y.float().mean().item() < .6


def test_fixed_four_and_no_eob_vocabulary():
    model = f.FixedKStudent(tiny(), rank=2, components=2)
    h, anchor_logits = model.encode(torch.ones(2,12,dtype=torch.long))
    weights, logits = model.tail(h, torch.tensor([0,1]))
    assert f.BLOCK_SIZE == 4
    assert logits.shape == (2,2,3,7)
    assert weights.shape == (2,2)


def test_anchor_path_stays_exact_after_all_trainable_weights_change():
    teacher = tiny()
    model = f.FixedKStudent(teacher, rank=2, components=2).eval()
    x = torch.randint(7,(2,12))
    expected = teacher(x)[0][:,-1]
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad: p.add_(torch.randn_like(p)*.1)
    actual = model.encode(x)[1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert f.state_hash(teacher) == f.state_hash(model.backbone)


def test_only_gate_lora_changes_in_reward_update():
    student = f.FixedKStudent(tiny(),rank=2,components=2).eval().requires_grad_(False)
    before = f.state_hash(student)
    gate = f.LoRAGate(6, rank=2)
    initial = f.state_hash(gate)
    opt = torch.optim.Adam([p for p in gate.parameters() if p.requires_grad],lr=.1)
    x = torch.randn(32,6)
    for _ in range(3):
        f.policy_step(gate, opt, x, torch.linspace(-1.,1.,32))
    assert f.state_hash(student) == before
    assert f.state_hash(gate) != initial
    assert all(n.endswith(('.A','.B')) for n,p in gate.named_parameters() if p.requires_grad)
    assert all(p.grad is None for p in student.parameters())


def test_actions_are_only_ar_or_fixed_block_no_partial_block():
    assert f.emitted_length(False,100) == 1
    assert f.emitted_length(True,100) == 4
    assert f.emitted_length(True,3) == 1
    with pytest.raises(ValueError): f.emitted_length(True,0)


def test_matched_budget_selects_exact_same_number_without_risk_input():
    scores=torch.tensor([.1,.8,.2,.9,.4,.3,.6,.5])
    keep=f.matched_mask(scores,2)
    assert keep.sum().item()==2
    assert keep.nonzero().flatten().tolist()==[1,3]
    with pytest.raises(ValueError): f.matched_mask(scores,9)


def test_full_teacher_distribution_keeps_low_probability_tokens():
    p=f.teacher_probs(torch.tensor([[0.,-10.]]))
    assert (p>0).all()
    torch.testing.assert_close(p.sum(-1),torch.ones(1))


def test_reward_has_no_length_action_and_penalizes_distribution_error():
    r=f.block_reward(torch.tensor([0.,1.,3.]),2.)
    torch.testing.assert_close(r,torch.tensor([3.,1.,-3.]))


def test_joint_nll_can_learn_correlated_blocks():
    weights=torch.zeros(4,2,requires_grad=True)
    logits=torch.randn(4,2,3,2,requires_grad=True)
    loss=-f.joint_log_prob(weights,logits,torch.tensor([[0,0,0],[1,1,1],[0,0,0],[1,1,1]])).mean()
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0


def test_counterfactual_kl_zero_for_identical_uniform_joint_models():
    from run_experiment import risk_bank
    teacher=f.GPT(f.GPTConfig(block_size=64,vocab_size=3,n_layer=1,n_head=1,n_embd=8,dropout=0,bias=True)).eval()
    with torch.no_grad():
        for p in teacher.parameters(): p.zero_()
    student=f.FixedKStudent(teacher,rank=2,components=2).eval().requires_grad_(False)
    bank=risk_bank(teacher,student,torch.zeros(2,64,dtype=torch.long),8,12)
    torch.testing.assert_close(bank['risk_samples'],torch.zeros(2,8),atol=2e-6,rtol=0)


def test_generate_commits_full_block_or_ar_at_end_never_a_truncated_block():
    from run_experiment import generate
    teacher=f.GPT(f.GPTConfig(block_size=64,vocab_size=7,n_layer=1,n_head=1,n_embd=8,dropout=0,bias=True)).eval()
    student=f.FixedKStudent(teacher,rank=2,components=2).eval().requires_grad_(False)
    prefix=torch.zeros(1,64,dtype=torch.long)
    tokens,calls,lengths=generate(student,None,prefix,9,'block',0.,1)
    assert tokens.shape==(1,9)
    assert calls==3 and lengths==[4,4,1]


def test_matched_diagnostic_has_equal_calls_tokens_and_blocks():
    from run_experiment import matched_diagnostic
    bank={'risk_samples':torch.arange(64,dtype=torch.float32)[:,None].expand(64,8),
          'confidence':-torch.arange(64,dtype=torch.float32)}
    result=matched_diagnostic(bank,bank['confidence'],1,boot=10)
    assert result['learned']['blocks']==result['confidence']['blocks']==8
    assert result['learned']['calls']==result['confidence']['calls']==64
    assert result['learned']['emitted_tokens']==result['confidence']['emitted_tokens']==88
