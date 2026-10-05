"""Acceptance tests for the adaptive block / LoRA feasibility experiment."""
import importlib.util
import pytest
import torch

def module():
    spec = importlib.util.find_spec('poc')
    assert spec is not None, 'poc implementation is missing'
    import poc
    return poc

def test_zero_lora_equals_base_and_merge_is_equivalent():
    p = module()
    torch.manual_seed(0)
    layer = torch.nn.Linear(8, 12)
    x = torch.randn(2, 5, 8)
    expected = layer(x).detach().clone()
    wrapped = p.LoRALinear(layer, rank=2, alpha=2)
    torch.testing.assert_close(wrapped(x), expected, rtol=0, atol=0)
    with torch.no_grad(): wrapped.B.normal_(std=0.02)
    actual = wrapped(x)
    torch.testing.assert_close(wrapped.merged()(x), actual, rtol=1e-5, atol=1e-6)

def test_lora_optimizer_cannot_change_frozen_base():
    p = module()
    layer = p.LoRALinear(torch.nn.Linear(8, 12), rank=2, alpha=2)
    original = {n:t.detach().clone() for n,t in layer.base.named_parameters()}
    opt = torch.optim.AdamW([q for q in layer.parameters() if q.requires_grad], lr=0.01)
    opt.zero_grad(); layer(torch.randn(4,8)).square().mean().backward(); opt.step()
    for n,t in layer.base.named_parameters():
        torch.testing.assert_close(t, original[n], rtol=0, atol=0)
        assert t.grad is None
    assert layer.B.detach().abs().sum().item() > 0

def test_block_model_shapes_and_trainable_whitelist():
    p = module()
    teacher = p.GPT(p.GPTConfig(block_size=16,vocab_size=17,n_layer=1,n_head=2,n_embd=16,dropout=0,bias=True))
    model = p.BlockStudent(teacher, rank=2)
    pred, hidden = model(torch.randint(0,17,(3,12)))
    assert pred.shape == (3,4,17)
    assert hidden.shape == (3,16)
    for n,q in model.named_parameters():
        if q.requires_grad:
            assert n.endswith(('.A','.B')) or n.startswith('future_heads.'), n

def test_backbone_has_no_future_leakage():
    p = module()
    teacher = p.GPT(p.GPTConfig(block_size=16,vocab_size=17,n_layer=1,n_head=2,n_embd=16,dropout=0,bias=True)).eval()
    a=torch.randint(0,17,(2,12)); b=a.clone(); b[:,8:]=(b[:,8:]+1)%17
    torch.testing.assert_close(p.hidden_states(teacher,a)[:,:8],p.hidden_states(teacher,b)[:,:8],rtol=0,atol=0)

def test_reward_depends_on_quality_and_not_only_length():
    p=module()
    r=p.action_rewards(torch.zeros(1,4), call_cost=0.1)
    assert r[0,2] > r[0,1] > r[0,0]
    r=p.action_rewards(torch.tensor([[0.,2.,3.,4.]]), call_cost=0.1)
    assert r[0,0] > r[0,1] > r[0,2]

def test_generation_truncates_exactly_without_teacher():
    p=module()
    teacher=p.GPT(p.GPTConfig(block_size=16,vocab_size=17,n_layer=1,n_head=2,n_embd=16,dropout=0,bias=True)).eval()
    model=p.BlockStudent(teacher,rank=2).eval()
    prefix=torch.randint(0,17,(1,12))
    out,info=p.generate_student(model,prefix,7,fixed_k=4)
    assert out.shape == (1,19)
    assert info['calls']==2
    assert info['committed']==7
    assert info['lengths']==[4,3]

def test_teacher_targets_are_sequential_greedy():
    p=module()
    teacher=p.GPT(p.GPTConfig(block_size=16,vocab_size=17,n_layer=1,n_head=2,n_embd=16,dropout=0,bias=True)).eval()
    x=torch.randint(0,17,(2,16))
    ys=p.teacher_targets(teacher,x,4)
    for j in range(4):
        z=torch.cat([x,ys[:,:j]],dim=1)[:,-16:]
        expected=teacher(z)[0][:,-1].argmax(-1)
        assert torch.equal(ys[:,j],expected)

def test_controller_lora_is_only_rl_trainable_parameter():
    p=module()
    controller=p.LengthController(24)
    controller.enable_lora(rank=2)
    params=[n for n,q in controller.named_parameters() if q.requires_grad]
    assert params
    assert all(n.endswith(('.A','.B')) for n in params)
    assert controller(torch.randn(3,24)).shape==(3,3)

def test_eob_student_emits_augmented_vocab_and_stops_at_first_eob():
    import eob_experiment as e
    p=module()
    teacher=p.GPT(p.GPTConfig(block_size=16,vocab_size=17,n_layer=1,n_head=2,n_embd=16,dropout=0,bias=True)).eval()
    model=e.EOBBlockStudent(teacher,rank=2,horizon=5).eval()
    x=torch.randint(0,17,(2,12))
    logits,_=model(x)
    assert logits.shape==(2,6,18)
    fake=torch.full((1,6,18),-100.0)
    fake[0,0,3]=10; fake[0,1,4]=10; fake[0,2,17]=10; fake[0,3,5]=10
    toks,k=e.decode_eob_logits(fake,eob_id=17,max_tokens=5)
    assert k.tolist()==[2]
    assert toks[0,:2].tolist()==[3,4]

def test_modal_prefix_targets_use_joint_prefix_mass_not_slotwise_majority():
    import eob_experiment as e
    roll=torch.tensor([[[1,2,7],[1,2,8],[1,3,9],[4,5,6]]])
    toks,mass=e.modal_prefix_path(roll)
    assert toks.shape==(1,3)
    assert toks[0,:2].tolist()==[1,2]
    assert mass[0,0].item()==pytest.approx(0.75)
    assert mass[0,1].item()==pytest.approx(0.50)
    assert mass[0,2].item()==pytest.approx(0.25)

def test_make_eob_targets_places_eob_after_last_safe_token():
    import eob_experiment as e
    modal=torch.tensor([[2,3,4,5]])
    mass=torch.tensor([[1.0,.95,.70,.40]])
    target, lengths=e.make_eob_targets(modal,mass,threshold=.8,eob_id=9)
    assert lengths.tolist()==[2]
    assert target[0,:3].tolist()==[2,3,9]
    assert target[0,3:].tolist()==[-100,-100]

def test_eob_bias_controls_stop_length_monotonically():
    import eob_experiment as e
    fake=torch.full((1,4,6),-20.0)
    fake[0,:,1]=1.0
    fake[0,0,5]=0.0; fake[0,1,5]=0.5; fake[0,2,5]=1.2; fake[0,3,5]=2.0
    _,k0=e.decode_eob_logits(fake,eob_id=5,max_tokens=3,eob_bias=0.0)
    _,k1=e.decode_eob_logits(fake,eob_id=5,max_tokens=3,eob_bias=1.0)
    assert k1.item() <= k0.item()

def test_eob_evaluate_uses_default_calibration_without_name_error():
    import eob_experiment as e
    p=module()
    teacher=p.GPT(p.GPTConfig(block_size=16,vocab_size=17,n_layer=1,n_head=2,n_embd=16,dropout=0,bias=True)).eval()
    model=e.EOBBlockStudent(teacher,rank=2,horizon=e.H).eval()
    x=torch.randint(0,17,(2,12))
    modal=torch.randint(0,17,(2,e.H))
    mass=torch.ones(2,e.H)
    target,lengths=e.make_eob_targets(modal,mass,.8,eob_id=17)
    metrics=e.evaluate(model,{'x':x,'modal':modal,'mass':mass,'target':target,'lengths':lengths},batch=2)
    assert 'ce' in metrics
