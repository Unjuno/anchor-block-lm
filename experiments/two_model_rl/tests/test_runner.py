import importlib.util
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_two_model import tiny

def api():
    assert importlib.util.find_spec('run_two_model') is not None,'runner missing'
    import run_two_model
    return run_two_model

def test_one_round_control_never_changes_content_joint_changes_only_lora():
    m=api();teacher,student=tiny()
    from adaptive_k import AdaptiveKGate
    import fixed_k as fk
    h,al=student.encode(torch.zeros(2,16,dtype=torch.long));w,l=student.tail(h,torch.zeros(2,dtype=torch.long))
    gate=AdaptiveKGate(student.features(h,torch.zeros(2,dtype=torch.long),w,l).shape[1])
    seq=torch.randint(7,(1024,));expected=fk.state_hash(teacher)
    for arm in ('gate_only','joint'):
        trained,policy,info=m.train_arm(teacher,student,gate,seq,arm,1.,torch.ones(4),rounds=1,updates=3,batch=4,seed=1)
        assert fk.state_hash(teacher)==expected
        assert info['content_changed']==(arm=='joint')
        assert info['teacher_unchanged'] and info['anchor_unchanged'] and info['gate_changed']
        assert info['trainable_content_names'] and all(n.endswith(('.A','.B')) for n in info['trainable_content_names']) if arm=='joint' else not info['trainable_content_names']

def test_free_running_eval_has_zero_kl_for_pure_ar_and_exact_length():
    m=api();teacher,student=tiny();x=torch.randint(7,(2,16))
    metrics,arrays=m.evaluate(teacher,student,None,x,draws=2,count=8,seed=2)
    assert metrics['tokens_per_call']==1.
    assert metrics['sequence_kl_nats_per_token']==pytest.approx(0.,abs=1e-6)
    assert arrays['tokens'].shape==(2,2,8) and arrays['log_ratio'].shape==(2,2)

def test_timing_summary_uses_equal_output_work_and_same_random_trace_repeats():
    m=api();teacher,student=tiny();x=torch.randint(7,(2,16))
    metrics=m.time_generation(student,None,x,count=4,seed=3,repeats=3)
    assert metrics['outputs_identical_across_repeats']
    assert metrics['tokens']==8 and metrics['calls']==8 and len(metrics['times_seconds'])==3
    assert metrics['median_seconds']>0
