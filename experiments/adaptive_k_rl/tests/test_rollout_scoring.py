import importlib.util
import math
import itertools
import pytest
import torch


def implementation():
    assert importlib.util.find_spec('evaluate_saved_policy') is not None, 'rollout evaluator not implemented'
    import evaluate_saved_policy as mod
    return mod


class Teacher(torch.nn.Module):
    def __init__(self, probs=(.5,.5), markov=False):
        super().__init__()
        self.register_buffer('probs', torch.tensor(probs,dtype=torch.float64))
        self.config = type('Config', (), {'block_size':4})()
        self.markov = markov
    def forward(self, x):
        p = self.probs.expand(len(x),-1)
        if self.markov:
            p = torch.where((x[:,-1] == 0)[:,None], p, p.flip(-1))
        return p.log()[:,None,:], None


class Student(torch.nn.Module):
    def __init__(self, teacher, component_probs):
        super().__init__()
        self.backbone = teacher
        self.register_buffer('component_probs',torch.tensor(component_probs,dtype=torch.float64))
    def encode(self, x):
        return torch.zeros(len(x),1), self.backbone(x)[0][:,-1]
    def tail(self, h, anchor):
        m = len(self.component_probs)
        return torch.zeros(len(h),m,dtype=torch.float64), self.component_probs.log()[None].expand(len(h),-1,-1,-1)


def test_exact_ar_has_zero_sequence_log_ratio_under_own_history():
    mod=implementation(); teacher=Teacher((.8,.2),markov=True)
    student=Student(teacher,[[[.6,.4]]*3])
    result=mod.score_sequence(teacher,student,torch.tensor([[0,1]]),torch.tensor([0,1,1]),[1,1,1])
    assert result['log_ratio_nats'] == pytest.approx(0.,abs=1e-12)
    assert result['teacher_logp'] == pytest.approx(math.log(.2*.2*.8))


def test_known_one_extra_token_log_ratio():
    mod=implementation(); t=Teacher(); s=Student(t,[[[.8,.2]]*3])
    r=mod.score_sequence(t,s,torch.tensor([[0]]),torch.tensor([0,0]),[2])
    assert r['log_ratio_nats'] == pytest.approx(math.log(1.6))


def test_prefix_mixture_is_normalized_after_truncating_a_block():
    mod=implementation(); t=Teacher(); s=Student(t,[[[.9,.1]]*3,[[.1,.9]]*3])
    logqs=[]
    for seq in itertools.product(range(2),repeat=3):
        r=mod.score_sequence(t,s,torch.tensor([[0]]),torch.tensor(seq),[3])
        logqs.append(r['student_logq'])
    assert sum(math.exp(x) for x in logqs) == pytest.approx(1.,abs=1e-12)
    assert math.exp(logqs[0]) == pytest.approx(.5*(.5*.9*.9+.5*.1*.1))


@pytest.mark.parametrize('tokens,lengths', [([0,1],[1]),([0],[0,1]),([0,1],[3]),([0]*5,[5])])
def test_rejects_invalid_segmentation(tokens,lengths):
    mod=implementation(); t=Teacher(); s=Student(t,[[[.5,.5]]*3])
    with pytest.raises(ValueError):
        mod.score_sequence(t,s,torch.tensor([[0]]),torch.tensor(tokens),lengths)


def test_zero_length_score_has_no_model_work():
    mod=implementation(); r=mod.score_sequence(None,None,torch.tensor([[0]]),torch.empty(0,dtype=torch.long),[])
    assert r == {'student_logq':0.,'teacher_logp':0.,'log_ratio_nats':0.}


def test_bootstrap_clusters_repeated_trajectories_by_prompt():
    mod=implementation()
    import numpy as np
    values=np.full((4,3),2.5)
    assert mod.prompt_cluster_interval(values,seed=7)==[2.5,2.5]
