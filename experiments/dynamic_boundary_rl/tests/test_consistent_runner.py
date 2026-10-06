import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def mod():
    assert importlib.util.find_spec('run_consistent_dual') is not None, 'runner missing'
    import run_consistent_dual
    return run_consistent_dual

def test_boundary_transition_retains_which_context_changed():
    m=mod();r=m.transitions([1,2,1,4],[2,1,3,4])
    assert r['increased']==2 and r['decreased']==1 and r['unchanged']==1
    assert r['matrix'][0][2]==1

def test_metric_does_not_claim_sampled_trace_is_marginal_kl():
    m=mod()
    a={'ratios':np.zeros((3,2)), 'calls':np.full((3,2),3), 'nll':np.ones((3,2))*6}
    r=m.rollout_summary(a,6,'sampled',2)
    assert r['score_kind']=='augmented_trace_reverse_kl_bound'
    assert r['tokens_per_call']==2
    assert m.rollout_summary(a,6,'greedy',2)['score_kind']=='token_sequence_reverse_kl'

def test_budget_and_indices_are_taken_from_archive_not_dev_tuned():
    m=mod()
    old={'protocol':{'kl_target':.31,'normalized_action_costs':[1.,1.2,1.3,1.4]},
         'training':{'final_dual_lambda':2.3},'dev_probe_starts':[1,3], 'evaluation_starts':[10,12]}
    assert m.resume_settings(old)['target']==.31
    assert m.resume_settings(old)['initial_dual']==2.3
    with pytest.raises(ValueError):m.resume_settings({'protocol':{'kl_target':float('nan')}})
