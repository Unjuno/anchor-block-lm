import importlib.util
import torch
from test_two_model import tiny

def api():
    assert importlib.util.find_spec('content_audit') is not None,'content-fidelity audit missing'
    import content_audit
    return content_audit

def test_uniform_teacher_and_student_have_zero_forward_kl():
    m=api();teacher,student=tiny()
    with torch.no_grad():
        for p in teacher.parameters():p.zero_()
        student.backbone.load_state_dict(teacher.state_dict())
    x=torch.zeros(3,16,dtype=torch.long)
    result,raw=m.score_contents(teacher,{'initial':student},x,samples=4,seed=8)
    assert abs(result['initial']['forward_kl_nats_per_tail'])<1e-6
    assert raw['initial'].shape==(3,4)

def test_same_teacher_samples_are_reused_across_identical_actors():
    m=api();teacher,student=tiny();x=torch.randint(7,(3,16))
    result,raw=m.score_contents(teacher,{'initial':student,'control':student},x,samples=4,seed=9)
    assert (raw['initial']==raw['control']).all()
    assert raw['contexts'].shape==(3,16)
