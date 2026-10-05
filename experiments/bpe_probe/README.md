# Natural-language BPE predictability probe

This is the first natural-language test of the Anchor Block LM hypothesis.

It intentionally does **not** train the block student yet. It tests the necessary precondition first:

> after one ordinary autoregressive anchor token is realized, does the teacher's future top-p distribution remain concentrated for more subsequent BPE tokens?

If the answer is no, there is little reason to spend compute training a block student.

## Dataset

The probe downloads a pinned Project Gutenberg *Romeo and Juliet* text snapshot from GitHub at runtime. The text itself is not vendored in this repository.

The corpus is split 80/10/10 by text position. A byte-level BPE tokenizer with target vocabulary 1024 is trained **only on the training split**, then dev/test are encoded with the frozen tokenizer.

## Probe

For each held-out test context:

1. sample 16 direct future trajectories from the teacher with top-p=0.95;
2. independently sample one ordinary anchor token from the same teacher;
3. commit that anchor to the context;
4. sample 16 future trajectories from the anchor-conditioned context;
5. compute the joint modal-prefix mass at each future depth;
6. compare the longest contiguous safe prefix before vs. after anchoring.

Results are reported for mass thresholds 0.60, 0.70, 0.80, 0.90, and 0.95, with paired bootstrap confidence intervals.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# install PyTorch appropriate for your platform

python prepare_data.py
python train_teacher.py --steps 1000
python probe_anchor_horizon.py --contexts 256 --rollouts 16 --horizon 6
```

Quick tests:

```bash
PYTHONPATH=. pytest -q
```

## Interpretation

A positive anchor-minus-direct safe-length difference supports the **predictability-horizon** premise. It does not yet show that a student can exploit the horizon efficiently; that is the next experiment.

A null or negative result is a useful stop signal: it would suggest that the strong synthetic result does not transfer cleanly to natural-language BPE at this scale.
