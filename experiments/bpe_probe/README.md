# Natural-language BPE experiments

This directory tests whether the synthetic Anchor Block LM mechanism survives on a small natural-language BPE model.

The goal is **mechanism validation, not scale or production speed**.

## Dataset and teacher

- Project Gutenberg *Romeo and Juliet* snapshot, downloaded from a pinned public source.
- 80/10/10 positional train/dev/test split.
- byte-level BPE, vocabulary 1024, trained on the train split only.
- tiny nanoGPT teacher: 2 layers, 4 heads, width 64, context 64.

The teacher is intentionally small. The positional split also creates noticeable distribution shift between dev and test. Results here should not be generalized to larger LMs without further experiments.

## What we learned

### 1. Exact rollout-prefix agreement is too strict

Using 16 top-p=0.95 rollouts and requiring a large fraction of complete sampled prefixes to remain identical produces almost no usable natural-language blocks.

At an 80% joint-prefix-mass threshold:

- direct mean safe length: 0.0156 BPE tokens
- after one sampled anchor: 0.0547 BPE tokens

The anchor effect is positive, but this label definition is too conservative for training.

### 2. The realized anchor carries substantial information about the next future token

A fair same-position probe compares the distribution of X_(t+2):

- before observing X_(t+1), versus
- after observing a sampled X_(t+1).

Across 256 held-out contexts with 32 sampled anchors per context:

- conditional mutual information: **1.083 nats / 1.562 bits**
- teacher top-1 probability: **13.8% -> 24.6%**
- positive information gain in **99.6%** of contexts

So the premise that an anchor can collapse future uncertainty is present even in this weak BPE teacher.

### 3. A two-backbone-call implementation is structurally wrong for this idea

The first BPE student used:

~~~text
AR backbone call -> anchor
second backbone call -> variable continuation block
~~~

The initial teacher-forced student achieved roughly:

- fixed continuation=2: 1.50 tokens/call, 61.9% local teacher agreement
- variable EOB: 1.33 tokens/call, 62.3% local teacher agreement

An on-policy refresh did not improve the Pareto frontier. More importantly, an immediate EOB wastes the second full-model call.

This motivated the one-pass design.

## One-pass architecture

~~~text
context
   |
one frozen backbone forward
   |
   +--> exact ordinary next-token logits --> anchor
   |
   +--> cheap continuation branch
        conditioned on (hidden state, realized anchor)
            |
            +--> continuation token
            +--> continuation token
            +--> ...
            +--> <EOB>
~~~

The base AR path is **fully frozen**. A low-rank residual and small heads exist only in the continuation branch. Therefore an immediate EOB is an exact fallback to ordinary AR generation.

The EOB head is calibrated after content training using the student's **actual consecutive acceptance length**: how many parallel continuation tokens match the teacher's greedy token under the student-induced prefix.

## Preselected expanded test

The main current result uses:

- 64 held-out test prompts
- 48 generated BPE tokens per prompt
- EOB bias **2.0**, frozen from an earlier dev diagnostic
- random-gating probability **0.132743**, also frozen before this expanded test
- no tuning on the expanded test

| Method | Tokens / backbone call | Local teacher agreement | Teacher NLL |
|---|---:|---:|---:|
| exact AR fallback | 1.000 | **100.00%** | 2.0125 |
| fixed +1 continuation | 2.000 | 64.29% | 3.1055 |
| **variable EOB** | **1.103** | **98.11%** | **2.0325** |
| speed-near random 0/1 gating | 1.128 | 93.33% | 2.1807 |

Variable gating exceeds the random-gating agreement by **+4.79 percentage points**.

Prompt-paired bootstrap, 10,000 resamples:

- mean difference: **+4.79 pp**
- 95% CI: **+3.65 to +5.89 pp**
- variable is better on **81.25%** of prompts

The random baseline is slightly faster (1.128 vs. 1.103 tokens/call), so this is not an exact equal-throughput comparison. The result nevertheless rejects the explanation that the gain comes merely from "occasionally emitting a second token at random."

The machine-readable snapshot is in [results/published_onepass_preregistered.json](results/published_onepass_preregistered.json).

### What this result does and does not show

It **does** show, in this small controlled natural-language experiment, that:

1. the AR fallback can remain exactly unchanged;
2. a learned continuation branch can safely commit extra tokens on a subset of states;
3. confidence-aware gating preserves teacher behavior much better than random gating at a nearby compression rate.

It **does not** yet show:

- GPU wall-clock speedup,
- exact preservation of the teacher sampling distribution,
- performance on a strong or large language model,
- robustness across independent training seeds.

The next required check is multi-seed replication.

## Why top-p is still useful

The experiments separate two roles:

1. **content target** — what continuation to imitate;
2. **uncertainty / horizon estimate** — how far to compress.

Best-of-N top-p samples are useful for probing the teacher's future distribution, but forcing a sampled block target while evaluating against the teacher's greedy mode creates an avoidable objective mismatch.

The current one-pass experiment therefore uses teacher-greedy content targets first. Top-p remains useful for uncertainty diagnostics and future distribution-aware calibration.

## Reproduce

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# install PyTorch appropriate for your platform

python prepare_data.py
python train_teacher.py --steps 1000

python train_onepass_anchor_student.py prepare
python train_onepass_anchor_student.py train --steps 1200

python calibrate_onepass_eob.py prepare
python calibrate_onepass_eob.py train --steps 500

python benchmark_onepass_preregistered.py --test-prompts 64 --count 48
~~~

The uncertainty probes remain available:

~~~bash
python probe_anchor_horizon.py --contexts 256 --rollouts 16 --horizon 6
python probe_bestofn_blocks.py --contexts 256 --rollouts 16 --horizon 6
python probe_anchor_information.py --contexts 256 --anchors-per-context 32
~~~

Quick tests:

~~~bash
PYTHONPATH=. pytest -q
~~~

## Current interpretation

The strongest supported statement is intentionally narrow:

> a realized autoregressive anchor token reduces uncertainty about its continuation, and a separately calibrated one-pass continuation branch can exploit part of that reduction while retaining an exact AR fallback.

The remaining question is how much this effect survives stronger teachers, additional datasets, independent seeds, and real inference systems.
