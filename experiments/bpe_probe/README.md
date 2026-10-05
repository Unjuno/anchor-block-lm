# Natural-language BPE experiments

This directory tests whether the synthetic Anchor Block LM result survives on a small natural-language BPE model.

The goal is still mechanism validation, not scale.

## Dataset and teacher

- Project Gutenberg *Romeo and Juliet* snapshot, downloaded from a pinned public source.
- 80/10/10 positional train/dev/test split.
- byte-level BPE, vocabulary 1024, trained on the train split only.
- tiny nanoGPT teacher: 2 layers, 4 heads, width 64, context 64.

The teacher is intentionally small. Its held-out NLL is high, so negative results here should not be generalized to larger LMs.

## What we learned so far

### 1. Exact rollout-prefix agreement is too strict

Using 16 top-p=0.95 rollouts and requiring a large fraction of complete sampled prefixes to remain identical produces almost no usable natural-language blocks.

At an 80% joint-prefix-mass threshold:

- direct mean safe length: 0.0156 BPE tokens
- after one sampled anchor: 0.0547 BPE tokens

The anchor effect is positive, but this label definition is not useful for training.

### 2. The anchor does contain real information about the next future token

A fair same-position probe compares the distribution of (X_{t+2}):

- before observing (X_{t+1}), versus
- after observing a sampled (X_{t+1}).

Across 256 held-out contexts with 32 sampled anchors per context:

- conditional mutual information: **1.083 nats / 1.562 bits**
- teacher top-1 probability: **13.8% -> 24.6%**
- positive information gain in **99.6%** of contexts

So the premise that an anchor can collapse future uncertainty is present even in this weak BPE teacher.

### 3. A two-backbone-call implementation is not enough

The first BPE student used:

```text
AR backbone call -> anchor
second backbone call -> variable continuation block
```

The initial teacher-forced student achieved roughly:

- fixed continuation=2: 1.50 tokens/call, 61.9% local teacher agreement
- variable EOB: 1.33 tokens/call, 62.3% local teacher agreement

An on-policy refresh did not improve the Pareto frontier. It also exposed a structural problem: when the block immediately emits EOB, the second backbone call advances zero tokens, so the method can fall below 1 token/call.

This is why the next implementation is **one-pass anchor-conditioned decoding**.

## Current architecture under test

```text
context
   |
one backbone forward
   |
   +--> ordinary next-token logits --> anchor
   |
   +--> cheap head conditioned on (hidden state, realized anchor)
            |
            +--> continuation token
            +--> continuation token
            +--> ...
            +--> <EOB>
```

Every backbone call therefore emits at least the ordinary anchor token. If the continuation head immediately stops, generation simply falls back to ordinary autoregressive progress instead of wasting a full backbone call.

The base nanoGPT weights remain frozen; LoRA and the small continuation modules are trainable.

## Why top-p is still useful

The experiments separate two roles that were conflated initially:

1. **content target** — a canonical continuation to imitate;
2. **uncertainty / horizon estimate** — how far it is safe to compress.

Best-of-N top-p samples are useful for exploring the teacher's continuation distribution, but forcing a sampled block target while evaluating against the teacher's greedy mode creates an avoidable objective mismatch.

The current one-pass experiment therefore uses teacher-greedy content targets first, while the top-p probes remain the uncertainty diagnostics. If the mechanism works, top-p will be reintroduced as data augmentation and horizon calibration.

## Run the probes

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# install PyTorch appropriate for your platform

python prepare_data.py
python train_teacher.py --steps 1000

python probe_anchor_horizon.py --contexts 256 --rollouts 16 --horizon 6
python probe_bestofn_blocks.py --contexts 256 --rollouts 16 --horizon 6
python probe_anchor_information.py --contexts 256 --anchors-per-context 32
```

## Run the one-pass student

```bash
python train_onepass_anchor_student.py prepare
python train_onepass_anchor_student.py train --steps 1200
python benchmark_onepass_anchor.py --dev-prompts 64 --test-prompts 128 --count 64
```

Quick tests:

```bash
PYTHONPATH=. pytest -q
```

## Interpretation

The current evidence supports a narrow statement:

> observing one realized autoregressive token substantially reduces uncertainty about the following token, but exploiting that information efficiently requires the continuation predictor to be integrated into the same backbone pass rather than invoked as a second full-model step.

The one-pass experiment tests the second half of that statement.
