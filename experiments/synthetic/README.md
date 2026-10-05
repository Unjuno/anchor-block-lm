# Synthetic anchor-block experiment

This directory contains the current controlled proof-of-concept for Anchor Block LM.

It trains a tiny nanoGPT teacher, samples anchor-conditioned continuations with top-p sampling, converts rollout agreement into variable-length `token ... <EOB>` targets, distills the block predictor with LoRA while keeping the base model frozen, then refreshes the student on its own rollout states.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run commands from this directory.

## Reproduce

```bash
python train_teacher.py --steps 1600
python anchor_block_experiment.py prepare
python eob_experiment.py prepare
python eob_experiment.py train --steps 1200
python onpolicy_eob_refresh.py prepare
python onpolicy_eob_refresh.py train 300
python benchmark_onpolicy.py
```

For a quick correctness check:

```bash
PYTHONPATH=. pytest -q
```

## Label definition

For each already-anchored context:

1. draw 16 continuation rollouts from the frozen teacher with `top_p=0.95`, `temperature=1.0`;
2. follow the observed modal prefix path;
3. measure the joint probability mass of samples still matching that prefix;
4. keep the longest contiguous prefix whose empirical mass is at least `0.80`;
5. append `<EOB>` immediately after that prefix.

This is an empirical self-distillation target, not a proof of distribution preservation.

## Current result

The checked-in result snapshot is in `results/published_result.json`.

The published result comes from CPU / FP32 / batch 1 / no KV-cache synthetic character-level experiments. It is a mechanism test, not a production benchmark.

## Upstream attribution

`third_party/nanogpt/model.py` is adapted from Karpathy's nanoGPT and is distributed under its MIT license. See `third_party/nanogpt/LICENSE`.
