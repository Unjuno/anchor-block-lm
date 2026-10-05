# Anchor Block LM

A research prototype for **adaptive language-model decoding**:

> keep the ordinary autoregressive token as an exact anchor, then use the same backbone state to emit a calibrated variable-length continuation block.

The working hypothesis is that full autoregressive decoding is most valuable at uncertain branching points. Once one token is realized, some following tokens become easier to predict and can sometimes be committed without another full backbone step.

## One-pass design

~~~text
context
   |
one frozen backbone forward
   |
   +--> exact ordinary next-token logits --> anchor
   |
   +--> lightweight continuation branch
        conditioned on (hidden state, realized anchor)
            |
            +--> continuation token
            +--> continuation token
            +--> ...
            +--> <EOB>
~~~

The autoregressive backbone and next-token path are fully frozen. Trainable low-rank updates exist only in the continuation branch. If the continuation branch stops immediately, the model reduces exactly to ordinary AR decoding for that step.

This is intentionally different from a two-model draft-and-verify loop: inference does not call a separate teacher to verify each block.

## Current evidence

### Synthetic mechanism test

The original character-level controlled experiment showed that a variable EOB policy can outperform a fixed continuation length at a similar backbone-call compression rate.

See [experiments/synthetic](experiments/synthetic).

### Natural-language BPE test

A small byte-level-BPE nanoGPT experiment on *Romeo and Juliet* now reproduces the mechanism.

The current preselected expanded test uses 64 held-out prompts × 48 generated BPE tokens. The EOB bias and random-gating probability were fixed before this expanded test.

| Method | Tokens / backbone call | Local teacher agreement |
|---|---:|---:|
| exact AR fallback | 1.000 | **100.00%** |
| fixed +1 continuation | 2.000 | 64.29% |
| **variable EOB** | **1.103** | **98.11%** |
| nearby-rate random 0/1 gating | 1.128 | 93.33% |

Variable gating improves prompt-paired teacher agreement over random gating by **+4.79 percentage points**, bootstrap 95% CI **+3.65 to +5.89 pp**.

Three independent teacher/student/calibration seeds also reproduce the effect:

| Metric | Mean ± sample std |
|---|---:|
| variable tokens / backbone call | **1.084 ± 0.032** |
| variable teacher agreement | **98.81% ± 0.47%** |
| random-gating teacher agreement | 93.83% ± 0.82% |
| variable - random agreement | **+4.98 ± 1.02 pp** |

Across all three seeds, the AR fallback remained exactly teacher-equivalent and variable gating beat random gating in teacher agreement.

See [experiments/bpe_probe](experiments/bpe_probe) for the full experiment history, failed variants, uncertainty probes, and reproduction commands.

## Important limitations

These results are **not** a production LLM speed benchmark.

They currently establish only an algorithmic mechanism:

- backbone-call count is reduced on a subset of states;
- the exact AR fallback is preserved;
- adaptive gating is better than random gating at a nearby compression rate.

They do **not** yet establish:

- GPU wall-clock speedup,
- exact preservation of the teacher sampling distribution,
- results on strong or large language models,
- cross-dataset generalization,
- optimized KV-cache behavior or kernels.

The natural-language teacher is deliberately tiny and the current corpus split has measurable distribution shift.

## Why an anchor helps

A same-position information probe asks how much knowing the realized token X_(t+1) changes uncertainty about X_(t+2).

On 256 held-out BPE contexts with 32 sampled anchors per context:

- conditional mutual information: **1.083 nats / 1.562 bits**
- teacher top-1 probability: **13.8% -> 24.6%**
- positive information gain in **99.6%** of contexts

This does not by itself imply that long blocks are easy, but it validates the central premise that the realized anchor contains useful information about the immediate future.

## Self-distillation

The current training path separates two problems:

1. **continuation content** — learn future token heads from the frozen AR teacher;
2. **commit horizon** — calibrate EOB from the student's actual consecutive agreement with the teacher under student-induced prefixes.

Top-p sampling is retained as an uncertainty probe and future source of distribution-aware training signals. Earlier experiments that directly used strict top-p rollout-prefix agreement were too conservative on natural language; those negative results are kept in the repository.

## Reproduce

Synthetic experiment:

~~~bash
cd experiments/synthetic
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -q
bash reproduce_synthetic.sh
~~~

Natural-language BPE experiment:

~~~bash
cd experiments/bpe_probe
python prepare_data.py
python train_teacher.py --steps 1000

python train_onepass_anchor_student.py prepare
python train_onepass_anchor_student.py train --steps 1200

python calibrate_onepass_eob.py prepare
python calibrate_onepass_eob.py train --steps 500

python benchmark_onepass_preregistered.py --test-prompts 64 --count 48
~~~

## Current status

- [x] Synthetic feasibility experiment
- [x] Natural-language BPE feasibility experiment
- [x] Exact frozen AR fallback
- [x] One-pass anchor-conditioned continuation branch
- [x] Learned/calibrated variable EOB
- [x] Nearby-rate random-gating baseline
- [x] Prompt-paired bootstrap
- [x] Three independent training seeds
- [x] CI and regression tests
- [ ] Better natural-language dataset split / additional corpus
- [ ] Stronger teacher
- [ ] KV-cache + GPU benchmark
- [ ] Kernel-level wall-clock optimization
- [ ] Larger-model validation

## Relationship to prior work

The project is related to multi-token prediction, blockwise parallel decoding, sequence-level/self-distillation, speculative decoding, adaptive computation, and optimal stopping.

The intended research direction emphasizes an **exact autoregressive anchor path plus a separately trainable, confidence-calibrated continuation branch**.

## Implementation

The first implementation uses a minimal subset of [nanoGPT](https://github.com/karpathy/nanoGPT). The upstream MIT license is preserved under experiments/synthetic/third_party/nanogpt.

## License

MIT License. See [LICENSE](LICENSE).
