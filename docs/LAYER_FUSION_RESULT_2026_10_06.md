# All-layer low-rank MLP: completed distillation probe, 2026-10-06

## Decision

Three archived tiny-nanoGPT teacher/student seeds were evaluated. All-layer fusion has a lower point estimate in two seeds and a slightly higher one in the third. The predeclared rule in experiments/layer_fusion_probe/README.md requires a lower mean dev forward KL in ALL three seeds before adding RL. That rule was not met. **No new RL training, free-running evaluation, or latency benchmark was performed in this stage.** This is not evidence of reward hacking, and not a rejection of all-layer adaptation in general.

## Architecture and controlled comparison

The teacher/AR backbone is frozen: 169,728 parameters, two transformer layers, four heads, width 64, context 64, BPE vocabulary 1024. Each last-position layer representation is projected from 64 to 8 dimensions. The two projections are concatenated and processed by an MLP with hidden width 64; its residual feeds the continuation branch only. This is a low-rank nonlinear side network reading all layers, NOT LoRA updates to every backbone weight.

Maximum prediction remains four total tokens (exact AR anchor plus three continuation tokens). No committed length is forced; this stage trains full-horizon content without invoking the adaptive policy. The intended 1/2/3/4 adaptive-length inference design remains unchanged.

Both conditions inherit identical archived student weights and initially produce identical scores. Both receive the SAME unfiltered teacher sample bank, minibatch order, 600 updates, batch 64 and learning rate 0.001. Teacher sampling is temperature 1/top-p 1. Training uses 1,536 train contexts x four full-tail samples; evaluation uses 128 dev contexts x 32 samples. The test split is not evaluated. The control continues training the existing final-layer head; the treatment also learns the added fusion network. KD updates all non-backbone continuation parameters: 30,724 control versus 36,996 treatment, including 6,272 extra parameters. This is not a parameter-matched experiment and not LoRA-only RL.

## Results

Teacher-to-student forward KL, nats per complete three-token continuation; lower is better.

| Seed | Before additional KD | Final-layer-only after KD | All-layer fusion after KD | Fusion minus control | Context-bootstrap 95% interval |
|---|---:|---:|---:|---:|---|
| 48017 | 4.365314 | 4.238553 | 4.230192 | -0.008360 | [-0.031300, +0.012414] |
| 48018 | 4.445005 | 4.334666 | 4.336834 | +0.002168 | [-0.022783, +0.026137] |
| 48019 | 4.284284 | 4.032153 | 4.011990 | -0.020163 | [-0.034175, -0.006446] |

Mean control 4.201791; mean fusion 4.193005, a relative reduction of about 0.209%. Only seed 48019 has a reported interval strictly favoring fusion. This is not a statistically established aggregate gain across training seeds. Continued KD alone improves every seed; the incremental all-layer effect is small and not consistent in sign. The predeclared all-three-seed GO rule is false.

Intervals use 4,000 resamples of 128 context means (each averages 32 continuations). Context starts are unique but windows can overlap, so these intervals are approximate and do not incorporate dependence across windows, different datasets, or the full training process. This is dev-set architecture exploration, not a fresh confirmatory test.

## Verification and reproducibility

Execution occurred locally during the current session, not through a new remote training run. CPU INTEL(R) XEON(R) PLATINUM 8573C; FP32, one CPU thread per training process, batch 64, no KV cache, unpinned clock. Three seed runs were concurrent, so their training durations are NOT latency benchmarks. Python 3.13.5, torch 2.10.0+cpu, numpy 2.3.5, tokenizers 0.22.2.

Source export: commit 109d61984039ae66d41f4b71a53d61af060fe359, Actions 37463799897, artifact 11413676823, SHA256 6f77778295d4f2df0e57c835c37106369ce2ffbf3525ef60b57a5d7ace224235.
Initial teacher/student checkpoints: Actions 37324518978, archive SHA256 14f8dcd6ad5574e39d693135c329f66238d6da0153db773fb01edcd7bedd209e. The corrected Gutenberg body-v2 source, saved BPE tokenizer and all split-array hashes were verified. Local execution reused the exact verified source cache through the original loader, with no replacement dataset.

The earlier original-LoRA-rank/fusion-rank loading bug was already repaired in the retrieved source. A second reproducibility issue was found: newly initialized fusion parameters lacked explicit global RNG seeding. Added a failing test, implemented seed_experiment, and reran all three named seeds. One initial unseeded pilot is retained separately but excluded from primary results. No learning rate, step count, architecture, or GO rule was selected from its outcome. The published runner blob 6f9ee689bf60774f8a7acdabf7057edc88e5982b exactly matches the locally executed file.

Six local suites passed: fusion 12, two-model 17, adaptive 21, fixed-horizon dependency 13, BPE/data integrity 37, synthetic 14: **114 passed, no skips**. All experiment Python files compile. The source export initially omitted hidden .github files, causing two BPE workflow-file tests to fail; restoring the exact existing workflow blob b5c80182030b36fe9cfec059c1c938c8bc840494 resolved them without changing those tests.

Raw score arrays were independently recomputed into means, paired differences and bootstrap intervals. Saved teacher and AR backbone tensors are unchanged. Both layers' projection weights genuinely updated. After training, exact AR logits were independently checked on twelve dev inputs per seed and matched bit-for-bit.

## Interpretation

The suggested all-layer MLP is trainable and preserves the original AR path. In this particular tiny two-layer model, it has not demonstrated enough additional distribution-fidelity benefit to pass the agreed gate into RL. Reward hacking remains unassessed here because no reward optimization was performed. General all-layer LoRA, larger numbers of layers, or alternative fusion structures are not tested by this result.

Representation learning, distributional distillation and inference cost remain distinct questions. No speed or semantic-quality preservation claim follows from the reported forward KL. A future reward diagnostic should measure attained reward and both KL directions on fixed states; no such new experiment is claimed here.
