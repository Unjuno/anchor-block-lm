# Fixed-K joint distillation + binary LoRA gate

This experiment follows the revised concept: **K is always 4**. A block includes one ordinary AR anchor and three continuation tokens. The only learned decision is AR-only versus committing the complete fixed block. There is no EOB, length controller, best-of-N target selection, or inference-time teacher verification.

## Distribution target

The teacher is sampled at temperature 1 with **no truncation (top-p=1)**. This is deliberate: top-p=0.95 would define a different, truncated teacher. Every sampled content target is retained, independently of safety/gating decisions. Greedy targets and majority-prefix filtering are not used.

The continuation is a normalized mixture of four product-categorical components. One shared component is sampled for the entire three-token tail, so the joint need not factor into independent marginal predictions. All continuation slots are computed in parallel. This finite mixture still approximates the teacher joint: fixed K does NOT guarantee exact distribution preservation.

The anchor's logits come from the unchanged nanoGPT backbone. The block branch is conditioned on its hidden state and the realized anchor. Content training updates a low-rank residual and new continuation heads. Content is then frozen and hashed. During reward learning **only gate LoRA A/B tensors are updated**.

## Gate learning and decision order

The gate observes prefix features, the realized anchor, and predicted-distribution summaries. It does not inspect a sampled candidate tail. Reward trades three saved sequential steps against an offline Monte Carlo estimate of the conditional reverse KL of the student tail relative to the full teacher tail. Individual log-ratio samples can be negative; they are not silently clamped.

Training is sampled-action REINFORCE on a contextual-bandit bank, not full long-horizon language-model RL. The exchange rate is set using training data only. Online deployment uses a deterministic threshold on the learned gate logit calibrated from dev scores, not the raw stochastic training policy. Any distribution claim must account for approximation error in the block branch.

## Frozen evaluation protocol (before the clean-BPE run)

- Existing tiny nanoGPT: 2 layers, 4 heads, width 64, context 64; CPU only.
- New corrected `gutenberg-body-v2` source, one train-only 1024-token BPE tokenizer shared by all seeds. Legacy checkpoints/results are never loaded.
- Teacher/content/gate steps: 600 / 1200 / 600. Seeds: 47017, 47018, 47019.
- Every content-training context contributes eight unfiltered teacher samples.
- Primary diagnostic: 64 non-overlapping held-out context windows and one teacher-sampled anchor per window. Estimate each conditional student-tail KL with 64 independent samples.
- Select exactly 8 full blocks, using learned scores, a marginal-entropy confidence rule, or uniformly random exact-size allocation. All policies have 64 backbone calls and 88 emitted tokens in this common-state diagnostic.
- Primary success: upper 95% nested-bootstrap bound for learned-minus-uniform selected KL is below zero. Confidence-rule comparison is separately reported. Do not change coverage or acceptance rules after seeing the test.
- The equal-budget diagnostic ranks an existing bank of contexts. It is an **offline allocation diagnostic**, not an online generation policy or latency guarantee. The uniform result is the expectation over exact-eight-block allocations; 32 actual random allocations are also summarized.
- Separate free-running diagnostic: dev-calibrated gate on 8 prompts, 48 tokens each, three timing repetitions. It imposes no quota. Whole blocks or AR tokens only; when fewer than four tokens remain, use AR rather than truncate a block. Realized workloads may differ and are reported as such.
- Save source/tokenizer/split hashes, checkpoints, test context offsets, candidate log-ratios, scores, full result JSON, and hardware/software conditions.

## Run

From this directory, after installing torch 2.10.0+cpu, numpy 2.3.5, tokenizers 0.22.2 and pytest:

```bash
PYTHONPATH=. pytest -q tests
python ../bpe_probe/prepare_data.py --out data --vocab-size 1024
python run_experiment.py --data data --out outputs/seed-47017 --seed 47017
```

Repeat with fresh output directories for the other two seeds; do not retrain the tokenizer between seeds. The dedicated workflow performs this bounded three-seed run. The synthetic `--smoke` option is explicitly a separate artificial grammar, not a replacement for BPE validation.

## Limits

Positive routing results do not establish sufficiently small absolute distribution error, task quality, exact teacher sampling, or production speedup. A frozen K alone does not prevent distribution changes. A prompt/Monte Carlo bootstrap does not replace independent training seeds, and three seeds on one play do not establish broad generality. The small head is an additional compute cost even when the gate falls back to AR.
