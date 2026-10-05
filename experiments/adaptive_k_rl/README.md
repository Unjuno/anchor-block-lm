# Adaptive-k RL policy experiment

This experiment keeps the **maximum distillation horizon fixed at H=4**, but lets a separate reward-trained policy choose how many tokens to commit at each state: `k in {1,2,3,4}`.

The distilled block generator is trained on every sampled teacher context and the full H=4 horizon. The chosen `k` does **not** filter or alter self-distillation data. After distillation the entire generator is frozen and hashed. Only LoRA tensors in the categorical commit-length policy are updated with contextual-bandit REINFORCE.

Reward:

`(k - 1) - lambda * estimated_prefix_reverse_KL`

`k=1` is the exact frozen AR anchor path. Larger `k` commits a longer prefix of the same already-distilled H=4 joint prediction. There is no EOB, no teacher verification during inference, and no best-of-N or beam reranking.

Primary outputs:

- mean selected `k` / tokens per backbone call on a held-out common-state bank;
- selected-prefix reverse-KL estimate;
- `k=1..4` histogram;
- mean predictive uncertainty for states assigned to each `k`;
- unforced free-running tokens/backbone-call and CPU timing versus ordinary AR.

This remains a tiny nanoGPT mechanism experiment, not a production speed benchmark.
