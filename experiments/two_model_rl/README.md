# Two-model on-policy LoRA experiment

Approved question: with a fixed nanoGPT evaluator, does updating the continuation generator through LoRA as well as the variable-length policy improve on policy-only updates?

The maximum prediction horizon remains four total tokens, while deployed length is confidence-informed and may be 1, 2, 3 or 4. No EOB, candidate reranking, fixed block requirement or inference-time teacher verification is introduced.

## Fixed protocol before results

Start from the three saved checkpoints in Actions run 37324518978, not from a newly chosen favorable teacher. Teacher, source text, tokenizer and split hashes are verified. The canonical tiny nanoGPT remains 2 layers, width 64, context 64, vocabulary 1024. This is exploratory reuse of a previously inspected held-out corpus, not a new confirmatory test.

Compare pure AR, the saved initial policy, policy-only on-policy RL, and joint continuation-LoRA plus policy learning. Both learning arms start at the same saved weights and use the same four-round, 64-update-per-round budget and hyperparameters. Each round recollects states from the current actor using training data only. The joint arm has additional evaluator work for content learning, so equal update counts are not equal total training compute. RNG streams and visited states can differ once the arms differ.

The evaluator stays entirely frozen. The original AR anchor path stays frozen. Existing head weights stay frozen; only 7,200 continuation A/B parameters and 812 categorical-policy A/B parameters are trainable in the joint arm. Only the latter are trainable in the policy-only arm.

Content objective: on-policy full-tail reverse-KL score-function REINFORCE plus 0.5 times the NLL of independent untruncated teacher-sampled tails. All three continuation slots are always trained, including beyond the deployed stopping point. No difficult-context filtering, top-p truncation or greedy-only target selection. A leave-one-out baseline excludes the scored sample.

Policy reward: emitted token count minus a measured initial action-cost estimate (in pure-AR step units), minus the existing training-only KL penalty times prefix distribution error. Initial measured costs are shared by both arms and frozen. This is a latency-informed proxy, not online optimization of wall-clock measurements. Policy learning is contextual-bandit REINFORCE rather than long-horizon sequence PPO.

Evaluate 16 saved held-out contexts x 4 stochastic samples x 48 output tokens per condition per seed. Report full-sequence log Q/P, prompt-cluster intervals, length histograms and tokens per backbone call. No claim of equal-quality/equal-throughput dominance is predefined. Fixed-step final checkpoints are evaluated without test-time tuning. Measure pure AR without extra heads; five timing replays per condition use identical within-condition random seeds and traces. Offline evaluator scoring is excluded. LoRA projections are merged into deployment copies before timing.

## Run

Download and extract the `adaptive-k-rl-evidence` artifact from run 37324518978 as `saved-evidence`, then run from this directory:

```bash
PYTHONPATH=. pytest -q tests
python run_two_model.py --artifact-root ../../saved-evidence --out results/seed-48017 --seed 48017
```

The workflow runs seeds 48017/48018/48019 independently and retains trained weights, per-sequence arrays, histories and protocol manifests. No large model or GPU scale-up is involved. Results are not yet claimed by this initial protocol.
