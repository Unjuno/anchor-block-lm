# Dynamic boundary LoRA + constrained RL

This is the direct test of the adaptive-block hypothesis.

## Model roles

- **Frozen evaluator / AR backbone**: never updated.
- **Continuation LoRA**: updated by unfiltered teacher KD on states actually visited by the current actor.
- **Length-policy LoRA**: chooses `k in {1,2,3,4}` and is updated by constrained REINFORCE.
- Maximum horizon H=4 is an implementation ceiling, not a fixed committed block size.

Confidence, entropy and hidden-state features are policy observations only. They are never used as the reward.

## Learning loop

For every round:

1. generate/recollect states with the *current* continuation model and current length policy;
2. train continuation LoRA on those states against the frozen teacher's full H=4 future samples;
3. train the length-policy LoRA using progress/cost reward minus an adaptive teacher-risk penalty;
4. update a Lagrange multiplier from the selected teacher risk;
5. re-evaluate the exact same fixed dev probe contexts.

Thus the block boundary can move as the actor changes. We save the selected k for every fixed probe context, not just an aggregate histogram.

## Quality constraint

The risk budget is fixed **before development evaluation** from a separate train-only probe: the initial policy's mean selected prefix reverse-KL estimate. Dev never sets the constraint.

The dual controller raises the divergence penalty when selected risk exceeds this budget and lowers it when the risk is below budget.

## Predeclared development interpretation

Three archived seeds are run independently.

A seed is a **speed/quality improvement** only if all of the following hold from round 0 to the final round:

1. fixed-dev-probe mean committed k increases;
2. fixed-dev-probe selected teacher risk increases by no more than 0.05 nat per decision;
3. final free-running sequence reverse-KL increases by no more than 0.02 nat/token.

We report the fraction of fixed probe contexts whose selected k changes, even when the mean or histogram happens to stay constant.

A **proxy-mismatch warning** is raised descriptively if the training selected constrained reward improves while the fixed-dev teacher risk or free-running sequence KL materially worsens. This is evidence of objective mismatch, not by itself proof of adversarial reward hacking.

The test split remains untouched. No held-out confirmation or production latency claim is made in this development experiment.
