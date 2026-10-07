# All-layer low-rank fusion probe

This experiment isolates the representation-capacity question before changing
the RL objective again.

The ordinary AR evaluator/backbone remains frozen.  The existing continuation
student reads only the final transformer representation.  The treatment adds a
separate continuation-only path:

```text
layer 1 last-position hidden -- rank-8 \
layer 2 last-position hidden -- rank-8  +--> concat --> small MLP --> residual
...                                            |
final frozen AR hidden ------------------------+--> existing H=4 continuation head
```

The exact AR next-token logits are unchanged.

## Paired protocol

For each archived seed 48017/48018/48019:

1. load the same corrected Gutenberg-body-v2 teacher and distilled source student;
2. create two exact-output-equivalent copies at step zero;
3. control: continue distilling the existing final-layer continuation head;
4. treatment: continue the same distillation but add all-layer rank-8 fusion;
5. use exactly the same train contexts, teacher samples, minibatch order, step count and learning rate;
6. evaluate both on the same dev contexts and the same teacher-sampled full three-token tails.

No RL update is performed in this stage.  The test split is not used.

Primary metric: teacher-to-student forward KL for the full three-token
continuation.  Lower is better.

**Predeclared Go rule:** only proceed to an RL experiment if the fusion
treatment has lower mean dev forward KL than the continued-baseline control in
all three seeds.  Confidence intervals are reported but are not used to
retroactively alter the rule.

This probe intentionally gives the fusion treatment extra trainable parameters;
it tests whether broader layer access can improve representational fidelity,
not parameter-matched efficiency.
