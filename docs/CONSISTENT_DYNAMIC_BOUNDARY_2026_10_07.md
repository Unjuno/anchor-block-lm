# Dynamic-boundary LoRA RL: consistent policy/dual continuation (2026-10-07 JST)

## Result

Continued training and free-running evaluation of all three archived actors completed locally. This is NOT a successful low-degradation speedup: teacher-distribution discrepancy fell in point estimates, but emitted lengths generally shortened and final latency remained worse than pure AR. Maximum prediction horizon remains four total tokens, and actual committed length remains variable from one to four. No inference-time teacher verification or forced length quota was introduced.

## Correction and scope

In the previous dynamic-boundary implementation, REINFORCE sampled a categorical length action, but the dual update evaluated only argmax. This can omit the risk of exploratory actions. The new code uses the SAME pre-update categorical probabilities for the policy update and expected-risk dual update; fresh actor-state collection also samples that policy. Adam moments now persist across rounds. The archive had no optimizer state, so these optimizers were initialized once at continuation start.

The fixed teacher and exact AR path are unchanged. Only 7,200 continuation-LoRA parameters and 812 length-policy-LoRA parameters change. Content is trained by unfiltered full-horizon teacher-sample KD, NOT content RL; the length policy is trained by REINFORCE. No all-layer fusion was added in this continuation. Confidence is a policy observation, not the reward.

Resume the final checkpoints from Actions run 37494321837; original teachers are from run 37324518978. Four rounds, 64 content plus 128 policy updates per round (256/512 total per seed), batch32, four teacher samples. Fresh states: 32 train starts times four cycles, recollected again after content updates. Learning rates .0003/.001; dual step .1. Prior train-derived quality targets, initial duals, and action-cost proxies are retained. No dev-based checkpoint or hyperparameter selection. This bundles continuation with several consistency changes, so it does not isolate the causal contribution of the dual change alone.

## Fixed 64-context development probes

The same contexts and sampled anchors are evaluated initially and after each content/policy phase, using 32 fresh tail samples per context. Risk is nats per committed decision, not an error percentage.

| Seed | Expected length before -> after | Argmax boundary changes | Expected teacher risk before -> after |
|---|---:|---:|---:|
| 48017 | 1.802717 -> 1.535465 | 10/64: 1 longer, 9 shorter | 0.878791 -> 0.429006 |
| 48018 | 1.578182 -> 1.578169 | 0/64 | 0.428945 -> 0.408816 |
| 48019 | 1.748440 -> 1.621269 | 5/64: 1 longer, 4 shorter | 0.761668 -> 0.553156 |

The second actor's probabilities and parameters changed despite unchanged argmax boundaries. Length growth is not forced or assumed monotonic.

## Free-running results with the stochastic policy used in training

16 dev prompts, four draws each, 48 tokens per draw. Initial/final comparisons use the same saved models and prompts, not a random-gating baseline.

| Seed | Tokens/backbone call before -> after | Augmented-trace discrepancy before -> after (nat/token) | Final speed/pure AR |
|---|---:|---:|---:|
| 48017 | 1.77984 -> 1.54062 | 0.50671 -> 0.33106 | 0.9354 |
| 48018 | 1.52684 -> 1.49927 | 0.28156 -> 0.26103 | 0.9348 |
| 48019 | 1.59834 -> 1.48621 | 0.41615 -> 0.34068 | 0.9219 |

For stochastic lengths the reported discrepancy is reverse KL of an augmented token/length trace against a teacher process with the same length policy. Policy factors cancel. Its expectation upper-bounds token-marginal KL; it is NOT an exact token-marginal KL measurement, nor does each finite sample necessarily upper-bound anything. It is not human-rated semantic quality.

Argmax deployment is evaluated separately for compatibility with earlier work. Its token-sequence reverse KL changes are 0.436749 -> 0.292720, 0.275885 -> 0.254621, and 0.330054 -> 0.291105 nat/token. Only the first seed's prompt-cluster 95% interval for that change excludes zero. Corresponding tokens/call decrease from 1.676856 -> 1.484775, 1.515540 -> 1.488372, and 1.513300 -> 1.444288.

Timing: AMD EPYC 9V74 80-Core Processor, one CPU thread, FP32, batch1, no KV cache, clock not pinned. All training ended before sequential exclusive timing. Eight prompts times 48 tokens, five repeats, excluded warmup, alternating method order; each method replays identical RNG/traces. Teacher scoring excluded. Pure AR bypasses continuation and policy computation. LoRA is merged into deployment copies. Final stochastic actor times are 0.291641, 0.291125, 0.290371 seconds versus pure-AR 0.272815, 0.272129, 0.267694 seconds (medians). This is not a production GPU benchmark.

## Constraint and reward diagnostics

Independent 32-sample risk estimates on 128 freshly collected train states:

| Seed | Unchanged budget | Final expected risk | 95% trajectory-cluster interval |
|---|---:|---:|---|
| 48017 | 0.548822 | 0.461850 | [0.350753, 0.571768] |
| 48018 | 0.389804 | 0.322693 | [0.225069, 0.433300] |
| 48019 | 0.531808 | 0.565507 | [0.439316, 0.704094] |

Two point estimates meet the original budget; all three intervals cross it. Constraint satisfaction is therefore NOT statistically established. The constraint is per decision, not a full-sequence guarantee. The retained greedy development screen (longer fixed-probe mean length, risk increase <= .05 nat/decision, sequence KL increase <= .02 nat/token) passes 0/3.

A reference reward is recomputed at every fixed probe using the FROZEN initial dual coefficient, separate from the changing training dual. It rises in all three point estimates while probe risk falls. Shorter outputs explain this as a quality/progress tradeoff; this experiment does not establish reward hacking or prove its absence.

## Verification and provenance

New local tests: 10 passed. Six related CI suites: 14 + 37 + 13 + 21 + 17 + 17 = 119 passed, no skips; Python compilation passed. CI run 37499394698, code commit 22c492265668b899a5060452e207d7e7c97a7218. Downloaded tracked-source export matches the locally executed new code, tests and imported dependencies byte-for-byte. Red/green logs are retained.

All three checkpoint ZIP SHA256 values match GitHub artifact digests. Teacher, AR backbone, and non-LoRA tensors remain unchanged. Continuation and policy change, with persistent optimizer counters 256/512. Post-training AR logits match the teacher bit-for-bit on checked inputs. Raw arrays independently reproduce reported risks, fixed-coefficient rewards, output counts and calls.

Local tokenizers was absent: execution used previously saved train/dev arrays whose exact bytes matched the original tokenizer/split manifest. Source, extracted body, tokenizer, split text and array SHA256 values were verified; no dataset replacement. The standard prepare command remains available with tokenizers 0.22.2 installed. Test was not used in updates or evaluation.

Limitations: continued development on one reused play/dev split, overlapping context windows, only three training seeds, Monte Carlo risk estimates, finite-mixture continuation approximation, local bandit-style policy objective rather than long-horizon sequence RL. No general or aggregate statistical superiority claim follows.
