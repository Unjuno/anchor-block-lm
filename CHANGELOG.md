# Changelog

## Related-work and novelty audit — 2026-10-07

Added a focused primary-source audit covering CLP, AdaMTP, EntMTP, K-Forcing, Parallel Token Prediction, MTP-RL, joint MTP-RL/OCC, Medusa and Hydra. The audit narrows the project's public novelty language: "backbone first token + adaptive extra-token length" is already covered closely by CLP, and policy-aligned MTP dynamics during RL are covered by MTP-RL.

The repository now positions its distinct research object more narrowly as the combination of full-horizon continuation distillation on all actor-visited contexts, a separate constrained-RL commit-length LoRA, repeated state recollection as continuation competence changes, and verifier-free direct commitment. No priority, world-first, lossless-acceleration, or state-of-the-art claim is made. No model training, hyperparameter change, or new performance measurement was performed in this audit.

## Research closeout — 2026-10-07

Consolidated the completed small-model experiment stack and all-layer distillation diagnostic without changing learning algorithms or re-running training. Added a claim ledger, explicit scope/status, theory with speed and KL derivations, a reproducibility guide, an experiment map and citation metadata. Kept original numeric reports and negative findings.

Corrected timing interpretation: use the timing trace's own committed-token count; effective overhead is not a head profile; inverse speed is the latency ratio. Larger-model acceleration is conditional and untested. Historical wrapper-contaminated BPE results are not headline evidence.

Added a standard-library timing-arithmetic checker and regression tests, plus integration CI for the seven existing suites. Preserved MIT/nanoGPT attribution and established file paths. No production-speed, exact-distribution, reward-hacking or full-RL equivalence claim was added. Existing experimental branches and discussions remain historical provenance.
