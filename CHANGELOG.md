# Changelog

## Research closeout — 2026-10-07

Consolidated the completed small-model experiment stack and all-layer distillation diagnostic without changing learning algorithms or re-running training. Added a claim ledger, explicit scope/status, theory with speed and KL derivations, a reproducibility guide, an experiment map and citation metadata. Kept original numeric reports and negative findings.

Corrected timing interpretation: use the timing trace's own committed-token count; effective overhead is not a head profile; inverse speed is the latency ratio. Larger-model acceleration is conditional and untested. Historical wrapper-contaminated BPE results are not headline evidence.

Added a standard-library timing-arithmetic checker and regression tests, plus integration CI for the seven existing suites. Preserved MIT/nanoGPT attribution and established file paths. No production-speed, exact-distribution, reward-hacking or full-RL equivalence claim was added. Existing experimental branches and discussions remain historical provenance.
