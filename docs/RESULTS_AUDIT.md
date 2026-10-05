# BPE data and evidence audit — 2026-10-05

## Disposition

Historical BPE JSON snapshots are preserved unchanged as execution records. Their interpretation as clean, body-only natural-language validation is withdrawn. No replacement training measurements are supplied by this patch. The synthetic experiment is a separate controlled corpus and is not invalidated by this particular source-parser defect.

## Confirmed defects

### Gutenberg boundary mismatch

The [historical source](https://github.com/phymooc/learn-python/blob/f0b657e7ec33de117d1ba5b5e8ae56c523ffc442/code/Romeo.txt) has a `START OF THIS PROJECT GUTENBERG` marker. The [old parser](https://github.com/Unjuno/anchor-block-lm/blob/f54e0a95c637d72ae0b756d713ecf277abd0e554/experiments/bpe_probe/prepare_data.py) recognizes only `START OF THE PROJECT GUTENBERG`. If either boundary is not recognized, it returns the complete source. Thus the corpus was not body-only; the source header and license appendix were retained before the positional split.

The amount of non-play text in individual legacy evaluation prompts has **not** been measured in this audit. Do not invent a contamination percentage or assume all prompts were affected identically. The retained non-play text is enough to invalidate the body-only claim, not to prove that every numeric measurement is fabricated or that the mechanism cannot work.

### Source description

The mirrored source also contains World Library copyright/noncommercial notices, including material within the nominal book body. Calling that edition simply public domain was not justified. The new default is [Project Gutenberg 1513](https://www.gutenberg.org/ebooks/1513), whose catalog labels it public domain in the USA. Original bytes/notices remain local and unchanged. This is source provenance, not legal advice or an assertion about all jurisdictions.

### Workflow paths

One path-filter string accidentally joined two Python filenames with a space. The fixed-policy benchmark was also absent from the compile check. Separate paths and a compile regression are added; concurrency cancels superseded runs of the same workflow/ref. The workflow now retains preprocessing metadata and the source manifest alongside result JSON.

## Interpretation corrections

**Unequal work:** historical variable and random policies had different realized tokens-per-call values. More exact teacher anchors and fewer approximate continuation tokens can themselves improve aggregate agreement. The published confidence intervals describe a difference between those particular policies; they do not remove the unequal-budget confound. The previous claim that this comparison rejects a random-emission explanation is withdrawn.

**Reused test material:** the fixed-policy diagnostic uses the same test-context seed, 11102, as earlier exploration. Freezing parameters for a subsequent invocation is not prospective registration of an independent holdout. Existing `preregistered` filenames are compatibility labels only. Three new training seeds still reuse the corpus and protocol; they cannot repair this issue.

**Quality proxy:** local teacher agreement conditions on each method's own prefix and includes anchors. It does not measure task correctness, human quality, sequence identity with an independent teacher rollout, or sampling-distribution preservation. Low teacher NLL can also arise from a different, easier generated trajectory.

**Efficiency proxy:** fewer backbone calls are not measured latency. Continuation heads, context processing, KV-cache work and runtime overhead are omitted from that count. Legacy artifacts do not contain enough hardware/clock detail to reconstruct a production benchmark.

**Information probe:** empirical entropy reduction concerns the particular sampled-anchor mixture. Entropy concavity makes nonnegativity unsurprising; a positive sign is not a novelty test, a guarantee for every anchor, or evidence of useful long continuation blocks. Finite anchor sampling and corpus contamination limit quantitative interpretation.

## Repair and verification scope

The parser now recognizes both marker wordings and fails closed for absent/ambiguous/reversed markers, empty bodies and remaining known publisher notices. Cache provenance and optional expected raw-byte hashes are checked. Existing derived outputs are not silently overwritten. Preprocessing v2 records cleaned-body, tokenizer and split fingerprints and must be used in a fresh checkout without legacy banks/checkpoints.

Offline tests reproduce the previous parser/workflow failures and verify the repair. A miniature tokenizer integration test checks that encoded splits reconstruct the intended body and that metadata hashes match actual files. These tests are software/data-contract evidence; they are not a replacement for training the model on the corrected corpus.

## Proposed next protocol — not executed here

| Item | Requirement |
|---|---|
| Hypothesis | Learned gating reduces continuation errors relative to context-independent gating at equal realized call budgets. |
| Minimum experiment | Existing tiny nanoGPT only; corrected source, one frozen tokenizer, independent training seeds and fresh held-out evaluation contexts. |
| Decision rule | Freeze quality and call-reduction thresholds before evaluation; require the matched-budget comparison and uncertainty interval to meet them. No retrospective threshold relaxation. |
| Competing explanation | Benefits disappear with cleaned text, equal anchor/continuation budgets, or a genuinely new holdout. |
| Uncertainty | Corpus/source choice, tokenizer identity, overlapping context windows, teacher/student/calibration seeds, random-gate seeds and hardware variability. A prompt bootstrap alone does not cover all of these. |

The computational footprint remains nanoGPT-scale. Large-model training is outside this repair's scope.
