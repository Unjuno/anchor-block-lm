# Natural-language BPE experiments

> **Audit status: historical body-only claims withdrawn.** The original mirrored source used `START OF THIS` markers, but the parser only recognized `START OF THE`. It silently retained the raw file, including headers and publisher/license material. The historical measurements remain in `results/` unchanged; see [the audit](../../docs/RESULTS_AUDIT.md).

## Current data contract

The default source is [Project Gutenberg edition 1513](https://www.gutenberg.org/ebooks/1513). Its catalog labels it public domain in the USA; that is not a worldwide license determination. The original mirrored edition includes a World Library notice and should not have been described indiscriminately as a public-domain text. Neither source is covered by this repository's MIT code license.

`prepare_data.py` now recognizes `THE` and `THIS` marker variants, normalizes line endings for its analysis copy, and rejects absent, reversed, ambiguous or empty boundaries. Known publisher/license notices inside the body trigger an error rather than being silently removed. Original source bytes remain untouched in the local cache.

Cache reuse requires a matching source URL and raw-byte hash. A cache without provenance is rejected. Existing derived data is not overwritten: use a fresh checkout/data directory, and do not mix old banks or model checkpoints with the new corpus.

The tokenizer is trained only on the 80% training portion; dev/test are 10% each by text position. This is still a one-work positional split, not an independent multi-document benchmark. Metadata now records preprocessing version, source hash, cleaned-body hash, tokenizer version/hash, split offsets and text/array hashes. The initial byte alphabet is sorted and tokenizer parallelism is disabled, but reproducibility still requires matching source bytes, tokenizer and runtime versions.

## Setup and offline regression tests

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q
```

The data-integrity tests use invented miniature text and a mocked download. They neither download a corpus nor train a language model. The integration test trains a tiny tokenizer when `tokenizers` is installed.

## Corrected pipeline — training results not yet released

Run in a fresh checkout. To enforce an existing source snapshot, add `--expected-source-sha256` with the recorded raw-byte digest to the data-preparation command. The default download URL itself is not content-pinned.

```bash
python prepare_data.py
python train_teacher.py --steps 1000
python train_onepass_anchor_student.py prepare
python train_onepass_anchor_student.py train --steps 1200
python calibrate_onepass_eob.py prepare
python calibrate_onepass_eob.py train --steps 500
python benchmark_onepass_preregistered.py --test-prompts 64 --count 48
```

This last script is only a **legacy fixed-policy diagnostic**. Its EOB bias and random-gating probability came from earlier exploration. Its filename does not make the run preregistered, its reused test seed does not create an independent holdout, and its comparator does not enforce equal test-time call counts. Do not use it as the final acceptance test for a new scientific claim.

## Architecture and interpretation

One frozen backbone forward produces a greedy anchor. A continuation-only low-rank residual and small dense heads predict later tokens and EOB. Calibration uses teacher labels offline; there is no per-block teacher-verification loop at inference.

The former two-backbone-call design could waste a call on immediate EOB. That motivated the one-pass implementation; it does not prove that every two-call design is intrinsically invalid.

Top-p trajectory, best-of-N and same-position information probes remain available in their `probe_*.py` scripts. Their historical numbers share the data issue. Positive empirical information gain alone does not establish useful long blocks or superiority over other multi-token predictors.

## Required next validation at the same small scale

Freeze preprocessing, tokenizer, quality thresholds and evaluation protocol before examining a fresh holdout. Retrain the teacher, continuation heads and EOB calibrator; compare learned and random gating at equal realized backbone-call budgets. Report continuation-only errors, complete-block agreement, per-prompt call counts, seeds and uncertainty. Wall-clock claims require separate measured latency with hardware, thread count, clock policy, batch and cache settings.

No clean-corpus performance improvement, equal-budget advantage or wall-clock speedup is claimed by the preprocessing repair.
