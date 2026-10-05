# Anchor Block LM

A nanoGPT-scale research prototype: emit an ordinary autoregressive anchor, then predict a variable-length continuation from the same backbone state.

> **Data-integrity correction — 2026-10-05:** the historical BPE preprocessing did not recognize the source's Gutenberg boundary markers. Headers and publisher/license material were retained. The archived BPE measurements are not validated body-only results. Clean-corpus retraining and an equal-budget comparison remain outstanding. See [the audit](docs/RESULTS_AUDIT.md).

## Mechanism

```text
context -> one frozen backbone pass -> ordinary greedy anchor
                                    -> anchor-conditioned continuation heads
                                       -> zero or more tokens -> <EOB>
```

The current one-pass model freezes the backbone and ordinary next-token head. A low-rank residual **and small trainable dense heads** live in the continuation branch; this is not a claim that every trainable parameter is LoRA. The EOB head is trained separately against the student's consecutive agreement with the frozen teacher on training/calibration contexts.

The teacher is used to construct training labels and evaluate outputs, not to verify each block during deployment. Immediate EOB retains the ordinary greedy anchor. Exact fallback tests apply to evaluation mode with the tested numerical backend; they are not a guarantee of bitwise identity across hardware or sampling-distribution preservation.

## Evidence status

| Evidence | Current interpretation |
|---|---|
| [Synthetic experiment](experiments/synthetic) | Separate controlled character-level proof of concept; not natural-language quality evidence. |
| [Historical BPE snapshots](experiments/bpe_probe/results) | Preserved unchanged for auditability; body-only interpretation withdrawn. |
| Three historical BPE training seeds | Repeat the same preprocessing and comparison limitations; repetition does not remove those limitations. |
| Corrected preprocessing | Boundary, cache, provenance and workflow regressions added. No corrected-corpus training results are released by this patch. |

Local teacher agreement measures whether each generated token equals the teacher's greedy choice **given that method's generated prefix**. It is not task accuracy, human-rated quality, or exact agreement with one independently generated teacher sequence. Teacher-produced anchors are included in the aggregate, so continuation-only metrics are also needed.

Tokens per backbone call is a count-based metric, **not wall-clock speed**. Head computation, context processing, cache behavior and runtime overhead must be measured separately. The historical random comparator emitted more tokens per call than the adaptive policy; its agreement difference does not establish superiority at equal compute.

## Scope and next experiment

Stay with the existing tiny nanoGPT: 2 layers, 4 attention heads, width 64, context 64. No large-model experiment is required.

The next meaningful experiment is clean-corpus retraining with a fixed tokenizer/source manifest, fresh held-out contexts and predeclared quality criteria, followed by an equal-call-budget random-gating control. Preserve per-prompt outputs and separate anchor from continuation errors. The scripts named `preregistered` are retained for compatibility but are historical fixed-policy diagnostics, not an independently preregistered confirmation.

## Run tests

```bash
cd experiments/bpe_probe
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q
```

## Prepare corrected data

Use a **fresh checkout/data directory**; do not reuse old tokenizers, banks or checkpoints. The default is Project Gutenberg edition 1513, not the historical mirrored edition 1112. Original source bytes and their notices are retained locally in `source.txt`.

```bash
cd experiments/bpe_probe
python prepare_data.py
```

The source URL is not immutable. Metadata records raw-byte, cleaned-body, tokenizer and split hashes. To enforce a known source snapshot, supply `--expected-source-sha256` with its recorded digest. Cross-version deterministic tokenization is not claimed merely because hashes are recorded.

See [BPE setup and limitations](experiments/bpe_probe) before running the training scripts. Existing numerical snapshots must not be presented as measurements produced by this corrected pipeline.

## Related areas

The project connects multi-token prediction, self-distillation, and selective prediction/optimal stopping. These connections motivate experiments, not a novelty claim or a proof of efficiency.

## Attribution and license

The model subset derives from [nanoGPT](https://github.com/karpathy/nanoGPT); its upstream MIT notice is preserved in `experiments/synthetic/third_party/nanogpt/LICENSE`. Repository code is under [MIT](LICENSE). Downloaded corpora retain their own terms and are not relicensed by this repository.
