# Reproduction and evidence retention

## 1. Read results without training

From the repository root:

```bash
python scripts/verify_snapshot.py
python -m unittest discover -s tests -v
```

These standard-library commands check archived timing arithmetic, not model performance. [Final timing evidence](evidence/final_snapshot_2026_10_07.json) records the input archive/member hashes and separates timing-trace call counts from the quality-evaluation sample.

## 2. Model-test environment

The recorded CPU experiments used Python 3.13, PyTorch 2.10.0+cpu, NumPy 2.3.5; tokenizers 0.22.2 is required for source-to-token reconstruction. Create a fresh virtual environment. CPU wheel installation is explicit so a CUDA build is not downloaded accidentally:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -r requirements-research.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
for folder in synthetic bpe_probe fixed_k_gate adaptive_k_rl two_model_rl dynamic_boundary_rl layer_fusion_probe; do
  (cd "experiments/$folder" && PYTHONPATH=. python -m pytest -q tests) || exit 1
done
python -m compileall -q experiments scripts tests
```

Suites run in separate processes because historical experiments use local module imports with overlapping names. There is no claim that a bare root-level `pytest` discovers an isolated combined model suite. The root `tests/` only tests snapshot arithmetic.

FP32, no quantization, no KV cache, no compilation acceleration, batch-one deployment and an unpinned CPU clock are the measured assumptions. CPU execution does not establish CUDA, Metal, alternate ABI, alternate kernel or lower-precision equivalence. Freeze/eval mode and identical tokenizer/window handling matter for AR equality. Actual confidence/logit values may change across backends even with unchanged source.

## 3. Canonical final continuation from the saved bundle

The result bundle is named `dynamic_boundary_consistent_rl_results.zip`. SHA256:

```text
1a02fbd7ef0876ae5ead98390d3c8de567ae62c041d6ac7a2d210c640faaecd7
```

The closing response also supplies this bundle separately. It contains `original/` teacher/student archives, `previous/seed-48017..48019/` dynamic checkpoints, `prepared/train_dev.pt`, `results/seed-*/` trained checkpoints and raw arrays, `PROTOCOL.md`, and logs. Extract it into a fresh directory outside the repository. Do not rename a legacy checkpoint into a current one or silently replace missing inputs with newly trained weights.

Use absolute paths; the following is run from this repository's root:

```bash
BUNDLE_ROOT=/absolute/path/to/extracted/dynamic-results
for seed in 48017 48018 48019; do
  python experiments/dynamic_boundary_rl/run_consistent_dual.py train --root "$BUNDLE_ROOT" --seed "$seed"
done
# Run only after ALL training has stopped.
for seed in 48017 48018 48019; do
  python experiments/dynamic_boundary_rl/run_consistent_dual.py evaluate --root "$BUNDLE_ROOT" --seed "$seed"
done
```

Training creates `new/seed-*/` and refuses an existing output directory. The supplied `prepared/` data are the exact verified frozen train/dev arrays; training does not require fitting another tokenizer. To replay evaluation only, in a clean extraction copy `results/` to `new/` first, then run only `evaluate`. Evaluation writes new measurements under `new/`; the original `results/` remains intact. Regenerating source arrays is optional:

```bash
python experiments/dynamic_boundary_rl/run_consistent_dual.py prepare --root "$BUNDLE_ROOT"
```

The source loader checks source, extracted-body, tokenizer and split hashes. A mismatch is an error, not permission to use a different corpus. Token arrays used in this continuation come from corrected `gutenberg-body-v2`, not the superseded wrapper-contaminated experiment.

## 4. GitHub Actions input archives

Original inputs are also identified in GitHub Actions:

| Run | Artifact | Purpose |
|---|---|---|
| 37324518978 | `adaptive-k-rl-evidence` | Corrected teacher/student/tokenizer inputs |
| 37494321837 | `dynamic-boundary-seed-48017`, `dynamic-boundary-seed-48018`, `dynamic-boundary-seed-48019` | Previous dynamic-boundary checkpoints |
| 37499394698 | `consistent-boundary-source-and-tests` | Exact tested source for the last continuation |

Artifacts have retention limits (the cited October archives reported January 2027 expiry). Source in git is not a substitute for weights or raw arrays. Preserve the supplied bundle and its hash; this closeout does not claim that a permanent GitHub Release asset or DOI deposit was created.

GitHub CLI users with appropriate access can retrieve the first two input families using `gh run download RUN_ID -R Unjuno/anchor-block-lm -n ARTIFACT_NAME -D DESTINATION`. Construct the exact `original/` and `previous/seed-*/` layout above. The final bundle is the simpler replay route because it also carries source caches, verified token arrays and completed continuation outputs.

## 5. Reproduction versus validation

A passing unit test verifies behavior and invariants, not the research hypothesis. Replaying one corpus/seed does not establish generality. CPU times are machine-dependent; use the same workload for both timing and call counts, exclude teacher scoring and warmup, report median and range, and never time concurrent training.

All published learned-quality comparisons remain exploratory. No hyperparameters, length quotas or acceptance thresholds are changed by the closing documentation. No claim is made that a fresh run must reproduce every floating-point value on a different backend.
