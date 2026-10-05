# Anchor Block LM

A research prototype for **adaptive language-model decoding**:

> commit one autoregressive anchor token, then generate the predictable continuation as a variable-length block.

The working hypothesis is that full autoregressive decoding is most valuable at high-uncertainty branching points. Once one token resolves a branch, part of the following continuation may become predictable enough to emit as a block.

## Core idea

```text
context
   │
   ├─ autoregressive step ──> anchor token
   │
   └─ anchor-conditioned block model
          ├─ token
          ├─ token
          ├─ ...
          └─ <EOB>
```

`<EOB>` (end of block) is learned as part of the continuation model, so block length is not chosen from a fixed set at inference time.

The current prototype keeps the base nanoGPT weights frozen and adapts the continuation behavior with **LoRA + self-distillation**.

## Training sketch

1. Train or load a standard autoregressive teacher.
2. Sample one anchor token normally.
3. From the anchor-conditioned context, draw multiple teacher continuations with top-p sampling.
4. Estimate how far the continuation remains sufficiently concentrated.
5. Train the student to emit the corresponding continuation followed by `<EOB>`.
6. Re-label student-generated contexts with the teacher and run an on-policy self-distillation refresh.

The important distinction from speculative decoding is that the deployed student does **not** require a separate draft-and-verify loop for every block.

## Preliminary result

Current results are **controlled synthetic character-level experiments**, not a production LLM benchmark.

Using a small nanoGPT model, the on-policy variable-EOB model reached approximately the same block compression ratio as a fixed continuation baseline while staying substantially closer to the teacher:

| Method | Tokens / backbone call | Local teacher agreement |
|---|---:|---:|
| Fixed continuation = 3 | 2.00 | 82.15% |
| Variable EOB + on-policy refresh | 1.97 | **95.12%** |

These numbers are only evidence that the mechanism is learnable in a controlled setting. They do **not** establish production speedups, distribution preservation, or natural-language quality.

## Current status

- [x] Synthetic feasibility experiment
- [x] Frozen base model + LoRA adaptation
- [x] Anchor-conditioned multi-token continuation
- [x] Learned `<EOB>` variable block length
- [x] On-policy self-distillation refresh
- [ ] Natural-language BPE experiment
- [ ] Multi-seed full training runs
- [ ] Ablations: anchor / EOB / on-policy refresh
- [ ] KV-cache + GPU benchmark
- [ ] Larger-model validation

## Research questions

1. How much does one committed anchor token reduce uncertainty over the following sequence?
2. Can a learned block model exploit that reduction better than a fixed block length?
3. How should safe block termination be calibrated?
4. How much of the gain survives on natural-language BPE models?
5. Can the method improve wall-clock latency once KV-cache and GPU kernels are included?

## Scope

This repository is intentionally a **research prototype**. The current goal is to establish whether the mechanism exists and can be trained reliably before scaling it to large models.

## Relationship to prior work

The project is related to:

- multi-token prediction,
- blockwise parallel decoding,
- sequence-level / on-policy distillation,
- speculative decoding,
- adaptive computation and optimal stopping.

The intended research direction differs in its emphasis on an **autoregressive anchor followed by a learned variable-length continuation block**.

## Implementation

The first implementation is based on the design and code structure of [nanoGPT](https://github.com/karpathy/nanoGPT). Attribution for reused or adapted upstream code will be preserved in source files.

Reproduction scripts and cleaned experiment code will be added next.

## License

MIT License. See [LICENSE](LICENSE).

## Disclaimer

Preliminary metrics in this README come from small synthetic experiments. Do not interpret them as evidence of equivalent results on production-scale LLMs.
