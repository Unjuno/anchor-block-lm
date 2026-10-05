from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
NANOGPT = HERE.parent / "synthetic" / "third_party" / "nanogpt"
sys.path.insert(0, str(NANOGPT))
from model import GPT, GPTConfig


def seed_all(seed: int, threads: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    seed_all(args.seed, args.threads)
    data_dir = HERE / "data"
    result_dir = HERE / "results"
    result_dir.mkdir(exist_ok=True)

    meta = json.loads((data_dir / "metadata.json").read_text())
    data = {
        split: torch.tensor(np.load(data_dir / f"{split}.npy").astype(np.int64))
        for split in ("train", "dev", "test")
    }
    cfg = GPTConfig(
        block_size=64,
        vocab_size=int(meta["actual_vocab_size"]),
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=True,
    )
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    rng = torch.Generator().manual_seed(args.seed + 100)

    def sample_batch(split: str, batch_size: int):
        seq = data[split]
        ix = torch.randint(0, len(seq) - cfg.block_size - 1, (batch_size,), generator=rng)
        arange = torch.arange(cfg.block_size)
        x = seq[ix[:, None] + arange[None, :]]
        y = seq[ix[:, None] + 1 + arange[None, :]]
        return x, y

    @torch.no_grad()
    def evaluate(split: str, batches: int = 20):
        model.eval()
        vals = []
        local = torch.Generator().manual_seed(1000 if split == "dev" else 1001)
        seq = data[split]
        arange = torch.arange(cfg.block_size)
        for _ in range(batches):
            ix = torch.randint(0, len(seq) - cfg.block_size - 1, (32,), generator=local)
            x = seq[ix[:, None] + arange[None, :]]
            y = seq[ix[:, None] + 1 + arange[None, :]]
            vals.append(model(x, y)[1].item())
        return float(np.mean(vals))

    start = time.perf_counter()
    best = float("inf")
    history = []
    print(json.dumps({"step": 0, "dev_nll": evaluate("dev")}), flush=True)

    for step in range(1, args.steps + 1):
        model.train()
        lr = 3e-3 * (0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = sample_batch("train", args.batch_size)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0 or step == args.steps:
            dev = evaluate("dev")
            row = {
                "step": step,
                "train_nll": loss.item(),
                "dev_nll": dev,
                "elapsed_s": time.perf_counter() - start,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if dev < best:
                best = dev
                torch.save(
                    {"model": model.state_dict(), "config": vars(cfg), "step": step, "seed": args.seed},
                    result_dir / "teacher.pt",
                )

    checkpoint = torch.load(result_dir / "teacher.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    final = {
        "best_step": checkpoint["step"],
        "dev_nll": evaluate("dev"),
        "test_nll": evaluate("test"),
        "elapsed_s": time.perf_counter() - start,
        "history": history,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (result_dir / "teacher_metrics.json").write_text(json.dumps(final, indent=2))
    print(json.dumps({"teacher_done": final}, indent=2))


if __name__ == "__main__":
    main()
