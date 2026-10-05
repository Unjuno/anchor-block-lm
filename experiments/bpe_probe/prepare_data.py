from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np

SOURCE_URL = (
    "https://raw.githubusercontent.com/phymooc/learn-python/"
    "f0b657e7ec33de117d1ba5b5e8ae56c523ffc442/code/Romeo.txt"
)


def split_text(text: str, train_fraction: float = 0.8, dev_fraction: float = 0.1):
    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be in (0,1)")
    if not (0 <= dev_fraction < 1) or train_fraction + dev_fraction >= 1:
        raise ValueError("invalid dev_fraction")
    n = len(text)
    a = int(n * train_fraction)
    b = int(n * (train_fraction + dev_fraction))
    return text[:a], text[a:b], text[b:]


def strip_gutenberg_wrapper(text: str) -> str:
    upper = text.upper()
    start_candidates = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
    ]
    end_candidates = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
    ]
    start = None
    for marker in start_candidates:
        i = upper.find(marker)
        if i >= 0:
            line_end = text.find("\n", i)
            start = line_end + 1 if line_end >= 0 else i
            break
    end = None
    for marker in end_candidates:
        i = upper.find(marker, start or 0)
        if i >= 0:
            end = i
            break
    if start is not None and end is not None and end > start:
        return text[start:end]
    return text


def train_bpe(train_text: str, tokenizer_path: Path, vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator([train_text], trainer=trainer)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--source-url", default=SOURCE_URL)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    raw_path = args.out / "source.txt"
    if raw_path.exists():
        raw = raw_path.read_text(encoding="utf-8")
    else:
        with urllib.request.urlopen(args.source_url, timeout=60) as response:
            raw = response.read().decode("utf-8")
        raw_path.write_text(raw, encoding="utf-8")

    text = strip_gutenberg_wrapper(raw).replace("\r\n", "\n")
    train_text, dev_text, test_text = split_text(text)
    tokenizer = train_bpe(train_text, args.out / "tokenizer.json", args.vocab_size)

    token_counts = {}
    for split, part in [("train", train_text), ("dev", dev_text), ("test", test_text)]:
        ids = np.asarray(tokenizer.encode(part).ids, dtype=np.uint16)
        np.save(args.out / f"{split}.npy", ids)
        token_counts[split] = int(ids.size)

    metadata = {
        "dataset": "romeo_and_juliet_project_gutenberg",
        "source_url": args.source_url,
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "clean_text_chars": len(text),
        "tokenizer": "byte-level BPE trained on train split only",
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "token_counts": token_counts,
        "split_fractions": {"train": 0.8, "dev": 0.1, "test": 0.1},
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
