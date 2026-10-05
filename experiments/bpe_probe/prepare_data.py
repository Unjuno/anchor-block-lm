"""Prepare a versioned, body-only Gutenberg corpus in a fresh data directory.

Historical BPE snapshots used a different edition and a broken wrapper parser.
They must not be compared to results produced by this preprocessing version.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request

import numpy as np

SOURCE_URL = "https://www.gutenberg.org/ebooks/1513.txt.utf-8"
PREPROCESSING_VERSION = "gutenberg-body-v2"
BOUNDARY = re.compile(
    r"^[ \t]*\*{3}[ \t]*(START|END)[ \t]+OF[ \t]+(?:THE|THIS)"
    r"[ \t]+PROJECT[ \t]+GUTENBERG[ \t]+(?:EBOOK|ETEXT)\b"
    r"[^\n]*\*{3}[ \t]*(?:\n|$)",
    re.IGNORECASE | re.MULTILINE,
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
    """Fail closed rather than silently train on headers, footers, or notices.

    The preserved source.txt is never edited. This returns an analysis copy,
    not a statement about redistribution rights or the source's legal status.
    """
    text = text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    boundaries = list(BOUNDARY.finditer(text))
    if len(boundaries) != 2 or [m.group(1).upper() for m in boundaries] != ["START", "END"]:
        raise ValueError("Expected exactly one ordered Gutenberg START/END marker pair")
    body = text[boundaries[0].end():boundaries[1].start()]
    if not body.strip():
        raise ValueError("Gutenberg body is empty")
    upper = body.upper()
    if any(phrase in upper for phrase in (
        "PROJECT GUTENBERG", "WORLD LIBRARY", "LIBRARY OF THE FUTURE",
    )):
        raise ValueError("Publisher/license notice remains inside the body; review the source")
    return body


def load_source(directory: Path, source_url: str, expected_sha256: str | None) -> bytes:
    """Keep original bytes and reject stale, unprovenanced, or changed caches."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "source.txt"
    manifest_path = directory / "source_manifest.json"
    if expected_sha256 is not None:
        expected_sha256 = expected_sha256.lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("Expected SHA-256 must be 64 hexadecimal characters")
    if raw_path.exists():
        if not manifest_path.exists():
            raise ValueError("Cached source has no provenance; use a fresh --out directory")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw = raw_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if manifest.get("source_url") != source_url or manifest.get("raw_sha256") != digest:
            raise ValueError("Source cache URL/hash mismatch; use a fresh --out directory")
    else:
        with urllib.request.urlopen(source_url, timeout=60) as response:
            raw = response.read()
        digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Source SHA-256 does not match the expected snapshot")
    raw.decode("utf-8-sig")  # Reject non-UTF-8 responses before caching.
    if not raw_path.exists():
        raw_path.write_bytes(raw)
        manifest_path.write_text(json.dumps({
            "source_url": source_url, "raw_sha256": digest,
        }, indent=2), encoding="utf-8")
    return raw


def train_bpe(train_text: str, tokenizer_path: Path, vocab_size: int):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=["<unk>"],
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()), show_progress=False,
    )
    tokenizer.train_from_iterator([train_text], trainer=trainer)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--source-url", default=SOURCE_URL)
    ap.add_argument("--expected-source-sha256", default=None)
    args = ap.parse_args(argv)
    if not 257 <= args.vocab_size <= 65536:
        ap.error("--vocab-size must be in [257,65536] for byte-level BPE / uint16 storage")
    derived = ("train.npy", "dev.npy", "test.npy", "tokenizer.json", "metadata.json")
    if any((args.out / name).exists() for name in derived):
        raise ValueError("Derived data already exists; use a fresh checkout / --out directory")

    raw = load_source(args.out, args.source_url, args.expected_source_sha256)
    text = strip_gutenberg_wrapper(raw.decode("utf-8-sig"))
    parts = dict(zip(("train", "dev", "test"), split_text(text)))
    if not all(parts.values()):
        raise ValueError("Corpus is too short for nonempty train/dev/test splits")
    tokenizer_path = args.out / "tokenizer.json"
    tokenizer = train_bpe(parts["train"], tokenizer_path, args.vocab_size)
    if tokenizer.get_vocab_size() > 65536:
        raise ValueError("Tokenizer vocabulary would overflow uint16 storage")

    token_counts, split_metadata = {}, {}
    offset = 0
    for split, part in parts.items():
        token_ids = tokenizer.encode(part).ids
        if token_ids and (min(token_ids) < 0 or max(token_ids) > 65535):
            raise ValueError("Token ID would overflow uint16 storage")
        ids = np.asarray(token_ids, dtype=np.uint16)
        np.save(args.out / f"{split}.npy", ids)
        token_counts[split] = int(ids.size)
        split_metadata[split] = {
            "character_start": offset, "character_end": offset + len(part),
            "text_sha256": hashlib.sha256(part.encode("utf-8")).hexdigest(),
            "array_sha256": hashlib.sha256((args.out / f"{split}.npy").read_bytes()).hexdigest(),
        }
        offset += len(part)
    import tokenizers
    metadata = {
        "dataset": "gutenberg_body_v2", "preprocessing_version": PREPROCESSING_VERSION,
        "source_url": args.source_url,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_source_sha256": args.expected_source_sha256,
        "clean_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "clean_text_chars": len(text),
        "tokenizer": "byte-level BPE trained on train split only",
        "tokenizers_version": tokenizers.__version__,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "token_counts": token_counts, "splits": split_metadata,
        "split_fractions": {"train": 0.8, "dev": 0.1, "test": 0.1},
        "legacy_results_comparable": False,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
