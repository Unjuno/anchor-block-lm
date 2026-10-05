"""Offline corpus-boundary regressions; no downloads or model training."""
import pytest
from prepare_data import split_text, strip_gutenberg_wrapper


def wrapped(body, article="THE", space=" "):
    return (
        "Metadata not belonging to the play.\n"
        f"***{space}START OF {article} PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        + body
        + f"***{space}END OF {article} PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        + "License appendix not belonging to the play.\n"
    )


@pytest.mark.parametrize("article", ["THE", "THIS"])
@pytest.mark.parametrize("space", [" ", ""])
def test_recognizes_both_marker_wordings(article, space):
    body = "ACT I.\nInvented dialogue for testing.\n"
    assert strip_gutenberg_wrapper(wrapped(body, article, space)) == body


@pytest.mark.parametrize("text", [
    "No book markers.\n",
    "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\nbody\n",
    "body\n*** END OF THIS PROJECT GUTENBERG EBOOK SAMPLE ***\n",
    wrapped(""),
    wrapped("body\n") + wrapped("second book\n"),
])
def test_unrecognized_or_ambiguous_boundaries_fail_closed(text):
    with pytest.raises(ValueError):
        strip_gutenberg_wrapper(text)


def test_crlf_and_bom_do_not_change_body_boundaries():
    raw = "\ufeff" + wrapped("ACT I.\nSome dialogue.\n", "THIS")
    assert strip_gutenberg_wrapper(raw.replace("\n", "\r\n")) == "ACT I.\nSome dialogue.\n"


def test_split_never_sees_header_or_footer():
    body = "Invented play text.\n" * 100
    parts = split_text(strip_gutenberg_wrapper(wrapped(body, "THIS")))
    assert "".join(parts) == body
    assert all("Metadata" not in part and "License appendix" not in part for part in parts)


def test_workflow_has_separate_benchmark_paths():
    import re
    from pathlib import Path
    workflow = Path(__file__).resolve().parents[3] / ".github/workflows/bpe-onepass.yml"
    paths = re.findall(r'^\s*- "(experiments/[^"\n]+)"$', workflow.read_text(), re.MULTILINE)
    assert all(" " not in path for path in paths), paths
    for name in ("benchmark_onepass_calibrated.py", "benchmark_onepass_preregistered.py"):
        assert f"experiments/bpe_probe/{name}" in paths


def test_workflow_compiles_the_fixed_policy_benchmark():
    import shlex
    from pathlib import Path
    workflow = Path(__file__).resolve().parents[3] / ".github/workflows/bpe-onepass.yml"
    command = next(line for line in workflow.read_text().splitlines() if "python -m py_compile" in line)
    assert "benchmark_onepass_preregistered.py" in shlex.split(command)


@pytest.mark.parametrize("notice", ["PROJECT GUTENBERG LICENSE", "WORLD LIBRARY", "LIBRARY OF THE FUTURE"])
def test_embedded_publisher_notice_is_not_silently_training_text(notice):
    with pytest.raises(ValueError, match="notice"):
        strip_gutenberg_wrapper(wrapped("ACT I.\n" + notice + "\n"))


def test_unknown_cached_source_is_refused(tmp_path):
    import prepare_data as prep
    (tmp_path / "source.txt").write_bytes(b"cached data")
    with pytest.raises(ValueError, match="provenance"):
        prep.load_source(tmp_path, "https://example.org/a", None)


@pytest.mark.parametrize("change", ["url", "bytes"])
def test_changed_cache_is_refused(tmp_path, change):
    import hashlib
    import json
    import prepare_data as prep
    raw = b"cached data"
    (tmp_path / "source.txt").write_bytes(raw)
    (tmp_path / "source_manifest.json").write_text(json.dumps({
        "source_url": "https://example.org/a",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
    }))
    if change == "bytes":
        (tmp_path / "source.txt").write_bytes(b"changed")
    url = "https://example.org/b" if change == "url" else "https://example.org/a"
    with pytest.raises(ValueError, match="mismatch"):
        prep.load_source(tmp_path, url, None)


def test_verified_cache_is_reusable_and_expected_hash_checked(tmp_path):
    import hashlib
    import json
    import prepare_data as prep
    raw = b"verified bytes\r\n"
    digest = hashlib.sha256(raw).hexdigest()
    (tmp_path / "source.txt").write_bytes(raw)
    (tmp_path / "source_manifest.json").write_text(json.dumps({
        "source_url": "https://example.org/a", "raw_sha256": digest,
    }))
    assert prep.load_source(tmp_path, "https://example.org/a", digest) == raw
    with pytest.raises(ValueError, match="SHA-256"):
        prep.load_source(tmp_path, "https://example.org/a", "0" * 64)


def test_download_hash_mismatch_does_not_create_cache(tmp_path, monkeypatch):
    import io
    import prepare_data as prep
    monkeypatch.setattr(prep.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"raw"))
    with pytest.raises(ValueError, match="SHA-256"):
        prep.load_source(tmp_path, "https://example.org/a", "0" * 64)
    assert not (tmp_path / "source.txt").exists()


def test_default_source_is_the_reviewed_edition():
    import prepare_data as prep
    assert prep.SOURCE_URL == "https://www.gutenberg.org/ebooks/1513.txt.utf-8"


def test_cli_records_body_split_and_tokenizer_provenance(tmp_path, monkeypatch):
    pytest.importorskip("tokenizers")
    import hashlib
    import io
    import json
    import numpy as np
    import prepare_data as prep
    body = "ACT I.\nInvented dialogue only for a software test.\n" * 100
    raw = wrapped(body, "THIS").encode("utf-8")
    monkeypatch.setattr(prep.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(raw))
    prep.main(["--out", str(tmp_path), "--vocab-size", "257",
               "--expected-source-sha256", hashlib.sha256(raw).hexdigest()])
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["preprocessing_version"] == "gutenberg-body-v2"
    assert meta["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert meta["clean_text_sha256"] == hashlib.sha256(body.encode()).hexdigest()
    assert meta["tokenizer_sha256"] == hashlib.sha256((tmp_path / "tokenizer.json").read_bytes()).hexdigest()
    assert meta["legacy_results_comparable"] is False
    assert (tmp_path / "source.txt").read_bytes() == raw
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    for split, part in zip(("train", "dev", "test"), split_text(body)):
        ids = np.load(tmp_path / f"{split}.npy")
        assert ids.dtype == np.uint16
        assert tokenizer.decode(ids.tolist()) == part
        assert meta["splits"][split]["text_sha256"] == hashlib.sha256(part.encode()).hexdigest()
    with pytest.raises(ValueError, match="Derived data already exists"):
        prep.main(["--out", str(tmp_path)])
