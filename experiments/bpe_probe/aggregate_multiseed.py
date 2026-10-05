from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(len(arr)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    paths = sorted(args.root.glob("multiseed-*/onepass_preregistered_benchmark.json"))
    if len(paths) < 3:
        raise SystemExit(f"expected >=3 seed results, found {len(paths)}")

    rows = []
    for path in paths:
        data = json.loads(path.read_text())
        rows.append({
            "artifact": path.parent.name,
            "variable_speed": data["variable_eob"]["tokens_per_backbone_call"],
            "variable_agreement": data["variable_eob"]["local_teacher_agreement"],
            "random_speed": data["random01"]["tokens_per_backbone_call"],
            "random_agreement": data["random01"]["local_teacher_agreement"],
            "agreement_gain": (
                data["variable_eob"]["local_teacher_agreement"]
                - data["random01"]["local_teacher_agreement"]
            ),
            "ar_agreement": data["ar"]["local_teacher_agreement"],
        })

    summary = {
        "seeds": rows,
        "aggregate": {
            "variable_speed": summarize([r["variable_speed"] for r in rows]),
            "variable_agreement": summarize([r["variable_agreement"] for r in rows]),
            "random_speed": summarize([r["random_speed"] for r in rows]),
            "random_agreement": summarize([r["random_agreement"] for r in rows]),
            "agreement_gain": summarize([r["agreement_gain"] for r in rows]),
            "all_ar_fallback_exact": all(r["ar_agreement"] == 1.0 for r in rows),
            "all_variable_better_than_random": all(
                r["agreement_gain"] > 0 for r in rows
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
