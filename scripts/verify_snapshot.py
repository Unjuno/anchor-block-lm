"""Recompute published timing identities; does not train or benchmark a model."""
from __future__ import annotations
import json
import math
from pathlib import Path


def timing_metrics(tokens: int, calls: int, ar_seconds: float, actor_seconds: float) -> dict:
    if (type(tokens) is not int or type(calls) is not int or not 0 < calls <= tokens
            or not all(math.isfinite(x) and x > 0 for x in (ar_seconds, actor_seconds))):
        raise ValueError('positive finite durations and integer 0 < calls <= tokens required')
    mean_k = tokens / calls
    speed = ar_seconds / actor_seconds
    return {'mean_k_timing': mean_k, 'speed': speed,
            'effective_overhead': mean_k / speed - 1,
            'latency_increase_percent': 100 * (1 / speed - 1),
            'throughput_drop_percent': 100 * (1 - speed)}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / 'docs/evidence/final_snapshot_2026_10_07.json').read_text())
    for row in data['seeds']:
        calc = timing_metrics(row['timing_tokens'], row['timing_actor_calls'],
                              row['ar_median_seconds'], row['actor_median_seconds'])
        for key, val in calc.items():
            if not math.isclose(val, row[key], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"seed {row['seed']}: inconsistent {key}")
        print(json.dumps({'seed': row['seed'], **calc}, sort_keys=True))
    print('Verified: same-trace timing identities; no new speed measurements.')

if __name__ == '__main__':
    main()
