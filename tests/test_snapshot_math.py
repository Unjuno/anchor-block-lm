import importlib.util
from pathlib import Path
import unittest


def module():
    p = Path(__file__).resolve().parents[1] / 'scripts' / 'verify_snapshot.py'
    assert p.exists(), 'snapshot verification implementation is missing'
    spec = importlib.util.spec_from_file_location('verify_snapshot', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class SnapshotMath(unittest.TestCase):
    def test_matched_trace_identity(self):
        r = module().timing_metrics(384, 246, 0.272815486, 0.291640946)
        self.assertAlmostEqual(r['mean_k_timing'], 384 / 246)
        self.assertAlmostEqual(r['speed'], r['mean_k_timing'] / (1 + r['effective_overhead']))

    def test_latency_is_reciprocal_not_throughput_drop(self):
        r = module().timing_metrics(100, 100, 1.0, 1.62)
        self.assertAlmostEqual(r['latency_increase_percent'], 62.0)
        self.assertAlmostEqual(r['throughput_drop_percent'], 100 * (1 - 1 / 1.62))

    def test_invalid_inputs_are_rejected(self):
        for values in ((0, 1, 1, 1), (10, 0, 1, 1), (10, 11, 1, 1), (10, 1, 0, 1), (10, 1, 1, float('nan'))):
            with self.assertRaises(ValueError):
                module().timing_metrics(*values)

    def test_approximate_aggregation_is_not_a_head_profiler(self):
        r = module().timing_metrics(100, 50, 1, 1)
        self.assertEqual(r['speed'], 1)
        self.assertEqual(r['effective_overhead'], 1)
        self.assertNotIn('head_seconds', r)

if __name__ == '__main__':
    unittest.main()
