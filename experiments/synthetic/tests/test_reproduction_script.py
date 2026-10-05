from pathlib import Path


def test_reproduction_trains_anchor_baseline_before_benchmark():
    script = (Path(__file__).resolve().parents[1] / 'reproduce_synthetic.sh').read_text()
    assert 'anchor_block_experiment.py train-anchor' in script
    assert script.index('anchor_block_experiment.py train-anchor') < script.index('benchmark_onpolicy.py')
