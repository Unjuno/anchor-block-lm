#!/usr/bin/env bash
set -euo pipefail

python train_teacher.py --steps 1600
python anchor_block_experiment.py prepare
python eob_experiment.py prepare
python eob_experiment.py train --steps 1200
python onpolicy_eob_refresh.py prepare
python onpolicy_eob_refresh.py train 300
python benchmark_onpolicy.py
