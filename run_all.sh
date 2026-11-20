#!/usr/bin/env bash
# Reproduce Q4-Q6 experiments (sequential so measured training times are comparable).
set -e
PY=${PY:-python}
mkdir -p results/logs
for m in none fixed adaptive hybrid; do
  $PY src/train.py --mode $m --seed 42 2>&1 | tee results/logs/$m.log
done
# ablation: same models with a skip connection (head sees Z + H)
for m in none fixed adaptive hybrid; do
  $PY src/train.py --mode $m --seed 42 --residual --tag ${m}_res 2>&1 | tee results/logs/${m}_res.log
done
