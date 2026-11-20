# AI651 PA3 — Spatio-temporal traffic forecasting on LargeST-SD

Code for Assignment 3 (Deep Learning for Space, Time and Graphs, LUMS, Fall 2026).

| Question | Script | Output |
|---|---|---|
| Q1 sum/mean/max aggregation | `src/q1_aggregation.py` | console (`results/q1_output.txt`) |
| Q2 ring road, direction | `src/q2_ring.py` | `figures/q2a_heatmaps.png`, console |
| Q3(a,b) extraction, resampling, graph | `src/prepare_data.py` | `data/processed/*` |
| Q3 plots & graph statistics | `src/q3_analysis.py` | `figures/q3*.png`, `results/q3_stats.json` |
| Q4/Q5 models & training | `src/models.py`, `src/train.py` | `results/<tag>.json/.pt/_test_pred.npy` |
| Q6 table, plots, learned-graph analysis | `src/q6_analysis.py` | `results/table_*.csv`, `figures/q5*/q6*.png` |

## 1. Environment

Python 3.10, PyTorch 2.6 (CUDA optional — everything runs on CPU, just slower).

```bash
pip install -r requirements.txt
```

## 2. Data

Download from Kaggle ([liuxu77/largest](https://www.kaggle.com/datasets/liuxu77/largest)) the files
`ca_his_raw_2020.h5`, `ca_his_raw_2021.h5`, `ca_meta.csv`, `ca_rn_adj.npy` into one folder
(default `~/datasets/LargeST`). ~15 GB; raw data is not committed.

```bash
python src/prepare_data.py --raw ~/datasets/LargeST   # -> data/processed/
```

* San Diego = rows of `ca_meta.csv` with `County == "San Diego"` (716 sensors); columns of the HDF5
  are matched by sensor ID; adjacency = `ca_rn_adj[ID2][:, ID2]` (same order).
* 5-min **vehicle counts** are **summed** to vehicles / 15 min.
* Missing values: an `observed` mask is kept and metrics/loss use observed targets only; inputs are
  filled by ≤1 h linear interpolation (per split) and otherwise by the sensor's 2020 (training)
  time-of-week mean.

## 3. Experiments

```bash
python src/q1_aggregation.py
python src/q2_ring.py
python src/q3_analysis.py
bash run_all.sh            # trains none/fixed/adaptive/hybrid (+ residual ablation), sequentially
python src/q6_analysis.py
```

Single run: `python src/train.py --mode {none,fixed,adaptive,hybrid} [--residual] [--seed 42]`.

### Configuration (defaults in `src/train.py`)

| Item | Value |
|---|---|
| Splits | train 2020-01-01–2020-12-31, val 2021-01-01–2021-03-31, test 2021-04-01–2021-12-31 (windows never cross a boundary) |
| Input / output | 8 × 15 min → 4 × 15 min, all 716 sensors |
| Normalisation | z-score, scalar mean/std of observed **training** values |
| Temporal encoder | Conv1d(1→32,k3)–ReLU–Conv1d(32→32,k3)–ReLU–Conv1d(32→32,k4)–ReLU, shared across sensors |
| Graph conv | `H = ReLU(Â Z W_g)`, d = 32 |
| Head | Linear(32→32)–ReLU–Linear(32→4) |
| Fixed Â | directed row normalisation `D_out⁻¹ Ã`, Ã = A with diagonal set to 1 |
| Adaptive Â | `softmax_row(ReLU(E1 E2ᵀ))`, E1, E2 ∈ R^{716×10}, dense (no top-k) |
| Hybrid Â | `λ Â_fixed + (1−λ) Â_adaptive`, λ = sigmoid(θ) learned, init 0.5 |
| Loss | masked MAE in original units |
| Optimiser | Adam, lr 1e-3, weight decay 1e-4, batch 64, grad-clip 5, ReduceLROnPlateau(0.5, 3) |
| Epochs | ≤ 50, early stopping on val MAE (patience 8), best checkpoint evaluated |
| Seed | 42 (Python, NumPy, PyTorch; cuDNN deterministic) |
