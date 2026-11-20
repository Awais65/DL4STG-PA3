"""Q6 - comparison table, predicted-vs-actual plots, learned-graph analysis."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROC, FIG, RES = (os.path.join(ROOT, p) for p in ("data/processed", "figures", "results"))
MODELS = [("none", "Temporal-only", "None"), ("fixed", "STGNN", "Fixed"),
          ("adaptive", "STGNN", "Adaptive"), ("hybrid", "STGNN", "Hybrid")]
K = 10


def load(tag):
    p = os.path.join(RES, f"{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def table(suffix=""):
    rows = []
    for tag, model, adj in MODELS:
        r = load(tag + suffix)
        if r is None:
            continue
        rows.append({
            "Model": model, "Adjacency": adj,
            "Val MAE": r["val"]["MAE"], "Val RMSE": r["val"]["RMSE"],
            "Test MAE": r["test"]["MAE"], "Test RMSE": r["test"]["RMSE"],
            **{f"Test MAE h{h + 1}": r["test"]["MAE_h"][h] for h in range(4)},
            "Params": r["n_params"], "Epochs (best)": f"{r['epochs_run']} ({r['best_epoch']})",
            "Train time (s)": r["train_time_sec"], "s/epoch": r["sec_per_epoch"],
            "lambda": r.get("lambda_final"),
        })
    return pd.DataFrame(rows)


def pred_vs_actual(times, flow, obs, test_lo, suffix=""):
    sensors = [49, 137, 305]
    week = (times >= "2021-05-03") & (times < "2021-05-10")
    preds = {tag: np.load(os.path.join(RES, f"{tag}{suffix}_test_pred.npy")) for tag, *_ in MODELS
             if os.path.exists(os.path.join(RES, f"{tag}{suffix}_test_pred.npy"))}
    meta = pd.read_csv(os.path.join(PROC, "sd_meta.csv"))
    colors = {"none": "tab:gray", "fixed": "tab:blue", "adaptive": "tab:orange", "hybrid": "tab:green"}
    for h, hname in ((0, "15 min"), (3, "60 min")):
        fig, axes = plt.subplots(len(sensors), 1, figsize=(13, 3.1 * len(sensors)), sharex=True)
        tidx = np.where(week)[0]
        for ax, j in zip(axes, sensors):
            y = np.where(obs[tidx, j], flow[tidx, j], np.nan)
            ax.plot(times[tidx], y, color="k", lw=1.6, label="actual")
            for tag, P in preds.items():
                k = tidx - test_lo - 8 - h                      # window index whose horizon h hits t
                ok = (k >= 0) & (k < len(P))
                lab = dict((t, f"{m} ({a})") for t, m, a in MODELS)[tag]
                ax.plot(times[tidx][ok], P[k[ok], h, j], lw=1, color=colors[tag], label=lab, alpha=.9)
            ax.set_title(f"sensor {meta.ID[j]} ({meta.Fwy[j]}), horizon {hname}" + (" - with skip connection" if suffix else ""))
            ax.set_ylabel("veh / 15 min"); ax.grid(alpha=.3)
        axes[0].legend(ncol=5, fontsize=8, loc="upper right")
        fig.savefig(os.path.join(FIG, f"q6a_pred_vs_actual{suffix}_h{h + 1}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def graph_analysis(tag, A_fix, geo, corr, meta, out):
    p = os.path.join(RES, f"{tag}_A_adaptive.npy")
    if not os.path.exists(p):
        return
    L = np.load(p).astype(np.float64)
    N = L.shape[0]
    B = A_fix > 0; np.fill_diagonal(B, False)
    off = L.copy(); np.fill_diagonal(off, -np.inf)               # self-weight analysed separately
    top = np.argsort(-off, axis=1)[:, :K]                        # top-K learned neighbours per row
    rows = np.repeat(np.arange(N), K)
    cols = top.ravel()
    in_phys = B[rows, cols]
    rng = np.random.default_rng(0)
    rand_c = rng.integers(0, N, size=len(rows)); keep = rand_c != rows
    st = {
        "self_weight_mean": float(np.diag(L).mean()),
        "row_entropy_mean_nats": float(-(L * np.log(L + 1e-30)).sum(1).mean()),
        "uniform_entropy_nats": float(np.log(N)),
        "topK_mass_mean": float(np.take_along_axis(L, top, 1).sum(1).mean()),
        f"top{K}_in_physical_frac": float(in_phys.mean()),
        "top1_in_physical_frac": float(B[np.arange(N), top[:, 0]].mean()),
        "physical_density_random_baseline": float(B.sum() / (N * (N - 1))),
        f"top{K}_median_km": float(np.median(geo[rows, cols])),
        "physical_edges_median_km": float(np.median(geo[B])),
        "random_pairs_median_km": float(np.median(geo[rows[keep], rand_c[keep]])),
        f"top{K}_within_4km_frac": float((geo[rows, cols] <= 4.0).mean()),
        f"top{K}_distinct_targets": int(len(np.unique(cols))),
        "max_in_degree_topK": int(np.bincount(cols, minlength=N).max()),
        # do learned edges NOT in the physical graph link sensors that co-fluctuate?
        "corr_learned_nonphysical": float(np.nanmean(corr[rows[~in_phys], cols[~in_phys]])),
        "corr_learned_physical": float(np.nanmean(corr[rows[in_phys], cols[in_phys]])) if in_phys.any() else None,
        "corr_physical_edges": float(np.nanmean(corr[B])),
        "corr_random_pairs": float(np.nanmean(corr[rows[keep], rand_c[keep]])),
        "same_freeway_frac_topK": float((meta.Fwy.values[rows] == meta.Fwy.values[cols]).mean()),
        "same_freeway_frac_physical": float((meta.Fwy.values[:, None] == meta.Fwy.values[None])[B].mean()),
    }
    # weight-level comparison: learned weight on physical vs non-physical pairs
    nondiag = ~np.eye(N, dtype=bool)
    st["mean_learned_weight_on_physical_edges"] = float(L[B].mean())
    st["mean_learned_weight_on_nonphysical_pairs"] = float(L[~B & nondiag].mean())
    out[tag] = st

    # ---- figures: heatmap of learned adjacency + side-by-side with fixed
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    Fn = A_fix.copy(); np.fill_diagonal(Fn, 1); Fn = Fn / Fn.sum(1, keepdims=True)
    for a, M, t in ((ax[0], Fn, "fixed  $D^{-1}\\tilde A$ (row-normalised)"),
                    (ax[1], L, f"learned $A_{{adaptive}}$ ({tag})")):
        im = a.imshow(np.where(M > 1e-4, M, np.nan), cmap="viridis", norm=LogNorm(1e-4, 1),
                      interpolation="nearest")
        a.set_title(t); a.set_xlabel("j")
    ax[0].set_ylabel("i"); fig.colorbar(im, ax=ax, fraction=.025, label="weight (log)")
    fig.savefig(os.path.join(FIG, f"q5b_learned_adj_{tag}.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- map: top-3 learned edges per node, coloured by whether they exist physically
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(meta.Lng, meta.Lat, s=4, c="lightgrey", zorder=1)
    t3 = top[:, :3]
    for i in range(N):
        for j in t3[i]:
            c = "tab:blue" if B[i, j] else "tab:red"
            ax.plot([meta.Lng[i], meta.Lng[j]], [meta.Lat[i], meta.Lat[j]], c=c, lw=.4, alpha=.35, zorder=2)
    ax.plot([], [], c="tab:blue", label="learned edge also in road graph")
    ax.plot([], [], c="tab:red", label="learned edge NOT in road graph")
    ax.set_aspect("equal"); ax.legend(loc="lower left"); ax.set_title(f"Top-3 learned neighbours per sensor ({tag})")
    fig.savefig(os.path.join(FIG, f"q6b_learned_edges_map_{tag}.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- distance distribution
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.linspace(0, 80, 81)
    ax.hist(geo[B], bins=bins, density=True, alpha=.6, label="road-graph edges")
    ax.hist(geo[rows, cols], bins=bins, density=True, alpha=.6, label=f"top-{K} learned ({tag})")
    ax.hist(geo[rows[keep], rand_c[keep]], bins=bins, density=True, histtype="step", color="k", label="random pairs")
    ax.set_xlabel("great-circle distance (km)"); ax.set_ylabel("density"); ax.legend()
    fig.savefig(os.path.join(FIG, f"q6b_distance_{tag}.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def training_curves():
    fig, ax = plt.subplots(figsize=(8, 4))
    for tag, model, adj in MODELS:
        r = load(tag)
        if r:
            ax.plot([h["val_MAE"] for h in r["history"]], label=f"{model} ({adj})")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation MAE (veh / 15 min)"); ax.legend(); ax.grid(alpha=.3)
    fig.savefig(os.path.join(FIG, "q6_val_curves.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    r = load("hybrid")
    if r:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot([h["lambda"] for h in r["history"]]); ax.set_xlabel("epoch"); ax.set_ylabel("$\\lambda$")
        ax.set_title("Hybrid: learned weight on fixed graph"); ax.grid(alpha=.3)
        fig.savefig(os.path.join(FIG, "q5c_lambda.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    d = np.load(os.path.join(PROC, "sd_2020_2021_15min.npz"), allow_pickle=True)
    flow, obs, times = d["flow"], d["observed"], pd.to_datetime(d["times"])
    test_lo = int(np.searchsorted(times.values, np.datetime64("2021-04-01")))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for suffix, name in (("", "main"), ("_res", "residual_ablation")):
        t = table(suffix)
        if len(t):
            t.to_csv(os.path.join(RES, f"table_{name}.csv"), index=False)
            print(f"\n== {name}\n", t.round(3).to_string(index=False))
    pred_vs_actual(times, flow, obs, test_lo)
    pred_vs_actual(times, flow, obs, test_lo, suffix="_res")
    training_curves()
    A = np.load(os.path.join(PROC, "adj_sd.npy"))
    geo = np.load(os.path.join(PROC, "geo_km.npy"))
    corr = np.load(os.path.join(PROC, "corr_anomaly_15min_change_train.npy"))
    meta = pd.read_csv(os.path.join(PROC, "sd_meta.csv"))
    out = {}
    for tag in ("adaptive", "hybrid", "adaptive_res", "hybrid_res"):
        graph_analysis(tag, A, geo, corr, meta, out)
    with open(os.path.join(RES, "q6b_graph_stats.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
