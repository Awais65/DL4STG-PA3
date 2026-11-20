"""Q3 - dataset plots, graph statistics and neighbourhood analysis."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROC, FIG, RES = (os.path.join(ROOT, p) for p in ("data/processed", "figures", "results"))
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)


def main():
    d = np.load(os.path.join(PROC, "sd_2020_2021_15min.npz"), allow_pickle=True)
    flow, obs, times = d["flow"], d["observed"], pd.to_datetime(d["times"])
    meta = pd.read_csv(os.path.join(PROC, "sd_meta.csv"))
    A = np.load(os.path.join(PROC, "adj_sd.npy"))
    N = A.shape[0]
    assert np.array_equal(meta.ID.values, d["sensor_ids"]), "sensor order mismatch"
    stats = {}

    # ---------------- Q3(a).5  three sensors over one week
    wk = (times >= "2020-02-03") & (times < "2020-02-10")         # Mon-Sun, pre-COVID
    full = obs[wk].mean(0) == 1
    # three fully observed sensors on three different freeways
    picks = []
    for fwy in meta.Fwy[full].value_counts().index:
        j = np.where(full & (meta.Fwy.values == fwy))[0]
        picks.append(int(j[len(j) // 2]))
        if len(picks) == 3:
            break
    fig, ax = plt.subplots(figsize=(12, 4))
    for j in picks:
        ax.plot(times[wk], flow[wk, j], lw=1.1,
                label=f"sensor {meta.ID[j]} ({meta.Fwy[j]}, idx {j})")
    ax.set_ylabel("flow (vehicles / 15 min)"); ax.set_title("Traffic flow, 3 Feb - 9 Feb 2020")
    ax.legend(); ax.grid(alpha=.3)
    fig.savefig(os.path.join(FIG, "q3a_week.png"), dpi=160, bbox_inches="tight"); plt.close(fig)
    stats["week_plot_sensors"] = picks

    # missingness over time (fraction of sensors missing per day)
    daily = pd.Series(1 - obs.mean(1), index=times).resample("D").mean()
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.plot(daily.index, daily.values * 100, lw=1)
    for t in ("2021-01-01", "2021-04-01"):
        ax.axvline(pd.Timestamp(t), color="k", ls="--", lw=.8)
    ax.set_ylabel("% missing"); ax.set_title("Daily missing rate (15-min, before filling); dashed = split boundaries")
    fig.savefig(os.path.join(FIG, "q3a_missing.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

    # ---------------- Q3(b)  graph statistics
    off = A.copy(); np.fill_diagonal(off, 0)
    B = off > 0
    stats["graph"] = {
        "nodes": N,
        "nonzero_entries_incl_diag": int((A > 0).sum()),
        "self_loops": int((np.diag(A) > 0).sum()),
        "directed_edges_excl_self": int(B.sum()),
        "symmetric_values": bool(np.allclose(A, A.T)),
        "asymmetric_pairs": int((B != B.T).sum() // 2),
        "reciprocal_pairs": int((B & B.T).sum() // 2),
        "max_abs_A_minus_AT": float(np.abs(A - A.T).max()),
        "weight_range": [float(off[B].min()), float(off[B].max())],
        "out_degree": {"min": int(B.sum(1).min()), "mean": float(B.sum(1).mean()),
                       "median": float(np.median(B.sum(1))), "max": int(B.sum(1).max())},
        "in_degree": {"min": int(B.sum(0).min()), "mean": float(B.sum(0).mean()),
                      "max": int(B.sum(0).max())},
        "isolated_nodes": int(((B.sum(1) + B.sum(0)) == 0).sum()),
        "density": float(B.sum() / (N * (N - 1))),
    }

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    im = ax[0].imshow(np.where(A > 0, A, np.nan), cmap="viridis", norm=LogNorm(1e-2, 1),
                      interpolation="nearest")
    ax[0].set_title("$A_{SD}$ (716 x 716), log colour, white = 0")
    ax[0].set_xlabel("j (sensor index)"); ax[0].set_ylabel("i (sensor index)")
    fig.colorbar(im, ax=ax[0], fraction=.046)
    zoom = slice(0, 120)
    im = ax[1].imshow(np.where(A[zoom, zoom] > 0, A[zoom, zoom], np.nan), cmap="viridis",
                      norm=LogNorm(1e-2, 1), interpolation="nearest")
    ax[1].set_title("zoom: first 120 sensors"); ax[1].set_xlabel("j")
    fig.savefig(os.path.join(FIG, "q3b_adjacency.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.arange(0, max(B.sum(1).max(), B.sum(0).max()) + 3, 2)
    ax.hist(B.sum(1), bins=bins, alpha=.7, label="out-degree (#j with $A_{ij}>0$)")
    ax.hist(B.sum(0), bins=bins, alpha=.5, label="in-degree (#i with $A_{ij}>0$)")
    ax.set_xlabel("number of neighbours (self-loop excluded)"); ax.set_ylabel("sensors"); ax.legend()
    fig.savefig(os.path.join(FIG, "q3b_degree.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

    # weights vs geographic distance (what a nonzero A_ij encodes)
    lat, lng = np.radians(meta.Lat.values), np.radians(meta.Lng.values)
    dlat = lat[:, None] - lat[None]; dlng = lng[:, None] - lng[None]
    h = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin(dlng / 2) ** 2
    geo = 2 * 6371 * np.arcsin(np.sqrt(h))
    np.save(os.path.join(PROC, "geo_km.npy"), geo.astype(np.float32))
    stats["graph"]["geo_km_of_edges"] = {
        "median": float(np.median(geo[B])), "p95": float(np.percentile(geo[B], 95)),
        "max": float(geo[B].max()), "corr_weight_vs_geo": float(np.corrcoef(off[B], geo[B])[0, 1])}
    stats["graph"]["same_direction_edge_frac"] = float(
        (meta.Direction.values[:, None] == meta.Direction.values[None])[B].mean())
    stats["graph"]["same_freeway_edge_frac"] = float(
        (meta.Fwy.values[:, None] == meta.Fwy.values[None])[B].mean())

    # ---------------- Q3(c)  one- and two-hop neighbourhoods
    deg = B.sum(1)
    v = int(np.argsort(np.abs(deg - np.median(deg)))[0])          # a typical (median-degree) sensor
    hop1 = set(np.where(B[v])[0])
    two = (B.astype(int) @ B.astype(int))[v] > 0
    hop2 = set(np.where(two | B[v])[0]) - {v}
    hop2_only = hop2 - hop1
    stats["neighbourhood"] = {
        "sensor_idx": v, "sensor_id": int(meta.ID[v]), "fwy": meta.Fwy[v],
        "one_hop": len(hop1), "two_hop_total": len(hop2), "two_hop_only": len(hop2_only),
        "one_hop_ids": sorted(int(meta.ID[j]) for j in hop1),
        "one_hop_freeways": meta.Fwy[sorted(hop1)].value_counts().to_dict(),
        "two_hop_only_freeways": meta.Fwy[sorted(hop2_only)].value_counts().to_dict(),
        "corr_with_one_hop_mean": None,
    }
    # Raw flows of all sensors share the daily cycle (corr ~0.9 everywhere), so compare
    # ANOMALIES: training-year flow minus each sensor's time-of-week mean, observed points only.
    tr = np.asarray(times < "2021-01-01")
    slot = (times.dayofweek * 96 + times.hour * 4 + times.minute // 15)[tr]
    F = np.where(obs[tr], flow[tr], np.nan)
    prof = pd.DataFrame(F).groupby(np.asarray(slot)).transform("mean").values
    R = F - prof
    # also remove each day's own mean per sensor, so the network-wide 2020 (COVID) level
    # shift does not make every pair look correlated -> intra-day anomaly
    day = np.asarray(times[tr].normalize().astype("int64"))
    R = R - pd.DataFrame(R).groupby(day).transform("mean").values
    rest = sorted(set(range(N)) - hop2 - {v})

    def corr(a, b):
        ok = ~np.isnan(a) & ~np.isnan(b)
        return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 1000 and b[ok].std() > 0 else np.nan

    c = np.array([corr(R[:, v], R[:, j]) for j in range(N)])
    for name, grp in (("one_hop", hop1), ("two_hop_only", hop2_only), ("beyond", set(rest))):
        stats["neighbourhood"][f"median_anomaly_corr_{name}"] = float(np.nanmedian(c[sorted(grp)]))
    stats["neighbourhood"].pop("corr_with_one_hop_mean")
    # same statistic averaged over ALL sensors: anomaly corr of i with its 1-hop vs non-neighbours
    good = obs[tr].mean(0) > 0.8
    pairs = B & good[:, None] & good[None]
    nonadj = ~B & ~np.eye(N, dtype=bool) & good[:, None] & good[None]
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, X in (("intraday_anomaly", R), ("anomaly_15min_change", np.diff(R, axis=0))):
            Z = np.nan_to_num((X - np.nanmean(X, 0)) / np.nanstd(X, 0))
            C = (Z.T @ Z) / len(Z)                                # approx. corr (NaN -> 0)
            stats["graph"][f"corr_{name}_adjacent_mean"] = float(C[pairs].mean())
            stats["graph"][f"corr_{name}_nonadjacent_mean"] = float(C[nonadj].mean())
            np.save(os.path.join(PROC, f"corr_{name}_train.npy"), C.astype(np.float32))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(meta.Lng, meta.Lat, s=5, c="lightgrey", label="other sensors")
    ax.scatter(meta.Lng[sorted(hop2_only)], meta.Lat[sorted(hop2_only)], s=14, c="tab:orange", label=f"2-hop only ({len(hop2_only)})")
    ax.scatter(meta.Lng[sorted(hop1)], meta.Lat[sorted(hop1)], s=18, c="tab:blue", label=f"1-hop ({len(hop1)})")
    ax.scatter(meta.Lng[v], meta.Lat[v], s=120, c="red", marker="*", label=f"sensor {meta.ID[v]} ({meta.Fwy[v]})")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_aspect("equal"); ax.legend(loc="lower left")
    ax.set_title("One- and two-hop neighbourhoods (directed, $A_{ij}>0$)")
    fig.savefig(os.path.join(FIG, "q3c_neighbourhood.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

    # sensor map coloured by freeway (used later as context)
    with open(os.path.join(RES, "q3_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
