"""Q3(a)/(b) - extract LargeST San Diego (716 sensors, 2020-2021), resample 5 -> 15 min,
handle missing values, and extract the matching adjacency submatrix.

Raw files (Kaggle liuxu77/largest), default location set by --raw:
  ca_his_raw_2020.h5, ca_his_raw_2021.h5   (pandas fixed-format HDF5, key 't')
  ca_meta.csv, ca_rn_adj.npy

Output: data/processed/sd_2020_2021_15min.npz  (flow, observed, times, sensor_ids)
        data/processed/adj_sd.npy, data/processed/sd_meta.csv, data/processed/missing_report.json
"""
import argparse
import json
import os
import warnings

import h5py
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "data", "processed")
SHORT_GAP = 4      # gaps of <= 4 x 15 min (1 h) are linearly interpolated


def read_year(path, sd_ids):
    with h5py.File(path, "r") as f:
        ids = f["t/axis0"][:].astype(str)
        pos = {s: i for i, s in enumerate(ids)}
        cols = np.array([pos[str(s)] for s in sd_ids])      # match by sensor ID, not position
        times = pd.to_datetime(f["t/axis1"][:])
        lo, hi = cols.min(), cols.max() + 1
        block = f["t/block0_values"][:, lo:hi]               # contiguous read, then reorder
        vals = block[:, cols - lo].astype(np.float32)
    assert (np.diff(times.values) == np.timedelta64(5, "m")).all(), "irregular 5-min grid"
    return times, vals


def to_15min(times, v5):
    """Values are vehicle COUNTS per 5-min interval (all integer, mean ~200/station), so the
    15-min value is the SUM of the three 5-min counts -> vehicles per 15 min.
    If 1-2 of the three sub-intervals are missing, the sum is estimated as
    3 x mean(available) (= the exact sum when all three exist); if all three are missing
    the 15-min value is missing."""
    assert times[0].minute % 15 == 0 and len(times) % 3 == 0
    T, N = v5.shape
    r = v5.reshape(T // 3, 3, N)
    n_obs = (~np.isnan(r)).sum(1)
    with np.errstate(invalid="ignore"):
        v15 = 3.0 * np.nanmean(r, axis=1)
    v15[n_obs == 0] = np.nan
    return times[::3], v15.astype(np.float32), n_obs


def interpolate_short_gaps(v, limit):
    """Linear interpolation inside gaps of length <= limit only (both ends observed)."""
    isna = np.isnan(v)
    out = pd.DataFrame(v).interpolate(method="linear", limit_area="inside", axis=0).values
    for j in range(v.shape[1]):
        col = isna[:, j]
        if not col.any():
            continue
        # run-length encode NaN runs; undo interpolation inside runs longer than `limit`
        d = np.diff(np.concatenate([[0], col.astype(int), [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            if e - s > limit:
                out[s:e, j] = np.nan
    return out


def tow_profile(times, v):
    """Per-sensor time-of-week mean (7 x 96 slots) from observed values."""
    slot = (times.dayofweek.values * 96 + times.hour.values * 4 + times.minute.values // 15)
    prof = np.full((672, v.shape[1]), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN slices -> NaN
        for s in range(672):
            prof[s] = np.nanmean(v[slot == s], axis=0)
    return prof, slot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.expanduser("~/datasets/LargeST"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    meta = pd.read_csv(os.path.join(args.raw, "ca_meta.csv"))
    sd = meta[meta.County == "San Diego"].reset_index(drop=True)
    print(f"San Diego sensors: {len(sd)}")
    sd.to_csv(os.path.join(OUT, "sd_meta.csv"), index=False)

    # ---------------- adjacency: rows/cols ID2 index the CA-wide matrix in meta order
    A_ca = np.load(os.path.join(args.raw, "ca_rn_adj.npy"), mmap_mode="r")
    A = np.asarray(A_ca[np.ix_(sd.ID2.values, sd.ID2.values)], dtype=np.float32)
    np.save(os.path.join(OUT, "adj_sd.npy"), A)

    # ---------------- traffic, same sensor order as sd (=> same order as A)
    T15, V15, NOBS = [], [], []
    report = {}
    for year in (2020, 2021):
        t5, v5 = read_year(os.path.join(args.raw, f"ca_his_raw_{year}.h5"), sd.ID.values)
        t15, v15, nobs = to_15min(t5, v5)
        report[year] = {
            "steps_5min": int(len(t5)), "steps_15min": int(len(t15)),
            "missing_frac_5min": float(np.isnan(v5).mean()),
            "missing_frac_15min": float(np.isnan(v15).mean()),
            "partial_15min_frac (1-2 of 3 present)": float(((nobs > 0) & (nobs < 3)).mean()),
            "sensors_all_missing": int(np.isnan(v15).all(0).sum()),
            "sensors_gt50pct_missing": int((np.isnan(v15).mean(0) > 0.5).sum()),
        }
        print(year, report[year])
        T15.append(t15); V15.append(v15)
        del v5
    times = T15[0].append(T15[1])
    flow = np.concatenate(V15)
    observed = ~np.isnan(flow)

    # ---------------- missing-value handling (no information flows across split boundaries)
    bounds = [("train", "2020-01-01", "2021-01-01"), ("val", "2021-01-01", "2021-04-01"),
              ("test", "2021-04-01", "2022-01-01")]
    tr = (times >= "2020-01-01") & (times < "2021-01-01")
    # step 1: short gaps (<= 1 h) -> linear interpolation, done within each split separately
    for name, lo, hi in bounds:
        sel = (times >= lo) & (times < hi)
        flow[sel] = interpolate_short_gaps(flow[sel], SHORT_GAP)
    after_interp = np.isnan(flow).mean()
    # step 2: long gaps -> the sensor's time-of-week mean from the TRAINING year only
    prof, slot = tow_profile(times[tr], np.where(observed[tr], flow[tr], np.nan))
    # sensors with no training data: average profile of road-graph neighbours with data,
    # else the network-wide training profile
    has = ~np.isnan(prof).all(0)
    glob = np.nanmean(prof[:, has], axis=1)
    for j in np.where(~has)[0]:
        nb = np.where((A[j] > 0) & has)[0]
        nb = nb[nb != j]
        prof[:, j] = np.nanmean(prof[:, nb], axis=1) if len(nb) else glob
    prof = np.where(np.isnan(prof), glob[:, None], prof)            # empty individual slots
    slot_all = times.dayofweek.values * 96 + times.hour.values * 4 + times.minute.values // 15
    miss = np.isnan(flow)
    flow[miss] = prof[slot_all][miss]
    assert not np.isnan(flow).any()

    report["combined"] = {
        "T_15min": int(len(times)), "N": int(flow.shape[1]),
        "observed_frac": float(observed.mean()),
        "missing_after_short_gap_interp": float(after_interp),
        "filled_by_tow_profile_frac": float(miss.mean()),
        "sensors_without_training_data": int((~has).sum()),
        "unit": "vehicles per 15 min (sum of three 5-min counts)",
    }
    print(report["combined"])
    with open(os.path.join(OUT, "missing_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    np.savez_compressed(os.path.join(OUT, "sd_2020_2021_15min.npz"), flow=flow.astype(np.float32),
                        observed=observed, times=times.values.astype("datetime64[ns]"),
                        sensor_ids=sd.ID.values)
    print("saved", flow.shape)


if __name__ == "__main__":
    main()
