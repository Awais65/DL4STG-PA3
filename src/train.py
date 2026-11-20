"""Q4/Q5 - train and evaluate one model variant.

usage: python src/train.py --mode {none,fixed,adaptive,hybrid} [--epochs 50] [--seed 42]
Outputs: results/<mode>.json (metrics, timing, config), results/<mode>.pt (best weights),
         results/<mode>_test_pred.npy (test predictions in original units, for plots).
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import torch

from models import STGNN

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
T_IN, T_OUT = 8, 4


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------ data
def load_data():
    d = np.load(os.path.join(PROC, "sd_2020_2021_15min.npz"), allow_pickle=True)
    return d["flow"].astype(np.float32), d["observed"], d["times"].astype("datetime64[ns]"), \
        np.load(os.path.join(PROC, "adj_sd.npy")).astype(np.float32)


def split_bounds(times):
    """Chronological splits. Returned as [start, end) index ranges on the 15-min axis."""
    def idx(ts):
        return int(np.searchsorted(times, np.datetime64(ts)))
    return {
        "train": (idx("2020-01-01"), idx("2021-01-01")),
        "val": (idx("2021-01-01"), idx("2021-04-01")),
        "test": (idx("2021-04-01"), idx("2022-01-01")),
    }


def window_starts(lo, hi):
    """Window = x[s : s+8] -> y[s+8 : s+12]; whole window (input AND target) inside [lo, hi),
    so no forecasting window crosses a split boundary."""
    return np.arange(lo, hi - (T_IN + T_OUT) + 1)


class WindowLoader:
    """Mini-batches of windows, gathered directly from device-resident tensors.
    x: normalised input (B,8,N,1); y: target in original units (B,4,N,1);
    m: 1 where the target was truly observed (filled values are excluded from loss/metrics)."""

    def __init__(self, xn, flow, obs, starts, batch, shuffle, seed):
        self.xn, self.flow, self.obs = xn, flow, obs
        self.starts = torch.as_tensor(starts, device=xn.device)
        self.batch, self.shuffle = batch, shuffle
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.off_in = torch.arange(T_IN, device=xn.device)
        self.off_out = torch.arange(T_IN, T_IN + T_OUT, device=xn.device)

    def __len__(self):
        return (len(self.starts) + self.batch - 1) // self.batch

    def __iter__(self):
        order = (torch.randperm(len(self.starts), generator=self.g).to(self.starts.device)
                 if self.shuffle else torch.arange(len(self.starts), device=self.starts.device))
        for k in range(0, len(order), self.batch):
            s = self.starts[order[k:k + self.batch]]
            ti, to = s[:, None] + self.off_in, s[:, None] + self.off_out
            yield self.xn[ti].unsqueeze(-1), self.flow[to].unsqueeze(-1), self.obs[to].unsqueeze(-1)


# ------------------------------------------------------------------ metrics
def masked_mae(pred, y, m):
    m = m.float()
    return (torch.abs(pred - y) * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, loader, mean, std, device, keep_pred=False):
    model.eval()
    abs_err = torch.zeros(T_OUT, dtype=torch.float64)
    sq_err = torch.zeros(T_OUT, dtype=torch.float64)
    cnt = torch.zeros(T_OUT, dtype=torch.float64)
    preds = []
    for x, y, m in loader:
        m = m.float()
        p = model(x) * std + mean
        e = (p - y) * m
        abs_err += e.abs().sum((0, 2, 3)).double().cpu()
        sq_err += (e ** 2).sum((0, 2, 3)).double().cpu()
        cnt += m.sum((0, 2, 3)).double().cpu()
        if keep_pred:
            preds.append(p.squeeze(-1).cpu().numpy().astype(np.float32))
    out = {
        "MAE": float(abs_err.sum() / cnt.sum()),
        "RMSE": float(torch.sqrt(sq_err.sum() / cnt.sum())),
        "MAE_h": (abs_err / cnt).tolist(),
        "RMSE_h": (sq_err / cnt).sqrt().tolist(),
    }
    return out, (np.concatenate(preds) if keep_pred else None)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["none", "fixed", "adaptive", "hybrid"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--d_e", type=int, default=10)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--lam_init", type=float, default=0.5)
    ap.add_argument("--residual", action="store_true", help="ablation: skip connection Z+H")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.mode
    os.makedirs(RES, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    flow, observed, times, A = load_data()
    b = split_bounds(times)

    # normalisation statistics: TRAINING period only, observed entries only
    tr = slice(*b["train"])
    mean = float(flow[tr][observed[tr]].mean())
    std = float(flow[tr][observed[tr]].std())
    xn = ((flow - mean) / std).astype(np.float32)

    xn_t = torch.from_numpy(xn).to(device)
    flow_t = torch.from_numpy(flow).to(device)
    obs_t = torch.from_numpy(observed).to(device)
    loaders = {name: WindowLoader(xn_t, flow_t, obs_t, window_starts(lo, hi), args.batch,
                                  shuffle=(name == "train"), seed=args.seed)
               for name, (lo, hi) in b.items()}
    n_win = {k: len(v.starts) for k, v in loaders.items()}
    print(f"[{tag}] windows: {n_win}  mean={mean:.2f} std={std:.2f}")

    N = flow.shape[1]
    model = STGNN(N, args.mode, A_fixed=torch.from_numpy(A), d=args.hidden, d_e=args.d_e,
                  topk=args.topk, lam_init=args.lam_init, residual=args.residual).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{tag}] trainable parameters: {n_params}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

    std_t = torch.tensor(std, device=device)
    mean_t = torch.tensor(mean, device=device)
    best, bad, history = float("inf"), 0, []
    ckpt = os.path.join(RES, f"{tag}.pt")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        te = time.perf_counter(); tot, nb = 0.0, 0
        for x, y, m in loaders["train"]:
            pred = model(x) * std_t + mean_t          # loss in original units (masked MAE)
            loss = masked_mae(pred, y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        val, _ = evaluate(model, loaders["val"], mean_t, std_t, device)
        sched.step(val["MAE"])
        rec = {"epoch": ep, "train_MAE": tot / nb, "val_MAE": val["MAE"], "val_RMSE": val["RMSE"],
               "epoch_sec": time.perf_counter() - te}
        if args.mode == "hybrid":
            rec["lambda"] = float(model.lam)
        history.append(rec)
        print(f"[{tag}] ep {ep:3d} train {rec['train_MAE']:.3f} val MAE {val['MAE']:.3f} "
              f"RMSE {val['RMSE']:.3f} ({rec['epoch_sec']:.1f}s)"
              + (f" lam={rec['lambda']:.3f}" if "lambda" in rec else ""), flush=True)
        if val["MAE"] < best - 1e-4:
            best, bad = val["MAE"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[{tag}] early stop at epoch {ep}")
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.perf_counter() - t0

    model.load_state_dict(torch.load(ckpt, map_location=device))
    val, _ = evaluate(model, loaders["val"], mean_t, std_t, device)
    test, test_pred = evaluate(model, loaders["test"], mean_t, std_t, device, keep_pred=True)
    np.save(os.path.join(RES, f"{tag}_test_pred.npy"), test_pred)
    if args.mode in ("adaptive", "hybrid"):
        with torch.no_grad():
            np.save(os.path.join(RES, f"{tag}_A_adaptive.npy"),
                    model.adaptive().cpu().numpy().astype(np.float32))

    out = {
        "mode": args.mode, "tag": tag, "config": vars(args), "n_params": n_params,
        "norm": {"mean": mean, "std": std}, "windows": n_win,
        "epochs_run": len(history), "best_epoch": int(np.argmin([h["val_MAE"] for h in history]) + 1),
        "train_time_sec": train_time,
        "sec_per_epoch": float(np.mean([h["epoch_sec"] for h in history])),
        "val": val, "test": test, "history": history,
        "hardware": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "cpu": platform.processor(), "torch": torch.__version__,
        },
    }
    if args.mode == "hybrid":
        out["lambda_final"] = float(model.lam)
    with open(os.path.join(RES, f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{tag}] DONE  val MAE {val['MAE']:.3f} RMSE {val['RMSE']:.3f} | "
          f"test MAE {test['MAE']:.3f} RMSE {test['RMSE']:.3f} | time {train_time:.1f}s")


if __name__ == "__main__":
    main()
