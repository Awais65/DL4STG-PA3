"""Q2 - Which way is the slowdown moving? (40-sensor ring road)"""
import os
import numpy as np
import matplotlib.pyplot as plt

N = 40
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)


def true_step(x):                      # x_i(t+1) = x_{i+1}(t)
    return np.roll(x, -1)


def sym_layer(x, a, b):                # a x_i + b (x_{i-1} + x_{i+1})
    return a * x + b * (np.roll(x, 1) + np.roll(x, -1))


# ---------------------------------------------------------------- (a)(i)
def fit_symmetric():
    x = np.zeros(N); x[20] = 1
    y = true_step(x)
    # design matrix: columns are x_i and (x_{i-1}+x_{i+1})
    X = np.stack([x, np.roll(x, 1) + np.roll(x, -1)], axis=1)
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    sse = np.sum((sym_layer(x, a, b) - y) ** 2)
    print(f"Q2(a)i  least squares: a={a:.4f}, b={b:.4f}, SSE={sse:.4f}")
    print(f"        preds at 19,20,21: {sym_layer(x, a, b)[19:22]}, targets: {y[19:22]}")
    return a, b


# ---------------------------------------------------------------- (a)(ii)
def rollout(a, b, steps=20):
    i = np.arange(N)
    x0 = np.exp(-(i - 20) ** 2 / 8)
    pred, true = [x0], [x0]
    for _ in range(steps):
        pred.append(sym_layer(pred[-1], a, b))
        true.append(true_step(true[-1]))
    pred, true = np.array(pred), np.array(true)

    vmax = max(pred.max(), true.max())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for k, (M, title) in enumerate([(pred, f"Symmetric layer (a={a:.2f}, b={b:.2f})"),
                                    (true, "True motion  $x_i(t+1)=x_{i+1}(t)$")]):
        im = ax[k].imshow(M, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=vmax,
                          extent=[-0.5, N - 0.5, -0.5, steps + 0.5])
        ax[k].set_title(title); ax[k].set_xlabel("sensor index i"); ax[k].set_yticks(range(0, steps + 1, 2))
    ax[0].set_ylabel("time step t")
    fig.colorbar(im, ax=ax, label="slowdown")
    fig.savefig(os.path.join(FIG, "q2a_heatmaps.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    com = lambda M: (M * i).sum(1) / M.sum(1)
    print(f"Q2(a)ii t=20  symmetric: peak={pred[-1].max():.3f} at {pred[-1].argmax()}, "
          f"centre of mass={com(pred)[-1]:.2f}, mass={pred[-1].sum():.3f}")
    print(f"        t=20  true:      peak={true[-1].max():.3f} at {true[-1].argmax()}, "
          f"mass={true[-1].sum():.3f}")


# ---------------------------------------------------------------- (b)
def directional():
    import torch
    a, wm, wp = 0.0, 0.0, 1.0
    rng = np.random.default_rng(0)
    for _ in range(100):
        x = rng.normal(size=N)
        pred = a * x + wm * np.roll(x, 1) + wp * np.roll(x, -1)
        assert np.allclose(pred, true_step(x))
    # same rule as a 1D convolution with circular padding
    conv = torch.nn.Conv1d(1, 1, 3, padding=1, padding_mode="circular", bias=False)
    with torch.no_grad():
        conv.weight[:] = torch.tensor([[[wm, a, wp]]])   # multiplies [x_{i-1}, x_i, x_{i+1}]
        x = torch.tensor(rng.normal(size=N), dtype=torch.float32)
        out = conv(x.view(1, 1, N)).view(-1).numpy()
    assert np.allclose(out, true_step(x.numpy()), atol=1e-6)
    print(f"Q2(b)   a={a}, w-={wm}, w+={wp}: exact on 100 random snapshots; "
          f"Conv1d kernel [0,0,1] + circular padding verified")


# ---------------------------------------------------------------- (c)
def history_layer(x_t, x_prev, a, b, c):
    return a * x_t + b * (np.roll(x_t, 1) + np.roll(x_t, -1)) + c * x_prev


def with_history():
    a, b, c = 0.0, 1.0, -1.0
    rng = np.random.default_rng(1)
    back = lambda x: np.roll(x, -1)    # x_i(t+1) = x_{i+1}(t)
    fwd = lambda x: np.roll(x, 1)      # x_i(t+1) = x_{i-1}(t)
    for name, step in (("backward", back), ("forward", fwd)):
        ok = True
        for _ in range(100):
            x_prev = rng.normal(size=N)
            x_t = step(x_prev)
            ok &= np.allclose(history_layer(x_t, x_prev, a, b, c), step(x_t))
        print(f"Q2(c)   a={a}, b={b}, c={c}: exact for {name} motion on 100 random pairs: {ok}")

    # bonus figure: the history layer rolled out on the bump tracks the true motion
    i = np.arange(N)
    x0 = np.exp(-(i - 20) ** 2 / 8)
    prev, cur, seq = np.roll(x0, 1), x0, [x0]   # x(-1) consistent with backward motion
    for _ in range(20):
        prev, cur = cur, history_layer(cur, prev, a, b, c)
        seq.append(cur)
    seq = np.array(seq)
    true = np.array([np.roll(x0, -t) for t in range(21)])
    print(f"        bump rollout (20 steps) max |error| vs truth: {np.abs(seq - true).max():.2e}")


if __name__ == "__main__":
    a, b = fit_symmetric()
    rollout(a, b)
    directional()
    with_history()
