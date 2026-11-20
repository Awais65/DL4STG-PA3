"""Q4/Q5 - lightweight spatio-temporal forecaster.

  X (B, T_in=8, N, 1)
   -> temporal encoder  (shared 1D conv over time, per sensor)  -> Z (B, N, d)
   -> graph conv        H = ReLU(A_hat Z W_g)                   -> H (B, N, d)
   -> forecast head     MLP d -> 4                              -> Y_hat (B, 4, N, 1)

The four model variants differ ONLY in how A_hat is obtained:
  none      : A_hat = I                         (temporal-only baseline, Q4c)
  fixed     : A_hat = D_out^-1 A~ (directed row-normalised road graph, Q5a, constant buffer)
  adaptive  : A_hat = softmax(relu(E1 E2^T))    (Q5b, learned)
  hybrid    : A_hat = lam*A_fixed + (1-lam)*A_adaptive, lam = sigmoid(theta) learned (Q5c)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """Two causal-free 'valid' convolutions over the time axis (kernel 3), then a conv
    whose kernel spans the remaining length collapses time into a d-dim feature."""

    def __init__(self, t_in=8, d=32, in_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, d, 3), nn.ReLU(),     # 8 -> 6
            nn.Conv1d(d, d, 3), nn.ReLU(),          # 6 -> 4
            nn.Conv1d(d, d, t_in - 4), nn.ReLU(),   # 4 -> 1
        )

    def forward(self, x):                           # x: (B, T, N, F)
        B, T, N, Fd = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * N, Fd, T)
        z = self.net(x).squeeze(-1)                 # (B*N, d)
        return z.view(B, N, -1)


class GraphConv(nn.Module):
    """H = ReLU(A_hat Z W). Multiplication over the node axis, independently per batch item."""

    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d, bias=True)

    def forward(self, z, A):                        # z: (B,N,d), A: (N,N)
        return F.relu(torch.einsum("nm,bmd->bnd", A, self.W(z)))


def row_normalize(A):
    """Directed random-walk normalisation A_hat = D_out^-1 A~, with A~ = A with self-loops of
    weight 1 (LargeST's road adjacency already has A_ii = 1, so the diagonal is SET to 1
    rather than adding I, which would double-count it). D_out = diag(row sums of A~).
    Row i of A_hat is a weighted average over i and the sensors j with A_ij > 0; no symmetry
    is assumed, and every row sums to 1 like the row-wise softmax of the adaptive graph."""
    A = A.clone()
    A.fill_diagonal_(1.0)
    deg = A.sum(1, keepdim=True)
    return A / deg.clamp(min=1e-12)


class AdaptiveAdjacency(nn.Module):
    """A_adaptive = Softmax_row(ReLU(E1 E2^T)), E1,E2 in R^{N x d_e} learned by backprop.
    Optional top-k: logits outside each row's top-k are set to -inf BEFORE the softmax,
    so each row is renormalised over its k kept neighbours and gradients flow only through
    the kept entries (the top-k selection itself is non-differentiable, like argmax)."""

    def __init__(self, n_nodes, d_e=10, topk=None):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(n_nodes, d_e) / d_e ** 0.5)
        self.E2 = nn.Parameter(torch.randn(n_nodes, d_e) / d_e ** 0.5)
        self.topk = topk

    def forward(self):
        logits = F.relu(self.E1 @ self.E2.T)
        if self.topk is not None:
            kth = logits.topk(self.topk, dim=1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        return torch.softmax(logits, dim=1)


class STGNN(nn.Module):
    def __init__(self, n_nodes, mode, A_fixed=None, t_in=8, t_out=4, d=32, d_e=10,
                 topk=None, lam_init=0.5, residual=False):
        super().__init__()
        self.residual = residual      # ablation only: head sees Z + H instead of H
        assert mode in ("none", "fixed", "adaptive", "hybrid")
        self.mode, self.t_out = mode, t_out
        self.encoder = TemporalEncoder(t_in, d)
        self.gconv = GraphConv(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, t_out))

        if mode in ("fixed", "hybrid"):
            self.register_buffer("A_fixed", row_normalize(A_fixed))     # never updated
        if mode in ("adaptive", "hybrid"):
            self.adaptive = AdaptiveAdjacency(n_nodes, d_e, topk)
        if mode == "hybrid":
            p = torch.tensor(lam_init)
            self.lam_logit = nn.Parameter(torch.log(p / (1 - p)))     # lam = sigmoid(.)
        if mode == "none":
            self.register_buffer("I", torch.eye(n_nodes))

    @property
    def lam(self):
        return torch.sigmoid(self.lam_logit)

    def adjacency(self):
        if self.mode == "none":
            return self.I
        if self.mode == "fixed":
            return self.A_fixed
        if self.mode == "adaptive":
            return self.adaptive()
        return self.lam * self.A_fixed + (1 - self.lam) * self.adaptive()

    def forward(self, x):                           # (B, T_in, N, 1)
        z = self.encoder(x)
        h = self.gconv(z, self.adjacency())
        if self.residual:
            h = h + z
        y = self.head(h)                            # (B, N, T_out)
        return y.permute(0, 2, 1).unsqueeze(-1)     # (B, T_out, N, 1)
