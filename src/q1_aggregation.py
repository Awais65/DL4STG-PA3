"""Q1 - What does a sum remember?

(a) affine messages collapse a neighbourhood to (n_v, S_v)
(b) place-value (base K+1) message makes the sum injective on trend counts
(c) one-hot messages under sum / mean / max aggregation
"""
import itertools
import numpy as np

TRENDS = (-1, 0, 1)


# ---------------------------------------------------------------- (a)
def affine_aggregate(neigh, w, c):
    return sum(w * x + c for x in neigh)          # = w*S_v + c*n_v


def part_a():
    N1 = [-1, +1, 0, 0, -1]     # the neighbourhood drawn in the figure: 2 steady
    N2 = [-1, 0, 0, 0, 0]       # 4 steady
    for N in (N1, N2):
        print(f"  N={N}: n_v={len(N)}, S_v={sum(N)}, steady={N.count(0)}")
    rng = np.random.default_rng(0)
    for _ in range(5):
        w, c = rng.normal(size=2)
        a1, a2 = affine_aggregate(N1, w, c), affine_aggregate(N2, w, c)
        assert np.isclose(a1, a2)
    print("  aggregated messages identical for random (w, c) -> any rho gives the same h_v")


# ---------------------------------------------------------------- (b)
def place_value_message(K):
    B = K + 1                                    # base: every count <= K < B, so no carries
    return {-1: 1, 0: B, 1: B ** 2}, B


def decode(s, B):
    return s % B, (s // B) % B, s // B ** 2      # (n_slow, n_steady, n_fast)


def all_count_triples(K):
    """Every (n_-, n_0, n_+) with 1 <= n_- + n_0 + n_+ <= K."""
    for a, b, c in itertools.product(range(K + 1), repeat=3):
        if 1 <= a + b + c <= K:
            yield a, b, c


def part_b(K=8):
    phi, B = place_value_message(K)
    print(f"  K={K}: phi(-1)={phi[-1]}, phi(0)={phi[0]}, phi(+1)={phi[1]} (base {B})")
    triples = list(all_count_triples(K))
    sums = {}
    for t in triples:
        s = t[0] * phi[-1] + t[1] * phi[0] + t[2] * phi[1]
        assert decode(s, B) == t, (t, s)
        assert s not in sums, f"collision {t} vs {sums[s]}"
        sums[s] = t
    print(f"  verified {len(triples)} count triples: all sums distinct and decoded exactly")

    # vectorised check over every actual multiset neighbourhood (same thing, NumPy only)
    T = np.array(triples)
    S = T @ np.array([phi[-1], phi[0], phi[1]])
    D = np.stack([S % B, (S // B) % B, S // B ** 2], axis=1)
    assert np.array_equal(D, T) and len(np.unique(S)) == len(S)

    # simpler target: number of non-steady neighbours, phi(x) = x^2
    for t in triples:
        neigh = [-1] * t[0] + [0] * t[1] + [1] * t[2]
        assert sum(x * x for x in neigh) == t[0] + t[2]
    print("  phi(x)=x^2 (equivalently |x|) sums to #non-steady for every neighbourhood")


# ---------------------------------------------------------------- (c)
ONE_HOT = {-1: np.array([1, 0, 0]), 0: np.array([0, 1, 0]), 1: np.array([0, 0, 1])}
AGG = {
    "sum": lambda M: M.sum(0),
    "mean": lambda M: M.mean(0),
    "max": lambda M: M.max(0),
}
TASKS = {
    "(i) count steady": lambda N: N.count(0),
    "(ii) >half slowing": lambda N: N.count(-1) > len(N) / 2,
    "(iii) any slowing": lambda N: N.count(-1) > 0,
}


def all_neighbourhoods(K=8):
    for a, b, c in all_count_triples(K):
        yield [-1] * a + [0] * b + [1] * c


def part_c(K=8):
    """Brute force: a task is solvable by an aggregator iff no two neighbourhoods
    share an aggregated vector but have different correct answers."""
    neighs = list(all_neighbourhoods(K))
    print(f"  {'task':22s} {'sum':>5s} {'mean':>5s} {'max':>5s}")
    for tname, task in TASKS.items():
        row = []
        for aname, agg in AGG.items():
            seen, witness = {}, None
            for N in neighs:
                key = tuple(np.round(agg(np.stack([ONE_HOT[x] for x in N])), 9))
                ans = task(N)
                if key in seen and seen[key][1] != ans:
                    witness = (seen[key][0], N, key)
                    break
                seen.setdefault(key, (N, ans))
            row.append("Yes" if witness is None else "No")
            if witness:
                A, Bn, key = witness
                print(f"     {aname:4s} counter-example: {A} -> {task(A)} vs {Bn} -> {task(Bn)}, agg={key}")
        print(f"  {tname:22s} {row[0]:>5s} {row[1]:>5s} {row[2]:>5s}")


if __name__ == "__main__":
    print("Q1(a)"); part_a()
    print("Q1(b)"); part_b(8)
    print("Q1(c)"); part_c(8)
