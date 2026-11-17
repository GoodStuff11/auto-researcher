#!/usr/bin/env python3
"""
P4.1 r21 item 2 (R22 `spin-tree`, falsifier 1): connected states of the Hubbard hop in a balanced coupling-tree basis.
===================================================================================================================

Run:  python3 tree_count.py [--samples INT (default 20)] [--seed INT (default 0)]

Basis |c, T>: site occupations c (0, 1, 2); singly occupied sites carry spin 1/2, others spin 0; T lists the spins of
the internal nodes of a fixed balanced k-d tree over the sites (square Lside x Lside open cluster, split alternately
along x and y into halves), root spin S = 0. Operator: one hop term sum_sigma c^dag_{j sigma} c_{i sigma} (i -> j), for
every ordered nearest-neighbour and next-nearest-neighbour (diagonal) pair. It is an SU(2) scalar supported in the
subtree of LCA(i, j), so nodes at or above the LCA keep their spin; a node containing exactly one of i, j changes its
number of singly occupied sites by +-1, so its spin changes by exactly +-1/2. The count of connected T' is the number
of path assignments (old spin +-1/2 on both paths up to the LCA's children) obeying every triangle rule with the fixed
siblings and the unchanged LCA spin (an upper bound on the nonzero matrix elements; accidental zeros of 6j symbols are
not removed). Hops with i empty or j doubly occupied give 0 and are counted as such.

Sampling (doping 1/8, N_e = 7L/8, S = 0):
  'uniform' : c from a uniform random S^z configuration (N_e/2 up, N_e/2 down, independent positions), the U = 0
              charge statistics; T uniform among valid trees given c (exact subtree-count weights, top-down).
  'nodbl'   : no doubly occupied sites (strong coupling): N_e singly occupied sites at random; T uniform given c.
Per configuration: total = sum over all ordered hops of the connected count; per_hop = total / (number of allowed
hops). Sizes L = 16, 64, 256, 1024. Fit: least squares of log(mean total) against log L over the four sizes.
Rule (RULE.md, R22 falsifier 1): KILL (G4) if the fitted exponent of the mean count against L exceeds 3; PASS if it
is at most 2.6; otherwise UNCLEAR, decided by the worst-case count at L = 1024. The card's lemma (O(L^2.5) connected
states per configuration) fixes the per-configuration total as the quantity the thresholds refer to; the per-hop
exponent (one lower) is printed as well. Writes results/tree_count.json.
"""
import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def kd_tree(coords):
    """returns parent, children (list of pairs or None), leaf_node[site], depth; nodes numbered, root 0."""
    parent, children, leaf_of = [-1], [None], {}
    stack = [(0, list(range(len(coords))), 0)]
    while stack:
        nd, sites, ax = stack.pop()
        if len(sites) == 1:
            leaf_of[sites[0]] = nd
            continue
        key = sorted(sites, key=lambda s: (coords[s][ax], coords[s][1 - ax]))
        h = len(key) // 2
        kids = []
        for part in (key[:h], key[h:]):
            parent.append(nd)
            children.append(None)
            kids.append(len(parent) - 1)
            stack.append((len(parent) - 1, part, 1 - ax))
        children[nd] = tuple(kids)
    return np.array(parent), children, np.array([leaf_of[s] for s in range(len(coords))])


def tri(a, b, c):  # doubled spins
    return abs(a - b) <= c <= a + b and (a + b + c) % 2 == 0


def sample_T(children, leafspin, rng):
    """uniform valid T given leaf (doubled) spins and root 0; returns doubled spins per node."""
    n = len(children)
    W = [None] * n  # node -> dict(2s -> weight), normalised per node
    order = []
    st = [0]
    while st:
        v = st.pop()
        order.append(v)
        if children[v] is not None:
            st.extend(children[v])
    for v in reversed(order):
        if children[v] is None:
            W[v] = {leafspin[v]: 1.0}
            continue
        a, b = children[v]
        acc = {}
        for s1, w1 in W[a].items():
            for s2, w2 in W[b].items():
                for s in range(abs(s1 - s2), s1 + s2 + 1, 2):
                    acc[s] = acc.get(s, 0.0) + w1 * w2
        mx = max(acc.values())
        W[v] = {s: w / mx for s, w in acc.items() if w > 1e-40 * mx}  # prune negligible tails (speed only)
    assert 0 in W[0], "no S = 0 tree"
    T = np.zeros(n, dtype=int)
    T[0] = 0
    for v in order:
        if children[v] is None:
            continue
        a, b = children[v]
        pairs, ws = [], []
        for s1, w1 in W[a].items():
            for s2, w2 in W[b].items():
                if tri(s1, s2, T[v]):
                    pairs.append((s1, s2))
                    ws.append(w1 * w2)
        ws = np.array(ws)
        k = rng.choice(len(pairs), p=ws / ws.sum())
        T[a], T[b] = pairs[k]
    return T


def path_counts(leaf, new_leaf_spin, top, parent, children, T):
    """dict: new doubled spin of node `top` (an LCA child) -> number of valid assignments along leaf..top."""
    cur = {new_leaf_spin: 1}
    v = leaf
    while v != top:
        p = parent[v]
        a, b = children[p]
        sib = b if a == v else a
        nxt = {}
        for sv, cnt in cur.items():
            for sp in (T[p] - 1, T[p] + 1):
                if sp >= 0 and tri(sv, T[sib], sp):
                    nxt[sp] = nxt.get(sp, 0) + cnt
        cur = nxt
        v = p
    return cur


def hop_count(i, j, occ, leaf_of, parent, children, T, anc):
    if occ[i] == 0 or occ[j] == 2:
        return None
    li, lj = leaf_of[i], leaf_of[j]
    ai = anc[li]
    lca = next(x for x in anc[lj] if x in ai)
    a, b = children[lca]
    ti = a if a in ai else b
    tj = b if ti == a else a
    si = 1 if occ[i] == 2 else 0       # new doubled spin of site i
    sj = 1 if occ[j] == 0 else 0
    ci = path_counts(li, si, ti, parent, children, T)
    cj = path_counts(lj, sj, tj, parent, children, T)
    return sum(x * y for s1, x in ci.items() for s2, y in cj.items() if tri(s1, s2, T[lca]))


def run(side, mode, nsamp, rng):
    L = side * side
    coords = [(x, y) for x in range(side) for y in range(side)]
    parent, children, leaf_of = kd_tree(coords)
    anc = {}
    for s in range(L):
        v, lst = leaf_of[s], []
        while v != -1:
            lst.append(v)
            v = parent[v]
        anc[leaf_of[s]] = lst
    idx = {c: k for k, c in enumerate(coords)}
    bonds = []
    for (x, y), k in idx.items():
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            q = (x + dx, y + dy)
            if q in idx:
                bonds += [(k, idx[q]), (idx[q], k)]
    Ne = 7 * L // 8
    tot, perhop, worst = [], [], 0
    for _ in range(nsamp):
        if mode == "uniform":
            up = rng.choice(L, Ne // 2, replace=False)
            dn = rng.choice(L, Ne // 2, replace=False)
            occ = np.zeros(L, dtype=int)
            occ[up] += 1
            occ[dn] += 1
        else:
            occ = np.zeros(L, dtype=int)
            occ[rng.choice(L, Ne, replace=False)] = 1
        leafspin = {leaf_of[s]: int(occ[s] == 1) for s in range(L)}
        T = sample_T(children, leafspin, rng)
        for s in range(L):
            T[leaf_of[s]] = leafspin[leaf_of[s]]
        cs = [hop_count(i, j, occ, leaf_of, parent, children, T, anc) for i, j in bonds]
        cs = [c for c in cs if c is not None]
        tot.append(sum(cs))
        perhop.append(sum(cs) / len(cs))
        worst = max(worst, max(cs))
    return dict(L=L, nbonds_ordered=len(bonds), mean_total=float(np.mean(tot)), max_total=int(max(tot)),
                se_total=float(np.std(tot) / np.sqrt(len(tot))), mean_per_hop=float(np.mean(perhop)),
                worst_single_hop=int(worst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    c0, t0 = time.process_time(), time.time()
    out = {}
    for mode in ("uniform", "nodbl"):
        rng = np.random.default_rng(a.seed)
        rows = [run(side, mode, a.samples, rng) for side in (4, 8, 16, 32)]
        Ls = np.log([r["L"] for r in rows])
        p_tot = float(np.polyfit(Ls, np.log([r["mean_total"] for r in rows]), 1)[0])
        p_hop = float(np.polyfit(Ls, np.log([r["mean_per_hop"] for r in rows]), 1)[0])
        p_last = float(np.log(rows[-1]["mean_total"] / rows[-2]["mean_total"]) / np.log(4))
        v = "KILL (G4)" if p_tot > 3 else "PASS" if p_tot <= 2.6 else "UNCLEAR"
        out[mode] = dict(rows=rows, exponent_total=p_tot, exponent_per_hop=p_hop, exponent_last_pair=p_last,
                         verdict=v)
        for r in rows:
            print(f"{mode:7s} L={r['L']:5d}: mean connected/config {r['mean_total']:.4e} (se {r['se_total']:.1e}, "
                  f"max {r['max_total']:.3e}); mean per allowed hop {r['mean_per_hop']:.3e}; worst hop "
                  f"{r['worst_single_hop']}")
        print(f"{mode:7s} fitted exponent: per configuration {p_tot:.3f} (last pair 256->1024: {p_last:.3f}); "
              f"per hop {p_hop:.3f} -> RULE (per-configuration): {v}")
    out["cpu_min"] = (time.process_time() - c0) / 60
    out["samples"], out["seed"] = a.samples, a.seed
    print(f"[{out['cpu_min']:.2f} CPU-min, {time.time() - t0:.0f} s wall]")
    json.dump(out, open(os.path.join(HERE, "results/tree_count.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
