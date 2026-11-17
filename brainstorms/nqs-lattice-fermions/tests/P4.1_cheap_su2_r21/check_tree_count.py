#!/usr/bin/env python3
"""
Check for tree_count.py (run: python3 check_tree_count.py; no options). On 4x4 and 8x8 clusters, for 5 random (c, T)
each and every ordered nn/nnn hop, enumerates all 2^(d_i + d_j) sign assignments (+-1/2) of the path nodes strictly
below the LCA, keeps those whose full tree (with the new leaf spins) obeys every triangle rule with root 0, and compares
that number with the dynamic-programming count tree_count.hop_count. Also checks every sampled T is valid.
Prints PASS/FAIL.
"""
import itertools

import numpy as np

import tree_count as tc

rng = np.random.default_rng(1)
bad = n = 0
for side in (4, 8):
    L = side * side
    coords = [(x, y) for x in range(side) for y in range(side)]
    parent, children, leaf_of = tc.kd_tree(coords)
    anc = {}
    for s in range(L):
        v, lst = leaf_of[s], []
        while v != -1:
            lst.append(v)
            v = parent[v]
        anc[leaf_of[s]] = lst
    idx = {c: k for k, c in enumerate(coords)}

    def valid(T):
        return T[0] == 0 and all(tc.tri(T[ch[0]], T[ch[1]], T[v]) for v, ch in enumerate(children) if ch)

    for _ in range(5):
        up = rng.choice(L, 7 * L // 16, replace=False)
        dn = rng.choice(L, 7 * L // 16, replace=False)
        occ = np.zeros(L, int)
        occ[up] += 1
        occ[dn] += 1
        ls = {leaf_of[s]: int(occ[s] == 1) for s in range(L)}
        T = tc.sample_T(children, ls, rng)
        assert valid(T)
        for (x, y), i in idx.items():
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                if (x + dx, y + dy) not in idx:
                    continue
                for a, b in ((i, idx[(x + dx, y + dy)]), (idx[(x + dx, y + dy)], i)):
                    c = tc.hop_count(a, b, occ, leaf_of, parent, children, T, anc)
                    if c is None:
                        continue
                    la, lb = leaf_of[a], leaf_of[b]
                    lca = next(v for v in anc[lb] if v in anc[la])
                    path = anc[la][1:anc[la].index(lca)] + anc[lb][1:anc[lb].index(lca)]
                    T2 = T.copy()
                    T2[la] = 1 if occ[a] == 2 else 0
                    T2[lb] = 1 if occ[b] == 0 else 0
                    cnt = 0
                    for sg in itertools.product((-1, 1), repeat=len(path)):
                        for v, s in zip(path, sg):
                            T2[v] = T[v] + s
                        if all(T2[v] >= 0 for v in path) and valid(T2):
                            cnt += 1
                    n += 1
                    bad += cnt != c
print(f"{n} hops checked, {bad} mismatches -> {'PASS' if bad == 0 else 'FAIL'}")
