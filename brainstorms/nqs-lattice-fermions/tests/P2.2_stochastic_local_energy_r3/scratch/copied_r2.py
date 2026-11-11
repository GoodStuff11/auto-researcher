def geometry(n_atoms, R, kind):
    if kind == "chain":
        return np.array([[i * R, 0.0, 0.0] for i in range(n_atoms)])
    rad = R / (2 * np.sin(np.pi / n_atoms))
    ang = 2 * np.pi * np.arange(n_atoms) / n_atoms
    return np.stack([rad * np.cos(ang), rad * np.sin(ang), 0 * ang], 1)

def rhf(S, h, G, nocc, maxit=1000, damp=0.5, lindep=1e-7):
    s, U = np.linalg.eigh(S)
    keep = s > lindep
    Xh = U[:, keep] / np.sqrt(s[keep])
    e, Cp = np.linalg.eigh(Xh.T @ h @ Xh)
    Cm = Xh @ Cp
    D = 2 * Cm[:, :nocc] @ Cm[:, :nocc].T
    Eold = 0.0
    for _ in range(maxit):
        F = h + np.einsum("pqrs,rs->pq", G, D) - 0.5 * np.einsum("prqs,rs->pq", G, D)
        e, Cp = np.linalg.eigh(Xh.T @ F @ Xh)
        Cm = Xh @ Cp
        Dn = 2 * Cm[:, :nocc] @ Cm[:, :nocc].T
        Eel = 0.5 * np.sum(D * (h + F))
        D = damp * D + (1 - damp) * Dn
        if abs(Eel - Eold) < 1e-11:
            break
        Eold = Eel
    return Cm, Eel, e


# ----------------------------------------------------------------------------
# Second-quantized Hamiltonian via Slater-Condon rows
# Spin-orbital order: alpha_0..alpha_{K-1}, beta_0..beta_{K-1}; ascending-order
# Jordan-Wigner phases.
# ----------------------------------------------------------------------------
class SQHam:
    def __init__(self, h, g, enuc, K, N):
        self.K, self.M, self.N, self.enuc = K, 2 * K, N, enuc
        M = self.M
        self.hso = np.zeros((M, M)); self.hso[:K, :K] = h; self.hso[K:, K:] = h
        gp = g.transpose(0, 2, 1, 3)  # <pq|rs> = (pr|qs)
        Gs = np.zeros((M, M, M, M))
        for s1 in (0, 1):
            for s2 in (0, 1):
                Gs[s1 * K:(s1 + 1) * K, s2 * K:(s2 + 1) * K, s1 * K:(s1 + 1) * K, s2 * K:(s2 + 1) * K] = gp
        self.gas = Gs - Gs.transpose(0, 1, 3, 2)
        del Gs
        self.Gd = np.einsum("pqpq->pq", self.gas).copy()  # <pq||pq>
        self.spin = (np.arange(M) >= K).astype(np.int64)
        self.iu_occ = np.triu_indices(N, 1)

    def diag(self, occ):
        return float(np.trace(self.hso[np.ix_(occ, occ)]) + 0.5 * self.Gd[np.ix_(occ, occ)].sum() + self.enuc)

    def row(self, code, occ):
        """All connected n' != n: dict with new codes, H_{n'n}, H_{n'n'}, excitation indices."""
        M, N = self.M, self.N
        o = np.zeros(M, np.int64); o[occ] = 1
        c = np.cumsum(o) - o
        virt = np.nonzero(o == 0)[0]
        u = self.Gd[:, occ].sum(1)
        f = np.diag(self.hso) + u
        Hnn = self.diag(occ)
        Fn = self.hso + self.gas[:, occ, :, occ].sum(0)
        # singles
        I, A = np.meshgrid(occ, virt, indexing="ij"); I = I.ravel(); A = A.ravel()
        m = self.spin[I] == self.spin[A]; I, A = I[m], A[m]
        hs = Fn[A, I] * (1 - 2 * ((c[I] + c[A] - (I < A)) & 1))
        dEs = f[A] - f[I] - self.Gd[A, I]
        # doubles
        oi, oj = occ[self.iu_occ[0]], occ[self.iu_occ[1]]
        vu = np.triu_indices(len(virt), 1)
        va, vb = virt[vu[0]], virt[vu[1]]
        nI = len(oi); nA = len(va)
        Ii = np.repeat(oi, nA); Jj = np.repeat(oj, nA)
        Aa = np.tile(va, nI); Bb = np.tile(vb, nI)
        m = (self.spin[Ii] + self.spin[Jj]) == (self.spin[Aa] + self.spin[Bb])
        Ii, Jj, Aa, Bb = Ii[m], Jj[m], Aa[m], Bb[m]
        val = self.gas[Aa, Bb, Ii, Jj]
        ph = c[Ii] + c[Jj] - 1 + c[Bb] - (Ii < Bb) - (Jj < Bb) + c[Aa] - (Ii < Aa) - (Jj < Aa)
        hd = val * (1 - 2 * (ph & 1))
        Gd = self.Gd
        dEd = (f[Aa] + f[Bb] - f[Ii] - f[Jj] + Gd[Ii, Jj] + Gd[Aa, Bb]
               - Gd[Aa, Ii] - Gd[Aa, Jj] - Gd[Bb, Ii] - Gd[Bb, Jj])
        one = np.int64(1)
        cs = code - (one << I) + (one << A)
        cd = code - (one << Ii) - (one << Jj) + (one << Aa) + (one << Bb)
        out = {
            "code": np.concatenate([cs, cd]),
            "h": np.concatenate([hs, hd]),
            "Hjj": Hnn + np.concatenate([dEs, dEd]),
            "rem": np.concatenate([np.stack([I, -np.ones_like(I)], 1), np.stack([Ii, Jj], 1)]),
            "add": np.concatenate([np.stack([A, -np.ones_like(A)], 1), np.stack([Aa, Bb], 1)]),
            "Hnn": Hnn,
        }
        keep = np.abs(out["h"]) > 1e-13
        for k in ("code", "h", "Hjj", "rem", "add"):
            out[k] = out[k][keep]
        return out


def build_space(K, na, nb):
    def strings(n):
        return [sum(1 << i for i in c) for c in combinations(range(K), n)]
    codes = np.array(sorted(a | (b << K) for a in strings(na) for b in strings(nb)), dtype=np.int64)
    M = 2 * K
    occ = ((codes[:, None] >> np.arange(M)) & 1).astype(np.int8)
    occ_idx = np.array([np.nonzero(o)[0] for o in occ])
    return codes, occ, occ_idx


def build_hamiltonian(ham, codes, occ_idx):
    D = len(codes)
    indptr = [0]; idx = []; dat = []
    for n in range(D):
        r = ham.row(codes[n], occ_idx[n])
        j = np.searchsorted(codes, r["code"])
        assert np.all(codes[j] == r["code"]), "connected config outside sector"
        idx.append(np.concatenate([[n], j]).astype(np.int32))
        dat.append(np.concatenate([[r["Hnn"]], r["h"]]))
        indptr.append(indptr[-1] + len(j) + 1)
    H = sp.csr_matrix((np.concatenate(dat), np.concatenate(idx), np.array(indptr, dtype=np.int64)), shape=(D, D))
    H.sort_indices()
    return H


def ground_states(H, k=2):
    D = H.shape[0]
    if D <= 2500:
        w, v = np.linalg.eigh(H.toarray())
        return w[:k], v[:, :k]
    from scipy.sparse.linalg import eigsh
    w, v = eigsh(H, k=k, which="SA", tol=1e-10)
    o = np.argsort(w)
    return w[o], v[:, o]


