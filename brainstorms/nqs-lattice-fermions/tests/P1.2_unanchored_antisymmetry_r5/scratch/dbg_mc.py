import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
import micro_test as T  # noqa: E402
import p21_harness as H21  # noqa: E402
from p11_core import energy  # noqa: E402
from tj_core import LRAnsatz  # noqa: E402

m = T.setup(8)
W, U = np.linalg.eigh(m.H.toarray())
sg, _ = H21.build_symmetry(m, U[:, 0])
refs = H21.references(m)
rng = np.random.default_rng(1)
for scale in (0.0, 0.3, 1.0):
    a = T.FRD(m, LRAnsatz(m, refs[0], "full", K=1, Hs=3, seed=1, w2_scale=0.3), "K", 3, 1)
    a.q["W2a"] = scale * rng.standard_normal(a.q["W2a"].shape); a.set_flat(a.flat())
    _, pc, _ = H21.state_of(a, m, sg, "chi", 0.0, 4096)
    E, Hp, nn = energy(m.H, pc)
    fex = a.force((Hp - E * pc) / nn, 4096)
    p = pc ** 2 / nn
    El_all = np.where(p > 0, Hp / np.where(pc != 0, pc, 1), 0)
    kurt = (p @ (El_all - E) ** 4) / (p @ (El_all - E) ** 2) ** 2
    Oall = a.per_sample_O_proj(sg, np.nonzero(p > 0)[0], pc, 64)
    sup = np.nonzero(p > 0)[0]
    pm = p[sup]; Om = (pm[:, None] * Oall).sum(0)
    fx2 = (pm[:, None] * (El_all[sup] - E)[:, None] * (Oall - Om)).sum(0)
    var_f = (pm[:, None] * ((El_all[sup] - E)[:, None] * (Oall - Om)) ** 2).sum(0)
    print(scale, "exact-sum-of-O vs force:", np.linalg.norm(fx2 - fex) / np.linalg.norm(fex),
          "kurt", kurt, "expected iid rel err (40000):", np.sqrt(var_f.sum() / 40000) / np.linalg.norm(fex))
    for how in ("iid", "mcmc"):
        if how == "iid":
            idx = rng.choice(m.D, size=40000, p=p)
        else:
            ch = T.Chains(m, 200, rng, pc ** 2); ch.step(pc ** 2, 200); idx = ch.draw(pc ** 2, 40000)
        El = Hp[idx] / pc[idx]
        O = a.per_sample_O_proj(sg, idx, pc, 64)
        f = ((El - El.mean())[:, None] * (O - O.mean(0))).mean(0)
        print("  ", how, np.linalg.norm(f - fex) / np.linalg.norm(fex))
