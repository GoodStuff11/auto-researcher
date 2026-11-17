#!/usr/bin/env python3
"""VOI probe: map grade-R amplitudes ETA=(a_in, a_junk) to the T3 error measures (t3_scrd.py lines 463-465):
eta = 1 - cos^2(u, y_exact), eta_low = same restricted to the window levels (omega <= 1 t). On the 4x3 OBC 4+4
toy (native; the T3 instance), DAV sources Q(H-E)^p phi of the 8 trial states, shift E_L1R - 0.1 t, pass 1.
Also reports f = |P_w unit(y)|^2, the window weight of the exact response."""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import importlib.util, resource  # noqa: E402
resource.setrlimit(resource.RLIMIT_CPU, (150, 155))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("r12", os.path.join(HERE, "..", "P3.1_order_parameter_error_r12", "t1_r12.py"))
r12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r12)
dav, t1s, t1, np, unit = r12.dav, r12.t1s, r12.t1, r12.np, r12.unit
size = os.environ.get("SIZE", "4x3")
I, info = dav.build(size, 15, None)
perts = t1.perturbations(I); rng = np.random.default_rng(7)
src = r12.make_sources(I, "DAV", None)
ETAS = [tuple(map(float, e.split(","))) for e in os.environ.get("ETAS", "0.3,0.5;0.9,0.5;2.0,1.0;3.0,0.5").split(";")]
res = {e: [] for e in ETAS}; fs = []
for si, lab in enumerate(list(perts)[:4]):
    psi = unit(t1s.trial_state(I, perts[lab], rng)); phi = psi
    _, EL1 = t1s.ritz_vec(I, [phi, t1s.resid(I, phi)]); s = EL1 - 0.1
    for b in src(phi):
        y = unit(I.solve(b, s, orth=phi)); yw = I.Uw.T @ y; fs.append(float(yw @ yw))
        for e in ETAS:
            dav.ETA = e
            for rep in range(3):
                u = dav.noisy(I, y, np.random.default_rng(1000 * si + rep), phi); uw = I.Uw.T @ u
                res[e].append((1 - float(u @ y) ** 2, 1 - float(uw @ yw) ** 2 / float((uw @ uw) * (yw @ yw))))
print(f"{size}: nwin {I.nwin}, window weight f of exact DAV responses: median {np.median(fs):.3f} range {min(fs):.3f}-{max(fs):.3f}")
for e in ETAS:
    a = np.array(res[e]); print(f"ETA={e}: median eta {np.median(a[:, 0]):.3f} median eta_low {np.median(a[:, 1]):.3f}")
print("measured trained Davidson (F3, job 705951): eta_low 0.90, eta 0.85")
