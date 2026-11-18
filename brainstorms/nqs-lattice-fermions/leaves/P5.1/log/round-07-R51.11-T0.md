# Round 7 — R51.11 Gate 4 T0: exact weight second moment of q ∝ |ψ|

## Pre-registration (written before any computation, 2026-11-17)
* **Quantity**: for p = |ψ|²/Z₂ and q = |ψ|/Z₁ over the full (sector) Hilbert space, E_q[w²] = Σ p²/q = Z₁ Σ|ψ|³ / Z₂², and
  the effective-sample-size fraction 1/E_q[w²]. Same for the guided mixture q_g = 0.8p + 0.2|ψ|/Z₁ (E = Σ p²/q_g).
  Reference: support size D (e^H with H the Shannon entropy of p, and the support count), to compare growth per site.
* **States**: stored exact or trained states at 5x2, 4x3, 6x2 (start states used by T1/kappa files; ground state too if stored).
* **Rule**: Gate 4 **FAIL** if E_q[w²] grows faster than polynomially across the three sizes: log E_q[w²] linear in V with
  slope > 0.1 per site, or growth factor per added site above that of e^H. **PASS** if bounded or consistent with
  polynomial growth (slope ≤ 0.1 per site and per-site factor below e^H's). **UNCLEAR** otherwise.
* **Caveat (stated in advance)**: three sizes (V = 10, 12, 12) of two geometries (Lx2 ladders, 4x3) is weak evidence on
  exponential vs polynomial; 4x3 and 6x2 share V, so the V-trend rests on 10 → 12 plus the 5x2 → 6x2 ladder pair.

## Result
Script `/tmp/r5111/t0w.py` (scratch, not kept): exact sums over the full (+,+) sector with configuration-space weights
size·|pr|^k, as `storm.py` samples; 24 CPU-s total. Trained = r04 exact-arm θ (5x2, 6x2) / `preflight_S4x3_exact` θ;
ground = exact (+,+) ground state (E₀ −6.5507 / −7.8286 / −7.6146).

| size | V | D | e^H (start) | E_q[w²], q∝\|ψ\|: start / trained / ground | ESS frac | E[w²] guided | ESS frac |
|---|---|---|---|---|---|---|---|
| 5x2 | 10 | 44,052 | 3,053 | 2.244 / 2.247 / 2.249 | 0.446 | 1.048 | 0.954 |
| 4x3 | 12 | 626,496 | 12,123 | 3.101 / 3.100 / 3.090 | 0.322 | 1.068 | 0.936 |
| 6x2 | 12 | 627,264 | 10,999 | 3.212 / 3.208 / 3.195 | 0.311 | 1.070 | 0.935 |

`kq.py`'s per-representative convention (√size·|pr|) gives 2.195 / 3.102 / 3.212, i.e. the same.
* Slope of ln E_q[w²] per site: 0.180 for the 5x2 → 6x2 ladder, 0.161 for 5x2 → 4x3. That is a factor of 1.20 per site,
  against 1.90 per site for e^H on the ladder (2.0 for 5x2 → 4x3). Guided: 0.010 per site, bounded by construction (w ≤ 1.25).
* The size sets the value, not the state: start, trained and ground state agree to 1%, and 4x3 ≈ 6x2 at equal V.

## Verdict under the pre-registered rule: **Gate 4 FAIL** (weak evidence)
* The slope clause fires (0.18 > 0.1 per site); the e^H clause does not (1.20 < 1.90 per site), so the FAIL rests on the
  slope criterion alone. With only V = 10 and 12, "linear in V" cannot be tested: the same points fit E_q[w²] ∝ V^2.0
  equally well (ln(3.21/2.24) = 2.0·ln 1.2). Three sizes of two geometries at two volumes is weak evidence, as stated in advance.
* The size of the effect today is small: ESS per draw falls from 45% to 31%, and the 4.7× κ advantage at both sizes
  already includes it. If the 0.18 per site rate held, E_q[w²] would be ~14 at V = 20 and ~2×10⁷ at V = 100; if it
  scales as V², ~220 at V = 100. Either way the reweighting cost is not "a constant factor" as the original Gate 4
  claim said. Guided's is.
* What would settle it: a third volume. Two options: exact at 7x2 N_e 12 (info file exists; check the sector size
  first), or sampled at 8x2 / 10x2, since E_q[w²] = E_p[|ψ|] / E_q[|ψ|] (unnormalised amplitudes, scale cancels),
  which can be estimated from Born draws and |ψ| draws of a trained state.
