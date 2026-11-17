# P4.2 r11: check of the reviewer's torus-capacity claim (T0/T1, about 1 CPU-min used)

Source data: `../P4.2_size_transfer_r5/results_close31/close31_s{0,1,2}.json` and `P4.2_r5_close31_451914_*.out` (job 451914); E0 also in `../P4.2_size_transfer_r3/P4.2_r3_zeroshot_410960_0.out`.

1. Numbers: CONFIRMED. Target 4x4 torus, 6+6, t'=-0.25, J=0.4, irrep Tx=Ty=Mx=My=+1, D=-1, n_irrep 13070,
   E0 = -13.93410177 (identical in 410960 and 451914). `scr` (MF init + 50 projected-SR steps, 122-parameter MFNG):
   eps(50) = 0.01858 / 0.01707 / 0.01828 t/site = 2.13 / 1.96 / 2.10 % relative error; median 0.01828 = 2.10 %.
   eps is the irrep-projected energy (the raw unprojected energy is 2-4x worse, e.g. 0.0397 for seed 0).
2. Family: NOT the R9.5 family. R3.1 used MFNG (r3_core.MFNG: channels 2, field 8, backflow 4, Jastrow 4; kernels
   summed over the cluster's own momentum grid). R9.5 names MFNG-O (obc_transfer.MFNGO: 4/16/16/16 = 426 parameters,
   fixed 8x8 quadrature kernel, coded for open clusters only). MFNG is not a parameter subfamily of MFNG-O (the
   kernel transform differs), and even a strict superset's SR plateau is not bounded by its subfamily's
   (R9.1 `fsym`, a strict superset, ended 1.9x worse). So 451914 is a valid capacity bound for MFNG on the torus,
   not for MFNG-O. Resolution: run R5.1 on the torus with the MFNG family itself (R3.1's grid-complete periodic
   sources 4x2 + t8), for which the bound is direct.
3. Convergence: NOT converged in 2 of 3 seeds. Drops over the last 10 steps: 1.2e-3 / 5e-5 / 3.5e-4 t/site
   (seed 1 flat). Projected SR with backtracking never raises the energy, so these are upper bounds:
   the MFNG torus plateau is <= 1.96 % whatever the convergence.

Next job drafted (not submitted): `submit_torus_pa.sbatch`, `voi.md`, `RULE.md`, `torus_pa.py`.
Gate check: PASS, cap 9.6 CPU-h (no auditor needed), digest 4d6cd30c2610.
Local T1: source ladder `source_ladder_preview.txt` (rung 1e-2 qualifies, 53 s), timing `timing_output.txt`,
end-to-end smoke `smoke_output.txt` (INVALID by design on the smoke instance: the 451914 init check).
