#!/usr/bin/env python3
"""
Write tasks.txt, the array table read by submit_vmc.sbatch (one line per array task, 0-based).

Usage:
    python3 make_tasks.py --timing PATH --pg "4:1,1;8:1,1;12:1,1;16:1,1" [--iters 3000] [--n-chains 1024]
                          [--out tasks.txt]

Options
-------
  --timing PATH   timing JSON from `vmc.py --mode timing` (required), measured WITHOUT P_G with the same
                  --n-chains. r_N = t_step(che_k8)/t_step(nnbf) at each size sets the nnbf equal-compute step
                  count n_eq(N) = round(iters * r_N). P_G multiplies every amplitude evaluation of every arm by
                  the same factor 4, so r_N is unchanged by it.
  --pg STR        per-size point-group irrep, "Lx:cMx,cMy;..." (required; characters from ED at Lx = 4 and
                  DMRG at Lx = 8, 12, 16, see the leaf doc). Every arm at that size is projected onto it.
  --iters INT     SR steps of every che arm (default 3000, the pre-registered value).
  --n-chains INT  samples per SR step (default 1024).
  --out PATH      output table (default <script dir>/tasks.txt).

Line format: "<Lx> <arm> <seed> <iters> <extra_checkpoints or -> <exact or -> <pg> <n_chains>".
Order: largest size first. Seeds: 0-2 at Lx = 4 (12 sites, exact final energy), 0-1 at Lx = 8 and 16,
0 only at Lx = 12 (amendment 2026-11-16, budget with P_G). nnbf lines carry extra checkpoint <iters>.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = {16: [0, 1], 12: [0], 8: [0, 1], 4: [0, 1, 2]}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--timing", required=True)
    p.add_argument("--pg", required=True)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--n-chains", type=int, default=1024)
    p.add_argument("--out", default=os.path.join(HERE, "tasks.txt"))
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    t = json.load(open(a.timing))["sizes"]
    pg = dict(item.split(":") for item in a.pg.split(";"))
    lines = []
    for Lx in [16, 12, 8, 4]:
        r = t[str(Lx)]["che_k8"]["t_step"] / t[str(Lx)]["nnbf"]["t_step"]
        neq = int(round(a.iters * r))
        ex = "exact" if Lx == 4 else "-"
        for arm in ["che_k8", "nnbf", "che_k4", "che_k2", "che"]:
            for s in SEEDS[Lx]:
                if arm == "nnbf":
                    lines.append(f"{Lx} nnbf {s} {neq} {a.iters} {ex} {pg[str(Lx)]} {a.n_chains}")
                else:
                    lines.append(f"{Lx} {arm} {s} {a.iters} - {ex} {pg[str(Lx)]} {a.n_chains}")
        print(f"Lx={Lx}: r = {r:.2f} -> nnbf n_eq = {neq}; irrep (Mx,My) = ({pg[str(Lx)]}); seeds {SEEDS[Lx]}")
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {a.out}: {len(lines)} tasks (array 0-{len(lines) - 1})")


if __name__ == "__main__":
    main()
