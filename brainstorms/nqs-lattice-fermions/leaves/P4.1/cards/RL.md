# RL — premise L: the SR-trained exact-$S$ states are optimiser-limited (MERGED)
* **The idea**: Measure, without sampling, the fraction of the gap one exact linear-method step in the trained state's own tangent space can remove; if it is large, the trained state is far from the best point of its own class, and every comparison of trained arms measures the optimiser.
* **Lineage**: Round 10, from the T1 observation that each parent's own constant directions remove 5 to 8% of its gap in one exact step, more than any new family. Dry run $L\ge0.199$ on 838 directions; job 707268 (full tangent, rank 2448 of 2638, all 627,264 configurations) gave $L=0.4895$ ($\Delta=1.095\times10^{-2}$ against a gap of $2.237\times10^{-2}$; by block W2 0.39, W1 and b1 0.12, P0 and b2 0.07, Jastrow 0.007): HOLDS.
* **Outcome**: MERGED into R18'' (as its premise).

