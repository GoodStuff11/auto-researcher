"""Print the key smoke-aggregate numbers (medians, delta, nu_W, cost ratios)."""
import json
import sys

s = open(sys.argv[1]).read()
i = s.index('{\n "R5.1'); j = s.rindex('}')
d = json.loads(s[i:j + 1])["R5.1 (WIBJ, Parts W+S)"]
for K, v in d["instances"]["H4_R4.4_d0.25_g1"]["byK"].items():
    print(K, {k: round(x, 3) for k, x in v["median_dE_mHa"].items()}, round(v["delta_med_mHa"], 3), round(v["nu_W_mHa"], 3))
for K, c in d["cost_byK"].items():
    print(K, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items() if k != "M_wall_CI"},
          [round(x, 2) for x in c["M_wall_CI"]])
print(d["controls"]["SCREEN_eps_star"], d["trend"]["H4_R4.4_d0.25_g1"]["delta_ext_K40"])
