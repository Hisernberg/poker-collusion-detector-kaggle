"""Final evidence recipe: rank-blend grid over bin/spec/sus weights."""
import sys, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa

paths = resolve_paths()
dh_all = pl.read_parquet(paths.labelled_hand_feats).to_pandas()
dhp = dh_all[dh_all["label"] == 1].copy().reset_index(drop=True)
dhp_sorted = dhp.sort_values(["pair_id", "hand_id"]).reset_index(drop=True)
y = dhp_sorted["is_evidence"].astype(int).to_numpy()

with open(paths.work / "s2_ckpt.pkl", "rb") as f:
    ck = pickle.load(f)
dh2 = ck["dh"]
dh2["k"] = list(zip(dh2["pair_id"], dh2["hand_id"]))
key = list(zip(dhp_sorted["pair_id"], dhp_sorted["hand_id"]))
m = dh2.merge(pd.DataFrame({"k": key, "row": np.arange(len(key))}), on="k").sort_values("row")

bin_score = m["hand_score"].to_numpy()
spec_cols = [f"hand_spec_{f}" for f in TARGET_BEHAVIORS]
S = m[spec_cols].to_numpy()
fam = m["behavior_family"].to_numpy()
prior = m.groupby("behavior_family")["is_evidence"].mean().to_dict()
spec_mix = np.array([S[i, TARGET_BEHAVIORS.index(f)] / max(prior[f], 1e-9) for i, f in enumerate(fam)])
sus = m["hand_sus"].to_numpy()

from scipy.stats import rankdata
def rk(a):
    return rankdata(a) / len(a)

b_rk, s_rk, u_rk = rk(bin_score), rk(spec_mix), rk(sus)


def map5(scores):
    df = dhp_sorted[["pair_id", "pot_bb", "hand_id"]].copy()
    df["s"] = scores
    df["ev"] = y
    df = df.sort_values(["pair_id", "s", "pot_bb", "hand_id"], ascending=[True, False, False, True], kind="mergesort")
    vals = []
    for _, g in df.groupby("pair_id", sort=False):
        r = g["ev"].to_numpy()
        n = r.sum()
        if n == 0:
            continue
        top = r[:5]
        vals.append(np.sum(np.cumsum(top) / np.arange(1, len(top) + 1) * top) / min(n, 5))
    return float(np.mean(vals))


best = (0, None)
print("wb   ws   wu   MAP@5")
for wb in [0.9, 0.8, 0.7, 0.6, 0.5]:
    for ws in [0.1, 0.2, 0.3, 0.4]:
        wu = round(1 - wb - ws, 2)
        if wu < 0 or wu > 0.2:
            continue
        sc = wb * b_rk + ws * s_rk + wu * u_rk
        v = map5(sc)
        flag = " <-- BEST" if v > best[0] else ""
        if v > best[0]:
            best = (v, (wb, ws, wu))
        print(f"{wb:.2f} {ws:.2f} {wu:.2f}  {v:.4f}{flag}")
print("BEST:", best)
# robustness check: ±0.05 around best
wb, ws, wu = best[1]
for dwb in [-0.05, 0.05]:
    for dws in [-0.05, 0.05]:
        w2b, w2s = round(wb + dwb, 2), round(ws + dws, 2)
        w2u = round(1 - w2b - w2s, 2)
        if w2u < 0:
            continue
        sc = w2b * b_rk + w2s * s_rk + w2u * u_rk
        print(f"neighbor {w2b}/{w2s}/{w2u}: {map5(sc):.4f}")
