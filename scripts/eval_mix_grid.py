"""Quick OOF tests: evidence mix weight grid + sus-model addition + seed-bag effect estimate."""
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
assert len(m) == len(dhp_sorted)

bin_score = m["hand_score"].to_numpy()
spec_cols = [f"hand_spec_{f}" for f in TARGET_BEHAVIORS]
S = m[spec_cols].to_numpy()
fam = m["behavior_family"].to_numpy()
prior = m.groupby("behavior_family")["is_evidence"].mean().to_dict()
spec_mix = np.array([S[i, TARGET_BEHAVIORS.index(f)] / max(prior[f], 1e-9) for i, f in enumerate(fam)])
sus = m["hand_sus"].to_numpy()


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


print(f"ref 0.6bin+0.4spec          : {map5(0.6*bin_score+0.4*spec_mix):.4f}")
for w in [0.4, 0.5, 0.7, 0.8]:
    print(f"{w:.1f}bin+{1-w:.1f}spec            : {map5(w*bin_score+(1-w)*spec_mix):.4f}")
# sus addition (rank-normalized)
from scipy.stats import rankdata


def rk(a):
    return rankdata(a) / len(a)


b_rk, s_rk, u_rk = rk(bin_score), rk(spec_mix), rk(sus)
for wu in [0.10, 0.15, 0.20, 0.30]:
    base = 0.6 * b_rk + 0.4 * s_rk
    bl = (1 - wu) * base + wu * u_rk
    print(f"0.6bin+0.4spec, sus w={wu:.2f}  : {map5(bl):.4f}")
# spec weight with sus fixed small
for ws in [0.3, 0.5]:
    for wu in [0.15]:
        bl = (1 - ws - wu) * b_rk + ws * s_rk + wu * u_rk
        print(f"{1-ws-wu:.2f}bin+{ws}spec+{wu}sus   : {map5(bl):.4f}")
