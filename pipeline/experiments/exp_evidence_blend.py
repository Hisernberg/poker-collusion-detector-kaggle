"""Evidence blend experiments on dev OOF (labelled positives).
Baseline: 0.6*spec(pred)+0.4*glob = 0.5235 (hard) / 0.5224 (soft).
Test: sus blend, rank-average, direction boost (pair-level), weight grid, family-specific weights.
"""
import sys, gc
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa

paths = resolve_paths()
dh = pl.read_parquet(paths.labelled_hand_feats).to_pandas()
print("rows:", len(dh))
dev_pf = pl.read_parquet(paths.dev_pair_features).to_pandas()
fam_prior = {fam: int((dev_pf["behavior_family"] == fam).sum()) for fam in TARGET_BEHAVIORS}

# OOF family probs are not stored per pair — recompute proxy: use OOF fam models? Instead use true family for oracle tests
# and pred-family via hard argmax on stored spec? We stored only hand_spec_* on rows. Use pair pred_family from pair features if present.
# For blend experiments use ORACLE family (upper bound) + GLOBAL family-agnostic variants (deployable).
G = dh["hand_score"].to_numpy()
S = dh[[f"hand_spec_{f}" for f in TARGET_BEHAVIORS]].to_numpy()
U = dh["hand_sus"].to_numpy()
fam = dh["behavior_family"].to_numpy()
fam_idx = pd.Series(fam).map(FAM_INDEX).fillna(-1).to_numpy(dtype=np.int64)
pos = dh["label"] == 1
dh_eval = dh[pos].copy()
Gp, Sp, Up = G[pos], S[pos], U[pos]
fam_ip = fam_idx[pos]
pid = dh_eval["pair_id"].to_numpy()
hid = dh_eval["hand_id"].to_numpy()
is_ev = dh_eval["is_evidence"].to_numpy()

def map5(score, pids=pid, hids=hid, ev=is_ev):
    df = pd.DataFrame({"pair_id": pids, "hand_id": hids, "s": score, "ev": ev, "pot": dh_eval["pot_bb"].to_numpy()})
    df = df.sort_values(["pair_id", "s", "pot", "hand_id"], ascending=[True, False, False, True], kind="mergesort")
    vals = []
    for _, g in df.groupby("pair_id", sort=False):
        r = g["ev"].to_numpy()
        n = r.sum()
        if n == 0:
            continue
        top = r[:5]
        hits = np.cumsum(top)
        vals.append(np.sum(hits / np.arange(1, len(top) + 1) * top) / min(n, 5))
    return float(np.mean(vals))

def spec_of(idx_mat, fam_codes):
    out = np.zeros(len(fam_codes))
    ok = fam_codes >= 0
    out[ok] = idx_mat[ok, fam_codes[ok]]
    return out

Spred = spec_of(Sp, fam_ip)
base = 0.6 * Spred + 0.4 * Gp
print(f"baseline hard 0.6/0.4:        {map5(base):.4f}")
print(f"global only:                  {map5(Gp):.4f}")
print(f"spec only (oracle):           {map5(Spred):.4f}")
print(f"sus only:                     {map5(Up):.4f}")

# weight grid with sus
for wg in [0.3, 0.4, 0.5]:
    for ws in [0.0, 0.1, 0.2, 0.3]:
        if wg + ws > 1.0:
            continue
        wspec = 1.0 - wg - ws
        sc = wg * Gp + ws * Up + wspec * Spred
        print(f"glob={wg} sus={ws} spec={wspec:.1f}:     {map5(sc):.4f}")

# rank-average (per pair percentile ranks)
def prank(x):
    df = pd.DataFrame({"p": pid, "s": x})
    return df.groupby("p")["s"].rank(pct=True).to_numpy()

for ws in [0.0, 0.15, 0.25]:
    sc = (1 - ws) * (0.6 * prank(Spred) + 0.4 * prank(Gp)) + ws * prank(Up)
    print(f"rankavg 0.6/0.4 sus={ws}:        {map5(sc):.4f}")

# direction boost: need per-hand transfer + loser side for eval rows — recompute from source
# (labelled_hand_feats has transfer_1_to_2/2_to_1/loser_is_p1 columns!)
t12 = dh_eval["transfer_1_to_2"].to_numpy()
t21 = dh_eval["transfer_2_to_1"].to_numpy()
loser_is_p1 = dh_eval["loser_is_p1"].to_numpy()
dts = fam_ip == 0
dir_score = np.zeros(len(pid))
df_dir = pd.DataFrame({"p": pid, "t12": t12, "t21": t21})
agg_dir = df_dir.groupby("p").sum()
dom_p1 = (agg_dir["t12"] > agg_dir["t21"]).reindex(pd.unique(pid)).fillna(False)
dom_map = dict(zip(agg_dir.index, (agg_dir["t12"] > agg_dir["t21"]).to_numpy()))
align = np.array([1.0 if dom_map.get(p, False) else 0.0 for p in pid])
# loser_is_p1==1 means p1 lost; p1 dominant-loser means t12>t21
match = ((loser_is_p1 == 1) & (align == 1)) | ((loser_is_p1 == 0) & (align == 0))
tmax = np.maximum(t12, t21)
tsum = df_dir.groupby("p")["t12"].sum() + df_dir.groupby("p")["t21"].sum()
boost = np.where(dts & match, np.log1p(np.minimum(tmax, 50)) / 3.0, 0.0)
for w in [0.5, 1.0, 1.5, 2.0]:
    sc = base + w * boost
    print(f"base + dirboost w={w}:          {map5(sc):.4f}")

# DT-only family specialist weights (per-family 0.6/0.4 tweaks)
for wdt, wsp, wci in [(0.8, 0.6, 0.4), (0.6, 0.6, 0.6), (0.7, 0.5, 0.5)]:
    w = np.select([fam_ip == 0, fam_ip == 1, fam_ip == 2], [wdt, wsp, wci], default=0.0)
    sc = w * Spred + (1 - w) * Gp
    print(f"fam-specific spec w={wdt}/{wsp}/{wci}: {map5(sc):.4f}")
