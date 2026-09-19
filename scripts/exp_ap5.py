"""AP@5 lambda-rank objective (ported from V12 notebook) for the evidence ranker.
Test OOF MAP@5 vs binary baseline 0.5189. Also test blend binary + AP5.
"""
import sys, math
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from numba import njit

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa

paths = resolve_paths()
dh = pl.read_parquet(paths.labelled_hand_feats).to_pandas()
dh[HAND_FEATS] = dh[HAND_FEATS].astype("float32")
pos_mask = (dh["label"] == 1).to_numpy()
dhp = dh[pos_mask].copy().reset_index(drop=True)
y = dhp["is_evidence"].astype(int).to_numpy()

tables = sorted(dhp["table_id"].unique().tolist())
rng = np.random.RandomState(SEED)
table_fold = {t: int(v) for t, v in zip(tables, rng.permutation(len(tables)) % N_FOLDS)}
folds = dhp["table_id"].map(table_fold).to_numpy()


@njit(cache=False)
def ap5_lambdas(scores, labels, ptr, k=5):
    grad = np.zeros(len(scores), np.float64)
    hess = np.full(len(scores), 1e-7, np.float64)
    for q in range(len(ptr) - 1):
        a = ptr[q]; b = ptr[q + 1]
        if b - a < 2:
            continue
        order = np.argsort(-scores[a:b], kind="mergesort")
        yy = labels[a:b][order].astype(np.int32)
        m = int(np.sum(yy)); denom = min(k, m)
        if m == 0 or m == b - a:
            continue
        prefix = np.cumsum(yy)
        pos = np.where(yy > 0)[0]; neg = np.where(yy == 0)[0]
        for p in pos:
            for v in neg:
                if min(p, v) >= k:
                    continue
                if p < v:
                    delta = prefix[p] / (p + 1.0)
                    for t in range(p + 1, min(v, k)):
                        if yy[t] > 0:
                            delta += 1.0 / (t + 1.0)
                    if v < k:
                        delta -= prefix[v] / (v + 1.0)
                else:
                    delta = (prefix[v] + 1.0) / (v + 1.0)
                    for t in range(v + 1, min(p, k)):
                        if yy[t] > 0:
                            delta += 1.0 / (t + 1.0)
                    if p < k:
                        delta -= prefix[p] / (p + 1.0)
                delta = max(0.0, delta / denom)
                if delta <= 0.0:
                    continue
                ip = a + order[p]; iv = a + order[v]
                diff = max(-40.0, min(40.0, scores[ip] - scores[iv]))
                rho = 1.0 / (1.0 + math.exp(diff))
                g = delta * rho
                hh = delta * rho * (1.0 - rho)
                grad[ip] -= g; grad[iv] += g
                hess[ip] += hh; hess[iv] += hh
    return grad, hess


class AP5Objective:
    def __init__(self, groups):
        self.ptr = np.r_[0, np.cumsum(groups)].astype(np.int64)

    def __call__(self, predictions, dataset):
        return ap5_lambdas(np.asarray(predictions, np.float64), dataset.get_label().astype(np.int8), self.ptr, 5)


def oof_map5(pred):
    df = dhp[["pair_id", "hand_id", "pot_bb"]].copy()
    df["s"] = pred
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


dhp_sorted = dhp.sort_values(["pair_id", "hand_id"]).reset_index(drop=True)
y_s = dhp_sorted["is_evidence"].astype(int).to_numpy()
folds_s = dhp_sorted["table_id"].map(table_fold).to_numpy()
groups_all = dhp_sorted.groupby("pair_id", sort=False).size().to_numpy()

import os as _o
NR = int(_o.environ.get("AP5_ROUNDS", "300"))
LR = float(_o.environ.get("AP5_LR", "0.08"))
pred_ap5 = np.zeros(len(dhp_sorted))
params = {**HAND_PARAMS, "learning_rate": LR, "num_threads": 2}
params.pop("objective", None)
for f in range(N_FOLDS):
    tr, te = folds_s != f, folds_s == f
    dtr = dhp_sorted.loc[tr]
    # group sizes in row order of dtr
    grp = dtr.groupby("pair_id", sort=False).size().to_numpy()
    ds = lgb.Dataset(dtr[HAND_FEATS], y_s[tr], group=grp, free_raw_data=False)
    obj = AP5Objective(grp)
    m = lgb.train({**params, "objective": obj}, ds, num_boost_round=NR)
    pred_ap5[te] = m.predict(dhp_sorted.loc[te, HAND_FEATS])
def oof_map5_sorted(pred):
    df = dhp_sorted[["pair_id", "hand_id", "pot_bb"]].copy()
    df["s"] = pred
    df["ev"] = y_s
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

print("AP5 objective OOF MAP@5:", round(oof_map5_sorted(pred_ap5), 4), flush=True)
np.save("/home/z/my-project/work/oof_ap5.npy", pred_ap5)
