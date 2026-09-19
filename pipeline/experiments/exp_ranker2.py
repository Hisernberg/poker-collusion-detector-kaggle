"""Validate rank-weighted 2-seed bag ranker vs pipeline baseline (0.5189)."""
import sys
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa

paths = resolve_paths()
dh = pl.read_parquet(paths.labelled_hand_feats).to_pandas()
dh[HAND_FEATS] = dh[HAND_FEATS].astype("float32")
pos_mask = (dh["label"] == 1).to_numpy()
dhp = dh[pos_mask].copy().reset_index(drop=True)
y = dhp["is_evidence"].astype(int).to_numpy()
er = dhp.merge(pl.read_csv(paths.data / "development_evidence.csv").to_pandas(), on=["pair_id", "hand_id"], how="left")
rank_w = er["evidence_rank"].fillna(6).to_numpy()

tables = sorted(dhp["table_id"].unique().tolist())
rng = np.random.RandomState(SEED)
table_fold = {t: int(v) for t, v in zip(tables, rng.permutation(len(tables)) % N_FOLDS)}
folds = dhp["table_id"].map(table_fold).to_numpy()


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


def run(name, rounds=400, lr=0.05, weight_mode="none", leaves=31, ff=0.8):
    pred = np.zeros(len(dhp))
    w = None
    if weight_mode == "rank":
        w = np.where(y == 1, 7.0 - np.minimum(rank_w, 6), 1.0)
    elif weight_mode == "rank_sqrt":
        w = np.where(y == 1, np.sqrt(7.0 - np.minimum(rank_w, 6)), 1.0)
    params = {**HAND_PARAMS, "learning_rate": lr, "num_leaves": leaves, "feature_fraction": ff}
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        ms = []
        for sd in HAND_SEEDS:
            ms.append(lgb.train({**params, "seed": sd}, lgb.Dataset(dhp.loc[tr, HAND_FEATS], y[tr], weight=None if w is None else w[tr]), num_boost_round=rounds))
        pred[te] = np.mean([m.predict(dhp.loc[te, HAND_FEATS]) for m in ms], axis=0)
    v = oof_map5(pred)
    print(f"{name:34s} {v:.4f}", flush=True)
    return v


run("bag2 base 400 (repro)")
run("bag2 rank-w 400", weight_mode="rank")
run("bag2 rank-sqrt 400", weight_mode="rank_sqrt")
run("bag2 rank-w 700 lr.03", weight_mode="rank", rounds=700, lr=0.03)
run("bag2 rank-w 400 leaves63", weight_mode="rank", leaves=63)
run("bag2 rank-w 400 ff0.7", weight_mode="rank", ff=0.7)
