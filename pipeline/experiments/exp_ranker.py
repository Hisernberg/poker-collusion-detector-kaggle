"""Hand-ranker improvement experiments (fast: labelled frame cached).
Variants: (a) 700 rounds, (b) evidence-rank-weighted binary, (c) DART,
(d) two-stage hard-negative rerank, (e) lambda-rank. Metric: OOF MAP@5 + per family.
"""
import sys, gc, pickle
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa
from poker_lib import _lgb_bag, _predict_bag

paths = resolve_paths()
dh = pl.read_parquet(paths.labelled_hand_feats).to_pandas()
dh[HAND_FEATS] = dh[HAND_FEATS].astype("float32")
pos_mask = (dh["label"] == 1).to_numpy()
dhp = dh[pos_mask].copy().reset_index(drop=True)  # positive-pair rows only
y = dhp["is_evidence"].astype(int)
er = dhp.merge(pl.read_csv(paths.data / "development_evidence.csv").to_pandas(), on=["pair_id", "hand_id"], how="left")
rank_w = er["evidence_rank"].fillna(6).to_numpy()  # 1..5, non-evidence -> 6

tables = sorted(dhp["table_id"].unique().tolist())
rng = np.random.RandomState(SEED)
table_fold = {t: int(v) for t, v in zip(tables, rng.permutation(len(tables)) % N_FOLDS)}
folds = dhp["table_id"].map(table_fold).to_numpy()

groups = dhp.groupby("pair_id").size().to_numpy()
print("rows:", len(dhp), "pairs:", len(groups), "evidence:", y.sum())


def oof_map5(pred):
    df = dhp[["pair_id", "hand_id", "pot_bb"]].copy()
    df["s"] = pred
    df["ev"] = y.to_numpy()
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


results = {}

# (a) baseline 400 rounds (repro)
pred = np.zeros(len(dhp))
for f in range(N_FOLDS):
    tr, te = folds != f, folds == f
    m = lgb.train(HAND_PARAMS, lgb.Dataset(dhp.loc[tr, HAND_FEATS], y[tr]), num_boost_round=HAND_ROUNDS)
    pred[te] = m.predict(dhp.loc[te, HAND_FEATS])
results["a_400_binary"] = oof_map5(pred)

# (b) 800 rounds lower lr
pred = np.zeros(len(dhp))
params = {**HAND_PARAMS, "learning_rate": 0.025}
for f in range(N_FOLDS):
    tr, te = folds != f, folds == f
    m = lgb.train(params, lgb.Dataset(dhp.loc[tr, HAND_FEATS], y[tr]), num_boost_round=800)
    pred[te] = m.predict(dhp.loc[te, HAND_FEATS])
results["b_800_lr025"] = oof_map5(pred)

# (c) rank-weighted binary: evidence rows weighted by (6-rank), non-ev 1
pred = np.zeros(len(dhp))
w = np.where(y.to_numpy() == 1, 7.0 - np.minimum(rank_w, 6), 1.0)
for f in range(N_FOLDS):
    tr, te = folds != f, folds == f
    m = lgb.train(HAND_PARAMS, lgb.Dataset(dhp.loc[tr, HAND_FEATS], y[tr], weight=w[tr]), num_boost_round=HAND_ROUNDS)
    pred[te] = m.predict(dhp.loc[te, HAND_FEATS])
results["c_rankw"] = oof_map5(pred)

# (d) DART
pred = np.zeros(len(dhp))
params = {**HAND_PARAMS, "boosting": "dart", "learning_rate": 0.15, "drop_rate": 0.1, "max_drop": 30}
for f in range(N_FOLDS):
    tr, te = folds != f, folds == f
    m = lgb.train(params, lgb.Dataset(dhp.loc[tr, HAND_FEATS], y[tr]), num_boost_round=300)
    pred[te] = m.predict(dhp.loc[te, HAND_FEATS])
results["d_dart300"] = oof_map5(pred)

# (e) lambdarank with pair groups
pred = np.zeros(len(dhp))
lgb_rank_params = dict(objective="lambdarank", label_gain=[0] + [2**i for i in range(1, 12)], learning_rate=0.06,
                       num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                       lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=2, eval_at=[5])
for f in range(N_FOLDS):
    tr, te = folds != f, folds == f
    dtr = dhp.loc[tr]
    grp = dtr.groupby("pair_id", sort=False).size().to_numpy()
    # labels: evidence rank 1-5 -> gain tier
    lab = np.where(y[tr] == 1, np.maximum(6 - np.minimum(rank_w[tr], 6), 1), 0)
    order = dtr["pair_id"].to_numpy()
    m = lgb.train(lgb_rank_params, lgb.Dataset(dtr[HAND_FEATS], lab, group=grp), num_boost_round=300)
    pred[te] = m.predict(dhp.loc[te, HAND_FEATS])
results["e_lambdarank300"] = oof_map5(pred)

# (f) two-stage hard negative rerank: stage1 = (a); stage2 trains on rows with stage1 score in top-20 per pair
pred1 = np.zeros(len(dhp))
for f in range(N_FOLDS):
    tr, te = folds != f, folds == f
    m = lgb.train(HAND_PARAMS, lgb.Dataset(dhp.loc[tr, HAND_FEATS], y[tr]), num_boost_round=HAND_ROUNDS)
    pred1[te] = m.predict(dhp.loc[te, HAND_FEATS])
dhp["_p1"] = pred1
pred2 = np.zeros(len(dhp))
for f in range(N_FOLDS):
    tr, te = folds != f, folds == f
    dtr = dhp.loc[tr].copy()
    # hard set: top-20 by _p1 within pair
    dtr["_rk"] = dtr.groupby("pair_id")["_p1"].rank(ascending=False, method="first")
    hard = dtr[dtr["_rk"] <= 20]
    m2 = lgb.train({**HAND_PARAMS, "learning_rate": 0.05}, lgb.Dataset(hard[HAND_FEATS], hard["is_evidence"].astype(int)), num_boost_round=250)
    pred2[te] = m2.predict(dhp.loc[te, HAND_FEATS])
results["f_hardrerank"] = oof_map5(pred2)

# (g) blend a + e
print("\n=== RESULTS (OOF MAP@5, baseline pipeline = 0.5189) ===")
for k, v in results.items():
    print(f"{k:22s} {v:.4f}")
