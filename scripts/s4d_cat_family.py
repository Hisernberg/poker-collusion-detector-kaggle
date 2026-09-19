"""Stage 4d: broad CatBoost family + robust equal-weight rank-average ensemble.
Experts: cat d4/i600, d5/i420 (have), d6/i500 (have), d5/i700-l2_8 (have), d8/i400, d5/i900.
Robust ensembles (NO weight search):
  A: rank-mean of all CATs
  B: 0.3*rank(PU) + 0.7*rank(mean-CATs)
  C: 0.2*rank(PU) + 0.5*rank(mean-CATs) + 0.3*rank(lgb_sub60)
Report OOF (mirrored + clean) for each vs v4 recipe (0.4 PU + 0.6 cat_d5).
"""
import sys, gc, pickle, json
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa

paths = resolve_paths()
log("S4d start")

dev_pf = pd.read_parquet(paths.dev_pair_features)
with open(paths.hand_models, "rb") as f:
    art = pickle.load(f)
if "fold" not in dev_pf.columns:
    dev_pf["fold"] = dev_pf["table_id"].map(art["table_fold"]).astype(np.int8)
y = dev_pf["label"].fillna(0).astype(int).to_numpy()
labm = dev_pf["is_labeled"].to_numpy()
folds = dev_pf["fold"].to_numpy().astype(int)
X = dev_pf[[c for c in dev_pf.columns if c not in PAIR_META and not c.startswith("sus_") and c != "top5s_same_loser"]].astype("float32")

with open(paths.work / "s4_stage2.pkl", "rb") as f:
    pu_oof = pickle.load(f)["oof"]
CK1 = paths.work / "s4_stage1.npy"
oof1 = np.load(CK1)
pos_oof = oof1[labm & (y == 1)]
thr_a, thr_p = np.quantile(pos_oof, 0.05), np.quantile(pos_oof, 0.50)
unl = ~labm
ambiguous = unl & (oof1 >= thr_a) & (oof1 < thr_p)
pseudo = unl & (oof1 >= thr_p)
y2 = np.where(pseudo, 1, y)
w2 = np.where(labm, np.where(y == 1, POS_WEIGHT, NEG_WEIGHT), np.where(pseudo, PSEUDO_POS_WEIGHT, np.where(ambiguous, 0.0, PU_WEIGHT)))
Xc = X[~ambiguous]; yc = y2[~ambiguous]; wc = w2[~ambiguous]; fc = folds[~ambiguous]
clean_full = ~(ambiguous | pseudo)


def run_cat(depth, iters, seed, l2=5.0):
    tag = f"cat_d{depth}i{iters}s{seed}"
    ck = paths.work / f"s4d_{tag}.pkl"
    if ck.exists():
        with open(ck, "rb") as _f:
            return pickle.load(_f)
    fck = paths.work / f"s4d_{tag}_folds"
    fck.mkdir(exist_ok=True)
    oof = np.zeros(len(yc))
    models = []
    for f in range(N_FOLDS):
        tr, te = fc != f, fc == f
        ff = fck / f"fold{f}.pkl"
        if ff.exists():
            with open(ff, "rb") as _f:
                m, pred = pickle.load(_f)
        else:
            m = cb.CatBoostClassifier(iterations=iters, depth=depth, learning_rate=0.04, loss_function="Logloss",
                                      verbose=False, random_seed=seed + f, thread_count=2, l2_leaf_reg=l2)
            m.fit(Xc[tr], yc[tr], sample_weight=wc[tr])
            pred = m.predict_proba(Xc[te])[:, 1]
            with open(ff, "wb") as _f:
                pickle.dump((m, pred), _f)
        oof[te] = pred
        models.append(m)
    full = cb.CatBoostClassifier(iterations=iters, depth=depth, learning_rate=0.04, loss_function="Logloss",
                                 verbose=False, random_seed=seed, thread_count=2, l2_leaf_reg=l2)
    full.fit(Xc, yc, sample_weight=wc)
    with open(ck, "wb") as _f:
        pickle.dump((oof, models, full), _f)
    return oof, models, full


cats = {}
oof, models, full = run_cat(4, 600, SEED + 300); cats["cat_d4"] = (oof, full)
log("cat_d4 done")
oof, models, full = run_cat(8, 400, SEED + 400); cats["cat_d8"] = (oof, full)
log("cat_d8 done")
oof, models, full = run_cat(5, 900, SEED + 500); cats["cat_d5i900"] = (oof, full)
log("cat_d5i900 done")

# existing cats
with open(paths.work / "pair_experts.pkl", "rb") as f:
    exp = pickle.load(f)
import glob as _glob
unamb = ~ambiguous
assert len(yc) == int(unamb.sum())
for tag, name in [("s4c_cat_d6i500s142.pkl", "cat_d6"), ("s4c_cat_d5i700s242.pkl", "cat_d5i700")]:
    p = paths.work / tag
    if p.exists():
        with open(p, "rb") as f:
            oof, _, full = pickle.load(f)
        if len(oof) == len(yc):
            cats[name] = (oof, full)
        elif len(oof) == len(y2):  # full-set oof -> subset
            cats[name] = (oof[unamb], full)
cats["cat_d5"] = (exp["oof_cat"][unamb] if len(exp["oof_cat"]) == len(y2) else exp["oof_cat"], exp["cat_full"])
assert all(len(v[0]) == len(yc) for v in cats.values()), {k: len(v[0]) for k, v in cats.items()}

cat_names = sorted(cats.keys())
M = np.column_stack([cats[k][0] for k in cat_names])
mean_cat = M.mean(axis=1)
R = np.column_stack([rankdata(Mf := M[:, j]) / len(Mf) for j in range(M.shape[1])])
mean_cat_rank = R.mean(axis=1)
pu_rank = rankdata(pu_oof[unamb]) / int(unamb.sum())
sub_path = paths.work / "s4c_lgb_1200_6_gbdt.pkl"
sub_rank = None
if sub_path.exists():
    with open(sub_path, "rb") as f:
        ck_sub = pickle.load(f)
    oof_sub = ck_sub["oof"] if isinstance(ck_sub, dict) and "oof" in ck_sub else (ck_sub[0] if isinstance(ck_sub, tuple) else None)
    if oof_sub is not None:
        if len(oof_sub) == len(y2):
            oof_sub = oof_sub[unamb]
        sub_rank = rankdata(oof_sub) / len(oof_sub)

yf = y2[~ambiguous]; cf = clean_full[~ambiguous]; lf = labm[~ambiguous]
def rep(name, ens):
    mir = average_precision_score(yf, ens)
    cln = average_precision_score(yf[cf], ens[cf])
    lab = average_precision_score(yf[lf], ens[lf])
    log(f"{name:38s} mir={mir:.4f} clean={cln:.4f} lab={lab:.4f}")
    return mir, cln, lab

rep("v4 recipe 0.4PU+0.6cat_d5", 0.4 * pu_rank + 0.6 * R[:, cat_names.index("cat_d5")])
rep("A rank-mean all cats", mean_cat_rank)
if sub_rank is not None:
    rep("C 0.2PU+0.5cats+0.3sub60", 0.2 * pu_rank + 0.5 * mean_cat_rank + 0.3 * sub_rank)
    rep("D 0.15PU+0.65cats+0.2sub60", 0.15 * pu_rank + 0.65 * mean_cat_rank + 0.2 * sub_rank)
rep("B1 0.3PU+0.7cats", 0.3 * pu_rank + 0.7 * mean_cat_rank)
rep("B2 0.2PU+0.8cats", 0.2 * pu_rank + 0.8 * mean_cat_rank)
rep("B3 0.1PU+0.9cats", 0.1 * pu_rank + 0.9 * mean_cat_rank)
rep("pure cats", mean_cat_rank)

artifact = dict(cat_names=cat_names,
                fulls={k: cats[k][1] for k in cat_names},
                sub=None if sub_rank is None or not sub_path.exists() else ck_sub)
with open(paths.work / "cat_family.pkl", "wb") as f:
    pickle.dump(artifact, f)
json.dump({"cat_names": cat_names}, open(paths.work / "s4d_metrics.json", "w"), indent=2)
log("S4d DONE")
