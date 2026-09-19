"""Stage 4b: extra pair experts for the risk ensemble (V12-style blend):
- PN model: labelled pairs only (positive-unlabelled-free)
- CatBoost model on all pairs (PU weights)
Saves OOF predictions + full models. Risk ensemble = 0.70*PU + 0.15*PN + 0.15*Cat.
"""
import sys, gc, pickle, json
import numpy as np
import polars as pl
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa
from poker_lib import _lgb_bag

paths = resolve_paths()
log("S4b start: PN + CatBoost pair experts")

dev_pf = pd.read_parquet(paths.dev_pair_features)
with open(paths.hand_models, "rb") as f:
    art = pickle.load(f)
if "fold" not in dev_pf.columns:
    dev_pf["fold"] = dev_pf["table_id"].map(art["table_fold"]).astype(np.int8)

pair_feats = [c for c in dev_pf.columns if c not in PAIR_META and not c.startswith("sus_") and c != "top5s_same_loser"]
y = dev_pf["label"].fillna(0).astype(int).to_numpy()
labm = dev_pf["is_labeled"].to_numpy()
folds = dev_pf["fold"].to_numpy().astype(int)
X = dev_pf[pair_feats].astype("float32")
folds_oof_pn = np.zeros(len(dev_pf))
folds_oof_cat = np.zeros(len(dev_pf))

# ---------------- PN expert (labelled only, class weight)
import catboost as cb
for f in range(N_FOLDS):
    tr, te = (folds != f) & labm, folds == f
    m = lgb.train({**PAIR_PARAMS, "scale_pos_weight": 3.0}, lgb.Dataset(X[tr], y[tr]), num_boost_round=600)
    folds_oof_pn[te] = m.predict(X[te])
pn_ap_lab = average_precision_score(y[labm], folds_oof_pn[labm])
pn_ap_mir = average_precision_score(y, folds_oof_pn)
log(f"PN expert: labelled AP={pn_ap_lab:.4f} eval-mirrored={pn_ap_mir:.4f}")

# full PN model
pn_full = lgb.train({**PAIR_PARAMS, "scale_pos_weight": 3.0}, lgb.Dataset(X[labm], y[labm]), num_boost_round=600)

# ---------------- CatBoost expert (PU weights like stage-2)
# reuse stage-2 weights: need pseudo labels from s4 artifact
with open(paths.pair_models, "rb") as f:
    part = pickle.load(f)
CK1 = paths.work / "s4_stage1.npy"
oof1 = np.load(CK1)
pos_oof = oof1[labm & (y == 1)]
thr_ambig, thr_pseudo = np.quantile(pos_oof, 0.05), np.quantile(pos_oof, 0.50)
unl = ~labm
ambiguous = unl & (oof1 >= thr_ambig) & (oof1 < thr_pseudo)
pseudo = unl & (oof1 >= thr_pseudo)
y2 = np.where(pseudo, 1, y)
w2 = np.where(labm, np.where(y == 1, POS_WEIGHT, NEG_WEIGHT), np.where(pseudo, PSEUDO_POS_WEIGHT, np.where(ambiguous, 0.0, PU_WEIGHT)))
Xc = X[~ambiguous]
yc = y2[~ambiguous]
wc = w2[~ambiguous]
fc = folds[~ambiguous]
oof_cat_sub = np.zeros(len(yc))
for f in range(N_FOLDS):
    tr, te = fc != f, fc == f
    m = cb.CatBoostClassifier(iterations=420, depth=5, learning_rate=0.04, loss_function="Logloss",
                              verbose=False, random_seed=SEED + f, thread_count=2,
                              scale_pos_weight=1.0, l2_leaf_reg=5.0)
    m.fit(Xc[tr], yc[tr], sample_weight=wc[tr])
    oof_cat_sub[te] = m.predict_proba(Xc[te])[:, 1]
folds_oof_cat[~ambiguous] = oof_cat_sub
cat_ap_mir = average_precision_score(y, folds_oof_cat)
log(f"CatBoost expert: eval-mirrored AP={cat_ap_mir:.4f}")

cat_full = cb.CatBoostClassifier(iterations=420, depth=5, learning_rate=0.04, loss_function="Logloss",
                                 verbose=False, random_seed=SEED, thread_count=2, l2_leaf_reg=5.0)
cat_full.fit(Xc, yc, sample_weight=wc)

# ---------------- ensemble search on dev (rank-normalized blend)
def rnorm(v):
    return rankdata(v) / len(v)

import pickle as _pk
with open(paths.work / "s4_stage2.pkl", "rb") as _f:
    _st = _pk.load(_f)
pu_oof = _st["oof"]
best = None
for w_pu in [0.6, 0.7, 0.8]:
    for w_pn in [0.1, 0.15, 0.2, 0.3]:
        w_cat = 1.0 - w_pu - w_pn
        if w_cat < 0:
            continue
        ens = w_pu * rnorm(pu_oof) + w_pn * rnorm(folds_oof_pn) + w_cat * rnorm(folds_oof_cat)
        ap_clean_mask = ~(ambiguous | pseudo)
        ap_mir = average_precision_score(y, ens)
        ap_clean = average_precision_score(y[ap_clean_mask], ens[ap_clean_mask])
        ap_lab = average_precision_score(y[labm], ens[labm])
        print(f"  w_pu={w_pu} w_pn={w_pn} w_cat={w_cat:.2f}: mirrored={ap_mir:.4f} clean={ap_clean:.4f} labelled={ap_lab:.4f}")
        key = (ap_clean, ap_mir)
        if best is None or key > best[0]:
            best = (key, (w_pu, w_pn, w_cat))

(w_pu, w_pn, w_cat) = best[1]
log(f"selected blend: PU={w_pu} PN={w_pn} CAT={w_cat:.2f} (by clean AP then mirrored)")

artifact = dict(
    pn_full=pn_full, cat_full=cat_full, oof_pn=folds_oof_pn, oof_cat=folds_oof_cat,
    w_pu=w_pu, w_pn=w_pn, w_cat=w_cat, pair_feats=pair_feats,
)
with open(paths.work / "pair_experts.pkl", "wb") as f:
    pickle.dump(artifact, f)
json.dump({"pn_mir": float(pn_ap_mir), "cat_mir": float(cat_ap_mir), "blend": [w_pu, w_pn, w_cat]},
          open(paths.work / "s4b_metrics.json", "w"), indent=2)
log("S4b DONE")
