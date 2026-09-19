"""Stage 4c: diverse expert pool for the risk ensemble.
Experts: CatBoost d5(420), d6(500), d5(700, seed2); LGB-DART; LGB-subspace(0.6); PU LGB (existing); PN (existing).
Extended rank-normalized ensemble grid -> save best weights + full models.
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
log("S4c start: expert pool")

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

with open(paths.work / "s4_stage2.pkl", "rb") as f:
    pu_oof = pickle.load(f)["oof"]
with open(paths.work / "pair_experts.pkl", "rb") as f:
    exp = pickle.load(f)
oof_pn, oof_cat = exp["oof_pn"], exp["oof_cat"]

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
clean = ~(ambiguous | pseudo)


def run_cat(depth, iters, seed, l2=5.0):
    tag = f"cat_d{depth}i{iters}s{seed}"
    ck = paths.work / f"s4c_{tag}.pkl"
    if ck.exists():
        with open(ck, "rb") as _f:
            return pickle.load(_f)
    oof = np.zeros(len(yc))
    models = []
    for f in range(N_FOLDS):
        tr, te = fc != f, fc == f
        m = cb.CatBoostClassifier(iterations=iters, depth=depth, learning_rate=0.04, loss_function="Logloss",
                                  verbose=False, random_seed=seed + f, thread_count=2, l2_leaf_reg=l2)
        m.fit(Xc[tr], yc[tr], sample_weight=wc[tr])
        oof[te] = m.predict_proba(Xc[te])[:, 1]
        models.append(m)
    full = cb.CatBoostClassifier(iterations=iters, depth=depth, learning_rate=0.04, loss_function="Logloss",
                                 verbose=False, random_seed=seed, thread_count=2, l2_leaf_reg=l2)
    full.fit(Xc, yc, sample_weight=wc)
    with open(ck, "wb") as _f:
        pickle.dump((oof, models, full), _f)
    return oof, models, full


def run_lgb(params, rounds, subspace=None):
    tag = f"lgb_{rounds}_{int(subspace * 10) if subspace else 'full'}_{params.get('boosting', 'gbdt')}"
    ck = paths.work / f"s4c_{tag}.pkl"
    if ck.exists():
        with open(ck, "rb") as _f:
            return pickle.load(_f)
    feats = pair_feats if subspace is None else [c for c in pair_feats if hash(c) % 10 < int(subspace * 10)]
    oof = np.zeros(len(yc))
    models = []
    for f in range(N_FOLDS):
        tr, te = fc != f, fc == f
        m = lgb.train(params, lgb.Dataset(Xc.loc[tr, feats], yc[tr], weight=wc[tr]), num_boost_round=rounds)
        oof[te] = m.predict(Xc.loc[te, feats])
        models.append(m)
    full = lgb.train(params, lgb.Dataset(Xc[feats], yc, weight=wc), num_boost_round=rounds)
    with open(ck, "wb") as _f:
        pickle.dump((oof, models, full, feats), _f)
    return oof, models, full, feats


EXPERTS = {}
EXPERTS["pu"] = (pu_oof, None, None, None)  # existing
EXPERTS["pn"] = (oof_pn, None, None, None)
EXPERTS["cat_d5"] = (oof_cat, None, exp["cat_full"], None)

oof, models, full = run_cat(6, 500, SEED + 100)
oof_full = np.full(len(dev_pf), np.nan); oof_full[~ambiguous] = oof
EXPERTS["cat_d6"] = (oof_full, models, full, None)
log(f"cat_d6 done")
oof, models, full = run_cat(5, 700, SEED + 200, l2=8.0)
oof_full2 = np.full(len(dev_pf), np.nan); oof_full2[~ambiguous] = oof
EXPERTS["cat_d5i700"] = (oof_full2, models, full, None)
log("cat_d5i700 done")

dart_params = {**PAIR_PARAMS, "boosting": "dart", "learning_rate": 0.08, "drop_rate": 0.15, "max_drop": 40}
oof, models, full, feats = run_lgb(dart_params, 350)
oof_full3 = np.full(len(dev_pf), np.nan); oof_full3[~ambiguous] = oof
EXPERTS["lgb_dart"] = (oof_full3, models, full, feats)
log("lgb_dart done")

oof, models, full, feats = run_lgb({**PAIR_PARAMS, "num_leaves": 63, "learning_rate": 0.02}, 1200, subspace=0.6)
oof_full4 = np.full(len(dev_pf), np.nan); oof_full4[~ambiguous] = oof
EXPERTS["lgb_sub60"] = (oof_full4, models, full, feats)
log("lgb_sub60 done")

# map sub-oofs back to full length (nan for ambiguous)
names = list(EXPERTS.keys())
M = np.column_stack([
    (np.nan_to_num(EXPERTS[k][0], nan=0.0) if k != "pu" and k != "pn" else EXPERTS[k][0])
    for k in names
])
active_mask = ~ambiguous
Mf = M[active_mask]
yf = y2[active_mask]
cf = clean[active_mask]
lf = labm[active_mask]
R = np.column_stack([rankdata(Mf[:, j]) / len(Mf) for j in range(Mf.shape[1])])

rng = np.random.RandomState(0)
best = None
# coordinate random search over weights
for it in range(4000):
    w = rng.dirichlet(np.ones(len(names)) * np.array([2.0 if k in ("pu", "cat_d5", "cat_d6") else 1.0 for k in names]))
    ens = R @ w
    cln = average_precision_score(yf[cf], ens[cf])
    mir = average_precision_score(yf, ens)
    score = cln + mir
    if best is None or score > best[0]:
        best = (score, w.copy(), cln, mir)
log(f"best weights: {dict(zip(names, np.round(best[1], 3)))} clean={best[2]:.4f} mir={best[3]:.4f}")

artifact = dict(names=names, weights=dict(zip(names, best[1].tolist())),
                full_models={k: (EXPERTS[k][2], EXPERTS[k][3]) for k in names if k not in ("pu", "pn")})
with open(paths.work / "expert_pool.pkl", "wb") as f:
    pickle.dump(artifact, f)
json.dump({"names": names, "weights": dict(zip(names, [float(x) for x in best[1]])), "clean": best[2], "mir": best[3]},
          open(paths.work / "s4c_metrics.json", "w"), indent=2)
log("S4c DONE")
