"""Test adding sus_ hand-model aggregate columns to the pair model (cat d5 + d6 protocol)."""
import sys, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
import catboost as cb

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa

paths = resolve_paths()
dev_pf = pd.read_parquet(paths.dev_pair_features)
with open(paths.hand_models, "rb") as f:
    art = pickle.load(f)
if "fold" not in dev_pf.columns:
    dev_pf["fold"] = dev_pf["table_id"].map(art["table_fold"]).astype(np.int8)
y = dev_pf["label"].fillna(0).astype(int).to_numpy()
labm = dev_pf["is_labeled"].to_numpy()
folds = dev_pf["fold"].to_numpy().astype(int)
with open(paths.work / "s4_stage1.npy", "rb") as f:
    oof1 = np.load(f)
pos_oof = oof1[labm & (y == 1)]
thr_a, thr_p = np.quantile(pos_oof, 0.05), np.quantile(pos_oof, 0.50)
unl = ~labm
ambiguous = unl & (oof1 >= thr_a) & (oof1 < thr_p)
pseudo = unl & (oof1 >= thr_p)
y2 = np.where(pseudo, 1, y)
w2 = np.where(labm, np.where(y == 1, POS_WEIGHT, NEG_WEIGHT), np.where(pseudo, PSEUDO_POS_WEIGHT, np.where(ambiguous, 0.0, PU_WEIGHT)))
unamb = ~ambiguous
clean_full = ~(ambiguous | pseudo)
X0 = dev_pf[[c for c in dev_pf.columns if c not in PAIR_META and not c.startswith("sus_") and c != "top5s_same_loser"]].astype("float32")
SUS_COLS = ["sus_mean", "sus_max", "sus_top3", "sus_top5", "sus_p90", "sus_n95", "sus_n99", "sus_f95", "sus_burst_max", "sus_burst_excess"]
X1 = dev_pf[[c for c in dev_pf.columns if c not in PAIR_META and c != "top5s_same_loser"]].astype("float32")
print("X0 cols:", X0.shape[1], " X1 cols:", X1.shape[1])

Xc0, Xc1 = X0[unamb], X1[unamb]
yc, wc, fc = y2[unamb], w2[unamb], folds[unamb]
cf = clean_full[unamb]


def run(Xc, depth, iters, seed):
    oof = np.zeros(len(yc))
    for f in range(N_FOLDS):
        tr, te = fc != f, fc == f
        m = cb.CatBoostClassifier(iterations=iters, depth=depth, learning_rate=0.04, loss_function="Logloss",
                                  verbose=False, random_seed=seed + f, thread_count=2, l2_leaf_reg=5.0)
        m.fit(Xc[tr], yc[tr], sample_weight=wc[tr])
        oof[te] = m.predict_proba(Xc[te])[:, 1]
    return oof


def rep(tag, oof):
    cln = average_precision_score(yc[cf], oof[cf])
    mir = average_precision_score(yc, oof)
    print(f"{tag:30s} clean={cln:.5f} mir={mir:.5f}", flush=True)


oof0 = run(Xc0, 5, 420, SEED + 100)
rep("d5 i420 base (v4 CAT)", oof0)
oof1_ = run(Xc1, 5, 420, SEED + 100)
rep("d5 i420 + sus_cols", oof1_)
oof0b = run(Xc0, 6, 500, SEED + 142)
rep("d6 i500 base", oof0b)
oof1b = run(Xc1, 6, 500, SEED + 142)
rep("d6 i500 + sus_cols", oof1b)
# blend check with PU
with open(paths.work / "s4_stage2.pkl", "rb") as f:
    pu_oof = pickle.load(f)["oof"]
pu_rank = rankdata(pu_oof[unamb]) / int(unamb.sum())
for tag, oof in [("0.4PU+0.6 d5+sus", oof1_), ("0.2PU+0.8 d6+sus", oof1b)]:
    r = rankdata(oof) / len(oof)
    rep(tag + " blend", 0.4 * pu_rank + 0.6 * r if "d5" in tag else 0.2 * pu_rank + 0.8 * r)
