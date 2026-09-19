"""CAT-weight ladder on HONEST OOF arrays (no full-model self-scoring)."""
import sys, pickle
import numpy as np
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata

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
with open(paths.work / "s4_stage1.npy", "rb") as f:
    oof1 = np.load(f)
pos_oof = oof1[labm & (y == 1)]
thr_a, thr_p = np.quantile(pos_oof, 0.05), np.quantile(pos_oof, 0.50)
unl = ~labm
ambiguous = unl & (oof1 >= thr_a) & (oof1 < thr_p)
pseudo = unl & (oof1 >= thr_p)
y2 = np.where(pseudo, 1, y)
unamb = ~ambiguous
clean_full = ~(ambiguous | pseudo)
yc = y2[unamb]
cf = clean_full[unamb]
print("rows:", len(yc), "clean rows:", int(cf.sum()))

# --- honest OOF arrays ---
with open(paths.work / "s4_stage2.pkl", "rb") as f:
    pu_oof = pickle.load(f)["oof"]
with open(paths.work / "pair_experts.pkl", "rb") as f:
    exp = pickle.load(f)
R = {}
R["pu"] = rankdata(pu_oof[unamb]) / int(unamb.sum())
R["cat_d5"] = rankdata(exp["oof_cat"][unamb]) / int(unamb.sum())

for tag, name in [("s4d_cat_d4i600s342.pkl", "cat_d4"), ("s4d_cat_d8i400s442.pkl", "cat_d8"),
                  ("s4d_cat_d5i900s542.pkl", "cat_d5i900"),
                  ("s4c_cat_d6i500s142.pkl", "cat_d6"), ("s4c_cat_d5i700s242.pkl", "cat_d5i700")]:
    with open(paths.work / tag, "rb") as f:
        oof, _, _ = pickle.load(f)
    assert len(oof) == len(yc), (tag, len(oof), len(yc))
    R[name] = rankdata(oof) / len(oof)

sub_path = paths.work / "s4c_lgb_1200_6_gbdt.pkl"
with open(sub_path, "rb") as f:
    ck_sub = pickle.load(f)
oof_sub = ck_sub["oof"] if isinstance(ck_sub, dict) and "oof" in ck_sub else ck_sub[0]
if len(oof_sub) == len(y2):
    oof_sub = oof_sub[unamb]
R["lgb_sub60"] = rankdata(oof_sub) / len(oof_sub)

cats5 = ["cat_d4", "cat_d5", "cat_d5i900", "cat_d8", "cat_d6", "cat_d5i700"]
R["cats_mean"] = np.mean([R[c] for c in cats5], axis=0)


def rep(tag, ens):
    cln = average_precision_score(yc[cf], ens[cf])
    mir = average_precision_score(yc, ens)
    print(f"{tag:36s} clean={cln:.5f} mir={mir:.5f}")
    return cln


print("--- ladder vs LB anchors: v4=0.4/0.6 d5 -> LB .81854 ; v5 7-expert search -> LB .81612 ---")
for w in [0.4, 0.3, 0.25, 0.2, 0.15, 0.1]:
    rep(f"{w:.2f}PU+{1-w:.2f}cat_d5", (1 - w) * R["cat_d5"] + w * R["pu"])
for w in [0.25, 0.2, 0.15, 0.1]:
    rep(f"{w:.2f}PU+{1-w:.2f}cat_d6", (1 - w) * R["cat_d6"] + w * R["pu"])
for w in [0.25, 0.2, 0.15]:
    rep(f"{w:.2f}PU+{1-w:.2f}cats_mean6", (1 - w) * R["cats_mean"] + w * R["pu"])
for w in [0.2, 0.15]:
    rep(f"{w:.2f}PU+{0.7:.2f}cats+{0.1:.2f}sub", (1 - w) * R["cats_mean"] * 0.7 / 0.7 + w * R["pu"] + 0.1 * R["lgb_sub60"] - w * R["pu"] * 0 + (0.7 - (1 - w - 0.1)) * 0)
print("--- pure family members ---")
for c in cats5:
    rep(f"pure {c}", R[c])
rep("pure lgb_sub60", R["lgb_sub60"])
