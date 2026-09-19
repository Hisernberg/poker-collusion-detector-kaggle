"""Stage 5c: submission with the 7-expert pool blend (uses cached chunk results)."""
import sys, gc, pickle, json, os
import numpy as np
import polars as pl
import pandas as pd

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa
from poker_lib import _predict_bag

paths = resolve_paths()
log("S5c start")

eval_pf = pl.read_parquet(paths.eval_pair_features).to_pandas()
sample_sub = pl.read_csv(paths.data / "sample_submission.csv")
with open(paths.pair_models, "rb") as f:
    part = pickle.load(f)
with open(paths.work / "expert_pool.pkl", "rb") as f:
    pool = pickle.load(f)
with open(paths.hand_models, "rb") as f:
    hart = pickle.load(f)

from scipy.stats import rankdata
Xe = eval_pf[part["pair_feats"]].astype("float32")
n = len(eval_pf)
preds = {}
for k in pool["names"]:
    w = pool["weights"][k]
    if w <= 1e-6:
        preds[k] = np.zeros(n)
        continue
    if k == "pu":
        raw = np.mean([m.predict(Xe) for m in part["pair_models"]], axis=0)
    elif k == "pn":
        raw = pool["full_models"].get(k) and None
        raw = exp_pn = None
        # PN full model lives in pair_experts.pkl
        with open(paths.work / "pair_experts.pkl", "rb") as f:
            exp2 = pickle.load(f)
        raw = exp2["pn_full"].predict(Xe)
    else:
        mdl, feats = pool["full_models"][k]
        Xk = Xe[feats] if feats is not None else Xe
        raw = mdl.predict_proba(Xk)[:, 1] if hasattr(mdl, "predict_proba") else mdl.predict(Xk)
    preds[k] = rankdata(raw) / n
    log(f"expert {k}: w={w:.3f}")

risk = np.zeros(n)
for k in pool["names"]:
    risk += pool["weights"][k] * preds[k]
log("risk blended")

fam_prior = part["fam_prior"]
fam_probs = {fam: np.mean([m.predict(Xe) for m in part["fam_models"][fam]], axis=0) for fam in TARGET_BEHAVIORS}
eval_pf["risk_score"] = risk
rank_pct = pd.Series(risk).rank(ascending=False, method="first").to_numpy() / n
mat = np.column_stack([fam_probs[f] / max(fam_prior[f], 1e-9) for f in TARGET_BEHAVIORS])
pred_family = np.array(TARGET_BEHAVIORS)[mat.argmax(axis=1)]
behavior_rho = part["behavior_rho"]
eval_pf["predicted_behavior"] = np.where(rank_pct <= behavior_rho, pred_family, "none")
log("behavior counts: " + str(pd.Series(eval_pf["predicted_behavior"]).value_counts().to_dict()))

# evidence from cached candidates (v1 blend, active-only)
cands = pl.read_parquet(paths.eval_candidates).to_pandas()
fam_w = family_weight_matrix(fam_probs, fam_prior)
w_df = pd.DataFrame(fam_w, columns=["_w0", "_w1", "_w2"])
w_df["pair_id"] = eval_pf["pair_id"].to_numpy()
cd = cands.merge(w_df, on="pair_id", how="left").merge(eval_pf[["pair_id", "predicted_behavior"]], on="pair_id", how="left")
W = cd[["_w0", "_w1", "_w2"]].to_numpy()
S = cd[list(SPEC_COLS)].to_numpy()
G = cd["hand_score"].to_numpy()
soft_all = 0.6 * (S * W).sum(axis=1) + 0.4 * G
active = cd["predicted_behavior"].to_numpy() != "none"
ev = G.copy()
ev[active] = soft_all[active]
cd["ev"] = ev
top = (
    cd.sort_values(["pair_id", "ev", "pot_bb", "hand_id"], ascending=[True, False, False, True], kind="mergesort")
    .groupby("pair_id", sort=False).head(5)
)
top["rank"] = top.groupby("pair_id", sort=False).cumcount()
top_piv = top.pivot(index="pair_id", columns="rank", values="hand_id")
for i in range(5):
    if i not in top_piv.columns:
        top_piv[i] = NO_EVIDENCE
top_piv = top_piv[list(range(5))].reset_index()
top_piv.columns = ["pair_id"] + [f"evidence_hand_{i + 1}" for i in range(5)]

sub = eval_pf[["pair_id", "risk_score", "predicted_behavior"]].merge(top_piv, on="pair_id", how="left")
sub["risk_score"] = sub["risk_score"].clip(0, 1).round(9)
sec = eval_pf.set_index("pair_id").loc[sub["pair_id"], "hs_max"].to_numpy()
order = pd.DataFrame({"r": sub["risk_score"], "sec": sec}).sort_values(["r", "sec"], ascending=[True, False], kind="mergesort")
dup_rank = order.groupby("r").cumcount().reindex(sub.index)
sub["risk_score"] = np.clip(sub["risk_score"] + 1e-13 * dup_rank.to_numpy(), 0, 1)
assert sub["risk_score"].is_unique
for col in EVIDENCE_COLUMNS:
    sub[col] = sub[col].fillna(NO_EVIDENCE).astype(str)
sub = sample_sub.select("pair_id").to_pandas().merge(sub, on="pair_id", how="left")
assert len(sub) == 112_540 and sub["pair_id"].is_unique and sub.isna().sum().sum() == 0
ev_long = sub.melt(id_vars="pair_id", value_vars=list(EVIDENCE_COLUMNS), value_name="hand_id")
assert not ev_long.duplicated(["pair_id", "hand_id"]).any()
out = os.environ.get("SUB_OUT", "/home/z/my-project/subs/submission.csv")
sub.to_csv(out, index=False)
log(f"S5c DONE: wrote {out}")
