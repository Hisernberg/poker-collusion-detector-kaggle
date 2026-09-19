"""Stage 4: two-step PU pair model + family detectors + rho selection (ported)."""
import sys, gc, pickle, json
import numpy as np
import polars as pl
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa
from poker_lib import _lgb_bag, _predict_bag, _spec_matrix, mix_evidence_scores, mix_evidence_soft, report_map5, map5_within_pairs, family_weight_matrix

paths = resolve_paths()
log("S4 start: pair models")

dev_pf = pd.read_parquet(paths.dev_pair_features)
evidence = pl.read_csv(paths.data / "development_evidence.csv")
with open(paths.hand_models, "rb") as f:
    art = pickle.load(f)
dh = pl.read_parquet(paths.labelled_hand_feats).to_pandas()

if "fold" not in dev_pf.columns:
    dev_pf["fold"] = dev_pf["table_id"].map(art["table_fold"]).astype(np.int8)
pair_feats = [c for c in dev_pf.columns if c not in PAIR_META and not c.startswith("sus_") and c != "top5s_same_loser"]
log(f"dev pair features: {dev_pf.shape}; {len(pair_feats)} features")
y = dev_pf["label"].fillna(0).astype(int).to_numpy()
labm = dev_pf["is_labeled"].to_numpy()
w_train = np.where(labm, np.where(y == 1, POS_WEIGHT, NEG_WEIGHT), PU_WEIGHT)
folds = dev_pf["fold"].to_numpy().astype(int)
X = dev_pf[pair_feats].astype("float32")


def fit_pair_cv(dev_pf, X, y, labm, w_fit, folds, tag, seeds=PAIR_SEEDS, rounds=PAIR_ROUNDS, with_family=True, ckpt=None):
    oof = np.zeros(len(dev_pf))
    models = []
    fam_models = {fam: [] for fam in TARGET_BEHAVIORS}
    oof_fam = {fam: np.zeros(len(dev_pf)) for fam in TARGET_BEHAVIORS}
    import pickle as _pk
    state = {"oof": oof, "oof_fam": oof_fam, "models": models, "fam_models": fam_models, "done": []}
    if ckpt and __import__("os").path.exists(ckpt):
        with open(ckpt, "rb") as _f:
            state = _pk.load(_f)
        oof, oof_fam, models, fam_models = state["oof"], state["oof_fam"], state["models"], state["fam_models"]
    for fold in range(N_FOLDS):
        if ckpt and fold in state["done"]:
            continue
        tr, te = folds != fold, folds == fold
        tr_w = tr & (w_fit > 0)
        for sd in seeds:
            model = lgb.train({**PAIR_PARAMS, "seed": sd}, lgb.Dataset(X[tr_w], y[tr_w], weight=w_fit[tr_w]), num_boost_round=rounds)
            oof[te] += model.predict(X[te]) / len(seeds)
            models.append(model)
        if with_family:
            tr_f = tr_w & ~((y == 1) & ~labm)
            for fam in TARGET_BEHAVIORS:
                yf = (dev_pf["behavior_family"] == fam).to_numpy().astype(int)
                mf = lgb.train({**PAIR_PARAMS, "num_leaves": 7}, lgb.Dataset(X[tr_f], yf[tr_f], weight=w_fit[tr_f]), num_boost_round=FAMILY_ROUNDS)
                oof_fam[fam][te] = mf.predict(X[te])
                fam_models[fam].append(mf)
        lab = labm & te
        state["done"].append(fold)
        if ckpt:
            with open(ckpt, "wb") as _f:
                _pk.dump(state, _f)
        log(f"{tag} fold {fold}: labelled AP={average_precision_score(y[lab], oof[lab]):.4f} | eval-mirrored AP={average_precision_score(y[te], oof[te]):.4f}")
    return oof, models, oof_fam, fam_models


CK1 = paths.work / "s4_stage1.npy"
if CK1.exists():
    oof1 = np.load(CK1)
else:
    oof1, _, _, _ = fit_pair_cv(dev_pf, X, y, labm, w_train, folds, "stage-1", seeds=PAIR_SEEDS[:1], rounds=PAIR_ROUNDS, with_family=False)
    np.save(CK1, oof1)
pos_oof = oof1[labm & (y == 1)]
thr_ambig, thr_pseudo = np.quantile(pos_oof, 0.05), np.quantile(pos_oof, 0.50)
unl = ~labm
ambiguous = unl & (oof1 >= thr_ambig) & (oof1 < thr_pseudo)
pseudo = unl & (oof1 >= thr_pseudo)
log(f"two-step PU: amb>={thr_ambig:.4f} pseudo>={thr_pseudo:.4f} | pseudo {int(pseudo.sum())}, amb {int(ambiguous.sum())}, reliable neg {int((unl & ~ambiguous & ~pseudo).sum())}")

y2 = np.where(pseudo, 1, y)
w2 = np.where(labm, np.where(y == 1, POS_WEIGHT, NEG_WEIGHT), np.where(pseudo, PSEUDO_POS_WEIGHT, np.where(ambiguous, 0.0, PU_WEIGHT)))
oof, pair_models, oof_fam, fam_models = fit_pair_cv(dev_pf, X, y2, labm, w2, folds, "stage-2", ckpt=str(paths.work / "s4_stage2.pkl"))
clean = ~ambiguous & ~pseudo
log(f"stage-1 eval-mirrored AP {average_precision_score(y, oof1):.4f} -> stage-2 {average_precision_score(y, oof):.4f}")
log(f"stage-2 AP w/o colluder-like unlabelled: {average_precision_score(y[clean], oof[clean]):.4f}")
log(f"stage-2 AP counting pseudo as positive: {average_precision_score(y2[~ambiguous], oof[~ambiguous]):.4f}")

labelled_ap = average_precision_score(y[labm], oof[labm])
pu_ap = average_precision_score(y, oof)
for fam in TARGET_BEHAVIORS:
    sel = (dev_pf["behavior_family"] == fam).to_numpy() | (y == 0)
    log(f"  {fam:22s} eval-mirrored AP = {average_precision_score(y[sel], oof[sel]):.4f}")

fam_prior = {fam: int((dev_pf["behavior_family"] == fam).sum()) for fam in TARGET_BEHAVIORS}
dev_pf = dev_pf.copy()
mat = np.column_stack([oof_fam[fam] / max(fam_prior[fam], 1e-9) for fam in TARGET_BEHAVIORS])
dev_pf["pred_family"] = np.array(TARGET_BEHAVIORS)[mat.argmax(axis=1)]
pos = dev_pf["label"] == 1
log(f"family accuracy on positives: {(dev_pf.loc[pos, 'pred_family'] == dev_pf.loc[pos, 'behavior_family']).mean():.3f}")

# evidence OOF with predicted-family soft weights (uses OOF fam probs on labelled rows)
pred_map = dev_pf.set_index("pair_id")["pred_family"]
dh["pred_family"] = dh["pair_id"].map(pred_map)
w_pair = family_weight_matrix(oof_fam, fam_prior)
w_map = {pid: w_pair[i] for i, pid in enumerate(dev_pf["pair_id"].to_numpy())}
dh = dh.sort_values(["pair_id", "hand_id"]).reset_index(drop=True)
w_hand = np.stack([w_map[p] for p in dh["pair_id"].to_numpy()])
spec = _spec_matrix(dh)
hard = mix_evidence_scores(dh["hand_score"], spec, dh["pred_family"])
soft = mix_evidence_soft(dh["hand_score"], spec, w_hand)
report_map5(dh.assign(hand_score_ev=hard), "hand_score_ev", "OOF evidence (hard pred_family mix)")
report_map5(dh.assign(hand_score_ev=soft), "hand_score_ev", "OOF evidence (SOFT all-pairs mix)")

# dev-population host metric + rho selection
top5 = (
    dh.sort_values(["pair_id", "hand_score_ev", "pot_bb", "hand_id"], ascending=[True, False, False, True])
    .groupby("pair_id")["hand_id"].apply(lambda s: list(s.head(5)))
)
sol = pd.DataFrame({"pair_id": dev_pf["pair_id"], "risk_score": y, "predicted_behavior": np.where(y == 1, dev_pf["behavior_family"], "none")})
ev_map = evidence.to_pandas().groupby("pair_id")["hand_id"].apply(list)
for i, col in enumerate(EVIDENCE_COLUMNS):
    sol[col] = [(ev_map[p][i] if (p in ev_map.index and i < len(ev_map[p])) else NO_EVIDENCE) for p in sol["pair_id"]]
sub_dev = pd.DataFrame({"pair_id": dev_pf["pair_id"], "risk_score": oof, "predicted_behavior": dev_pf["pred_family"]})
for i, col in enumerate(EVIDENCE_COLUMNS):
    sub_dev[col] = [(top5[p][i] if (p in top5.index and i < len(top5[p])) else NO_EVIDENCE) for p in sub_dev["pair_id"]]
comp = host_score(sol, sub_dev, return_components=True)
log("host metric on dev population: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in comp.items()}, default=str))

rank_pct = pd.Series(oof).rank(ascending=False, method="first").to_numpy() / len(oof)
none_scores = {}
for rho in NONE_GRID:
    sd = sub_dev.copy()
    sd["predicted_behavior"] = np.where(rank_pct <= rho, dev_pf["pred_family"], "none")
    none_scores[rho] = host_score(sol, sd, return_components=True)
    print(f"  rho={rho:<7} final={none_scores[rho]['final']:.4f} behavior_map={none_scores[rho]['behavior_map']:.4f}")
best = max(v["final"] for v in none_scores.values())
behavior_rho = max(r for r in NONE_GRID if r <= 0.05 and none_scores[r]["final"] >= best - 0.0005)
comp = none_scores[behavior_rho]
log(f"selected rho={behavior_rho}")

imp = pd.Series(np.mean([m.feature_importance("gain") for m in pair_models], axis=0), index=pair_feats).sort_values(ascending=False)
print("top pair features (gain):\n", imp.head(30).round(0).to_string())

artifact = dict(
    pair_models=pair_models, fam_models=fam_models, pair_feats=pair_feats,
    behavior_rho=float(behavior_rho), fam_prior=fam_prior,
    labelled_ap=float(labelled_ap), pu_ap=float(pu_ap),
    thr_ambig=float(thr_ambig), thr_pseudo=float(thr_pseudo),
)
with open(paths.pair_models, "wb") as f:
    pickle.dump(artifact, f)
json.dump({"labelled_ap": labelled_ap, "pu_ap": pu_ap, "rho": behavior_rho,
           "oof_map5": art["oof_map5"], "oof_map5_ev": art["oof_map5_ev"]},
          open(paths.work / "s4_metrics.json", "w"), indent=2)
log("S4 DONE")
