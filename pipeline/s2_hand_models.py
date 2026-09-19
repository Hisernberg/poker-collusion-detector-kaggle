"""Stage 2: build hand features for LABELLED pairs only -> train global/sus/specialist hand models.
Persists: labelled_hand_feats.parquet (with OOF scores), hand_models.pkl.
"""
import sys, gc, pickle, json
import numpy as np
import polars as pl
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa
from poker_lib import _lgb_bag, _predict_bag, _spec_matrix, mix_evidence_scores, mix_evidence_soft, report_map5, map5_within_pairs, family_weight_matrix
from pairfeat import build_pair_hand_features

paths = resolve_paths()
log("S2 start: labelled hand features")

dev_pairs = pl.read_parquet(paths.dev_pairs)
labelled_ids = set(dev_pairs.filter(pl.col("is_labeled"))["pair_id"].to_list())
evidence = pl.read_csv(paths.data / "development_evidence.csv")

if not paths.labelled_hand_feats.exists():
    ph = pl.scan_parquet(paths.dev_pair_hands).filter(pl.col("pair_id").is_in(list(labelled_ids)))
    lab_hands = ph.select("hand_id").unique().collect(engine="streaming")["hand_id"].to_list()
    wide = build_pair_hand_features(ph, paths, paths.player_hands_full, hand_ids=lab_hands)
    lab = wide.collect(engine="streaming")
    log(f"labelled pair-hand rows: {lab.height:,}")
    lab = lab.join(dev_pairs.select(["pair_id", "label", "behavior_family", "is_labeled"]), on="pair_id", how="left")
    lab = lab.join(evidence.select(["pair_id", "hand_id", pl.lit(True).alias("is_evidence")]), on=["pair_id", "hand_id"], how="left").with_columns(pl.col("is_evidence").fill_null(False))
    assert int(lab["is_evidence"].sum()) == 1817, f"evidence reconstruction {lab['is_evidence'].sum()} != 1817"
    lab.write_parquet(paths.labelled_hand_feats)
    del wide, lab
    gc.collect()

dh = pl.read_parquet(paths.labelled_hand_feats).to_pandas()
dh[HAND_FEATS] = dh[HAND_FEATS].astype("float32")
log(f"labelled frame: {dh.shape}; positives rows {(dh['label'] == 1).sum():,}")

rng = np.random.RandomState(SEED)
tables = sorted(dev_pairs["table_id"].unique().to_list())
table_fold = {t: int(v) for t, v in zip(tables, rng.permutation(len(tables)) % N_FOLDS)}
if "fold_x" in dh.columns:
    dh = dh.drop(columns=["fold_x"])
import os as _os2
RANK_W = _os2.environ.get("S2_RANK_WEIGHTS", "0") == "1"
_suffix = "_rw" if RANK_W else ""
rank_models, sus_models, spec_models = {}, {}, {f: {} for f in range(N_FOLDS)}
dh["fold"] = dh["table_id"].map(table_fold).astype(np.int8)
if RANK_W:
    _ev = pl.read_csv(paths.data / "development_evidence.csv").to_pandas()
    _rw = dh[["pair_id", "hand_id"]].merge(_ev, on=["pair_id", "hand_id"], how="left")["evidence_rank"].fillna(6).to_numpy()
    _w_all = np.where(dh["is_evidence"].to_numpy(), 7.0 - np.minimum(_rw, 6), 1.0)
else:
    _w_all = None

pos_mask = (dh["label"] == 1).to_numpy()
dh["hand_score"] = 0.0
dh["hand_sus"] = 0.0
for col in SPEC_COLS:
    dh[col] = 0.0

import os as _os
CKPT = paths.work / f"s2_ckpt{_suffix}.pkl"
if CKPT.exists():
    import pickle as _p
    with open(CKPT, "rb") as _fh:
        _ck = _p.load(_fh)
    rank_models, sus_models, spec_models = _ck["r"], _ck["s"], _ck["sp"]
    dh = _ck["dh"]
    print(f"resumed from checkpoint with folds {sorted(rank_models.keys())}")
for fold in range(N_FOLDS):
    if fold in rank_models:
        continue
    tr_pos = pos_mask & (dh["fold"] != fold).to_numpy()
    tr_all = (dh["fold"] != fold).to_numpy()
    te = (dh["fold"] == fold).to_numpy()
    X_te = dh.loc[te, HAND_FEATS]
    _w_pos = _w_all[tr_pos] if _w_all is not None else None
    rank_models[fold] = [
        lgb.train({**HAND_PARAMS, "seed": sd}, lgb.Dataset(dh.loc[tr_pos, HAND_FEATS], dh.loc[tr_pos, "is_evidence"].astype(int), weight=_w_pos), num_boost_round=HAND_ROUNDS)
        for sd in HAND_SEEDS
    ]
    sus_models[fold] = lgb.train(SUS_PARAMS, lgb.Dataset(dh.loc[tr_all, HAND_FEATS], dh.loc[tr_all, "is_evidence"].astype(int)), num_boost_round=HAND_ROUNDS)
    dh.loc[te, "hand_score"] = _predict_bag(rank_models[fold], X_te)
    dh.loc[te, "hand_sus"] = sus_models[fold].predict(X_te)
    for fam in TARGET_BEHAVIORS:
        tr_fam = tr_pos & (dh["behavior_family"] == fam).to_numpy()
        if int(tr_fam.sum()) < 200:
            spec_models[fold][fam] = rank_models[fold]
        else:
            _w_fam = _w_all[tr_fam] if _w_all is not None else None
            spec_models[fold][fam] = [
                lgb.train({**HAND_PARAMS, "seed": sd}, lgb.Dataset(dh.loc[tr_fam, HAND_FEATS], dh.loc[tr_fam, "is_evidence"].astype(int), weight=_w_fam), num_boost_round=HAND_ROUNDS)
                for sd in HAND_SEEDS
            ]
        dh.loc[te, f"hand_spec_{fam}"] = _predict_bag(spec_models[fold][fam], X_te)
    log(f"hand models fold {fold}: global MAP@5 on positive pairs = {map5_within_pairs(dh[te & pos_mask]):.4f}")
    import pickle as _p2
    with open(CKPT, "wb") as _f:
        _p2.dump({"r": rank_models, "s": sus_models, "sp": spec_models, "dh": dh}, _f)

oof_map5, fam_map5 = report_map5(dh, "hand_score", "OOF evidence (global ranker)")
dh["hand_score_ev"] = mix_evidence_scores(dh["hand_score"], _spec_matrix(dh), dh["behavior_family"])
oof_map5_ev, fam_map5_ev = report_map5(dh, "hand_score_ev", "OOF evidence (oracle family mix 0.60/0.40)")

# full models
_w_pos_full = _w_all[pos_mask] if _w_all is not None else None
rank_full = [
    lgb.train({**HAND_PARAMS, "seed": sd}, lgb.Dataset(dh.loc[pos_mask, HAND_FEATS], dh.loc[pos_mask, "is_evidence"].astype(int), weight=_w_pos_full), num_boost_round=HAND_ROUNDS)
    for sd in HAND_SEEDS
]
sus_full = lgb.train(SUS_PARAMS, lgb.Dataset(dh[HAND_FEATS], dh["is_evidence"].astype(int)), num_boost_round=HAND_ROUNDS)
spec_full = {}
for fam in TARGET_BEHAVIORS:
    mask = pos_mask & (dh["behavior_family"] == fam).to_numpy()
    _w_spf = _w_all[mask] if _w_all is not None else None
    spec_full[fam] = ([
        lgb.train({**HAND_PARAMS, "seed": sd}, lgb.Dataset(dh.loc[mask, HAND_FEATS], dh.loc[mask, "is_evidence"].astype(int), weight=_w_spf), num_boost_round=HAND_ROUNDS)
        for sd in HAND_SEEDS
    ] if int(mask.sum()) >= 200 else rank_full)

neg_scores = dh.loc[(dh["label"] == 0).to_numpy(), "hand_score"].to_numpy()
neg_sus = dh.loc[(dh["label"] == 0).to_numpy(), "hand_sus"].to_numpy()
imp = pd.Series(np.mean([m.feature_importance("gain") for m in rank_full], axis=0), index=HAND_FEATS).sort_values(ascending=False)
print("top evidence-ranker features (gain):\n", imp.head(25).round(0).to_string())

thr = dict(
    thr95=float(np.quantile(neg_scores, 0.95)), thr99=float(np.quantile(neg_scores, 0.99)),
    sus95=float(np.quantile(neg_sus, 0.95)), sus99=float(np.quantile(neg_sus, 0.99)),
)
artifact = dict(
    rank_models=rank_models, sus_models=sus_models, spec_models=spec_models,
    rank_full=rank_full, sus_full=sus_full, spec_full=spec_full,
    table_fold=table_fold, thr=thr,
    oof_map5=oof_map5, fam_map5=fam_map5, oof_map5_ev=oof_map5_ev, fam_map5_ev=fam_map5_ev,
)
with open(paths.hand_models, "wb") as f:
    pickle.dump(artifact, f)
dh.to_parquet(paths.labelled_hand_feats, index=False)
log(f"S2 DONE. OOF map5={oof_map5:.4f} ev_mix={oof_map5_ev:.4f}")
json.dump({"oof_map5": oof_map5, "oof_map5_ev": oof_map5_ev, "thr": thr}, open(paths.work / "s2_metrics.json", "w"), indent=2)
