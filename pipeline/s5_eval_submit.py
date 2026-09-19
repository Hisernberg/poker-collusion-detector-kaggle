"""Stage 5: stream eval pair-hands -> score -> aggregate -> evidence candidates -> submission.csv"""
import sys, gc, pickle, json
import numpy as np
import polars as pl
import pandas as pd

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa
from poker_lib import _lgb_bag, _predict_bag, _spec_matrix, mix_evidence_scores, mix_evidence_soft, report_map5, map5_within_pairs, family_weight_matrix
from pairfeat import build_pair_hand_features
from aggfeat import aggregate_pairs

paths = resolve_paths()
log("S5 start: eval scoring + submission")

eval_prep = pl.read_parquet(paths.eval_pairs)
sample_sub = pl.read_csv(paths.data / "sample_submission.csv")
with open(paths.hand_models, "rb") as f:
    hart = pickle.load(f)
with open(paths.pair_models, "rb") as f:
    part = pickle.load(f)
thr = hart["thr"]
rank_full, sus_full, spec_full = hart["rank_full"], hart["sus_full"], hart["spec_full"]
pair_feats = part["pair_feats"]
behavior_rho = part["behavior_rho"]
fam_prior = part["fam_prior"]

baselines = pl.read_parquet(paths.work / "baselines.parquet")
base_eval = baselines.filter(pl.col("phase") == "evaluation").drop("phase")
phf = paths.player_hands_full

pair_table = pl.scan_parquet(paths.eval_pair_hands).group_by("pair_id").agg(pl.col("table_id").first()).collect(engine="streaming")
eval_prep = eval_prep.join(pair_table, on="pair_id", how="left")
assert eval_prep["table_id"].null_count() == 0
tables = sorted(eval_prep["table_id"].unique().to_list())
N_CHUNK = 24
chunks = [tables[i::N_CHUNK] for i in range(N_CHUNK)]

import pathlib
PART_DIR = paths.work / "s5_parts"
PART_DIR.mkdir(exist_ok=True)
eval_pf_parts = []
cand_parts = []
CAND_TOP = 30
for ci, chunk_tables in enumerate(chunks):
    part_pf = PART_DIR / f"pf_{ci:02d}.parquet"
    part_cand = PART_DIR / f"cand_{ci:02d}.parquet"
    if part_pf.exists() and part_cand.exists():
        eval_pf_parts.append(pl.read_parquet(part_pf))
        cand_parts.append(pl.read_parquet(part_cand))
        log(f"chunk {ci + 1}/{N_CHUNK}: loaded from cache")
        continue
    ph = pl.scan_parquet(paths.eval_pair_hands).filter(pl.col("table_id").is_in(chunk_tables))
    wide_lf = build_pair_hand_features(ph, paths, phf, table_ids=chunk_tables)
    hf = wide_lf.collect(engine="streaming")
    del wide_lf
    gc.collect()
    n = hf.height
    s1 = np.zeros(n, np.float32); s2 = np.zeros(n, np.float32)
    spec = {fam: np.zeros(n, np.float32) for fam in TARGET_BEHAVIORS}
    SLICE = 250_000
    for start in range(0, n, SLICE):
        end = min(start + SLICE, n)
        X = hf.slice(start, end - start).select(HAND_FEATS).to_pandas().astype("float32")
        s1[start:end] = np.mean([m.predict(X) for m in rank_full], axis=0)
        s2[start:end] = sus_full.predict(X)
        for fam in TARGET_BEHAVIORS:
            spec[fam][start:end] = np.mean([m.predict(X) for m in spec_full[fam]], axis=0)
        del X
        gc.collect()
    hf = hf.with_columns(
        pl.Series("hand_score", s1.astype(np.float32)), pl.Series("hand_sus", s2.astype(np.float32)),
        *[pl.Series(f"hand_spec_{fam}", spec[fam].astype(np.float32)) for fam in TARGET_BEHAVIORS],
    )
    # per-pair candidate union: top by global, each spec, and proxy blend
    spec_max = np.maximum.reduce([spec[f] for f in TARGET_BEHAVIORS])
    proxy = 0.6 * spec_max + 0.4 * s1
    hf2 = hf.with_columns(pl.Series("_proxy", proxy.astype(np.float32)))
    cols_keep = ["pair_id", "hand_id", "pot_bb", "hand_score", "hand_sus", *SPEC_COLS]
    frames = []
    for sc in ["hand_score", "_proxy", *SPEC_COLS]:
        t = (
            hf2.sort([sc, "hand_id"], descending=[True, False])
            .group_by("pair_id", maintain_order=True).head(CAND_TOP)
            .select(cols_keep + ([] if sc == "hand_score" else []))
        )
        frames.append(t)
    cand = pl.concat(frames).unique(subset=["pair_id", "hand_id"])
    cand_parts.append(cand.select(cols_keep))
    del hf2, cand, frames, proxy, spec_max
    gc.collect()

    dp_chunk = eval_prep.filter(pl.col("table_id").is_in(chunk_tables))
    pf = aggregate_pairs(hf, base_eval, dp_chunk, thr)
    pl.from_pandas(pf).write_parquet(part_pf)
    cand_parts[-1].write_parquet(part_cand)
    eval_pf_parts.append(pl.from_pandas(pf))
    log(f"chunk {ci + 1}/{N_CHUNK}: {hf.height:,} hands -> {len(pf):,} pairs, cand {cand_parts[-1].height:,}")
    del hf, s1, s2, spec, pf
    gc.collect()

eval_pf_pl = pl.concat(eval_pf_parts)
assert eval_pf_pl.height == 112_540, eval_pf_pl.height
eval_pf_pl.write_parquet(paths.eval_pair_features)
eval_pf = eval_pf_pl.to_pandas()
cands = pl.concat(cand_parts).unique(subset=["pair_id", "hand_id"])
cands.write_parquet(paths.eval_candidates)
log(f"candidates: {cands.height:,}")

# ---- risk + behavior
Xe = eval_pf[pair_feats].astype("float32")
risk = np.mean([m.predict(Xe) for m in part["pair_models"]], axis=0)
import os as _os
if _os.environ.get("RISK_ENSEMBLE", "0") == "1":
    from scipy.stats import rankdata
    with open(paths.work / "pair_experts.pkl", "rb") as _f:
        _exp = pickle.load(_f)
    pu_r = rankdata(risk) / len(risk)
    pn_pred = _exp["pn_full"].predict(Xe)
    cat_pred = _exp["cat_full"].predict_proba(Xe)[:, 1]
    w_pu = float(_os.environ.get("W_PU", _exp["w_pu"]))
    w_pn = float(_os.environ.get("W_PN", _exp["w_pn"]))
    w_cat = float(_os.environ.get("W_CAT", _exp["w_cat"]))
    risk = (w_pu * pu_r
            + w_pn * rankdata(pn_pred) / len(risk)
            + w_cat * rankdata(cat_pred) / len(risk))
    log(f"risk ensemble applied: w={w_pu}/{w_pn}/{w_cat}")
fam_probs = {fam: np.mean([m.predict(Xe) for m in part["fam_models"][fam]], axis=0) for fam in TARGET_BEHAVIORS}
eval_pf["risk_score"] = risk
rank_pct = pd.Series(risk).rank(ascending=False, method="first").to_numpy() / len(risk)
mat = np.column_stack([fam_probs[f] / max(fam_prior[f], 1e-9) for f in TARGET_BEHAVIORS])
pred_family = np.array(TARGET_BEHAVIORS)[mat.argmax(axis=1)]
eval_pf["predicted_behavior"] = np.where(rank_pct <= behavior_rho, pred_family, "none")
log("behavior counts: " + str(pd.Series(eval_pf["predicted_behavior"]).value_counts().to_dict()))

# ---- evidence blend on candidates
cd = cands.to_pandas()
fam_w = family_weight_matrix(fam_probs, fam_prior)
w_df = pd.DataFrame(fam_w, columns=["_w0", "_w1", "_w2"])
w_df["pair_id"] = eval_pf["pair_id"].to_numpy()
cd = cd.merge(w_df, on="pair_id", how="left")
cd = cd.merge(eval_pf[["pair_id", "predicted_behavior"]], on="pair_id", how="left")
W = cd[["_w0", "_w1", "_w2"]].to_numpy()
S = cd[list(SPEC_COLS)].to_numpy()
G = cd["hand_score"].to_numpy()
soft_all = 0.6 * (S * W).sum(axis=1) + 0.4 * G
active = cd["predicted_behavior"].to_numpy() != "none"
ev_active = G.copy()
ev_active[active] = soft_all[active]
cd["ev_active"] = ev_active
cd["ev_soft_all"] = soft_all
VARIANT = __import__("os").environ.get("EVIDENCE_VARIANT", "active")
score_col = "ev_active" if VARIANT == "active" else "ev_soft_all"
log(f"evidence variant: {VARIANT}")

top = (
    cd.sort_values(["pair_id", score_col, "pot_bb", "hand_id"], ascending=[True, False, False, True], kind="mergesort")
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

assert len(sub) == 112_540 and sub["pair_id"].is_unique
assert sub.isna().sum().sum() == 0
assert sub["risk_score"].between(0, 1).all()
assert set(sub["predicted_behavior"]) <= set(TARGET_BEHAVIORS) | {"none"}
ev_long = sub.melt(id_vars="pair_id", value_vars=list(EVIDENCE_COLUMNS), value_name="hand_id")
assert not ev_long.duplicated(["pair_id", "hand_id"]).any()
shared_keys = cands.select(["pair_id", "hand_id"]).to_pandas()
mk = ev_long[ev_long["hand_id"] != NO_EVIDENCE].merge(shared_keys, on=["pair_id", "hand_id"], how="left", indicator=True)
assert mk["_merge"].eq("both").all(), "evidence hand not a shared eval hand"

out = __import__("os").environ.get("SUB_OUT", "/home/z/my-project/subs/submission.csv")
sub.to_csv(out, index=False)
chk = pd.read_csv(out, dtype={c: str for c in EVIDENCE_COLUMNS})
assert chk.shape == (112_540, 8)
log(f"S5 DONE: wrote {out} ({__import__('os').path.getsize(out) / 1e6:.1f} MB)")
print(chk.head(3).to_string())
print(json.dumps({"rho": behavior_rho, "labelled_ap": part["labelled_ap"], "pu_ap": part["pu_ap"]}, indent=2))
