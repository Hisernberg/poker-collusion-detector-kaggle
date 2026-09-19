"""Day-2 submission builder.
- Evidence: re-score candidate hands with NEW 4-seed hand models; rank-blend 0.70*bin + 0.30*spec(pred fam / prior).
- Risk: recipe switch (env RISK_RECIPE) on OLD eval_pair_features (train-consistent).
- Full 12-point validation battery before writing.
Usage: RISK_RECIPE=s1 SUB_OUT=path python s6_day2.py
"""
import sys, gc, pickle, json, os
import numpy as np
import polars as pl
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa
from pairfeat import build_pair_hand_features

paths = resolve_paths()
log("S6 start")

RISK_RECIPE = os.environ.get("RISK_RECIPE", "s1")
SUB_OUT = os.environ.get("SUB_OUT", f"/home/z/my-project/subs/sub_day2_{RISK_RECIPE}.csv")

# ---------- 1. family prediction + risk inputs (OLD eval_pair_features, unchanged) ----------
eval_pf = pl.read_parquet(paths.eval_pair_features).to_pandas()
with open(paths.pair_models, "rb") as f:
    part = pickle.load(f)
pair_feats = part["pair_feats"]
behavior_rho = part["behavior_rho"]
fam_prior = part["fam_prior"]
Xe = eval_pf[pair_feats].astype("float32")
fam_probs = {fam: np.mean([m.predict(Xe) for m in part["fam_models"][fam]], axis=0) for fam in TARGET_BEHAVIORS}
mat = np.column_stack([fam_probs[f] / max(fam_prior[f], 1e-9) for f in TARGET_BEHAVIORS])
pred_family_arr = np.array(TARGET_BEHAVIORS)[mat.argmax(axis=1)]
fam_by_pair = pd.Series(pred_family_arr, index=eval_pf["pair_id"].to_numpy())
log(f"families: {pd.Series(pred_family_arr).value_counts().to_dict()}")

# ---------- 2. candidate keys + table assignment ----------
cands_old = pl.read_parquet(paths.eval_candidates)
cand_keys = cands_old.select(["pair_id", "hand_id"]).unique(subset=["pair_id", "hand_id"])
log(f"old candidate rows: {cand_keys.height:,}")
eh = pl.scan_parquet(paths.eval_pair_hands)
key_tab = eh.join(cand_keys.lazy(), on=["pair_id", "hand_id"], how="semi").group_by("hand_id").agg(pl.col("table_id").first()).collect(engine="streaming")
hand_tab = dict(zip(key_tab["hand_id"].to_list(), key_tab["table_id"].to_list()))
hand_ids_all = np.array(sorted(hand_tab.keys()))
log(f"unique candidate hands: {len(hand_ids_all):,}")
del cands_old, key_tab
gc.collect()

pair_tab = eh.group_by("pair_id").agg(pl.col("table_id").first()).collect(engine="streaming")
eval_prep = pl.read_parquet(paths.eval_pairs).join(pair_tab, on="pair_id", how="left")
assert eval_prep["table_id"].null_count() == 0
tables = sorted(eval_prep["table_id"].unique().to_list())
N_CHUNK = 24
chunks = [tables[i::N_CHUNK] for i in range(N_CHUNK)]

with open(paths.hand_models, "rb") as f:
    hart = pickle.load(f)
rank_full, spec_full = hart["rank_full"], hart["spec_full"]
assert len(rank_full) == 4, f"expected 4-seed bags, got {len(rank_full)}"

import pathlib
TOP5 = paths.work / "s6_top5.parquet"
PART_DIR = paths.work / "s6_parts"
PART_DIR.mkdir(exist_ok=True)
parts = []
hand_id_set_cache = None
for ci, chunk_tables in enumerate(chunks):
    if TOP5.exists():
        break
    part_f = PART_DIR / f"ev_{ci:02d}.parquet"
    if part_f.exists():
        parts.append(pl.read_parquet(part_f))
        log(f"chunk {ci+1}/{N_CHUNK}: cached")
        continue
    ct = set(chunk_tables)
    hids = [h for h, t in hand_tab.items() if t in ct]
    hid_lf = pl.LazyFrame({"hand_id": hids})
    ph = eh.filter(pl.col("table_id").is_in(chunk_tables)).join(hid_lf, on="hand_id", how="semi")
    wide = build_pair_hand_features(ph, paths, paths.player_hands_full, hand_ids=set(hids), table_ids=set(chunk_tables))
    hf = wide.collect(engine="streaming")
    del wide
    gc.collect()
    n = hf.height
    s_bin = np.zeros(n, np.float32)
    s_spec = np.zeros(n, np.float32)
    pid_np = hf["pair_id"].to_numpy()
    fam_arr = pd.Series(pid_np).map(fam_by_pair).fillna("none").to_numpy()
    SLICE = 250_000
    Xall = hf.select(HAND_FEATS)
    for start in range(0, n, SLICE):
        end = min(start + SLICE, n)
        X = Xall.slice(start, end - start).to_pandas().astype("float32")
        s_bin[start:end] = np.mean([m.predict(X) for m in rank_full], axis=0)
        del X
        gc.collect()
    # spec score: per-row predicted family's specialist, batched per family then sliced
    for fam in TARGET_BEHAVIORS:
        idx = np.where(fam_arr == fam)[0]
        if len(idx) == 0:
            continue
        preds = np.zeros(len(idx), np.float32)
        for start in range(0, len(idx), SLICE):
            sl = idx[start:start + SLICE]
            X = Xall[sl.tolist()].to_pandas().astype("float32")
            preds[start:start + SLICE] = np.mean([m.predict(X) for m in spec_full[fam]], axis=0)
            del X
            gc.collect()
        s_spec[idx] = preds / max(fam_prior[fam], 1e-9)
    out = hf.select(["pair_id", "hand_id", "pot_bb"]).with_columns(
        pl.Series("s_bin", s_bin.astype(np.float32)), pl.Series("s_spec", s_spec.astype(np.float32)))
    out.write_parquet(part_f)
    parts.append(out)
    log(f"chunk {ci+1}/{N_CHUNK}: {n:,} rows scored")
    del hf, Xall, s_bin, s_spec, out
    gc.collect()

if not TOP5.exists():
    ev = pl.concat(parts)
    assert ev.height >= cand_keys.height, (ev.height, cand_keys.height)
    log(f"scored candidate rows: {ev.height:,}")

    # ---------- 3. evidence rank blend (recipe-independent) ----------
    n_rows = ev.height
    ev = (ev.with_columns([
                pl.col("s_bin").rank(method="ordinal").alias("_b"),
                pl.col("s_spec").rank(method="ordinal").alias("_s"),
            ])
            .with_columns(((0.70 * pl.col("_b") + 0.30 * pl.col("_s")) / n_rows).alias("ev_score"))
            .drop(["_b", "_s"]))
    top = (ev.sort(["pair_id", "ev_score", "pot_bb", "hand_id"], descending=[False, True, True, False], maintain_order=False)
             .group_by("pair_id", maintain_order=True).head(5))
    top_pd = top.to_pandas()
    top_pd["rank"] = top_pd.groupby("pair_id", sort=False).cumcount()
    top_piv = top_pd.pivot(index="pair_id", columns="rank", values="hand_id")
    for i in range(5):
        if i not in top_piv.columns:
            top_piv[i] = NO_EVIDENCE
    top_piv = top_piv[list(range(5))].reset_index()
    top_piv.columns = ["pair_id"] + [f"evidence_hand_{i+1}" for i in range(5)]
    pl.from_pandas(top_piv).write_parquet(TOP5)
    log(f"top5 cached: {len(top_piv):,} pairs")
    del top, top_pd, ev, parts
    gc.collect()
else:
    top_piv = pl.read_parquet(TOP5).to_pandas()
    log(f"loaded cached top5 ({len(top_piv):,} pairs)")

# ---------- 4. risk per recipe ----------
pu_r = rankdata(np.mean([m.predict(Xe) for m in part["pair_models"]], axis=0)) / len(eval_pf)
with open(paths.work / "pair_experts.pkl", "rb") as f:
    exp = pickle.load(f)
cat_d5_r = rankdata(exp["cat_full"].predict_proba(Xe)[:, 1]) / len(eval_pf)


def load_oof_full(p):
    with open(paths.work / p, "rb") as f:
        oof, _, full = pickle.load(f)
    return full


cat_d6 = load_oof_full("s4c_cat_d6i500s142.pkl")
cat_d4 = load_oof_full("s4d_cat_d4i600s342.pkl")
cat_d8 = load_oof_full("s4d_cat_d8i400s442.pkl")
cat_d5i900 = load_oof_full("s4d_cat_d5i900s542.pkl")
cat_d5i700 = load_oof_full("s4c_cat_d5i700s242.pkl")
with open(paths.work / "s4c_lgb_1200_6_gbdt.pkl", "rb") as f:
    _ck = pickle.load(f)
sub_full, sub_feats = _ck[2], _ck[3]
sub_r = rankdata(sub_full.predict(Xe[sub_feats].astype("float32"))) / len(eval_pf)

cat_d5_full = exp["cat_full"]
cat6_ranks = [rankdata(m.predict_proba(Xe)[:, 1]) / len(eval_pf) for m in [cat_d4, cat_d5_full, cat_d5i900, cat_d8, cat_d6, cat_d5i700]]
cats6_r = np.mean(cat6_ranks, axis=0)
cat_d6_r = cat6_ranks[4]

RECIPES = {
    "s1": 0.4 * pu_r + 0.6 * cat_d5_r,          # v4 replica (LB .81854) + new evidence
    "s2": 0.2 * pu_r + 0.8 * cats6_r,            # equal-weight cat family
    "s3": 0.15 * pu_r + 0.75 * cats6_r + 0.10 * sub_r,
    "s4": 0.3 * pu_r + 0.7 * cat_d6_r,           # d6 ladder probe
}
assert RISK_RECIPE in RECIPES, RISK_RECIPE
risk = RECIPES[RISK_RECIPE]
log(f"risk recipe {RISK_RECIPE} applied")

# ---------- 5. behavior (rho cut on FINAL recipe risk rank, same as s5/v4 logic) ----------
rank_pct = pd.Series(risk).rank(ascending=False, method="first").to_numpy() / len(risk)
eval_pf["risk_score"] = risk
eval_pf["predicted_behavior"] = np.where(rank_pct <= behavior_rho, pred_family_arr, "none")

# ---------- 6. assemble ----------
sub = eval_pf[["pair_id", "risk_score", "predicted_behavior"]].merge(top_piv, on="pair_id", how="left")
sub["risk_score"] = sub["risk_score"].clip(0, 1).round(9)
sec = eval_pf.set_index("pair_id").loc[sub["pair_id"], "hs_max"].to_numpy()
order = pd.DataFrame({"r": sub["risk_score"], "sec": sec}).sort_values(["r", "sec"], ascending=[True, False], kind="mergesort")
dup_rank = order.groupby("r").cumcount().reindex(sub.index)
sub["risk_score"] = np.clip(sub["risk_score"] + 1e-13 * dup_rank.to_numpy(), 0, 1)
for col in EVIDENCE_COLUMNS:
    sub[col] = sub[col].fillna(NO_EVIDENCE).astype(str)
sample_sub = pl.read_csv(paths.data / "sample_submission.csv")
sub = sample_sub.select("pair_id").to_pandas().merge(sub, on="pair_id", how="left")

# ---------- 7. validation battery (12 green lights) ----------
G = []
G.append(("V1 shape+cols", sub.shape == (112_540, 8) and list(sub.columns) == ["pair_id", "risk_score", "predicted_behavior"] + list(EVIDENCE_COLUMNS)))
G.append(("V2 pair_id set == sample", set(sub["pair_id"]) == set(sample_sub["pair_id"].to_list()) and sub["pair_id"].is_unique))
G.append(("V3 risk in [0,1] unique", sub["risk_score"].between(0, 1).all() and sub["risk_score"].is_unique))
G.append(("V4 behavior valid", set(sub["predicted_behavior"]) <= set(TARGET_BEHAVIORS) | {"none"}))
G.append(("V5 no NaN", sub.isna().sum().sum() == 0))
ev_long = sub.melt(id_vars="pair_id", value_vars=list(EVIDENCE_COLUMNS), value_name="hand_id")
G.append(("V6 no dup evidence in row", not ev_long.duplicated(["pair_id", "hand_id"]).any()))
mk = ev_long[ev_long["hand_id"] != NO_EVIDENCE]
# V7 (merge-based, OOM-safe): every evidence (pair,hand) must be a shared eval pair-hand
mk_pl = pl.from_pandas(mk[["pair_id", "hand_id"]].drop_duplicates())
shared_hits = eh.join(mk_pl.lazy(), on=["pair_id", "hand_id"], how="semi").select("hand_id").collect(engine="streaming")
G.append(("V7 evidence in shared hands", shared_hits.height == mk_pl.height))
# V8: evidence hands in evaluation phase (semi-join on unique hands)
ev_hands = mk_pl.select("hand_id").unique()
hm = pl.scan_parquet(paths.hands_meta).select(["hand_id", "phase"]).join(ev_hands.lazy(), on="hand_id", how="semi").collect(engine="streaming")
G.append(("V8 evidence in eval phase", hm.height == ev_hands.height and (hm["phase"] == "evaluation").all()))
# V9 (vectorized, seats.parquet): both pair members actually seated in every evidence hand
need = mk[["pair_id", "hand_id"]].drop_duplicates()
pp = pl.read_parquet(paths.eval_pairs)
seats_ev = (pl.scan_parquet(paths.data / "seats.parquet")
            .select(["hand_id", "player_id"])
            .join(pl.from_pandas(need[["hand_id"]].drop_duplicates()).lazy(), on="hand_id", how="semi")
            .collect(engine="streaming").to_pandas())
chk9 = need.merge(pd.DataFrame({"pair_id": pp["pair_id"], "_p1": pp["player_1"], "_p2": pp["player_2"]}), on="pair_id", how="left")
chk9 = chk9.merge(seats_ev, on="hand_id", how="left")
agg = chk9.assign(_h1=lambda d: d["player_id"] == d["_p1"], _h2=lambda d: d["player_id"] == d["_p2"]).groupby(["pair_id", "hand_id"])[["_h1", "_h2"]].any()
G.append(("V9 both pair members seated", bool(agg["_h1"].all() and agg["_h2"].all())))
# V10 (vectorized): NO_EVIDENCE must form a suffix in every row
rows_ev = sub[list(EVIDENCE_COLUMNS)].to_numpy()
is_ne = rows_ev == NO_EVIDENCE
bad10 = (is_ne[:, :-1] & ~is_ne[:, 1:]).any()
G.append(("V10 NO_EVIDENCE trailing only", not bad10))
ne_counts = is_ne.sum(axis=1)
G.append(("V11 top-risk pairs have evidence", (ne_counts[sub["risk_score"].rank(ascending=False) <= behavior_rho * len(sub)] == 0).all()))
os.makedirs(os.path.dirname(SUB_OUT), exist_ok=True)
sub.to_csv(SUB_OUT, index=False)
chk = pd.read_csv(SUB_OUT, dtype={c: str for c in EVIDENCE_COLUMNS})
G.append(("V12 CSV roundtrip", chk.shape == (112_540, 8) and not chk.isna().sum().sum() and (chk["risk_score"].between(0, 1)).all()))
for name, ok in G:
    log(f"{'GREEN' if ok else '*** FAIL ***'}: {name}")
assert all(ok for _, ok in G), "validation battery failed"
log(f"S6 DONE: {SUB_OUT} ({os.path.getsize(SUB_OUT)/1e6:.1f} MB)")
print(chk.head(3).to_string())
