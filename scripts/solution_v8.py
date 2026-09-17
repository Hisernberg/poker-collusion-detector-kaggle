"""
Poker Collusion Detector v8 — Polars Lazy Streaming Self-Join
==============================================================
Use polars lazy API to push down the pairs filter through the self-join,
minimizing intermediate memory usage.
"""
import os, gc, time, json, sys, warnings, pickle, heapq
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import polars as pl
import pandas as pd

warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

DATA = Path("/home/z/my-project/kaggle/comp-data")
OUT  = Path("/home/z/my-project/kaggle/runs")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260917
N_FOLDS = 5
np.random.seed(SEED)

print("="*80)
print("STAGE 0 — Setup")
print("="*80)
t0 = time.time()

dev_labels = pd.read_csv(DATA / "development_labels.csv")
dev_evidence = pd.read_csv(DATA / "development_evidence.csv")
eval_pairs  = pd.read_csv(DATA / "evaluation_pairs.csv")
sample_sub  = pd.read_csv(DATA / "sample_submission.csv")
players_df  = pd.read_parquet(DATA / "players.parquet")
print(f"dev_labels: {dev_labels.shape}, eval_pairs: {eval_pairs.shape}")

# Build pair_lookup with sorted (p1, p2) tuple keys
pair_lookup = {}
pair_to_players = {}
for r in dev_labels.itertuples():
    p1, p2 = sorted([r.player_1, r.player_2])
    pair_lookup[(p1, p2)] = r.pair_id
    pair_to_players[r.pair_id] = (p1, p2)
for r in eval_pairs.itertuples():
    p1, p2 = sorted([r.player_1, r.player_2])
    pair_lookup[(p1, p2)] = r.pair_id
    pair_to_players[r.pair_id] = (p1, p2)
print(f"Total pairs: {len(pair_lookup):,}")

needed_players = set()
for p1, p2 in pair_to_players.values():
    needed_players.add(p1); needed_players.add(p2)
print(f"Players needed: {len(needed_players):,}")

# Build pairs_df (polars) - sorted p1, p2
pairs_df = pl.DataFrame({
    "p1": [k[0] for k in pair_lookup.keys()],
    "p2": [k[1] for k in pair_lookup.keys()],
    "pair_id": list(pair_lookup.values()),
})
print(f"pairs_df: {pairs_df.shape}")

print(f"Setup: {time.time()-t0:.1f}s")

# =========================================================================
# Build pair-hand records via polars self-join with early filtering
# =========================================================================
print()
print("="*80)
print("Building pair-hand records (polars lazy self-join)")
print("="*80)

needed_players_list = list(needed_players)

# Lazy scan of seats
seats_l = pl.scan_parquet(DATA / "seats.parquet").filter(
    pl.col("player_id").is_in(needed_players_list)
)

# We need pair-hand records: (pair_id, hand_id, p1, p2, seat_data_1, seat_data_2)
# Strategy: do a cross-join within each hand_id, filter by (p1, p2) in pairs_df

# We can use polars to do this efficiently by:
# 1. Join seats with pairs_df on p1 = player_id
# 2. Then join with seats again on (hand_id, p2 = player_id)
# This avoids the O(N^2) self-join

print("Step 1: Join seats (filtered to needed players) with pairs_df on p1...")
# For each seat row, check if its player_id is a p1 in any pair
seats_p1 = seats_l.rename({
    "player_id": "p1",
    "starting_stack": "stack1",
    "total_contribution": "contrib1",
    "net_chips": "net1",
    "folded": "folded1",
    "went_to_showdown": "showdown1",
    "won_share": "won1",
}).join(
    pairs_df.lazy(), on="p1", how="inner"
)
print(f"  seats_p1 (lazy): constructed")

print("Step 2: Join result with seats again on (hand_id, p2)...")
# Now join with seats filtered for p2 = player_id
seats_p2 = seats_l.rename({
    "player_id": "p2",
    "starting_stack": "stack2",
    "total_contribution": "contrib2",
    "net_chips": "net2",
    "folded": "folded2",
    "went_to_showdown": "showdown2",
    "won_share": "won2",
})

# Join: seats_p1 (with pair_id) joined with seats_p2 on (hand_id, p2)
# Filter p1 < p2 to avoid duplicates
pair_hand_lazy = seats_p1.join(
    seats_p2, on=["hand_id", "p2"], how="inner"
).filter(
    pl.col("p1") < pl.col("p2")
)

print("Step 3: Collecting pair_hand records (streaming)...")
t_join = time.time()
try:
    pair_hand = pair_hand_lazy.collect(streaming=True)
    print(f"pair_hand: {pair_hand.shape}, time={time.time()-t_join:.1f}s, mem={pair_hand.estimated_size('mb'):.1f}MB")
except Exception as e:
    print(f"Streaming failed: {e}")
    print("Trying without streaming...")
    pair_hand = pair_hand_lazy.collect()
    print(f"pair_hand (non-streaming): {pair_hand.shape}, time={time.time()-t_join:.1f}s, mem={pair_hand.estimated_size('mb'):.1f}MB")

print(pair_hand.head(3))

# Free memory
del seats_l
gc.collect()

# =========================================================================
# Join with hands.parquet for big_blind, final_pot
# =========================================================================
print()
print("Loading hands.parquet...")
hands_pl = pl.read_parquet(DATA / "hands.parquet", columns=[
    "hand_id", "phase", "big_blind", "final_pot", "players_dealt"
])
print(f"hands: {hands_pl.shape}")

pair_hand = pair_hand.join(hands_pl, on="hand_id", how="left")
print(f"pair_hand (with hands): {pair_hand.shape}")
del hands_pl
gc.collect()

# =========================================================================
# Compute per-(pair, hand) features (vectorized in polars)
# =========================================================================
print()
print("="*80)
print("Computing per-(pair, hand) features (polars vectorized)")
print("="*80)

pair_hand = pair_hand.with_columns([
    (pl.col("net1") / pl.col("big_blind").clip(1)).alias("net1_bb"),
    (pl.col("net2") / pl.col("big_blind").clip(1)).alias("net2_bb"),
    (pl.col("contrib1") / pl.col("big_blind").clip(1)).alias("contrib1_bb"),
    (pl.col("contrib2") / pl.col("big_blind").clip(1)).alias("contrib2_bb"),
    (pl.col("stack1") / pl.col("big_blind").clip(1)).alias("stack1_bb"),
    (pl.col("stack2") / pl.col("big_blind").clip(1)).alias("stack2_bb"),
    (pl.col("final_pot") / pl.col("big_blind").clip(1)).alias("pot_bb"),
    ((pl.col("net1") - pl.col("net2")).abs() / pl.col("big_blind").clip(1)).alias("abs_net_diff_bb"),
    ((pl.col("folded1").cast(pl.Int32)) & (pl.col("folded2").cast(pl.Int32))).alias("both_folded"),
    ((pl.col("folded1").cast(pl.Int32)) | (pl.col("folded2").cast(pl.Int32))).alias("either_folded"),
    ((pl.col("showdown1").cast(pl.Int32)) & (pl.col("showdown2").cast(pl.Int32))).alias("both_showdown"),
    (pl.col("net1") > 0).cast(pl.Int32).alias("p1_won"),
    (pl.col("net2") > 0).cast(pl.Int32).alias("p2_won"),
    (pl.col("net1") > pl.col("net2")).cast(pl.Int32).alias("p1_won_more"),
    (pl.col("net2") > pl.col("net1")).cast(pl.Int32).alias("p2_won_more"),
    (pl.col("net1") + pl.col("net2")).alias("transfer_signal"),
    (
        ((pl.col("net1") - pl.col("net2")).abs() > pl.col("big_blind") * 5) &
        ((pl.col("net1") > 0) != (pl.col("net2") > 0))
    ).cast(pl.Int32).alias("asymmetric_outcome"),
    (
        (pl.col("showdown1") & pl.col("showdown2")) &
        (pl.col("final_pot") < pl.col("big_blind") * 4)
    ).cast(pl.Int32).alias("passive_outcome"),
])

# Compute suspicion score per hand for evidence selection
pair_hand = pair_hand.with_columns(
    (
        pl.col("asymmetric_outcome") * 1.5 +
        pl.col("passive_outcome") * 2.0 +
        pl.col("abs_net_diff_bb") * 0.5 +
        pl.col("either_folded") * 0.5 +
        pl.col("transfer_signal").abs() * 0.3
    ).alias("suspicion_score")
)

# Save pair-hand features (for evidence selection)
pair_hand.write_parquet(OUT / "pair_hand_features.parquet")
print(f"Saved pair_hand_features.parquet: {pair_hand.shape}")

# =========================================================================
# Top-5 evidence hands per pair (in polars)
# =========================================================================
print()
print("Computing top-5 evidence hands per pair...")
# Sort by pair_id, suspicion_score desc; take top 5 per pair
top5_per_pair = pair_hand.sort(["pair_id", "suspicion_score"], descending=[False, True]).group_by("pair_id", maintain_order=True).head(5)
print(f"top5_per_pair: {top5_per_pair.shape}")

# =========================================================================
# Aggregate to pair-level features
# =========================================================================
print()
print("="*80)
print("STAGE A2 — Aggregating pair-level features")
print("="*80)

agg_exprs = [
    pl.len().alias("n_hands"),
    pl.col("net1_bb").sum().alias("net1_bb_sum"),
    pl.col("net1_bb").mean().alias("net1_bb_mean"),
    pl.col("net1_bb").max().alias("net1_bb_max"),
    pl.col("net1_bb").min().alias("net1_bb_min"),
    pl.col("net1_bb").std().alias("net1_bb_std"),
    pl.col("net2_bb").sum().alias("net2_bb_sum"),
    pl.col("net2_bb").mean().alias("net2_bb_mean"),
    pl.col("net2_bb").max().alias("net2_bb_max"),
    pl.col("net2_bb").min().alias("net2_bb_min"),
    pl.col("net2_bb").std().alias("net2_bb_std"),
    pl.col("contrib1_bb").sum().alias("contrib1_bb_sum"),
    pl.col("contrib1_bb").mean().alias("contrib1_bb_mean"),
    pl.col("contrib2_bb").sum().alias("contrib2_bb_sum"),
    pl.col("contrib2_bb").mean().alias("contrib2_bb_mean"),
    pl.col("stack1_bb").mean().alias("stack1_bb_mean"),
    pl.col("stack2_bb").mean().alias("stack2_bb_mean"),
    pl.col("pot_bb").mean().alias("pot_bb_mean"),
    pl.col("pot_bb").max().alias("pot_bb_max"),
    pl.col("pot_bb").sum().alias("pot_bb_sum"),
    pl.col("abs_net_diff_bb").mean().alias("abs_net_diff_bb_mean"),
    pl.col("abs_net_diff_bb").max().alias("abs_net_diff_bb_max"),
    pl.col("abs_net_diff_bb").sum().alias("abs_net_diff_bb_sum"),
    pl.col("abs_net_diff_bb").std().alias("abs_net_diff_bb_std"),
    pl.col("won1").sum().alias("won1_sum"),
    pl.col("won1").mean().alias("won1_mean"),
    pl.col("won2").sum().alias("won2_sum"),
    pl.col("won2").mean().alias("won2_mean"),
    pl.col("both_showdown").sum().alias("both_showdown_sum"),
    pl.col("both_showdown").mean().alias("both_showdown_rate"),
    pl.col("either_folded").sum().alias("either_folded_sum"),
    pl.col("either_folded").mean().alias("either_folded_rate"),
    pl.col("both_folded").sum().alias("both_folded_sum"),
    pl.col("both_folded").mean().alias("both_folded_rate"),
    pl.col("folded1").cast(pl.Int32).sum().alias("folded1_sum"),
    pl.col("folded1").cast(pl.Int32).mean().alias("folded1_rate"),
    pl.col("folded2").cast(pl.Int32).sum().alias("folded2_sum"),
    pl.col("folded2").cast(pl.Int32).mean().alias("folded2_rate"),
    pl.col("p1_won").sum().alias("p1_won_sum"),
    pl.col("p1_won").mean().alias("p1_won_rate"),
    pl.col("p2_won").sum().alias("p2_won_sum"),
    pl.col("p2_won").mean().alias("p2_won_rate"),
    pl.col("p1_won_more").sum().alias("p1_won_more_sum"),
    pl.col("p1_won_more").mean().alias("p1_dominates"),
    pl.col("p2_won_more").sum().alias("p2_won_more_sum"),
    pl.col("p2_won_more").mean().alias("p2_dominates"),
    pl.col("transfer_signal").sum().alias("transfer_signal_sum"),
    pl.col("transfer_signal").mean().alias("transfer_signal_mean"),
    pl.col("transfer_signal").min().alias("transfer_signal_min"),
    pl.col("asymmetric_outcome").sum().alias("asymmetric_outcome_sum"),
    pl.col("asymmetric_outcome").mean().alias("asymmetric_outcome_rate"),
    pl.col("passive_outcome").sum().alias("passive_outcome_sum"),
    pl.col("passive_outcome").mean().alias("passive_outcome_rate"),
]

pair_features_pl = pair_hand.group_by("pair_id").agg(agg_exprs)
print(f"pair_features_pl: {pair_features_pl.shape}")

# Derived features
pair_features_pl = pair_features_pl.with_columns([
    (pl.col("net1_bb_sum") - pl.col("net2_bb_sum")).abs().alias("net_asymmetry_sum"),
    (pl.col("net1_bb_mean") - pl.col("net2_bb_mean")).abs().alias("net_asymmetry_mean"),
    (pl.col("contrib1_bb_sum") - pl.col("contrib2_bb_sum")).abs().alias("contribution_asymmetry"),
    (pl.col("folded1_sum") - pl.col("folded2_sum")).abs().alias("fold_asymmetry"),
    (pl.col("p1_won_rate") - pl.col("p2_won_rate")).abs().alias("win_asymmetry"),
    (pl.col("p1_dominates") - pl.col("p2_dominates")).abs().alias("dominance_asymmetry"),
    (pl.col("won1_mean") - pl.col("won2_mean")).abs().alias("won_share_asymmetry"),
    (pl.col("net_asymmetry_sum") / pl.col("n_hands").clip(1)).alias("chips_transferred_rate"),
    (pl.col("contrib1_bb_sum") + pl.col("contrib2_bb_sum")).alias("total_wagered_bb"),
])

# Add player profile features (in polars)
players_p = pl.from_pandas(players_df)
for c in ["experience_hands_bucket","preferred_stake","region_bucket","client_family"]:
    players_p = players_p.with_columns(pl.col(c).cast(pl.Categorical).to_physical().alias(f"{c}_code"))

# Add p1, p2 columns
pair_features_pl = pair_features_pl.join(pairs_df, on="pair_id", how="left")

p1_prof = players_p.select([
    pl.col("player_id").alias("p1"),
    pl.col("account_age_days").alias("p1_account_age_days"),
    pl.col("experience_hands_bucket_code").alias("p1_exp_code"),
    pl.col("preferred_stake_code").alias("p1_stake_code"),
    pl.col("region_bucket_code").alias("p1_region_code"),
    pl.col("client_family_code").alias("p1_client_code"),
])
p2_prof = players_p.select([
    pl.col("player_id").alias("p2"),
    pl.col("account_age_days").alias("p2_account_age_days"),
    pl.col("experience_hands_bucket_code").alias("p2_exp_code"),
    pl.col("preferred_stake_code").alias("p2_stake_code"),
    pl.col("region_bucket_code").alias("p2_region_code"),
    pl.col("client_family_code").alias("p2_client_code"),
])

pair_features_pl = pair_features_pl.join(p1_prof, on="p1", how="left")
pair_features_pl = pair_features_pl.join(p2_prof, on="p2", how="left")

pair_features_pl = pair_features_pl.with_columns([
    (pl.col("p1_region_code") == pl.col("p2_region_code")).cast(pl.Int32).alias("same_region"),
    (pl.col("p1_stake_code") == pl.col("p2_stake_code")).cast(pl.Int32).alias("same_stake"),
    (pl.col("p1_client_code") == pl.col("p2_client_code")).cast(pl.Int32).alias("same_client"),
    (pl.col("p1_account_age_days") - pl.col("p2_account_age_days")).abs().alias("account_age_diff"),
    (pl.col("p1_account_age_days") + pl.col("p2_account_age_days")).alias("account_age_sum"),
])

# Add shared_hands from eval_pairs
eval_pairs_pl = pl.from_pandas(eval_pairs[["pair_id","shared_hands"]])
pair_features_pl = pair_features_pl.join(eval_pairs_pl, on="pair_id", how="left")
pair_features_pl = pair_features_pl.with_columns(
    pl.col("shared_hands").fill_null(pl.col("n_hands"))
)

print(f"pair_features (final): {pair_features_pl.shape}")
pair_features_pl.write_parquet(OUT / "pair_features.parquet")
print("Saved pair_features.parquet")

# Convert to pandas for sklearn
pair_features_pd = pair_features_pl.to_pandas()
del pair_features_pl
gc.collect()

# =========================================================================
# STAGE B — Train LightGBM models
# =========================================================================
print()
print("="*80)
print("STAGE B — Train LightGBM models")
print("="*80)

train_df = pair_features_pd.merge(dev_labels[["pair_id","label","behavior_family"]], on="pair_id", how="inner")
print(f"Train pairs: {train_df.shape}, positives: {int(train_df.label.sum())}")

drop_cols = {"pair_id","p1","p2","label","behavior_family"}
feat_cols = [c for c in train_df.columns if c not in drop_cols and train_df[c].dtype != object and str(train_df[c].dtype) != "string"]
print(f"# feature columns: {len(feat_cols)}")

for c in feat_cols:
    if train_df[c].isna().any():
        train_df[c] = train_df[c].fillna(train_df[c].median() if not train_df[c].isna().all() else 0)

X = train_df[feat_cols].values.astype(np.float32)
y = train_df.label.values.astype(int)
families = train_df.behavior_family.values

y_dt = (families == "directed_transfer").astype(int)
y_sp = (families == "soft_play").astype(int)
y_ci = (families == "coordinated_isolation").astype(int)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def lgb_params(num_leaves=31, n_estimators=400, lr=0.04):
    return dict(
        objective="binary", num_leaves=num_leaves,
        learning_rate=lr, n_estimators=n_estimators,
        min_child_samples=8, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.05, reg_lambda=0.1,
        random_state=SEED, n_jobs=2, verbose=-1,
    )

def train_family_models(y_family, family_name):
    print(f"\n--- Training {family_name} models ---")
    oof = np.zeros(len(train_df))
    fold_scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y_family)):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y_family[tr], y_family[va]
        model = lgb.LGBMClassifier(**lgb_params())
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
        oof[va] = model.predict_proba(X_va)[:, 1]
        if y_va.sum() > 0:
            score = average_precision_score(y_va, oof[va])
        else:
            score = 0.0
        fold_scores.append(score)
        print(f"  fold {fold}: AP={score:.4f}, best_iter={model.best_iteration_}")
    print(f"  Mean AP: {np.mean(fold_scores):.4f}")
    return oof

oof_dt = train_family_models(y_dt, "directed_transfer")
oof_sp = train_family_models(y_sp, "soft_play")
oof_ci = train_family_models(y_ci, "coordinated_isolation")
oof_risk = train_family_models(y, "any_suspicious")

oof_combined = np.maximum.reduce([oof_dt, oof_sp, oof_ci, oof_risk * 0.6])
overall_ap = average_precision_score(y, oof_combined)
print(f"\nOverall OOF AP (composite risk): {overall_ap:.4f}")

for fam, oof_arr in [("directed_transfer", oof_dt), ("soft_play", oof_sp), ("coordinated_isolation", oof_ci)]:
    y_fam = (families == fam).astype(int)
    if y_fam.sum() > 0:
        fam_ap = average_precision_score(y_fam, oof_arr)
        print(f"  {fam} OOF AP: {fam_ap:.4f}")

# Train final models on full data
print("\n--- Training final models on full data ---")
final_dt = lgb.LGBMClassifier(**lgb_params())
final_dt.fit(X, y_dt)
final_sp = lgb.LGBMClassifier(**lgb_params())
final_sp.fit(X, y_sp)
final_ci = lgb.LGBMClassifier(**lgb_params())
final_ci.fit(X, y_ci)
final_risk = lgb.LGBMClassifier(**lgb_params())
final_risk.fit(X, y)

with open(OUT / "models.pkl", "wb") as f:
    pickle.dump({"dt":final_dt, "sp":final_sp, "ci":final_ci, "risk":final_risk, "feat_cols":feat_cols}, f)
print("Saved models")

# =========================================================================
# STAGE C — Predict + build submission
# =========================================================================
print()
print("="*80)
print("STAGE C — Predict on evaluation pairs")
print("="*80)

eval_df = pair_features_pd[pair_features_pd.pair_id.isin(set(eval_pairs.pair_id))].copy()
print(f"Eval pairs: {eval_df.shape}")

for c in feat_cols:
    if eval_df[c].isna().any():
        eval_df[c] = eval_df[c].fillna(train_df[c].median() if not train_df[c].isna().all() else 0)

X_eval = eval_df[feat_cols].values.astype(np.float32)

p_dt = final_dt.predict_proba(X_eval)[:, 1]
p_sp = final_sp.predict_proba(X_eval)[:, 1]
p_ci = final_ci.predict_proba(X_eval)[:, 1]
p_risk = final_risk.predict_proba(X_eval)[:, 1]

risk_combined = np.maximum.reduce([p_dt, p_sp, p_ci, p_risk * 0.6])

behaviors = []
fam_names = ["directed_transfer", "soft_play", "coordinated_isolation"]
for i in range(len(eval_df)):
    probs = [p_dt[i], p_sp[i], p_ci[i]]
    fam_idx = int(np.argmax(probs))
    if max(probs) < 0.4 and p_risk[i] < 0.4:
        behaviors.append("none")
    else:
        behaviors.append(fam_names[fam_idx])
print(f"Behavior distribution: {Counter(behaviors)}")

# Build evidence from top5_per_pair
print("\n--- Building evidence from top-5 per pair ---")
top5_pd = top5_per_pair.to_pandas()
# For each pair, sort hands by suspicion_score (desc), take top 5
top5_pd = top5_pd.sort_values(["pair_id", "suspicion_score"], ascending=[True, False])
top5_grouped = top5_pd.groupby("pair_id")["hand_id"].apply(list).to_dict()

eval_pair_id_to_idx = {pid: i for i, pid in enumerate(eval_df.pair_id.values)}

sub_rows = []
for r in eval_pairs.itertuples():
    pid = r.pair_id
    if pid not in eval_pair_id_to_idx:
        sub_rows.append({
            "pair_id": pid, "risk_score": 0.0, "predicted_behavior": "none",
            "evidence_hand_1": "NO_EVIDENCE", "evidence_hand_2": "NO_EVIDENCE",
            "evidence_hand_3": "NO_EVIDENCE", "evidence_hand_4": "NO_EVIDENCE",
            "evidence_hand_5": "NO_EVIDENCE",
        })
        continue
    idx = eval_pair_id_to_idx[pid]
    top5 = top5_grouped.get(pid, [])
    while len(top5) < 5:
        top5.append("NO_EVIDENCE")
    sub_rows.append({
        "pair_id": pid,
        "risk_score": float(risk_combined[idx]),
        "predicted_behavior": behaviors[idx],
        "evidence_hand_1": top5[0],
        "evidence_hand_2": top5[1],
        "evidence_hand_3": top5[2],
        "evidence_hand_4": top5[3],
        "evidence_hand_5": top5[4],
    })

sub_df = pd.DataFrame(sub_rows)
sub_df.to_csv(OUT / "submission.csv", index=False)
print(f"Saved submission: {sub_df.shape}")
print(sub_df.head(5))
print(f"\nRisk score stats:")
print(sub_df.risk_score.describe())
print(f"\nPredicted behaviors:")
print(sub_df.predicted_behavior.value_counts())

cv_results = {
    "overall_oof_ap": float(overall_ap),
    "n_train": int(len(train_df)),
    "n_positives": int(y.sum()),
    "n_features": int(len(feat_cols)),
    "n_eval": int(len(eval_df)),
    "feature_cols": feat_cols,
}
with open(OUT / "cv_results.json", "w") as f:
    json.dump(cv_results, f, indent=2, default=str)

print()
print("="*80)
print(f"DONE in {time.time()-t0:.1f}s")
print(f"Submission: {OUT / 'submission.csv'}")
print(f"OOF AP: {overall_ap:.4f}")
print("="*80)
