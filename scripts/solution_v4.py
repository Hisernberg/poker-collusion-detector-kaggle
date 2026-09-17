"""
Poker Collusion Detector v4 — Polars Lazy Streaming
====================================================
Uses polars lazy self-join with streaming to avoid loading all data in memory.
"""
import os, gc, time, json, sys, warnings, pickle
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import polars as pl
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["POLARS_STREAMING_CHUNK_SIZE"] = "50000"

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

# Build pair_lookup with sorted (p1, p2)
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

# Players needed
needed_players = set()
for p1, p2 in pair_to_players.values():
    needed_players.add(p1); needed_players.add(p2)
print(f"Players needed: {len(needed_players):,}")

# Build polars DataFrame of pairs
pairs_df = pl.DataFrame({
    "p1": [k[0] for k in pair_lookup.keys()],
    "p2": [k[1] for k in pair_lookup.keys()],
    "pair_id": list(pair_lookup.values()),
})
print(f"pairs_df: {pairs_df.shape}, mem={pairs_df.estimated_size('mb'):.1f}MB")

# Gold evidence
gold_evidence = defaultdict(set)
for r in dev_evidence.itertuples():
    gold_evidence[r.pair_id].add(r.hand_id)

print(f"Setup: {time.time()-t0:.1f}s")

# =========================================================================
# Self-join seats to build (pair_id, hand_id, seat1_data, seat2_data)
# =========================================================================
print()
print("="*80)
print("Self-joining seats to build pair-hand records (streaming)...")
print("="*80)

needed_players_list = list(needed_players)

# Lazy scan seats
seats_l = pl.scan_parquet(DATA / "seats.parquet").filter(
    pl.col("player_id").is_in(needed_players_list)
).select([
    "hand_id", "player_id", "starting_stack", "total_contribution",
    "net_chips", "folded", "went_to_showdown", "won_share"
])

# Self-join with p1 < p2 (alphabetical order)
seats_left = seats_l.select([
    pl.col("hand_id"),
    pl.col("player_id").alias("p1"),
    pl.col("starting_stack").alias("stack1"),
    pl.col("total_contribution").alias("contrib1"),
    pl.col("net_chips").alias("net1"),
    pl.col("folded").alias("folded1"),
    pl.col("went_to_showdown").alias("showdown1"),
    pl.col("won_share").alias("won1"),
])
seats_right = seats_l.select([
    pl.col("hand_id"),
    pl.col("player_id").alias("p2"),
    pl.col("starting_stack").alias("stack2"),
    pl.col("total_contribution").alias("contrib2"),
    pl.col("net_chips").alias("net2"),
    pl.col("folded").alias("folded2"),
    pl.col("went_to_showdown").alias("showdown2"),
    pl.col("won_share").alias("won2"),
])

pair_hand_lazy = seats_left.join(seats_right, on="hand_id").filter(
    pl.col("p1") < pl.col("p2")
).join(
    pairs_df.lazy(), on=["p1", "p2"], how="inner"
)

t_join = time.time()
print("Starting streaming self-join (this may take 1-2 minutes)...")
pair_hand = pair_hand_lazy.collect(streaming=True)
print(f"pair_hand: {pair_hand.shape}, time={time.time()-t_join:.1f}s, mem={pair_hand.estimated_size('mb'):.1f}MB")

# Now we have (pair_id, hand_id, p1, p2, seat data) for all needed pairs
print(f"Sample:")
print(pair_hand.head(3))

# =========================================================================
# Load hands (full, small)
# =========================================================================
print()
print("Loading hands.parquet...")
hands = pl.read_parquet(DATA / "hands.parquet", columns=[
    "hand_id", "phase", "big_blind", "final_pot", "players_at_showdown"
])
print(f"hands: {hands.shape}, mem={hands.estimated_size('mb'):.1f}MB")

# Join pair_hand with hands to get big_blind, final_pot, etc.
pair_hand_full = pair_hand.join(hands, on="hand_id", how="left")
print(f"pair_hand_full: {pair_hand_full.shape}")
del pair_hand, hands
gc.collect()

# =========================================================================
# Compute per-(pair, hand) features (vectorized in polars)
# =========================================================================
print()
print("="*80)
print("Computing per-(pair, hand) features (polars vectorized)")
print("="*80)

# Build features using polars expressions
pair_hand_full = pair_hand_full.with_columns([
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

# Save pair-hand features
pair_hand_full.write_parquet(OUT / "pair_hand_features.parquet")
print(f"Saved pair_hand_features.parquet: {pair_hand_full.shape}")

# =========================================================================
# Aggregate to pair-level features
# =========================================================================
print()
print("="*80)
print("STAGE A2 — Aggregating pair-level features")
print("="*80)

agg_dict = {
    "net1_bb": ["sum", "mean", "max", "min", "std"],
    "net2_bb": ["sum", "mean", "max", "min", "std"],
    "contrib1_bb": ["sum", "mean"],
    "contrib2_bb": ["sum", "mean"],
    "stack1_bb": ["mean"],
    "stack2_bb": ["mean"],
    "pot_bb": ["mean", "max", "sum"],
    "abs_net_diff_bb": ["mean", "max", "sum", "std"],
    "won1": ["sum", "mean"],
    "won2": ["sum", "mean"],
    "both_showdown": ["sum", "mean"],
    "either_folded": ["sum", "mean"],
    "both_folded": ["sum", "mean"],
    "folded1": ["sum", "mean"],
    "folded2": ["sum", "mean"],
    "p1_won": ["sum", "mean"],
    "p2_won": ["sum", "mean"],
    "p1_won_more": ["sum", "mean"],
    "p2_won_more": ["sum", "mean"],
    "transfer_signal": ["sum", "mean", "min"],
    "asymmetric_outcome": ["sum", "mean"],
    "passive_outcome": ["sum", "mean"],
}

# Count of hands per pair (for normalization)
pair_features = pair_hand_full.group_by("pair_id").agg([
    pl.len().alias("n_hands"),
    *[pl.col(c).agg(f) for c, funcs in agg_dict.items() for f in (funcs if isinstance(funcs, list) else [funcs])]
])
print(f"pair_features after agg: {pair_features.shape}")
print(f"Columns: {pair_features.columns[:10]}... ({len(pair_features.columns)} total)")

# Polars suffix handling: when using agg with multiple functions, columns are suffixed
# Actually polars agg with a single function applied to a list works differently
# Let me try a different approach: do each aggregation separately
print("Doing more controlled aggregation...")

# Better approach: list each aggregation explicitly
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
    pl.col("folded1").sum().alias("folded1_sum"),
    pl.col("folded1").mean().alias("folded1_rate"),
    pl.col("folded2").sum().alias("folded2_sum"),
    pl.col("folded2").mean().alias("folded2_rate"),
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

pair_features = pair_hand_full.group_by("pair_id").agg(agg_exprs)
print(f"pair_features: {pair_features.shape}")

# Derived features
pair_features = pair_features.with_columns([
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
# Encode categoricals
for c in ["experience_hands_bucket","preferred_stake","region_bucket","client_family"]:
    players_p = players_p.with_columns(pl.col(c).cast(pl.Categorical).to_physical().alias(f"{c}_code"))

# Get p1, p2 per pair from pairs_df
pair_features = pair_features.join(pairs_df, on="pair_id", how="left")

# Join with players (for p1 and p2 separately)
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

pair_features = pair_features.join(p1_prof, on="p1", how="left")
pair_features = pair_features.join(p2_prof, on="p2", how="left")

# Add same-region, same-stake, etc indicators
pair_features = pair_features.with_columns([
    (pl.col("p1_region_code") == pl.col("p2_region_code")).cast(pl.Int32).alias("same_region"),
    (pl.col("p1_stake_code") == pl.col("p2_stake_code")).cast(pl.Int32).alias("same_stake"),
    (pl.col("p1_client_code") == pl.col("p2_client_code")).cast(pl.Int32).alias("same_client"),
    (pl.col("p1_account_age_days") - pl.col("p2_account_age_days")).abs().alias("account_age_diff"),
    (pl.col("p1_account_age_days") + pl.col("p2_account_age_days")).alias("account_age_sum"),
])

# Add shared_hands from eval_pairs
eval_pairs_pl = pl.from_pandas(eval_pairs[["pair_id","shared_hands"]])
pair_features = pair_features.join(eval_pairs_pl, on="pair_id", how="left")
# Fill missing shared_hands with n_hands (for dev pairs)
pair_features = pair_features.with_columns(
    pl.col("shared_hands").fill_null(pl.col("n_hands"))
)

print(f"pair_features (with player profile): {pair_features.shape}")
print(f"Total time so far: {time.time()-t0:.1f}s")

# Save
pair_features.write_parquet(OUT / "pair_features.parquet")
print("Saved pair_features.parquet")

# =========================================================================
# STAGE B — Train LightGBM models
# =========================================================================
print()
print("="*80)
print("STAGE B — Train LightGBM models")
print("="*80)

# Convert to pandas for sklearn/lightgbm
pair_features_pd = pair_features.to_pandas()
del pair_features
gc.collect()

train_df = pair_features_pd.merge(dev_labels[["pair_id","label","behavior_family"]], on="pair_id", how="inner")
print(f"Train pairs: {train_df.shape}, positives: {int(train_df.label.sum())}")

drop_cols = {"pair_id","p1","p2","label","behavior_family"}
feat_cols = [c for c in train_df.columns if c not in drop_cols and train_df[c].dtype != object and str(train_df[c].dtype) != "string"]
print(f"# feature columns: {len(feat_cols)}")

# Impute NaN
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

# Train final models
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

# Evidence selection
print("\n--- Computing per-hand evidence scores ---")
phf_pd = pl.read_parquet(OUT / "pair_hand_features.parquet")
# Filter to eval pairs
eval_pair_ids_set = set(eval_pairs.pair_id)
phf_eval = phf_pd.filter(pl.col("pair_id").is_in(list(eval_pair_ids_set)))

# Suspicion score per hand
phf_eval = phf_eval.with_columns(
    (
        pl.col("asymmetric_outcome") * 1.5 +
        pl.col("passive_outcome") * 2.0 +
        pl.col("abs_net_diff_bb") * 0.5 +
        pl.col("either_folded") * 0.5 +
        pl.col("transfer_signal").abs() * 0.3
    ).alias("suspicion_score")
)

# Convert to pandas for evidence selection
phf_eval_pd = phf_eval.select(["pair_id", "hand_id", "suspicion_score"]).to_pandas()
del phf_pd, phf_eval
gc.collect()

# Per pair, sort hands by suspicion, take top 5
pair_to_hand_scores = defaultdict(list)
for r in phf_eval_pd.itertuples():
    pair_to_hand_scores[r.pair_id].append((r.hand_id, r.suspicion_score))

# Build submission
sub_rows = []
eval_pair_id_to_idx = {pid: i for i, pid in enumerate(eval_df.pair_id.values)}

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
    hand_scores = pair_to_hand_scores.get(pid, [])
    hand_scores.sort(key=lambda x: -x[1])
    top5 = [h for h, s in hand_scores[:5]]
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
