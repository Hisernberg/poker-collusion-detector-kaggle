"""
Poker Collusion Detector v5 — Row-Group Streaming
===================================================
Process seats.parquet row-group by row-group using pyarrow.
For each hand in a row group, generate all pairs of players and check
against pair_lookup. Only emit records for known pairs.

This avoids the memory blowup of self-joining the full seats table.
"""
import os, gc, time, json, sys, warnings, pickle
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
import numpy as np
import polars as pl
import pandas as pd
import pyarrow.parquet as pq

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

gold_evidence = defaultdict(set)
for r in dev_evidence.itertuples():
    gold_evidence[r.pair_id].add(r.hand_id)

print(f"Setup: {time.time()-t0:.1f}s")

# =========================================================================
# PASS 1 — Process seats.parquet row-group by row-group
# Build pair_hand_records: list of (pair_id, hand_id, p1, p2, seat_data)
# where seat_data = (stack, contrib, net, folded, showdown, won_share) for each
# =========================================================================
print()
print("="*80)
print("PASS 1 — Streaming seats.parquet to build pair-hand records")
print("="*80)

# First pass: build hand -> {player_id: seat_data} for needed players only
needed_players_array = set(needed_players)  # for fast lookup

# Stream seats.parquet row-group by row-group
pf_seats = pq.ParquetFile(DATA / "seats.parquet")
print(f"seats.parquet: {pf_seats.metadata.num_rows:,} rows, {pf_seats.num_row_groups} row groups")

# We'll build a dict: hand_id -> {player_id: (stack, contrib, net, folded, showdown, won_share)}
# Memory: 2M hands × ~6 players × ~40 bytes = ~480MB. OK.
hand_seat_data = defaultdict(dict)
t_seats = time.time()
rows_read = 0

for rg_idx in range(pf_seats.num_row_groups):
    # Read only the columns we need
    table = pf_seats.read_row_group(rg_idx, columns=[
        "hand_id", "player_id", "starting_stack", "total_contribution",
        "net_chips", "folded", "went_to_showdown", "won_share"
    ])
    # Convert to pandas (small for one row group)
    df = table.to_pandas()
    # Filter to needed players
    df = df[df.player_id.isin(needed_players)]
    rows_read += len(df)
    # Accumulate per-hand
    for r in df.itertuples():
        hand_seat_data[r.hand_id][r.player_id] = (
            int(r.starting_stack), int(r.total_contribution), int(r.net_chips),
            bool(r.folded), bool(r.went_to_showdown),
            float(r.won_share) if pd.notna(r.won_share) else 0.0
        )
    if (rg_idx + 1) % 20 == 0 or rg_idx == pf_seats.num_row_groups - 1:
        elapsed = time.time() - t_seats
        print(f"  row group {rg_idx+1}/{pf_seats.num_row_groups}: read {rows_read:,} rows, elapsed={elapsed:.1f}s", flush=True)

print(f"Pass 1 done in {time.time()-t_seats:.1f}s. {len(hand_seat_data):,} hands indexed.")
print(f"Current mem: {len(hand_seat_data) * 6 * 60 / 1e6:.1f}MB (estimated)")

# =========================================================================
# Build pair-hand records from hand_seat_data
# =========================================================================
print()
print("="*80)
print("Building pair-hand records (per-hand pair enumeration)")
print("="*80)
t_idx = time.time()

pair_hand_records = []  # list of (pair_id, hand_id, p1, p2, seat_data1, seat_data2)

for hand_id, players_dict in hand_seat_data.items():
    if len(players_dict) < 2:
        continue
    # Sort player IDs to ensure p1 < p2
    players_list = sorted(players_dict.keys())
    # Generate all C(n, 2) pairs
    for i in range(len(players_list)):
        p1 = players_list[i]
        s1 = players_dict[p1]
        for j in range(i+1, len(players_list)):
            p2 = players_list[j]
            key = (p1, p2)
            pid = pair_lookup.get(key)
            if pid is not None:
                s2 = players_dict[p2]
                pair_hand_records.append((pid, hand_id, p1, p2, s1, s2))

print(f"Pair-hand records: {len(pair_hand_records):,}, time={time.time()-t_idx:.1f}s")

# Free hand_seat_data
del hand_seat_data
gc.collect()

# =========================================================================
# Load hands.parquet (full, small) and join
# =========================================================================
print()
print("Loading hands.parquet...")
hands = pl.read_parquet(DATA / "hands.parquet", columns=[
    "hand_id", "phase", "big_blind", "final_pot", "players_at_showdown"
])
print(f"hands: {hands.shape}")

# Build a dict hand_id -> (phase, big_blind, final_pot, players_at_showdown)
hands_dict = {}
hands_pd = hands.to_pandas()
del hands
gc.collect()
for r in hands_pd.itertuples():
    hands_dict[r.hand_id] = (r.phase, int(r.big_blind), int(r.final_pot), int(r.players_at_showdown))
del hands_pd
gc.collect()
print(f"hands_dict: {len(hands_dict):,}")

# =========================================================================
# Compute per-(pair, hand) features
# =========================================================================
print()
print("="*80)
print("Computing per-(pair, hand) features")
print("="*80)
t_feat = time.time()

pair_hand_features = []
for pid, hand_id, p1, p2, s1, s2 in pair_hand_records:
    if hand_id not in hands_dict:
        continue
    phase, big_blind, final_pot, n_showdown = hands_dict[hand_id]
    bb = max(big_blind, 1)

    stack1, contrib1, net1, folded1, showdown1, won1 = s1
    stack2, contrib2, net2, folded2, showdown2, won2 = s2

    net1_bb = net1 / bb
    net2_bb = net2 / bb
    contrib1_bb = contrib1 / bb
    contrib2_bb = contrib2 / bb
    pot_bb = final_pot / bb
    abs_net_diff = abs(net1 - net2) / bb

    # Asymmetric outcome: one wins, other loses, big difference
    asymmetric = 1 if (abs(net1 - net2) > bb * 5) and ((net1 > 0) != (net2 > 0)) else 0
    # Passive outcome: both reach showdown with small pot
    passive = 1 if (showdown1 and showdown2 and final_pot < bb * 4) else 0
    # Transfer signal: net sum (negative = chip transfer to outsiders)
    transfer = (net1 + net2) / bb
    # Soft play: both checked down (passive)
    both_showdown = 1 if (showdown1 and showdown2) else 0
    either_folded = 1 if (folded1 or folded2) else 0
    both_folded = 1 if (folded1 and folded2) else 0
    p1_won = 1 if net1 > 0 else 0
    p2_won = 1 if net2 > 0 else 0
    p1_won_more = 1 if net1 > net2 else 0
    p2_won_more = 1 if net2 > net1 else 0

    pair_hand_features.append({
        "pair_id": pid,
        "hand_id": hand_id,
        "phase": phase,
        "net1_bb": net1_bb,
        "net2_bb": net2_bb,
        "contrib1_bb": contrib1_bb,
        "contrib2_bb": contrib2_bb,
        "stack1_bb": stack1 / bb,
        "stack2_bb": stack2 / bb,
        "pot_bb": pot_bb,
        "abs_net_diff_bb": abs_net_diff,
        "won1": won1,
        "won2": won2,
        "both_showdown": both_showdown,
        "either_folded": either_folded,
        "both_folded": both_folded,
        "folded1": int(folded1),
        "folded2": int(folded2),
        "p1_won": p1_won,
        "p2_won": p2_won,
        "p1_won_more": p1_won_more,
        "p2_won_more": p2_won_more,
        "transfer_signal": transfer,
        "asymmetric_outcome": asymmetric,
        "passive_outcome": passive,
    })

print(f"Total pair-hand features: {len(pair_hand_features):,} in {time.time()-t_feat:.1f}s")

phf = pd.DataFrame(pair_hand_features)
del pair_hand_records, pair_hand_features
gc.collect()
print(f"phf shape: {phf.shape}, mem={phf.memory_usage(deep=True).sum()/1e6:.1f}MB")
phf.to_parquet(OUT / "pair_hand_features.parquet")
print("Saved pair_hand_features.parquet")

# =========================================================================
# Aggregate to pair-level features
# =========================================================================
print()
print("="*80)
print("STAGE A2 — Aggregating pair-level features")
print("="*80)

agg_funcs = {
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

pair_features = phf.groupby("pair_id").agg(agg_funcs)
pair_features.columns = ["_".join(c) if isinstance(c, tuple) else c for c in pair_features.columns]
pair_features = pair_features.reset_index()
pair_features["n_hands"] = pair_features["net1_bb_sum"].notna().astype(int)  # placeholder; we'll set n_hands below
# Actually count distinct hands per pair
n_hands_per_pair = phf.groupby("pair_id").size().rename("n_hands")
pair_features = pair_features.merge(n_hands_per_pair, on="pair_id", how="left")
pair_features = pair_features.drop(columns=["n_hands_x"], errors="ignore").rename(columns={"n_hands_y": "n_hands"} if "n_hands_y" in pair_features.columns else pair_features)
# Clean up: keep n_hands
if "n_hands" not in pair_features.columns:
    pair_features["n_hands"] = pair_features["net1_bb_sum"].notna().astype(int)

print(f"pair_features: {pair_features.shape}")

# Derived features
pair_features["net_asymmetry_sum"] = (pair_features["net1_bb_sum"] - pair_features["net2_bb_sum"]).abs()
pair_features["net_asymmetry_mean"] = (pair_features["net1_bb_mean"] - pair_features["net2_bb_mean"]).abs()
pair_features["contribution_asymmetry"] = (pair_features["contrib1_bb_sum"] - pair_features["contrib2_bb_sum"]).abs()
pair_features["fold_asymmetry"] = (pair_features["folded1_sum"] - pair_features["folded2_sum"]).abs()
pair_features["win_asymmetry"] = (pair_features["p1_won_mean"] - pair_features["p2_won_mean"]).abs()
pair_features["dominance_asymmetry"] = (pair_features["p1_won_more_mean"] - pair_features["p2_won_more_mean"]).abs()
pair_features["won_share_asymmetry"] = (pair_features["won1_mean"] - pair_features["won2_mean"]).abs()
pair_features["chips_transferred_rate"] = pair_features["net_asymmetry_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["chips_transferred_bb"] = pair_features["net_asymmetry_sum"]
pair_features["fold_rate_p1"] = pair_features["folded1_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["fold_rate_p2"] = pair_features["folded2_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["both_showdown_rate"] = pair_features["both_showdown_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["passive_rate"] = pair_features["passive_outcome_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["asymmetric_rate"] = pair_features["asymmetric_outcome_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["p1_dominates"] = pair_features["p1_won_more_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["p2_dominates"] = pair_features["p2_won_more_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["win_rate_p1"] = pair_features["p1_won_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["win_rate_p2"] = pair_features["p2_won_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["total_wagered_bb"] = pair_features["contrib1_bb_sum"] + pair_features["contrib2_bb_sum"]

# Add player profile features
all_player_map = pd.concat([
    dev_labels[["pair_id","player_1","player_2"]].rename(columns={"player_1":"p1","player_2":"p2"}),
    eval_pairs[["pair_id","player_1","player_2"]].rename(columns={"player_1":"p1","player_2":"p2"})
]).drop_duplicates(subset=["pair_id"])
pair_features = pair_features.merge(all_player_map, on="pair_id", how="left")

# Player profile features
players_df_p = players_df.copy()
for c in ["experience_hands_bucket","preferred_stake","region_bucket","client_family"]:
    players_df_p[c+"_code"] = players_df_p[c].astype("category").cat.codes

p1_prof = players_df_p.rename(columns={"player_id":"p1","account_age_days":"p1_account_age_days",
    "experience_hands_bucket_code":"p1_exp_code","preferred_stake_code":"p1_stake_code",
    "region_bucket_code":"p1_region_code","client_family_code":"p1_client_code"})[
    ["p1","p1_account_age_days","p1_exp_code","p1_stake_code","p1_region_code","p1_client_code"]]
p2_prof = players_df_p.rename(columns={"player_id":"p2","account_age_days":"p2_account_age_days",
    "experience_hands_bucket_code":"p2_exp_code","preferred_stake_code":"p2_stake_code",
    "region_bucket_code":"p2_region_code","client_family_code":"p2_client_code"})[
    ["p2","p2_account_age_days","p2_exp_code","p2_stake_code","p2_region_code","p2_client_code"]]

pair_features = pair_features.merge(p1_prof, on="p1", how="left")
pair_features = pair_features.merge(p2_prof, on="p2", how="left")

pair_features["same_region"] = (pair_features["p1_region_code"] == pair_features["p2_region_code"]).astype(int)
pair_features["same_stake"] = (pair_features["p1_stake_code"] == pair_features["p2_stake_code"]).astype(int)
pair_features["same_client"] = (pair_features["p1_client_code"] == pair_features["p2_client_code"]).astype(int)
pair_features["account_age_diff"] = (pair_features["p1_account_age_days"] - pair_features["p2_account_age_days"]).abs()
pair_features["account_age_sum"] = pair_features["p1_account_age_days"] + pair_features["p2_account_age_days"]

# Add shared_hands from eval_pairs
pair_features = pair_features.merge(eval_pairs[["pair_id","shared_hands"]], on="pair_id", how="left")
pair_features["shared_hands"] = pair_features["shared_hands"].fillna(pair_features["n_hands"])

print(f"pair_features final: {pair_features.shape}")
pair_features.to_parquet(OUT / "pair_features.parquet")
print("Saved pair_features.parquet")

# =========================================================================
# STAGE B — Train LightGBM models
# =========================================================================
print()
print("="*80)
print("STAGE B — Train LightGBM models")
print("="*80)

train_df = pair_features.merge(dev_labels[["pair_id","label","behavior_family"]], on="pair_id", how="inner")
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

eval_df = pair_features[pair_features.pair_id.isin(set(eval_pairs.pair_id))].copy()
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
phf_eval = phf[phf.pair_id.isin(set(eval_pairs.pair_id))].copy()

phf_eval["suspicion_score"] = (
    phf_eval["asymmetric_outcome"] * 1.5 +
    phf_eval["passive_outcome"] * 2.0 +
    phf_eval["abs_net_diff_bb"] * 0.5 +
    phf_eval["either_folded"] * 0.5 +
    phf_eval["transfer_signal"].abs() * 0.3
).values

pair_to_hand_scores = defaultdict(list)
for r in phf_eval.itertuples():
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
