"""
Kaggle Notebook: Poker Collusion Detector v1
==============================================
Run on Kaggle for competition submission.

Approach:
  1. Read all competition data from /kaggle/input/
  2. Process per-pair hand-level features (using seats + hands, no actions for memory)
  3. Train 3 behavior-specific LightGBM models + 1 generic risk model
  4. Predict on evaluation_pairs
  5. Select top-5 evidence hands per pair
  6. Output submission.csv
"""
import os, gc, time, json, sys, warnings, pickle, heapq
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

# Kaggle paths
DATA = Path("/kaggle/input/detect-suspicious-value-transfers-in-poker")
OUT  = Path("/kaggle/working")
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

print(f"Setup: {time.time()-t0:.1f}s")

# =========================================================================
# Load hands.parquet (full)
# =========================================================================
print()
print("Loading hands.parquet...")
hands_pl = pl.read_parquet(DATA / "hands.parquet", columns=[
    "hand_id", "phase", "big_blind", "final_pot", "players_dealt"
])
print(f"hands_pl: {hands_pl.shape}, mem={hands_pl.estimated_size('mb'):.1f}MB")

hands_dict = {}
for row in hands_pl.iter_rows(named=True):
    hands_dict[row["hand_id"]] = (
        row["phase"], int(row["big_blind"]), int(row["final_pot"]), int(row["players_dealt"])
    )
del hands_pl
gc.collect()
print(f"hands_dict: {len(hands_dict):,}")

# =========================================================================
# Load seats (filtered to needed players) using polars
# =========================================================================
print()
print("Loading seats.parquet (filtered)...")
needed_players_list = list(needed_players)
seats_pl = pl.scan_parquet(DATA / "seats.parquet").filter(
    pl.col("player_id").is_in(needed_players_list)
).select([
    "hand_id", "player_id", "starting_stack", "total_contribution",
    "net_chips", "folded", "went_to_showdown", "won_share"
]).collect()
print(f"seats_pl: {seats_pl.shape}, mem={seats_pl.estimated_size('mb'):.1f}MB")

# =========================================================================
# Now process: per-hand pair enumeration, build pair features
# Use polars group_by for efficiency
# =========================================================================
print()
print("="*80)
print("Per-hand processing")
print("="*80)

# Group seats by hand_id
seats_grouped = seats_pl.group_by("hand_id").agg([
    pl.col("player_id").alias("players"),
    pl.col("starting_stack").alias("stacks"),
    pl.col("total_contribution").alias("contribs"),
    pl.col("net_chips").alias("nets"),
    pl.col("folded").alias("foldeds"),
    pl.col("went_to_showdown").alias("showdowns"),
    pl.col("won_share").alias("won_shares"),
])
print(f"seats_grouped: {seats_grouped.shape}")

# Convert to pandas for easier iteration (2M rows is OK on Kaggle with 16GB)
seats_grouped_pd = seats_grouped.to_pandas()
del seats_pl, seats_grouped
gc.collect()
print(f"seats_grouped_pd: {seats_grouped_pd.shape}")

# Free seats memory
gc.collect()

# Per-pair accumulators
def new_accumulator():
    return {
        "n_hands": 0,
        "net1_sum": 0.0, "net1_sq_sum": 0.0, "net1_max": -1e18, "net1_min": 1e18,
        "net2_sum": 0.0, "net2_sq_sum": 0.0, "net2_max": -1e18, "net2_min": 1e18,
        "contrib1_sum": 0.0, "contrib2_sum": 0.0,
        "stack1_sum": 0.0, "stack2_sum": 0.0,
        "pot_sum": 0.0, "pot_max": 0.0,
        "abs_net_diff_sum": 0.0, "abs_net_diff_max": 0.0, "abs_net_diff_sq_sum": 0.0,
        "won1_sum": 0.0, "won2_sum": 0.0,
        "both_showdown_count": 0, "either_folded_count": 0, "both_folded_count": 0,
        "folded1_count": 0, "folded2_count": 0,
        "p1_won_count": 0, "p2_won_count": 0,
        "p1_won_more_count": 0, "p2_won_more_count": 0,
        "transfer_signal_sum": 0.0, "transfer_signal_min": 1e18,
        "asymmetric_outcome_count": 0, "passive_outcome_count": 0,
    }

pair_accum = defaultdict(new_accumulator)
pair_top5 = defaultdict(list)

# Iterate through seats_grouped_pd (one row per hand, with player lists)
print("Iterating per-hand...")
t_iter = time.time()
hands_processed = 0
total_pairs = 0

for row in seats_grouped_pd.itertuples():
    hand_id = row.hand_id
    if hand_id not in hands_dict:
        continue
    phase, big_blind, final_pot, players_dealt = hands_dict[hand_id]
    bb = max(big_blind, 1)

    players_list = list(row.players)
    n = len(players_list)
    if n < 2:
        continue

    stacks = list(row.stacks)
    contribs = list(row.contribs)
    nets = list(row.nets)
    foldeds = list(row.foldeds)
    showdowns = list(row.showdowns)
    won_shares = list(row.won_shares)

    # Build players_dict for fast lookup
    players_dict = {}
    for i in range(n):
        players_dict[players_list[i]] = (
            int(stacks[i]), int(contribs[i]), int(nets[i]),
            bool(foldeds[i]), bool(showdowns[i]),
            float(won_shares[i]) if won_shares[i] is not None else 0.0
        )

    # Sort players to ensure p1 < p2
    sorted_players = sorted(players_dict.keys())

    # Enumerate pairs
    for i in range(len(sorted_players)):
        p1 = sorted_players[i]
        s1 = players_dict[p1]
        for j in range(i+1, len(sorted_players)):
            p2 = sorted_players[j]
            key = (p1, p2)
            pid = pair_lookup.get(key)
            if pid is None:
                continue
            s2 = players_dict[p2]
            stack1, contrib1, net1, folded1, showdown1, won1 = s1
            stack2, contrib2, net2, folded2, showdown2, won2 = s2
            net1_bb = net1 / bb
            net2_bb = net2 / bb
            contrib1_bb = contrib1 / bb
            contrib2_bb = contrib2 / bb
            stack1_bb = stack1 / bb
            stack2_bb = stack2 / bb
            pot_bb = final_pot / bb
            abs_net_diff = abs(net1 - net2) / bb
            transfer = (net1 + net2) / bb
            both_showdown = 1 if (showdown1 and showdown2) else 0
            either_folded = 1 if (folded1 or folded2) else 0
            both_folded = 1 if (folded1 and folded2) else 0
            p1_won = 1 if net1 > 0 else 0
            p2_won = 1 if net2 > 0 else 0
            p1_won_more = 1 if net1 > net2 else 0
            p2_won_more = 1 if net2 > net1 else 0
            asymmetric = 1 if (abs(net1 - net2) > bb * 5) and ((net1 > 0) != (net2 > 0)) else 0
            passive = 1 if (showdown1 and showdown2 and final_pot < bb * 4) else 0
            acc = pair_accum[pid]
            acc["n_hands"] += 1
            acc["net1_sum"] += net1_bb
            acc["net1_sq_sum"] += net1_bb * net1_bb
            if net1_bb > acc["net1_max"]: acc["net1_max"] = net1_bb
            if net1_bb < acc["net1_min"]: acc["net1_min"] = net1_bb
            acc["net2_sum"] += net2_bb
            acc["net2_sq_sum"] += net2_bb * net2_bb
            if net2_bb > acc["net2_max"]: acc["net2_max"] = net2_bb
            if net2_bb < acc["net2_min"]: acc["net2_min"] = net2_bb
            acc["contrib1_sum"] += contrib1_bb
            acc["contrib2_sum"] += contrib2_bb
            acc["stack1_sum"] += stack1_bb
            acc["stack2_sum"] += stack2_bb
            acc["pot_sum"] += pot_bb
            if pot_bb > acc["pot_max"]: acc["pot_max"] = pot_bb
            acc["abs_net_diff_sum"] += abs_net_diff
            if abs_net_diff > acc["abs_net_diff_max"]: acc["abs_net_diff_max"] = abs_net_diff
            acc["abs_net_diff_sq_sum"] += abs_net_diff * abs_net_diff
            acc["won1_sum"] += won1
            acc["won2_sum"] += won2
            acc["both_showdown_count"] += both_showdown
            acc["either_folded_count"] += either_folded
            acc["both_folded_count"] += both_folded
            acc["folded1_count"] += int(folded1)
            acc["folded2_count"] += int(folded2)
            acc["p1_won_count"] += p1_won
            acc["p2_won_count"] += p2_won
            acc["p1_won_more_count"] += p1_won_more
            acc["p2_won_more_count"] += p2_won_more
            acc["transfer_signal_sum"] += transfer
            if transfer < acc["transfer_signal_min"]: acc["transfer_signal_min"] = transfer
            acc["asymmetric_outcome_count"] += asymmetric
            acc["passive_outcome_count"] += passive
            suspicion_score = (
                asymmetric * 1.5 +
                passive * 2.0 +
                abs_net_diff * 0.5 +
                either_folded * 0.5 +
                abs(transfer) * 0.3
            )
            heap = pair_top5[pid]
            if len(heap) < 5:
                heapq.heappush(heap, (suspicion_score, hand_id))
            else:
                heapq.heappushpop(heap, (suspicion_score, hand_id))
            total_pairs += 1

    hands_processed += 1
    if hands_processed % 200_000 == 0:
        elapsed = time.time() - t_iter
        print(f"  processed {hands_processed:,} hands, {total_pairs:,} pair-hand records, elapsed={elapsed:.1f}s", flush=True)

print(f"Iteration done in {time.time()-t_iter:.1f}s. {hands_processed:,} hands, {total_pairs:,} pair-hand records.")

# Free seats_grouped_pd and hands_dict
del seats_grouped_pd, hands_dict
gc.collect()

# =========================================================================
# Build pair-level features DataFrame
# =========================================================================
print()
print("="*80)
print("Building pair-level features DataFrame")
print("="*80)

pair_ids_with_data = list(pair_accum.keys())
print(f"Pairs with data: {len(pair_ids_with_data):,}")

rows = []
for pid in pair_ids_with_data:
    acc = pair_accum[pid]
    n = acc["n_hands"]
    if n == 0:
        continue
    net1_mean = acc["net1_sum"] / n
    net2_mean = acc["net2_sum"] / n
    net1_var = max(0.0, acc["net1_sq_sum"] / n - net1_mean * net1_mean)
    net2_var = max(0.0, acc["net2_sq_sum"] / n - net2_mean * net2_mean)
    abs_diff_mean = acc["abs_net_diff_sum"] / n
    abs_diff_var = max(0.0, acc["abs_net_diff_sq_sum"] / n - abs_diff_mean * abs_diff_mean)
    row = {
        "pair_id": pid,
        "n_hands": n,
        "net1_sum": acc["net1_sum"], "net1_mean": net1_mean,
        "net1_max": acc["net1_max"], "net1_min": acc["net1_min"],
        "net1_std": float(np.sqrt(net1_var)),
        "net2_sum": acc["net2_sum"], "net2_mean": net2_mean,
        "net2_max": acc["net2_max"], "net2_min": acc["net2_min"],
        "net2_std": float(np.sqrt(net2_var)),
        "contrib1_sum": acc["contrib1_sum"], "contrib1_mean": acc["contrib1_sum"] / n,
        "contrib2_sum": acc["contrib2_sum"], "contrib2_mean": acc["contrib2_sum"] / n,
        "stack1_mean": acc["stack1_sum"] / n,
        "stack2_mean": acc["stack2_sum"] / n,
        "pot_sum": acc["pot_sum"], "pot_mean": acc["pot_sum"] / n,
        "pot_max": acc["pot_max"],
        "abs_net_diff_sum": acc["abs_net_diff_sum"], "abs_net_diff_mean": abs_diff_mean,
        "abs_net_diff_max": acc["abs_net_diff_max"], "abs_net_diff_std": float(np.sqrt(abs_diff_var)),
        "won1_sum": acc["won1_sum"], "won1_mean": acc["won1_sum"] / n,
        "won2_sum": acc["won2_sum"], "won2_mean": acc["won2_sum"] / n,
        "both_showdown_count": acc["both_showdown_count"], "both_showdown_rate": acc["both_showdown_count"] / n,
        "either_folded_count": acc["either_folded_count"], "either_folded_rate": acc["either_folded_count"] / n,
        "both_folded_count": acc["both_folded_count"], "both_folded_rate": acc["both_folded_count"] / n,
        "folded1_count": acc["folded1_count"], "folded1_rate": acc["folded1_count"] / n,
        "folded2_count": acc["folded2_count"], "folded2_rate": acc["folded2_count"] / n,
        "p1_won_count": acc["p1_won_count"], "p1_won_rate": acc["p1_won_count"] / n,
        "p2_won_count": acc["p2_won_count"], "p2_won_rate": acc["p2_won_count"] / n,
        "p1_won_more_count": acc["p1_won_more_count"], "p1_dominates": acc["p1_won_more_count"] / n,
        "p2_won_more_count": acc["p2_won_more_count"], "p2_dominates": acc["p2_won_more_count"] / n,
        "transfer_signal_sum": acc["transfer_signal_sum"], "transfer_signal_mean": acc["transfer_signal_sum"] / n,
        "transfer_signal_min": acc["transfer_signal_min"],
        "asymmetric_outcome_count": acc["asymmetric_outcome_count"], "asymmetric_outcome_rate": acc["asymmetric_outcome_count"] / n,
        "passive_outcome_count": acc["passive_outcome_count"], "passive_outcome_rate": acc["passive_outcome_count"] / n,
    }
    row["net_asymmetry_sum"] = abs(acc["net1_sum"] - acc["net2_sum"])
    row["net_asymmetry_mean"] = abs(net1_mean - net2_mean)
    row["contribution_asymmetry"] = abs(acc["contrib1_sum"] - acc["contrib2_sum"])
    row["fold_asymmetry"] = abs(acc["folded1_count"] - acc["folded2_count"])
    row["win_asymmetry"] = abs(row["p1_won_rate"] - row["p2_won_rate"])
    row["dominance_asymmetry"] = abs(row["p1_dominates"] - row["p2_dominates"])
    row["won_share_asymmetry"] = abs(row["won1_mean"] - row["won2_mean"])
    row["chips_transferred_rate"] = row["net_asymmetry_sum"] / n
    row["chips_transferred_bb"] = row["net_asymmetry_sum"]
    row["total_wagered_bb"] = acc["contrib1_sum"] + acc["contrib2_sum"]
    row["pot_mean_bb"] = row["pot_mean"]
    rows.append(row)

pair_features = pd.DataFrame(rows)
del rows, pair_accum
gc.collect()
print(f"pair_features: {pair_features.shape}")

# Add player IDs and player profile features
all_player_map = pd.concat([
    dev_labels[["pair_id","player_1","player_2"]].rename(columns={"player_1":"p1","player_2":"p2"}),
    eval_pairs[["pair_id","player_1","player_2"]].rename(columns={"player_1":"p1","player_2":"p2"})
]).drop_duplicates(subset=["pair_id"])
pair_features = pair_features.merge(all_player_map, on="pair_id", how="left")

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
        random_state=SEED, n_jobs=-1, verbose=-1,
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

# Build evidence from top-5 heaps
print("\n--- Building evidence from top-5 heaps ---")
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
    heap = pair_top5.get(pid, [])
    sorted_hands = sorted(heap, key=lambda x: -x[0])
    top5 = [h for s, h in sorted_hands[:5]]
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
}
with open(OUT / "cv_results.json", "w") as f:
    json.dump(cv_results, f, indent=2, default=str)

print()
print("="*80)
print(f"DONE in {time.time()-t0:.1f}s")
print(f"Submission: {OUT / 'submission.csv'}")
print(f"OOF AP: {overall_ap:.4f}")
print("="*80)
