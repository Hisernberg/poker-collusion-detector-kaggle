"""
Poker Collusion Detector v3 — Lean Memory-Efficient Version
============================================================
Memory budget: 4GB RAM. Skip actions.parquet (18.6M rows = too big).
Use only seats.parquet + hands.parquet for per-pair features.

For evidence selection, use seat-level signals (net_chips, contributions, won_share).
This should yield ~0.80 baseline; we'll iterate up from there.
"""
import os, gc, time, json, sys, warnings, pickle
from pathlib import Path
from collections import defaultdict, Counter
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
print("STAGE 0 — Loading small CSVs + building pair indices")
print("="*80)
t0 = time.time()

dev_labels = pd.read_csv(DATA / "development_labels.csv")
dev_evidence = pd.read_csv(DATA / "development_evidence.csv")
eval_pairs  = pd.read_csv(DATA / "evaluation_pairs.csv")
sample_sub  = pd.read_csv(DATA / "sample_submission.csv")
players_df  = pd.read_parquet(DATA / "players.parquet")

print(f"dev_labels: {dev_labels.shape}  ({dev_labels.label.sum()} positives)")
print(f"eval_pairs: {eval_pairs.shape}")

# Build pair_lookup
pair_lookup = {}
pair_to_players = {}
for r in dev_labels.itertuples():
    pair_lookup[frozenset((r.player_1, r.player_2))] = r.pair_id
    pair_to_players[r.pair_id] = (r.player_1, r.player_2)
for r in eval_pairs.itertuples():
    pair_lookup[frozenset((r.player_1, r.player_2))] = r.pair_id
    pair_to_players[r.pair_id] = (r.player_1, r.player_2)
print(f"Total pairs: {len(pair_lookup):,}")

# Players needed
needed_players = set()
for p1, p2 in pair_to_players.values():
    needed_players.add(p1); needed_players.add(p2)
print(f"Players needed: {len(needed_players):,}")

# Gold evidence and family
gold_evidence = defaultdict(set)
for r in dev_evidence.itertuples():
    gold_evidence[r.pair_id].add(r.hand_id)
gold_family = {r.pair_id: r.behavior_family for r in dev_labels.itertuples() if r.label == 1}

print(f"Setup time: {time.time()-t0:.1f}s")

# =========================================================================
# Load seats (filtered to needed players) + hands (full)
# =========================================================================
print()
print("="*80)
print("Loading seats.parquet (filtered) and hands.parquet (full)...")
print("="*80)

needed_players_list = list(needed_players)
t_seats = time.time()
seats = pl.scan_parquet(DATA / "seats.parquet").filter(
    pl.col("player_id").is_in(needed_players_list)
).select([
    "hand_id", "player_id", "starting_stack", "total_contribution",
    "net_chips", "folded", "went_to_showdown", "won_share"
]).collect(streaming=True)
print(f"seats: {seats.shape}, time={time.time()-t_seats:.1f}s, mem={seats.estimated_size('mb'):.1f}MB")

hands = pl.read_parquet(DATA / "hands.parquet", columns=[
    "hand_id", "phase", "small_blind", "big_blind",
    "board_cards", "final_pot", "players_at_showdown"
])
print(f"hands: {hands.shape}, mem={hands.estimated_size('mb'):.1f}MB")

# =========================================================================
# Build pair-hand index: for each hand, which pairs of needed players are present?
# =========================================================================
print()
print("Building pair-hand index...")

# Convert seats to pandas for grouping (we already have it filtered)
seats_pd = seats.to_pandas()
del seats
gc.collect()
print(f"seats_pd: {seats_pd.shape}, mem={seats_pd.memory_usage(deep=True).sum()/1e6:.1f}MB")

# Build hand -> set of players map
hand_players = seats_pd.groupby("hand_id")["player_id"].agg(set).to_dict()
print(f"hand_players: {len(hand_players):,}")

# Build (pair, hand) index
t_idx = time.time()
pair_hands = defaultdict(list)
pair_hand_player = {}
for hand_id, players_in_hand in hand_players.items():
    if len(players_in_hand) < 2:
        continue
    players_list = list(players_in_hand)
    for i in range(len(players_list)):
        for j in range(i+1, len(players_list)):
            key = frozenset((players_list[i], players_list[j]))
            pid = pair_lookup.get(key)
            if pid is not None:
                pair_hands[pid].append(hand_id)
                pair_hand_player[(pid, hand_id)] = (players_list[i], players_list[j])
print(f"pair-hand index: {len(pair_hands):,} pairs, time={time.time()-t_idx:.1f}s")
total_pair_hand = sum(len(v) for v in pair_hands.values())
print(f"Total (pair, hand) records: {total_pair_hand:,}")

# Free hand_players (no longer needed)
del hand_players
gc.collect()

# =========================================================================
# Convert hands to indexed for fast lookup
# =========================================================================
hands_pd = hands.to_pandas()
del hands
gc.collect()
hands_indexed = hands_pd.set_index("hand_id")
del hands_pd
gc.collect()
print(f"hands_indexed: {len(hands_indexed):,}")

# =========================================================================
# Build seat_lookup dict: (hand_id, player_id) -> seat row
# =========================================================================
print("Building seat lookup...")
t_sl = time.time()
# We can use a dict keyed by (hand_id, player_id)
seat_lookup = {}
for r in seats_pd.itertuples():
    seat_lookup[(r.hand_id, r.player_id)] = (
        int(r.starting_stack), int(r.total_contribution), int(r.net_chips),
        bool(r.folded), bool(r.went_to_showdown), float(r.won_share) if r.won_share is not None else 0.0
    )
print(f"seat_lookup: {len(seat_lookup):,} entries, time={time.time()-t_sl:.1f}s, mem~{sys.getsizeof(seat_lookup)/1e6:.1f}MB")

# Free seats_pd
del seats_pd
gc.collect()

# =========================================================================
# Compute per-(pair, hand) features
# =========================================================================
print()
print("="*80)
print("Computing per-(pair, hand) features...")
print("="*80)
t_feat = time.time()
pair_hand_features = []

# Iterate over all (pair, hand) records
for pid, hand_list in pair_hands.items():
    p1, p2 = pair_to_players[pid]
    for hand_id in hand_list:
        key = (pid, hand_id)
        if key not in pair_hand_player: continue
        hp1, hp2 = pair_hand_player[key]

        # Look up seats
        s1 = seat_lookup.get((hand_id, hp1))
        s2 = seat_lookup.get((hand_id, hp2))
        if s1 is None or s2 is None: continue

        # Look up hand info
        if hand_id not in hands_indexed: continue
        hand_row = hands_indexed.loc[hand_id]
        big_blind = max(int(hand_row.big_blind), 1)
        phase = hand_row.phase
        final_pot = int(hand_row.final_pot)

        # Seat data
        s1_stack, s1_contrib, s1_net, s1_folded, s1_showdown, s1_won = s1
        s2_stack, s2_contrib, s2_net, s2_folded, s2_showdown, s2_won = s2

        # Per-pair-per-hand features (only from seats+hands, no actions)
        f = {
            "pair_id": pid,
            "hand_id": hand_id,
            "phase": phase,
            "n_shared": len(hand_list),
            "net1_bb": float(s1_net) / big_blind,
            "net2_bb": float(s2_net) / big_blind,
            "contrib1_bb": float(s1_contrib) / big_blind,
            "contrib2_bb": float(s2_contrib) / big_blind,
            "stack1_bb": float(s1_stack) / big_blind,
            "stack2_bb": float(s2_stack) / big_blind,
            "pot_bb": float(final_pot) / big_blind,
            "abs_net_diff_bb": abs(float(s1_net) - float(s2_net)) / big_blind,
            "won_share1": s1_won,
            "won_share2": s2_won,
            "both_showdown": int(s1_showdown and s2_showdown),
            "either_folded": int(s1_folded or s2_folded),
            "p1_folded": int(s1_folded),
            "p2_folded": int(s2_folded),
            "p1_won": int(s1_net > 0),
            "p2_won": int(s2_net > 0),
            "p1_won_more": int(s1_net > s2_net),
            "p2_won_more": int(s2_net > s1_net),
            # Strong transfer signal: one wins big, other loses big
            "transfer_signal": float(s1_net + s2_net) / big_blind,  # negative = transfer to outsiders, positive = transfer from outsiders
            "asymmetric_outcome": int(abs(s1_net - s2_net) > big_blind * 5 and (s1_net > 0) != (s2_net > 0)),
            # Soft play signal: both check down (no showdown but both see)
            "passive_outcome": int(s1_showdown and s2_showdown and final_pot < big_blind * 4),
        }

        pair_hand_features.append(f)

print(f"Total: {len(pair_hand_features):,} pair-hand features in {time.time()-t_feat:.1f}s")

phf = pd.DataFrame(pair_hand_features)
del pair_hand_features
gc.collect()
print(f"phf: {phf.shape}, mem={phf.memory_usage(deep=True).sum()/1e6:.1f}MB")
phf.to_parquet(OUT / "pair_hand_features.parquet")
print("Saved pair_hand_features.parquet")

# Free seat_lookup (no longer needed for pair-hand features)
# But we need it for evidence selection later — actually we can re-derive from phf
# Let's keep it for now
print(f"Current mem usage: {sum(v for k,v in __import__('psutil').Process(os.getpid()).memory_info()._asdict().items() if k in ['rss'])//1024//1024} MB" if 'psutil' in sys.modules else "")

# =========================================================================
# STAGE A2 — Aggregate to pair-level features
# =========================================================================
print()
print("="*80)
print("STAGE A2 — Aggregating pair-level features")
print("="*80)

# Aggregate per-pair
agg_dict = {
    "n_shared": ["first"],
    "net1_bb": ["sum", "mean", "max", "min", "std"],
    "net2_bb": ["sum", "mean", "max", "min", "std"],
    "contrib1_bb": ["sum", "mean"],
    "contrib2_bb": ["sum", "mean"],
    "stack1_bb": ["mean"],
    "stack2_bb": ["mean"],
    "pot_bb": ["mean", "max", "sum"],
    "abs_net_diff_bb": ["mean", "max", "sum", "std"],
    "won_share1": ["sum", "mean"],
    "won_share2": ["sum", "mean"],
    "both_showdown": ["sum", "mean"],
    "either_folded": ["sum", "mean"],
    "p1_folded": ["sum", "mean"],
    "p2_folded": ["sum", "mean"],
    "p1_won": ["sum", "mean"],
    "p2_won": ["sum", "mean"],
    "p1_won_more": ["sum", "mean"],
    "p2_won_more": ["sum", "mean"],
    "transfer_signal": ["sum", "mean", "min"],
    "asymmetric_outcome": ["sum", "mean"],
    "passive_outcome": ["sum", "mean"],
}

pair_features = phf.groupby("pair_id").agg(agg_dict)
pair_features.columns = ["_".join(c) if isinstance(c, tuple) else c for c in pair_features.columns]
pair_features = pair_features.reset_index()
print(f"pair_features (raw): {pair_features.shape}")

# Derived features
pair_features["n_hands"] = pair_features["n_shared_first"]
pair_features["net_asymmetry_sum"] = (pair_features["net1_bb_sum"] - pair_features["net2_bb_sum"]).abs()
pair_features["net_asymmetry_mean"] = (pair_features["net1_bb_mean"] - pair_features["net2_bb_mean"]).abs()
pair_features["chips_transferred_bb"] = pair_features["net_asymmetry_sum"]
pair_features["contribution_asymmetry"] = (pair_features["contrib1_bb_sum"] - pair_features["contrib2_bb_sum"]).abs()
pair_features["chips_transferred_rate"] = pair_features["net_asymmetry_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["fold_rate_p1"] = pair_features["p1_folded_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["fold_rate_p2"] = pair_features["p2_folded_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["fold_asymmetry"] = (pair_features["p1_folded_sum"] - pair_features["p2_folded_sum"]).abs()
pair_features["both_showdown_rate"] = pair_features["both_showdown_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["passive_rate"] = pair_features["passive_outcome_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["asymmetric_outcome_rate"] = pair_features["asymmetric_outcome_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["win_rate_p1"] = pair_features["p1_won_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["win_rate_p2"] = pair_features["p2_won_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["win_asymmetry"] = (pair_features["win_rate_p1"] - pair_features["win_rate_p2"]).abs()
pair_features["p1_dominates"] = pair_features["p1_won_more_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["p2_dominates"] = pair_features["p2_won_more_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["dominance_asymmetry"] = (pair_features["p1_dominates"] - pair_features["p2_dominates"]).abs()
pair_features["won_share_asymmetry"] = (pair_features["won_share1_mean"] - pair_features["won_share2_mean"]).abs()
pair_features["pot_mean_bb"] = pair_features["pot_bb_mean"]
pair_features["total_wagered_bb"] = pair_features["contrib1_bb_sum"] + pair_features["contrib2_bb_sum"]

# Add player profile features
all_player_map = pd.concat([
    dev_labels[["pair_id","player_1","player_2"]].rename(columns={"player_1":"p1","player_2":"p2"}),
    eval_pairs[["pair_id","player_1","player_2"]].rename(columns={"player_1":"p1","player_2":"p2"})
]).drop_duplicates(subset=["pair_id"])
pair_features = pair_features.merge(all_player_map, on="pair_id", how="left")

# Player profile features (encode categoricals)
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

# Save features
pair_features.to_parquet(OUT / "pair_features.parquet")
print(f"Saved pair_features: {pair_features.shape}")

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
        objective="binary",
        num_leaves=num_leaves,
        learning_rate=lr,
        n_estimators=n_estimators,
        min_child_samples=8,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.05,
        reg_lambda=0.1,
        random_state=SEED,
        n_jobs=2,
        verbose=-1,
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

# Composite risk OOF
oof_combined = np.maximum.reduce([oof_dt, oof_sp, oof_ci, oof_risk * 0.6])
overall_ap = average_precision_score(y, oof_combined)
print(f"\nOverall OOF AP (composite risk): {overall_ap:.4f}")

# Per-family AP
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

# Save models
with open(OUT / "models.pkl", "wb") as f:
    pickle.dump({"dt":final_dt, "sp":final_sp, "ci":final_ci, "risk":final_risk, "feat_cols":feat_cols}, f)
print("Saved models")

# =========================================================================
# STAGE C — Predict on evaluation pairs + evidence selection
# =========================================================================
print()
print("="*80)
print("STAGE C — Predict on evaluation pairs")
print("="*80)

eval_df = pair_features[pair_features.pair_id.isin(set(eval_pairs.pair_id))].copy()
print(f"Eval pairs in features: {eval_df.shape}")

for c in feat_cols:
    if eval_df[c].isna().any():
        eval_df[c] = eval_df[c].fillna(train_df[c].median() if not train_df[c].isna().all() else 0)

X_eval = eval_df[feat_cols].values.astype(np.float32)

p_dt = final_dt.predict_proba(X_eval)[:, 1]
p_sp = final_sp.predict_proba(X_eval)[:, 1]
p_ci = final_ci.predict_proba(X_eval)[:, 1]
p_risk = final_risk.predict_proba(X_eval)[:, 1]

risk_combined = np.maximum.reduce([p_dt, p_sp, p_ci, p_risk * 0.6])

# Predicted behavior (argmax across families, with threshold)
behaviors = []
fam_names = ["directed_transfer", "soft_play", "coordinated_isolation"]
for i in range(len(eval_df)):
    probs = [p_dt[i], p_sp[i], p_ci[i]]
    fam_idx = int(np.argmax(probs))
    if max(probs) < 0.4 and p_risk[i] < 0.4:
        behaviors.append("none")
    else:
        behaviors.append(fam_names[fam_idx])
print(f"Predicted behavior distribution: {Counter(behaviors)}")

# Evidence selection using pair_hand_features
print("\n--- Computing per-hand evidence scores ---")

# Per-hand suspicion score
phf_for_eval = phf[phf.pair_id.isin(set(eval_pairs.pair_id))].copy()

phf_for_eval["suspicion_score"] = (
    phf_for_eval["asymmetric_outcome"] * 1.5 +
    phf_for_eval["passive_outcome"] * 2.0 +
    phf_for_eval["abs_net_diff_bb"] * 0.5 +
    phf_for_eval["either_folded"] * 0.5 +
    np.abs(phf_for_eval["transfer_signal"]) * 0.3
).values

pair_to_hand_scores = defaultdict(list)
for r in phf_for_eval.itertuples():
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
print(sub_df.head(10))
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
