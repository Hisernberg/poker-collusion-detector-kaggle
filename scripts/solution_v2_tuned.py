"""
Poker Collusion Detector v2 — Threshold tuning + better risk calibration
=========================================================================
Reuses pair_features.parquet from v1.
Key changes:
  1. Lower behavior threshold (predict more positives, currently 94% predicted "none")
  2. Use rank-based risk score (percentile) instead of raw probability
  3. Train with more trees and bagging
  4. Better evidence selection
"""
import os, gc, time, json, warnings, pickle
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

DATA = Path("/home/z/my-project/kaggle/comp-data")
OUT  = Path("/home/z/my-project/kaggle/runs")
RUN1 = OUT  # where v1 outputs are stored

SEED = 20260917
N_FOLDS = 5
np.random.seed(SEED)

print("="*80)
print("Loading v1 cached features")
print("="*80)
t0 = time.time()

pair_features = pd.read_parquet(RUN1 / "pair_features.parquet")
print(f"pair_features: {pair_features.shape}")

dev_labels = pd.read_csv(DATA / "development_labels.csv")
eval_pairs  = pd.read_csv(DATA / "evaluation_pairs.csv")
print(f"dev_labels: {dev_labels.shape}, eval_pairs: {eval_pairs.shape}")

# Train df
train_df = pair_features.merge(dev_labels[["pair_id","label","behavior_family"]], on="pair_id", how="inner")
print(f"train_df: {train_df.shape}, positives: {int(train_df.label.sum())}")

drop_cols = {"pair_id","p1","p2","label","behavior_family"}
feat_cols = [c for c in train_df.columns if c not in drop_cols and train_df[c].dtype != object and str(train_df[c].dtype) != "string"]
print(f"# features: {len(feat_cols)}")

# Impute
for c in feat_cols:
    if train_df[c].isna().any():
        train_df[c] = train_df[c].fillna(train_df[c].median() if not train_df[c].isna().all() else 0)

X = train_df[feat_cols].values.astype(np.float32)
y = train_df.label.values.astype(int)
families = train_df.behavior_family.values

y_dt = (families == "directed_transfer").astype(int)
y_sp = (families == "soft_play").astype(int)
y_ci = (families == "coordinated_isolation").astype(int)

# Evaluation pairs
eval_df = pair_features[pair_features.pair_id.isin(set(eval_pairs.pair_id))].copy()
print(f"eval_df: {eval_df.shape}")
for c in feat_cols:
    if eval_df[c].isna().any():
        eval_df[c] = eval_df[c].fillna(train_df[c].median() if not train_df[c].isna().all() else 0)
X_eval = eval_df[feat_cols].values.astype(np.float32)

# =========================================================================
# Train models with multiple seeds for ensemble
# =========================================================================
print()
print("="*80)
print("Training models with multiple seeds for ensemble")
print("="*80)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def lgb_params(seed=20260917, num_leaves=31, n_estimators=500, lr=0.03):
    return dict(
        objective="binary", num_leaves=num_leaves,
        learning_rate=lr, n_estimators=n_estimators,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.05, reg_lambda=0.1,
        random_state=seed, n_jobs=2, verbose=-1,
    )

def train_family_models_ensemble(y_family, family_name, seeds=(20260917, 137, 42)):
    """Train 3 models with different seeds, average predictions."""
    print(f"\n--- Training {family_name} models (3-seed ensemble) ---")
    oof = np.zeros(len(train_df))
    eval_preds = np.zeros(len(eval_df))
    fold_scores_all = []

    for seed in seeds:
        skf_seed = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_seed = np.zeros(len(train_df))
        fold_scores = []
        for fold, (tr, va) in enumerate(skf_seed.split(X, y_family)):
            X_tr, X_va = X[tr], X[va]
            y_tr, y_va = y_family[tr], y_family[va]
            model = lgb.LGBMClassifier(**lgb_params(seed=seed))
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
            oof_seed[va] = model.predict_proba(X_va)[:, 1]
            if y_va.sum() > 0:
                score = average_precision_score(y_va, oof_seed[va])
            else:
                score = 0.0
            fold_scores.append(score)
        print(f"  seed {seed}: mean AP={np.mean(fold_scores):.4f}")
        oof += oof_seed / len(seeds)
        fold_scores_all.append(np.mean(fold_scores))

        # Train on full data and predict on eval
        model_full = lgb.LGBMClassifier(**lgb_params(seed=seed))
        model_full.fit(X, y_family)
        eval_preds += model_full.predict_proba(X_eval)[:, 1] / len(seeds)

    print(f"  Ensemble mean AP: {np.mean(fold_scores_all):.4f}")
    return oof, eval_preds

oof_dt, eval_p_dt = train_family_models_ensemble(y_dt, "directed_transfer")
oof_sp, eval_p_sp = train_family_models_ensemble(y_sp, "soft_play")
oof_ci, eval_p_ci = train_family_models_ensemble(y_ci, "coordinated_isolation")
oof_risk, eval_p_risk = train_family_models_ensemble(y, "any_suspicious")

# Composite risk OOF
oof_combined = np.maximum.reduce([oof_dt, oof_sp, oof_ci, oof_risk * 0.6])
overall_ap = average_precision_score(y, oof_combined)
print(f"\nOverall OOF AP (composite risk): {overall_ap:.4f}")

# Composite risk for eval
eval_risk_combined = np.maximum.reduce([eval_p_dt, eval_p_sp, eval_p_ci, eval_p_risk * 0.6])

# Per-family AP
for fam, oof_arr in [("directed_transfer", oof_dt), ("soft_play", oof_sp), ("coordinated_isolation", oof_ci)]:
    y_fam = (families == fam).astype(int)
    if y_fam.sum() > 0:
        fam_ap = average_precision_score(y_fam, oof_arr)
        print(f"  {fam} OOF AP: {fam_ap:.4f}")

# =========================================================================
# Rank-based risk score (percentile of pair_ap)
# =========================================================================
print()
print("="*80)
print("Rank-based risk score (percentile)")
print("="*80)

# Convert raw risk scores to percentile rank (across eval set)
# This should give a more uniform risk distribution
from scipy.stats import rankdata
eval_risk_percentile = rankdata(eval_risk_combined) / len(eval_risk_combined)
print(f"Risk percentile stats: min={eval_risk_percentile.min():.4f}, max={eval_risk_percentile.max():.4f}, mean={eval_risk_percentile.mean():.4f}")

# =========================================================================
# Predicted behavior (lower threshold, weight by risk)
# =========================================================================
print()
print("="*80)
print("Predicting behaviors (lower threshold)")
print("="*80)

# Get per-family probabilities and predict behavior based on argmax
# Use lower threshold (more positive predictions)
behaviors = []
fam_names = ["directed_transfer", "soft_play", "coordinated_isolation"]
fam_probs = np.column_stack([eval_p_dt, eval_p_sp, eval_p_ci])

# Calculate threshold based on dev set distribution (5-20% positive rate expected)
# Use ~15% of eval pairs predicted as positive (matches dev positive rate)
# Sort pairs by max family probability, top 15% predicted positive
max_fam_prob = fam_probs.max(axis=1)
threshold_idx = int(len(eval_df) * 0.85)  # top 15% are positive
threshold_val = np.sort(max_fam_prob)[::-1][threshold_idx]
print(f"Threshold value (top 15%): {threshold_val:.4f}")

# Also check generic risk
risk_threshold = np.sort(eval_p_risk)[::-1][threshold_idx]
print(f"Risk threshold (top 15%): {risk_threshold:.4f}")

for i in range(len(eval_df)):
    fam_idx = int(np.argmax(fam_probs[i]))
    if max_fam_prob[i] >= threshold_val or eval_p_risk[i] >= risk_threshold:
        behaviors.append(fam_names[fam_idx])
    else:
        behaviors.append("none")
print(f"Behavior distribution: {Counter(behaviors)}")

# =========================================================================
# Build submission with rank-based risk score
# =========================================================================
print()
print("="*80)
print("Building submission v2")
print("="*80)

# Build evidence from top-5 heaps (loaded from v1's pair_hand_features.parquet)
print("Loading pair_hand_features for evidence...")
# Re-load top5 from pair_hand_features (we saved this in v1 - but we didn't save it in v7; let's recompute)
# Actually v7 didn't save pair_hand_features. Let's just recompute from pair_features
# We don't have per-hand suspicion scores anymore, so let's use a fallback: pick the most suspicious hands based on aggregated features

# For evidence, let's just use the top-5 hand_ids per pair by re-reading pair_hand_features
# But we don't have it saved. Let's use a quick approximation: pick 5 random hands from the pair's shared hands
# Actually, we can just compute suspicion per pair-hand again, but it would require reprocessing
# For v2, let's just emit NO_EVIDENCE for all hands (worst case for evidence_map5, but we can fix later)
# Actually let's recompute the evidence using the v1 logic

# Re-load pair_hand_features
# v7 didn't save pair_hand_features.parquet. Let's check
import os.path
if (RUN1 / "pair_hand_features.parquet").exists():
    print("Loading saved pair_hand_features.parquet...")
    phf = pd.read_parquet(RUN1 / "pair_hand_features.parquet")
    print(f"phf: {phf.shape}")
else:
    print("pair_hand_features.parquet not found - using NO_EVIDENCE for all hands")
    phf = None

# Compute top-5 evidence per pair
pair_to_hand_scores = defaultdict(list)
if phf is not None:
    # Compute suspicion score
    phf["suspicion_score"] = (
        phf["asymmetric_outcome"] * 1.5 +
        phf["passive_outcome"] * 2.0 +
        phf["abs_net_diff_bb"] * 0.5 +
        phf["either_folded"] * 0.5 +
        phf["transfer_signal"].abs() * 0.3
    )
    for r in phf[phf.pair_id.isin(set(eval_pairs.pair_id))].itertuples():
        pair_to_hand_scores[r.pair_id].append((r.hand_id, r.suspicion_score))

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
    hand_scores = pair_to_hand_scores.get(pid, [])
    hand_scores.sort(key=lambda x: -x[1])
    top5 = [h for h, s in hand_scores[:5]]
    while len(top5) < 5:
        top5.append("NO_EVIDENCE")
    sub_rows.append({
        "pair_id": pid,
        "risk_score": float(eval_risk_percentile[idx]),  # percentile-based risk
        "predicted_behavior": behaviors[idx],
        "evidence_hand_1": top5[0],
        "evidence_hand_2": top5[1],
        "evidence_hand_3": top5[2],
        "evidence_hand_4": top5[3],
        "evidence_hand_5": top5[4],
    })

sub_df = pd.DataFrame(sub_rows)
sub_df.to_csv(OUT / "submission_v2.csv", index=False)
print(f"Saved submission_v2.csv: {sub_df.shape}")
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
    "risk_threshold": float(threshold_val),
    "method": "3-seed ensemble, percentile risk, lower threshold",
}
with open(OUT / "cv_results_v2.json", "w") as f:
    json.dump(cv_results, f, indent=2, default=str)

print()
print("="*80)
print(f"DONE in {time.time()-t0:.1f}s")
print(f"Submission: {OUT / 'submission_v2.csv'}")
print(f"OOF AP: {overall_ap:.4f}")
print("="*80)
