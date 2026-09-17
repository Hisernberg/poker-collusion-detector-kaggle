"""
Poker Collusion Detector v3 — Better threshold + reuse v1 evidence
==================================================================
Improvements over v2:
  1. Predict top 20% of pairs as positive (matches dev positive rate)
  2. Among positives, assign behavior by argmax of family probabilities
  3. Reuse evidence hands from v1 submission (which has actual hand_ids)
  4. Use percentile-based risk score (better discrimination)
"""
import os, gc, time, json, warnings, pickle
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

DATA = Path("/home/z/my-project/kaggle/comp-data")
OUT  = Path("/home/z/my-project/kaggle/runs")

SEED = 20260917
N_FOLDS = 5
np.random.seed(SEED)

print("="*80)
print("Loading v1 cached features and submission")
print("="*80)
t0 = time.time()

pair_features = pd.read_parquet(OUT / "pair_features.parquet")
print(f"pair_features: {pair_features.shape}")

# Load v1 submission to get evidence hands
v1_submission = pd.read_csv(OUT / "submission.csv")
print(f"v1_submission: {v1_submission.shape}")
evidence_cols = ["evidence_hand_1","evidence_hand_2","evidence_hand_3","evidence_hand_4","evidence_hand_5"]
v1_evidence = v1_submission.set_index("pair_id")[evidence_cols].to_dict("index")
print(f"v1_evidence: {len(v1_evidence):,} pairs")

dev_labels = pd.read_csv(DATA / "development_labels.csv")
eval_pairs  = pd.read_csv(DATA / "evaluation_pairs.csv")
print(f"dev_labels: {dev_labels.shape}, eval_pairs: {eval_pairs.shape}")

train_df = pair_features.merge(dev_labels[["pair_id","label","behavior_family"]], on="pair_id", how="inner")
print(f"train_df: {train_df.shape}, positives: {int(train_df.label.sum())}")

drop_cols = {"pair_id","p1","p2","label","behavior_family"}
feat_cols = [c for c in train_df.columns if c not in drop_cols and train_df[c].dtype != object and str(train_df[c].dtype) != "string"]
print(f"# features: {len(feat_cols)}")

for c in feat_cols:
    if train_df[c].isna().any():
        train_df[c] = train_df[c].fillna(train_df[c].median() if not train_df[c].isna().all() else 0)

X = train_df[feat_cols].values.astype(np.float32)
y = train_df.label.values.astype(int)
families = train_df.behavior_family.values

y_dt = (families == "directed_transfer").astype(int)
y_sp = (families == "soft_play").astype(int)
y_ci = (families == "coordinated_isolation").astype(int)

eval_df = pair_features[pair_features.pair_id.isin(set(eval_pairs.pair_id))].copy()
for c in feat_cols:
    if eval_df[c].isna().any():
        eval_df[c] = eval_df[c].fillna(train_df[c].median() if not train_df[c].isna().all() else 0)
X_eval = eval_df[feat_cols].values.astype(np.float32)

# =========================================================================
# Train models (3-seed ensemble)
# =========================================================================
print()
print("="*80)
print("Training models with 3-seed ensemble")
print("="*80)

def lgb_params(seed=20260917, num_leaves=31, n_estimators=500, lr=0.03):
    return dict(
        objective="binary", num_leaves=num_leaves,
        learning_rate=lr, n_estimators=n_estimators,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.05, reg_lambda=0.1,
        random_state=seed, n_jobs=2, verbose=-1,
    )

def train_family_models_ensemble(y_family, family_name, seeds=(20260917, 137, 42)):
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

oof_combined = np.maximum.reduce([oof_dt, oof_sp, oof_ci, oof_risk * 0.6])
overall_ap = average_precision_score(y, oof_combined)
print(f"\nOverall OOF AP (composite risk): {overall_ap:.4f}")

# Composite risk for eval (max across all family + risk models)
eval_risk_combined = np.maximum.reduce([eval_p_dt, eval_p_sp, eval_p_ci, eval_p_risk * 0.6])

# Per-family AP
for fam, oof_arr in [("directed_transfer", oof_dt), ("soft_play", oof_sp), ("coordinated_isolation", oof_ci)]:
    y_fam = (families == fam).astype(int)
    if y_fam.sum() > 0:
        fam_ap = average_precision_score(y_fam, oof_arr)
        print(f"  {fam} OOF AP: {fam_ap:.4f}")

# =========================================================================
# Risk score: rank-based (percentile of composite score across eval set)
# =========================================================================
print()
print("="*80)
print("Computing rank-based risk score")
print("="*80)

# Convert raw scores to percentile ranks
eval_risk_percentile = rankdata(eval_risk_combined) / len(eval_risk_combined)
print(f"Percentile stats: min={eval_risk_percentile.min():.4f}, max={eval_risk_percentile.max():.4f}, mean={eval_risk_percentile.mean():.4f}")

# =========================================================================
# Behavior prediction: top 20% pairs are positive (matches dev distribution)
# Among positives, assign family by argmax of family-specific probabilities
# =========================================================================
print()
print("="*80)
print("Predicting behaviors (top 20% positive)")
print("="*80)

# Distribution from dev set:
# directed_transfer: 148/372 = 39.8% of positives
# soft_play: 132/372 = 35.5% of positives
# coordinated_isolation: 92/372 = 24.7% of positives
# Total positive rate: 20%

# Strategy: take top 20% pairs by max family probability
# Among them, assign family by argmax of family probs, weighted by dev distribution
fam_probs = np.column_stack([eval_p_dt, eval_p_sp, eval_p_ci])
max_fam_prob = fam_probs.max(axis=1)

# Sort indices by max family prob (descending)
sorted_idx = np.argsort(-max_fam_prob)
n_pos = int(len(eval_df) * 0.20)  # top 20% are positive
positive_indices = set(sorted_idx[:n_pos])

behaviors = []
fam_names = ["directed_transfer", "soft_play", "coordinated_isolation"]
# Family prior weights from dev distribution
fam_weights = np.array([0.398, 0.355, 0.247])

# For each positive pair, assign family by argmax of weighted prob
# For non-positive, assign "none"
for i in range(len(eval_df)):
    if i in positive_indices:
        # Weighted argmax (use dev distribution as prior)
        weighted_probs = fam_probs[i] * fam_weights
        fam_idx = int(np.argmax(weighted_probs))
        behaviors.append(fam_names[fam_idx])
    else:
        behaviors.append("none")

print(f"Behavior distribution: {Counter(behaviors)}")

# =========================================================================
# Build submission with rank-based risk + v1 evidence
# =========================================================================
print()
print("="*80)
print("Building submission v3")
print("="*80)

eval_pair_id_to_idx = {pid: i for i, pid in enumerate(eval_df.pair_id.values)}

sub_rows = []
for r in eval_pairs.itertuples():
    pid = r.pair_id
    if pid not in eval_pair_id_to_idx:
        # Pair not in features - shouldn't happen
        sub_rows.append({
            "pair_id": pid, "risk_score": 0.0, "predicted_behavior": "none",
            "evidence_hand_1": "NO_EVIDENCE", "evidence_hand_2": "NO_EVIDENCE",
            "evidence_hand_3": "NO_EVIDENCE", "evidence_hand_4": "NO_EVIDENCE",
            "evidence_hand_5": "NO_EVIDENCE",
        })
        continue
    idx = eval_pair_id_to_idx[pid]
    # Get evidence from v1 submission
    ev = v1_evidence.get(pid, {})
    sub_rows.append({
        "pair_id": pid,
        "risk_score": float(eval_risk_percentile[idx]),
        "predicted_behavior": behaviors[idx],
        "evidence_hand_1": ev.get("evidence_hand_1", "NO_EVIDENCE"),
        "evidence_hand_2": ev.get("evidence_hand_2", "NO_EVIDENCE"),
        "evidence_hand_3": ev.get("evidence_hand_3", "NO_EVIDENCE"),
        "evidence_hand_4": ev.get("evidence_hand_4", "NO_EVIDENCE"),
        "evidence_hand_5": ev.get("evidence_hand_5", "NO_EVIDENCE"),
    })

sub_df = pd.DataFrame(sub_rows)
sub_df.to_csv(OUT / "submission_v3.csv", index=False)
print(f"Saved submission_v3.csv: {sub_df.shape}")
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
    "method": "3-seed ensemble, percentile risk, top-20% positive threshold, dev-distribution-weighted behavior",
}
with open(OUT / "cv_results_v3.json", "w") as f:
    json.dump(cv_results, f, indent=2, default=str)

print()
print("="*80)
print(f"DONE in {time.time()-t0:.1f}s")
print(f"Submission: {OUT / 'submission_v3.csv'}")
print(f"OOF AP: {overall_ap:.4f}")
print("="*80)
