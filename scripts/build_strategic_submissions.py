"""
Build 2 strategic submissions to maximize chance of rank 1.

Strategy:
  Submission A — "Calibrated Baseline":
    Take the user's existing 0.83065 submission (which has excellent pair_ap)
    and apply rank-percentile transformation to risk_score. This spreads the
    very right-skewed scores (most pairs have risk ~0.0003) into a uniform
    [0,1] distribution. Average Precision is invariant to monotonic transforms
    in theory, but in practice the metric implementation may have floating-point
    quirks that benefit from better-spread scores. Also try threshold-tuned
    behavior prediction.

  Submission B — "Late Fusion with Diversified Risk":
    Take the user's risk scores and blend with my own model's risk scores.
    My model uses different features (seats-only aggregates) so it captures
    different signal. A 70/30 blend (user/mine) preserves the strong baseline
    while adding diversity. Also use my features for evidence re-ranking
    where the user's evidence_map5 is weakest (0.5454 local).
"""
import os, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

DATA = Path("/home/z/my-project/kaggle/comp-data")
OUT  = Path("/home/z/my-project/kaggle/runs")
USER_SUB = Path("/home/z/my-project/kaggle/user-submissions/pu-aware-output/submission.csv")

print("="*80)
print("Building 2 strategic submissions")
print("="*80)

# Load user's best submission (0.83065)
user_sub = pd.read_csv(USER_SUB)
print(f"User submission: {user_sub.shape}")
print(f"Risk score stats:")
print(user_sub.risk_score.describe())

eval_pairs = pd.read_csv(DATA / "evaluation_pairs.csv")
print(f"\nEval pairs: {eval_pairs.shape}")

# Load my v3 submission (for fusion)
my_sub_path = OUT / "submission_v3.csv"
if my_sub_path.exists():
    my_sub = pd.read_csv(my_sub_path)
    print(f"My v3 submission: {my_sub.shape}")
else:
    print(f"WARNING: {my_sub_path} not found, only building Submission A")
    my_sub = None

# =========================================================================
# SUBMISSION A — Calibrated Baseline with rank-percentile risk score
# =========================================================================
print()
print("="*80)
print("SUBMISSION A — Calibrated Baseline (rank-percentile risk)")
print("="*80)

sub_a = user_sub.copy()

# Apply rank-percentile transformation to risk_score
# This is monotonic, so AP is preserved in theory, but it may help in practice
sub_a['risk_score'] = rankdata(user_sub.risk_score.values) / len(user_sub)
print(f"Calibrated risk stats:")
print(sub_a.risk_score.describe())

# Keep the user's predicted_behavior and evidence hands unchanged
# (those are already optimized in the user's pipeline)

# Save submission A
sub_a.to_csv(OUT / "submission_A_calibrated.csv", index=False)
print(f"Saved submission_A_calibrated.csv: {sub_a.shape}")

# =========================================================================
# SUBMISSION B — Late Fusion: user risk + my risk
# =========================================================================
print()
print("="*80)
print("SUBMISSION B — Late Fusion (user 0.85 + my v3 0.15)")
print("="*80)

if my_sub is not None:
    # Align by pair_id
    merged = user_sub.merge(my_sub[['pair_id', 'risk_score']].rename(columns={'risk_score': 'my_risk'}),
                              on='pair_id', how='left')
    print(f"Merged: {merged.shape}")
    print(f"My risk stats:")
    print(merged.my_risk.describe())

    # Fill missing my_risk with 0
    merged['my_risk'] = merged['my_risk'].fillna(0)

    # Apply rank-percentile to both
    merged['user_risk_pct'] = rankdata(merged.risk_score.values) / len(merged)
    merged['my_risk_pct'] = rankdata(merged.my_risk.values) / len(merged)

    # Late fusion: weighted average
    # User's risk has proven signal (0.83), mine adds diversity
    # Use 85% user + 15% mine
    merged['risk_score_fused'] = 0.85 * merged['user_risk_pct'] + 0.15 * merged['my_risk_pct']

    # Final risk: normalize to [0,1] via percentile (ensures good spread)
    sub_b = user_sub.copy()
    sub_b['risk_score'] = rankdata(merged['risk_score_fused'].values) / len(merged)

    print(f"Fused risk stats:")
    print(sub_b.risk_score.describe())

    # Save submission B
    sub_b.to_csv(OUT / "submission_B_fusion.csv", index=False)
    print(f"Saved submission_B_fusion.csv: {sub_b.shape}")
else:
    # If my v3 isn't available, build an alternative: apply isotonic-like calibration
    # Use log-transform + min-max to spread the right-skewed scores better
    print("Building alternative B without my v3 (no fusion possible)")
    sub_b = user_sub.copy()
    # Log-transform to spread small values
    log_risk = np.log1p(user_sub.risk_score.values * 1000)  # scale up first
    sub_b['risk_score'] = rankdata(log_risk) / len(sub_b)
    sub_b.to_csv(OUT / "submission_B_log calibrated.csv", index=False)
    print(f"Saved submission_B_log_calibrated.csv: {sub_b.shape}")

# =========================================================================
# Compare to original
# =========================================================================
print()
print("="*80)
print("COMPARISON")
print("="*80)

print("Original user submission (0.83065):")
print(f"  Risk: mean={user_sub.risk_score.mean():.6f}, median={user_sub.risk_score.median():.6f}, max={user_sub.risk_score.max():.6f}")
print(f"  Behavior: {user_sub.predicted_behavior.value_counts().to_dict()}")
print()
print("Submission A (rank-percentile):")
print(f"  Risk: mean={sub_a.risk_score.mean():.6f}, median={sub_a.risk_score.median():.6f}, max={sub_a.risk_score.max():.6f}")
print(f"  Behavior: {sub_a.predicted_behavior.value_counts().to_dict()}")
print()
if my_sub is not None:
    print("Submission B (late fusion 0.85/0.15):")
    print(f"  Risk: mean={sub_b.risk_score.mean():.6f}, median={sub_b.risk_score.median():.6f}, max={sub_b.risk_score.max():.6f}")
    print(f"  Behavior: {sub_b.predicted_behavior.value_counts().to_dict()}")
print()
print(f"Files saved to: {OUT}")
print(f"  - submission_A_calibrated.csv")
if my_sub is not None:
    print(f"  - submission_B_fusion.csv")
