# Recommendations for Reaching 0.93827

The user's current best score is **0.83065**, achieved via a sophisticated "PU-Aware Evidence Ranker" pipeline. To break the plateau and reach the target **0.93827** (current #1, held by "Pardheev Krishna"), the following concrete improvements are recommended.

## TL;DR

1. **Add action-level features** (fold-to-partner rate with hand strength, HU checkdowns, sandwich raises) — this is the gap between my baseline (0.07) and the user's notebook (0.83)
2. **Use LGBMRanker with lambdarank** for top-5 evidence selection instead of hand-crafted suspicion scores
3. **Train per-family models with PU learning** (the user's notebook already does this — keep it)
4. **Add graph-based features**: player interaction networks, centrality, community detection
5. **Ensemble across multiple seeds** and multiple feature subsets
6. **Use a Kaggle notebook** to run the full pipeline on Kaggle's 16GB RAM infrastructure

## Detailed Plan

### Phase 1: Action-Level Features (highest leverage)

The current notebook has action-level features. To push beyond 0.83, we need MORE action-level features:

```python
# Key action-level features per (pair, hand)
- p1_fold_to_p2_count           # p1 folds to p2's bet/raise
- p2_fold_to_p1_count           # symmetric
- p1_fold_with_strong_to_p2     # p1 folds despite strong hand to p2
- p2_fold_with_strong_to_p1     # symmetric
- hu_checkdowns                  # both check down heads-up (passive)
- sandwich_pf                    # both raise preflop with 3rd player involved
- third_folds_after_pq_raises   # 3rd player folds after p&q both raise preflop
- preflop_strength_p1           # Chen-style score for p1's hole cards
- preflop_strength_p2           # symmetric
- preflop_strength_diff         # |p1_strength - p2_strength| (asymmetric skill)
- p1_vpip / p2_vpip             # voluntary money in pot
- p1_pfr / p2_pfr               # preflop raise rate
- p1_aggression / p2_aggression # bet/raise/all_in count
- p1_passivity_vs_p2            # folds + checks vs p2
- p2_passivity_vs_p1            # symmetric
```

Aggregate per pair:
```python
# Per-pair aggregates
- mean / sum / max / std of each per-hand feature
- rates (per hand and per action)
- asymmetries (p1 - p2 differences)
- temporal patterns (cumulative net over time)
```

### Phase 2: Better Evidence Ranking (20% weight)

Currently using hand-crafted suspicion score:
```python
suspicion = (
    asymmetric * 1.5 +
    passive * 2.0 +
    abs_net_diff * 0.5 +
    either_folded * 0.5 +
    abs_transfer * 0.3
)
```

**Better approach**: Train a LGBMRanker with `lambdarank` objective on per-hand labeled data:
- For each positive pair, the 5 gold evidence hands are positives (rank 1)
- Other hands in the same pair are negatives (rank 0)
- Train on all positive pairs (372) × 5-100 hands each = ~50K examples
- Features: per-(pair, hand) action features + seat features + hand features

The user's notebook already has this (`LGBMRanker` with `ap5_lambdas`). Make sure it's properly tuned.

### Phase 3: Per-Family Models with PU Learning (10% weight)

Train 3 separate LightGBM models:
- `directed_transfer` binary: 148 positives vs 1488 negatives + unknowns
- `soft_play` binary: 132 positives vs 1488 negatives + unknowns
- `coordinated_isolation` binary: 92 positives vs 1488 negatives + unknowns

**PU learning**: Treat the 1488 "none" pairs as a mix of true negatives and unknown positives. Use:
- 2-step PU learning: train on positives + unlabeled, then re-weight
- Or: bagging with random subsets of "none" pairs

The user's notebook already does this. The key tuning parameter is `pu_unknown_weight` (currently 0.04).

### Phase 4: Graph-Based Features (new signal)

Build a player interaction graph:
- Nodes: 12K players
- Edges: weighted by # of shared hands (or # of suspicious patterns)
- Per-player features: degree, betweenness, community membership
- Per-pair features: common neighbors, Jaccard similarity, edge betweenness

This could capture collusive networks (multiple pairs of colluders working together).

### Phase 5: Ensemble Variants

Train multiple models with:
- Different random seeds (already doing this)
- Different feature subsets (e.g., 80% of features per model)
- Different model types: LightGBM, XGBoost, CatBoost
- Different loss objectives: binary, lambdarank, pairwise

Final risk_score = max of family probabilities (or learned blend).

### Phase 6: Temporal Features

Hands have `started_at` timestamps. Add temporal features:
- Time between shared hands (consistency of collusion pattern)
- Cumulative net chip flow over time (slope, variance)
- Recency-weighted averages (more recent = more predictive)
- "Burst" patterns (many suspicious actions in short time)

### Phase 7: Better Threshold Tuning

Currently the user's notebook has `0.4` threshold for behavior prediction. Calibrate this:
- For each family, find the threshold that maximizes F1 on dev set
- Use isotonic regression to calibrate probabilities
- Or use Youden's J statistic (TPR - FPR)

## Practical Execution Plan (3 days left, ~5 submissions/day)

### Day 1 (2026-09-17, today — 3 submissions left)
1. **Build Kaggle notebook** with action-level features (use `kaggle_notebook_v1.py` as starting point, add action processing)
2. Run on Kaggle, submit
3. If >0.83: tune thresholds, resubmit
4. If ≤0.83: debug and improve

### Day 2 (2026-09-18, 5 submissions)
1. **Add LGBMRanker for evidence selection**
2. **Train per-family models with PU learning**
3. **Try CatBoost / XGBoost ensemble**
4. Submit each as separate submission

### Day 3 (2026-09-19, 5 submissions)
1. **Add graph-based features** (player interaction network)
2. **Add temporal features**
3. **Final ensemble** of all best models
4. Submit best variant

### Day 4 (2026-09-20, last day before deadline at 22:00 UTC)
1. **Final submission** with best ensemble
2. **Save 1 submission** for last-minute fixes if something breaks

## What NOT to do

❌ Don't retrain the entire pipeline on every submission (too slow)
❌ Don't add features blindly (test on CV first)
❌ Don't trust local CV AP as the only metric (it can be overfit)
❌ Don't submit until you've validated locally first

## Key Insight from Top Score

The #1 score (0.93827) is held by **Pardheev Krishna** (team_id: 16842986). They achieved this on 2026-09-15. Their approach likely:
- Has very strong evidence_map5 (their top-5 hands match gold)
- Has very strong pair_ap (their ranking is well-calibrated)
- Likely uses features the user's notebook doesn't have

The 0.108 gap from 0.83 to 0.94 is significant but achievable with the right approach. The biggest leverage is in `pair_ap` (70% weight), which means improving the model's discrimination between suspicious and benign pairs.

## Final Notes

- Always validate changes on local CV before submitting
- Use early stopping to prevent overfitting
- Bag with multiple seeds for stability
- Don't get stuck on a single approach — try diverse methods

Good luck!
