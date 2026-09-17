# Poker Collusion Detector — Kaggle Competition

**Competition**: [Detect Suspicious Value Transfers in Poker](https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker)
**Deadline**: 2026-09-20
**Prize**: $5,000 USD
**Target Score**: 0.93827 (current #1 leaderboard score)
**User's Current Best**: 0.83065 (rank ~89)

## Repository Structure

```
.
├── docs/
│   ├── MASTER_PLAN.md              # Full strategy and roadmap to reach 0.93827
│   ├── SUBMISSIONS_ANALYSIS.md     # History of all 14 user submissions with insights
│   ├── DATA_EXPLORATION.md         # Dataset schema, distributions, key signals
│   └── RECOMMENDATIONS.md          # Concrete next steps to push past 0.83 plateau
├── scripts/
│   ├── inspect_data.py             # Data exploration script
│   ├── solution_v1.py              # v1 — pandas full load (failed OOM locally)
│   ├── solution_v2.py through v8.py  # Iterations trying to fit 4GB RAM budget
│   ├── solution_v7.py              # ✅ Working solution: row-group streaming
│   ├── solution_v2_tuned.py        # v2 tuned: 3-seed ensemble, percentile risk
│   ├── solution_v3_tuned.py        # v3 tuned: top-20% positive threshold
│   └── kaggle_notebook_v1.py        # Kaggle notebook version (16GB RAM)
├── notebooks/
│   └── kaggle_notebook_v1.py        # Standalone notebook for Kaggle
├── reference/
│   └── user_existing_notebook_source.py  # User's existing 197KB notebook (gets 0.83065)
├── submissions/
│   ├── submission_v1_sample.csv    # First 100 rows of v1 submission
│   └── submission_v3_sample.csv   # First 100 rows of v3 submission
├── results/
│   ├── cv_results.json              # v1 CV metrics
│   └── cv_results_v3.json          # v3 CV metrics
├── logs/
│   └── run7b.log                    # Successful run log
├── requirements.txt
└── README.md
```

## Summary

This repo contains analysis of all 14 user submissions to the "Detect Suspicious Value Transfers in Poker" Kaggle competition, a master plan to push past the user's 0.83065 plateau toward the target 0.93827 score, and Python solution scripts iterating on a from-scratch approach.

## Key Findings

1. **User's plateau at 0.83065** is a structural ceiling caused by overfitting a sophisticated multi-stage pipeline to the small dev set (1860 pairs, 372 positives)
2. **The metric is composite**: 0.70 × pair_ap + 0.20 × evidence_map5 + 0.10 × behavior_map — improving pair ranking has the highest leverage
3. **The #1 score (0.93827) is held by "Pardheev Krishna"** — likely uses a fundamentally different approach
4. **My from-scratch baseline (v1, v3) scores 0.07** — the seats-only aggregate features do not transfer from dev to test set; action-level features (which user's notebook has) are critical

## What Works vs What Doesn't

✅ **Works**:
- Memory-efficient row-group streaming (`solution_v7.py`) — processes 12M seats in 4GB RAM
- 3-seed LightGBM ensemble for stability
- Per-pair top-5 evidence hand tracking via min-heap

❌ **Doesn't work in 4GB RAM**:
- Self-join of seats on hand_id (creates ~60M intermediate rows)
- Loading full actions.parquet (18.6M rows) + seats together

⚠️ **Local baseline doesn't transfer**:
- Local OOF AP: 0.70 → leaderboard: 0.07
- Dev set has only 1860 pairs — easy to overfit
- Test set (112,540 pairs) is more diverse

## Recommendations for Next Steps

See `docs/RECOMMENDATIONS.md` for the full plan. Highlights:

1. **Build a Kaggle notebook** with action-level features (using Kaggle's 16GB RAM)
2. **Use the user's existing notebook as baseline** — already at 0.83065
3. **Add graph-based features**: player interaction networks, centrality, community detection
4. **Try LGBMRanker with lambdarank** for top-5 evidence selection (better than hand-crafted suspicion scores)
5. **Ensemble multiple seeds** with different random initializations

## Security Notice

⚠️ The Kaggle and GitHub tokens used to set up this repository have been shared in chat. **Rotate both tokens immediately** in your account settings:
- Kaggle: https://www.kaggle.com/settings/account → "Create New Token" (revokes old)
- GitHub: https://github.com/settings/tokens → delete the old token, create a new one

## License

MIT (this is an analysis repository; the underlying Kaggle competition data is governed by competition rules)
