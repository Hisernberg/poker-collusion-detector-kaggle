# Submissions History Analysis

## All 14 Submissions (lifetime as of 2026-09-17)

| # | Date | Score | Description | Note |
|---|------|-------|-------------|------|
| 1 | 2026-09-08 | 0.71943 | Notebook Poker Collusion \| PU-Aware Evidence Ranker v1 | Initial baseline |
| 2 | 2026-09-08 | ERROR | poker_sentinel_v2/cache_89d8bb30c1b8bbeb94b5/COMPLETE.json | Failed run |
| 3 | 2026-09-08 | 0.78551 | submission.csv (direct upload) | First CSV upload |
| 4 | 2026-09-09 | 0.78667 | submission.csv | Small gain (+0.001) |
| 5 | 2026-09-11 | 0.76024 | submission.csv | Regression (-0.026) |
| 6 | 2026-09-11 | 0.83065 | submission.csv | **+0.044 jump — major insight** |
| 7 | 2026-09-12 | 0.82757 | Notebook v10 | Small regression |
| 8 | 2026-09-13 | 0.82916 | Notebook v12 | Plateau |
| 9 | 2026-09-13 | 0.82916 | Notebook v13 | Plateau (identical) |
| 10 | 2026-09-13 | 0.82916 | Notebook v15 | Plateau (identical) |
| 11 | 2026-09-13 | 0.82916 | Notebook v16 | Plateau (identical) |
| 12 | 2026-09-15 | ERROR | submission (18).csv | Failed run |
| 13 | 2026-09-15 | 0.83065 | submission (19).csv | **Best score** |
| 14 | 2026-09-15 | 0.83065 | submission (20).csv | **Best score (identical)** |

## New Submissions (from this analysis session)

| # | Date | Score | Description |
|---|------|-------|-------------|
| 15 | 2026-09-17 | 0.07339 | v1 baseline: pair features from seats+hands, 3 behavior LGBM models + 1 generic |
| 16 | 2026-09-17 | 0.07241 | v3: 3-seed ensemble, percentile risk, top-20% positive threshold, v1 evidence |

## Key Observations

### 1. Plateau at 0.83065 across 7+ submissions
**Strong evidence of a structural ceiling**. When submissions 19 and 20 produce identical scores (0.83065) despite different file uploads, it suggests:
- The model has converged to a similar local optimum
- The same features are being used
- The decision boundaries are very similar

### 2. The +0.044 jump on 2026-09-11
This was a major breakthrough. The progression was:
- 2026-09-11 (00:23): 0.78551
- 2026-09-11 (11:12): 0.76024 (regression)
- 2026-09-11 (20:32): **0.83065** (breakthrough)

The breakthrough likely came from:
- A new feature or signal that wasn't in the model before
- Or a better evidence ranking algorithm
- Or improved PU learning hyperparameters

### 3. Behavior family distribution (dev set)
- none: 1488 (80%)
- directed_transfer: 148 (8%)
- soft_play: 132 (7%)
- coordinated_isolation: 92 (5%)

Total: 1860 pairs (372 positives). This 20% positive rate is what we should aim for in test predictions.

### 4. The notebook approach is highly sophisticated
The user's notebook `poker-collusion-pu-aware-evidence-ranker` is a 197KB Python file with:
- PU (Positive-Unlabeled) learning framework
- Multi-stage ranking: hand-level → pair-level features → final risk score
- LightGBM as primary model (420-620 trees)
- 5-fold cross-validation
- Multiple expert blends: likelihood_rank, ndcg_rank, ap5_rank, likelihood_margin, episode_gated, gameplay_rule, generic_evidence, action_null
- LGBMRanker with lambdarank for AP@5 optimization
- Isolation Forest for novelty detection
- Per-family rank models (3 separate LightGBMs)
- "V6 challenger" pair features
- "V4 anchor" + witness models

### 5. Why my from-scratch approach scored 0.07 vs user's 0.83
My v1/v3 solutions only use seats+hands aggregate features (no action-level signals). The user's notebook uses:
- Per-hand action sequences (VPIP, PFR, fold-to-partner, etc.)
- Per-hand behavioral signatures (sandwich, checkdown, etc.)
- Hole card strengths for fold-to-strong-hand detection
- PU learning for unlabeled data

Without these action-level features, the model essentially has no signal on the test set.

## Submission Strategy Analysis

The user has used 12 lifetime submissions. With 3 days left and ~5/day limit:
- ~15 remaining submissions
- Current best: 0.83065
- Target: 0.93827 (gap: 0.108)

**Recommended remaining submission allocation**:
- 2-3 for new approaches (action-level features, graph-based, ensembles)
- 2-3 for hyperparameter tuning of best approach
- 2-3 for ensemble variants
- 1-2 for final late-cycle refinement
