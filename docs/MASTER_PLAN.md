# Master Plan: Win "Detect Suspicious Value Transfers in Poker"

## Competition Overview
- **Name**: Detect Suspicious Value Transfers in Poker
- **URL**: https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker
- **Prize**: $5,000 USD
- **Deadline**: 2026-09-20 22:00:00 UTC (3 days from now)
- **Teams**: 294
- **Current user rank**: 89
- **Current user best score**: 0.83065 (plateau across submissions 19, 20)
- **Target score**: 0.93827 (current #1, held by "Pardheev Krishna")

## Task Structure
Detect collusion / chip-dumping between pairs of poker players. For each pair in `evaluation_pairs.csv`, we must output:
- `risk_score`: continuous 0-1 (how suspicious)
- `predicted_behavior`: one of {none, directed_transfer, soft_play, coordinated_isolation, other_coordination}
- `evidence_hand_1..5`: top 5 `hand_id` values that best evidence the suspicious behavior (or "NO_EVIDENCE")

## Metric (best-known proxy from public spec)
```
composite = 0.70 * pair_ap + 0.20 * evidence_map5 + 0.10 * behavior_map
```
- `pair_ap`: Average Precision of risk_score on labeled dev pairs
- `evidence_map5`: Mean AP@5 over top-5 evidence hands vs. gold evidence
- `behavior_map`: Mean AP per behavior family

→ Improving pair ranking matters most (70% weight).

## Data Files
| File | Rows | Size | Purpose |
|------|------|------|---------|
| actions.parquet | 18,609,028 | 150 MB | Action-level log (street, action, amount, pot) |
| hands.parquet | 2,000,000 | 48 MB | Per-hand metadata (phase, board, pot, showdown) |
| seats.parquet | 12,000,000 | 94 MB | Per-hand-per-player (stack, hole cards, net, won_share) |
| players.parquet | 12,000 | 137 KB | Player profile (age, region, stake, client) |
| development_labels.csv | 1,860 | 132 KB | 1860 pairs, label (1/0), behavior_family |
| development_evidence.csv | 1,817 | 87 KB | Top-5 gold evidence hands per positive pair (372 pairs) |
| evaluation_pairs.csv | 112,540 | 5 MB | Test pairs (pair_id, player_1, player_2, shared_hands) |
| sample_submission.csv | 112,540 | 9 MB | Output template |

## Label Distribution (dev)
- none: 1488 (80%)
- directed_transfer: 148 (8%)
- soft_play: 132 (7%)
- coordinated_isolation: 92 (5%)

## Analysis of User's Submissions (12 lifetime, 5 remaining today)

| # | Date | Score | Description | Note |
|---|------|-------|-------------|------|
| 1 | 09-08 | 0.71943 | Notebook Poker Collusion \| PU-Aware Evidence Ranker v1 | Initial baseline |
| 2 | 09-08 | ERROR | poker_sentinel_v2/cache_89d8bb30c1b8bbeb94b5/COMPLETE.json | Failure |
| 3 | 09-08 | 0.78551 | submission.csv | Early iteration |
| 4 | 09-09 | 0.78667 | submission.csv | Small gain |
| 5 | 09-11 | 0.76024 | submission.csv | Regression |
| 6 | 09-11 | 0.83065 | submission.csv | BIG jump (+0.044) |
| 7 | 09-12 | 0.82757 | Notebook v10 | Small regression |
| 8-11 | 09-13 to 09-14 | 0.82916 (x4) | Notebook v12, v13, v15, v16 | Plateau |
| 12-14 | 09-15 | 0.83065 (x3) | submission.csv (18,19,20) | Final plateau |

### Key observations
1. **Plateau at 0.83065** across 7+ submissions — strong ceiling
2. The user has a complex multi-stage pipeline ("sentinel-cpu-7.0.0-consensus")
3. Multiple LightGBM models (420-620 trees each), 5-fold CV
4. Uses PU learning, isolation forest, multiple expert blends
5. **Likely issue**: Over-engineering → overfitting to dev set, structural ceiling

## Master Plan to Beat 0.93827

### Phase 1: Clean Baseline Replication (today)
- Build a focused, working pipeline that reproduces ~0.83 locally
- Submit to verify pipeline works
- Get a reproducible artifact

### Phase 2: Key Improvements (next 2 days)
1. **Better behavioral signals** from action-level data
   - Per-street fold-to-partner rates with hand strength
   - Checkdown rate in heads-up pots
   - "Sandwich" preflop raise patterns
   - Net chip transfer direction consistency
2. **Player-pair temporal features**
   - Chip flow over time (cumulative net per player across shared hands)
   - Variance / consistency of suspicious patterns
3. **Better evidence ranking**
   - Per-hand behavioral score (per family)
   - Use LGBMRanker with lambdarank for top-5 retrieval
4. **Behavior-specific models** (3 separate LightGBMs)
   - directed_transfer model
   - soft_play model
   - coordinated_isolation model
   - Final risk_score = max(probability from each family model)
5. **Ensemble** with different feature subsets & seeds
6. **Tune decision threshold for behavior prediction**

### Phase 3: Final Submission (day 3)
- Submit best ensemble
- Save 1 daily submission for late-cycle refinement

## Submission Strategy
- 5 submissions remaining today (2026-09-17)
- 3 days × 5/day = ~15 total shots remaining
- Use 1 today for baseline verification
- 2-3 today for first improvements
- 4-5 per day tomorrow & day after for iteration
- 1-2 final day for best ensemble

## Risk Management
- If new approach fails, fall back to known 0.83065 baseline
- Always submit only after local validation on dev set
- Use early stopping and CV to detect overfitting
- Monitor submission feedback (publicScore) closely
