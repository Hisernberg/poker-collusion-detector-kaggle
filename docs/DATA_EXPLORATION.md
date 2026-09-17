# Data Exploration Notes

## Dataset Overview

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

## Submission Format

Output CSV with 8 columns:
- `pair_id` (string ID)
- `risk_score` (float 0-1)
- `predicted_behavior` (one of: none, directed_transfer, soft_play, coordinated_isolation, other_coordination)
- `evidence_hand_1..5` (hand_id or "NO_EVIDENCE")

## Metric (proxy from public spec)

```
composite = 0.70 × pair_ap + 0.20 × evidence_map5 + 0.10 × behavior_map
```

- `pair_ap`: Average Precision of risk_score on labeled dev pairs (70% weight — most important)
- `evidence_map5`: Mean AP@5 over top-5 evidence hands vs. gold evidence (20% weight)
- `behavior_map`: Mean AP per behavior family (10% weight)

## Dev Label Distribution

| behavior_family | count | % of dev | % of positives |
|-----------------|-------|----------|----------------|
| none | 1488 | 80.0% | — |
| directed_transfer | 148 | 8.0% | 39.8% |
| soft_play | 132 | 7.1% | 35.5% |
| coordinated_isolation | 92 | 5.0% | 24.7% |
| **TOTAL** | **1860** | **100%** | **372 positives (20%)** |

## Player Counts

- Total players: 12,000 (in players.parquet)
- Players needed (in any dev or eval pair): 11,996 (99.97%)
- Total pairs (dev + eval): 114,400

## Key Behavioral Signatures (from user's EDA notebook)

### directed_transfer (148 positive pairs)
**Signal**: One player consistently folds to another, especially with strong hands.
- `p1_fold_to_p2 > 0` AND `p1_fold_to_other_best > 0` (folding with strong hand)
- High `net_chips` asymmetry (one wins, other loses consistently)
- Average evidence per pair: ~5 hands (top-5 most suspicious)

### soft_play (132 positive pairs)
**Signal**: Passive play between partners in heads-up pots.
- `hu_checkdown_pq >= 2` (multiple HU checkdowns where both check)
- High `both_postflop` rate
- Low `pot_bb` (small pots between partners)
- Low `p1_aggression` vs `p2_aggression` (mutual passivity)

### coordinated_isolation (92 positive pairs)
**Signal**: Two players cooperate to isolate a third player preflop.
- `sandwich_pf == 1` (both p&q raise preflop with ≥3 distinct raisers)
- `third_folds_after_pq_raises > 0` (third player folds after p&q raise)
- High `n_raisers_pf` (multiple preflop raises)

## Memory Constraints (Local Sandbox)

The local environment has only **4GB RAM and 2 CPUs**. This constrained the solution approach:

| Operation | Memory Required | Local OK? |
|-----------|----------------|-----------|
| Load full seats.parquet (12M rows × 11 cols as pandas) | ~1.9 GB | ⚠️ Tight |
| Load full actions.parquet (18.6M rows × 11 cols) | ~1.6 GB | ⚠️ Tight |
| Polars seats filtered (12M × 8) | 689 MB | ✅ |
| Polars hands (2M × 7) | 121 MB | ✅ |
| pandas filter on 2M-element set (isin) | spike to 3 GB+ | ❌ |
| Self-join of seats on hand_id | ~9 GB intermediate | ❌ |

**Solution**: Stream row-groups from pyarrow, build per-pair running accumulators, never materialize the full pair-hand DataFrame.

## Final Pipeline (v7)

1. **Stream seats.parquet row-group by row-group** (200 groups × ~60K rows each)
2. For each row group:
   - Filter to needed players (12K of 12K = essentially all)
   - Group by hand_id
   - For each hand, enumerate pairs of needed players and check pair_lookup
   - For each known pair: update per-pair accumulator + top-5 evidence heap
3. **Build pair-level features** from accumulators (means, sums, max, std for net_chips, contributions, pots, etc.)
4. **Train 4 LightGBM models** (3 family-specific + 1 generic), 5-fold CV
5. **Predict on evaluation pairs**: argmax across families for behavior, max for risk
6. **Output submission.csv** with top-5 evidence hands from heaps

Total runtime: ~16 minutes locally (1 CPU bottleneck).
