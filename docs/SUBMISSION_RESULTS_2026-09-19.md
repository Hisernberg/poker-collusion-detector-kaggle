# Daily Report — 2026-09-19 (Day 1 of final sprint, 5/5 submissions used)

## TL;DR
- Rebuilt the entire proven 0.81-floor pipeline from scratch as a **memory-safe, resumable 5-stage pipeline** (our box: 2 CPU / 4 GB RAM) and reproduced the reference stack exactly (pair index, OOF metrics all match).
- **Best new score: 0.81854 (v4)** — up from 0.81137 (baseline reproduction), via a rank-normalized **PU-LGB + CatBoost ensemble** (0.4 / 0.6).
- Forensic conclusion of the day: **no deterministic planted-action fingerprint exists** (mined 11,520 action templates; best precision 0.15 at high recall; episode-window features AUC ≈ 0.5). Evidence MAP@5 is a hard ranking problem; incremental blend tweaks are worth ≤ +0.002.
- OOF weight-search over 7 experts **overfit** (v5 scored 0.81612 < v4). Keep blends simple.

## Scores (public LB)

| # | File | Risk model | Evidence | Score |
|---|------|------------|----------|-------|
| v1 | sub_v1_port81_iso.csv | PU 2-step LGB (ported 0.81 stack + street-split contrasts + iso flags) | 0.6 spec(pred) + 0.4 global, active-only | 0.81137 |
| v2 | sub_v2_evsoft.csv | = v1 | soft family weights on ALL pairs | 0.81119 |
| v3 | sub_v3_ensemble.csv | 0.6 PU + 0.1 PN + 0.3 Cat(d5) rank-blend | = v1 | 0.81283 |
| **v4** | **sub_v4_catheavy.csv** | **0.4 PU + 0.6 Cat(d5,i420) rank-blend** | = v1 | **0.81854** |
| v5 | sub_v5_pool.csv | 7-expert OOF-searched blend (cat_d6 0.69 …) | = v1 | 0.81612 |

Previous user best: 0.83065 (V12 notebook). Current top LB: 0.93827.

## What we learned
1. **Pair AP responds to model diversity, not to more PU tuning.** CatBoost (depth 5, PU weights) alone reaches OOF eval-mirrored AP 0.8156 vs PU-LGB 0.7443; blending 0.4/0.6 gave the day's best.
2. **Evidence has no silver bullet.** Verified by: (a) template mining grid (street × action × size × strength × facing-partner), (b) rolling-window "episode heat" features (AUC≈0.5), (c) AP@5 LambdaMart objective (port of V12 `ap5_lambdas`), (d) evidence-rank-weighted targets, (e) rank-average blends. All ≤ +0.006 OOF MAP@5.
3. **OOF-searched blend weights overfit the LB** (v5 < v4). Tomorrow: simple weight ladder around the v4 recipe (CAT weight 0.6 → 0.75 → 0.85).
4. Pipeline engineering: sharded (24-way hand-hash) stage builds + subprocess-per-shard survive a 4 GB box; chunk caches make every stage resumable.

## OOF reference numbers (dev population)
- Stage-2 PU pair model: eval-mirrored AP 0.7443 (pessimistic; treats all unlabelled as negative), clean AP 0.9834
- Evidence: global ranker MAP@5 0.5189; oracle-family mix 0.5242; family accuracy on positives 0.997
- CatBoost expert: eval-mirrored AP 0.8156; ensemble 0.4/0.6 → 0.8141 mirrored / 0.9860 clean

## Plan for tomorrow (5 submissions)
1. CAT-weight ladder on the v4 recipe: 0.25/0.75, 0.15/0.85 (find the knee).
2. Evidence: percentile-normalized blend (V12-style) as a low-risk swap.
3. Risk: extra pair features (context-matched decision residuals) if time permits; pick best combo for the final slot.
