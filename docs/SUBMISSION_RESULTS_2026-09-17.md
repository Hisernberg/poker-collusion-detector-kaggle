# Submission Results — Evidence Re-ranking Experiment (2026-09-17)

## Goal
Beat user's plateau score of 0.83065 by improving evidence selection (which is the user's weakest dimension: local evidence_map5 = 0.5454).

## Approach
Built 2 strategic submissions that:
- **Keep user's risk_score unchanged** (preserves pair_ap signal that achieves ~0.99 locally)
- **Keep user's predicted_behavior unchanged** (preserves behavior_map signal that achieves 1.0 locally)
- **Replace evidence hands** with model-free behavioral signals:
  - **Submission A**: Top-5 hands by chip transfer asymmetry (one wins big, other loses big)
  - **Submission B**: Behavior-specific — passive signal for soft_play, aggressive for coordinated_isolation, transfer for directed_transfer

## Results

| Submission | Public Score | Notes |
|------------|--------------|-------|
| User's existing 0.83065 (baseline) | **0.83065** | Best so far |
| **Submission A** (chip-transfer evidence) | **0.75238** | -0.078 |
| **Submission B** (behavior-specific evidence) | **0.74941** | -0.081 |
| My v1 (seats+hands features) | 0.07339 | Floor |
| My v3 (3-seed ensemble) | 0.07241 | Floor |

## Conclusion

**The user's evidence ranker is actually quite good.** Replacing the model-selected evidence with raw chip-transfer signals dropped the score by ~0.08.

This validates that the user's PU-aware evidence ranker (with LGBMRanker, lambdarank, multiple expert blends) is doing real work — simple behavioral signals can't replace it.

## What this tells us about the gap to 0.93827

If even replacing the user's evidence (15% overlap with original) drops score by 0.08, then:
- The user's evidence_map5 on the public test set is much higher than 0.15 (probably 0.30-0.40)
- The gap from 0.83 to 0.94 must come from improvements in **pair_ap** (which has 70% weight)
- OR from a fundamentally different evidence signal we haven't tried

## Recommended next steps

1. **Don't replace user's evidence** — their model is better than heuristics
2. **Focus on pair_ap improvement** — that's where 70% of the score lives
3. **Try ensembling** user's risk_score with another model's risk_score (different signal)
4. **Add graph-based features** (player interaction network, centrality) for pair-level discrimination

## Submissions used today
- 2 of 5 daily limit used
- 3 remaining today, ~13 total remaining over 3 days
