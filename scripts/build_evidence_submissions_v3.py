"""
Build evidence-focused submissions using POLARS lazy (no Python loops over rows).
Memory-efficient: uses polars streaming + group_by aggregation.
"""
import os, gc, time, json, heapq
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import polars as pl

DATA = Path("/home/z/my-project/kaggle/comp-data")
OUT  = Path("/home/z/my-project/kaggle/runs")
USER_SUB = Path("/home/z/my-project/kaggle/user-submissions/pu-aware-output/submission.csv")

print("="*80)
print("Building evidence-focused submissions (polars lazy streaming)")
print("="*80)
t0 = time.time()

user_sub = pd.read_csv(USER_SUB)
print(f"User submission: {user_sub.shape}")

eval_pairs = pd.read_csv(DATA / "evaluation_pairs.csv")
print(f"Eval pairs: {eval_pairs.shape}")

# Build pair_lookup
pair_lookup = {}
pair_to_players = {}
for r in eval_pairs.itertuples():
    p1, p2 = sorted([r.player_1, r.player_2])
    pair_lookup[(p1, p2)] = r.pair_id
    pair_to_players[r.pair_id] = (p1, p2)

needed_players = set()
for p1, p2 in pair_to_players.values():
    needed_players.add(p1); needed_players.add(p2)
print(f"Needed players: {len(needed_players):,}")

# Build pairs DataFrame (polars)
pairs_df = pl.DataFrame({
    "p1": [k[0] for k in pair_lookup.keys()],
    "p2": [k[1] for k in pair_lookup.keys()],
    "pair_id": list(pair_lookup.values()),
})

# =========================================================================
# Use polars lazy: self-join seats with pairs, then join with hands
# All operations push down to streaming for low memory
# =========================================================================
print()
print("="*80)
print("Streaming polars self-join of seats + pairs + hands")
print("="*80)

needed_players_list = list(needed_players)

# Strategy: for each pair (p1, p2), find all hands where both are present,
# then compute per-(pair, hand) features for evidence selection

# Step 1: For each hand, find all (p1, p2) pairs that are both in the hand AND are eval pairs
# This requires self-joining seats on hand_id, then filtering by pairs_df

# Lazy scan: seats filtered by needed players
seats_l = pl.scan_parquet(DATA / "seats.parquet").filter(
    pl.col("player_id").is_in(needed_players_list)
).select([
    "hand_id", "player_id", "starting_stack", "total_contribution",
    "net_chips", "folded", "went_to_showdown", "won_share"
])

# Self-join approach: rename players to p1, p2 and join on hand_id
# Then join with pairs_df to filter to known pairs
print("Step 1: Build seats_p1 (rename player_id -> p1)...")
seats_p1 = seats_l.rename({
    "player_id": "p1",
    "starting_stack": "stack1",
    "total_contribution": "contrib1",
    "net_chips": "net1",
    "folded": "folded1",
    "went_to_showdown": "showdown1",
    "won_share": "won1",
})

print("Step 2: Build seats_p2 (rename player_id -> p2)...")
seats_p2 = seats_l.rename({
    "player_id": "p2",
    "starting_stack": "stack2",
    "total_contribution": "contrib2",
    "net_chips": "net2",
    "folded": "folded2",
    "went_to_showdown": "showdown2",
    "won_share": "won2",
})

print("Step 3: Join seats_p1 with pairs_df to filter p1...")
# This gives us all seats where player_id is a p1 of some eval pair
seats_p1_pairs = seats_p1.join(
    pairs_df.lazy(), on="p1", how="inner"
)

print("Step 4: Join with seats_p2 on (hand_id, p2)...")
# Now for each (hand, p1, pair_id, p2), get the seat data for p2 in the same hand
pair_hand_lazy = seats_p1_pairs.join(
    seats_p2, on=["hand_id", "p2"], how="inner"
).filter(
    pl.col("p1") < pl.col("p2")  # avoid duplicates
)

# Step 5: Join with hands for big_blind, final_pot
print("Step 5: Join with hands.parquet...")
hands_l = pl.scan_parquet(DATA / "hands.parquet").select([
    "hand_id", "big_blind", "final_pot"
])
pair_hand_lazy = pair_hand_lazy.join(hands_l, on="hand_id", how="left")

# Step 6: Compute evidence scores per (pair, hand)
print("Step 6: Compute evidence scores...")
pair_hand_lazy = pair_hand_lazy.with_columns([
    (pl.col("net1") - pl.col("net2")).abs().alias("abs_net_diff"),
    ((pl.col("net1") > 0) != (pl.col("net2") > 0)).alias("one_wins_one_loses"),
    (pl.col("net1") + pl.col("net2")).alias("transfer_signal"),
    (pl.col("showdown1") & pl.col("showdown2")).alias("both_showdown"),
])

# Compute three scores
pair_hand_lazy = pair_hand_lazy.with_columns([
    # Transfer score: high when one wins big, other loses big
    (pl.col("abs_net_diff") / pl.col("big_blind").clip(1) *
     pl.when(pl.col("one_wins_one_loses")).then(2.0).otherwise(0.3)
    ).alias("transfer_score"),
    # Passive score: high when both showdown with small pot
    (pl.when(pl.col("both_showdown") & (pl.col("final_pot") < pl.col("big_blind") * 4))
       .then(3.0 - pl.col("final_pot") / pl.col("big_blind").clip(1) / 4.0)
       .when(pl.col("both_showdown") & (pl.col("final_pot") < pl.col("big_blind") * 10))
       .then(1.0 - pl.col("final_pot") / pl.col("big_blind").clip(1) / 10.0)
       .otherwise(0.0)
     + pl.when((pl.col("contrib1") < pl.col("big_blind") * 2) & (pl.col("contrib2") < pl.col("big_blind") * 2))
       .then(0.5).otherwise(0.0)
    ).alias("passive_score"),
    # Aggressive score: high when big pot and asymmetric outcome
    (pl.when((pl.col("final_pot") > pl.col("big_blind") * 20) & pl.col("one_wins_one_loses") & (pl.col("abs_net_diff") > pl.col("big_blind") * 10))
       .then(pl.min_horizontal([5.0, pl.col("abs_net_diff") / pl.col("big_blind").clip(1) / 5.0]))
       .otherwise(0.0)
    ).alias("aggressive_score"),
])

# Step 7: For each pair, sort by each score desc, take top 5
print("Step 7: Top 5 per pair per score (collecting in chunks)...")
t_collect = time.time()
try:
    # Use the new engine="streaming" parameter (polars 1.25+)
    pair_hand = pair_hand_lazy.collect(engine="streaming")
    print(f"Collected: {pair_hand.shape}, time={time.time()-t_collect:.1f}s, mem={pair_hand.estimated_size('mb'):.1f}MB")
except Exception as e:
    print(f"Streaming failed: {e}")
    try:
        pair_hand = pair_hand_lazy.collect(engine="gpu")  # try gpu
        print(f"Collected (GPU engine): {pair_hand.shape}")
    except Exception as e2:
        print(f"GPU failed: {e2}")
        print("Trying without streaming...")
        pair_hand = pair_hand_lazy.collect()
        print(f"Collected (non-streaming): {pair_hand.shape}, time={time.time()-t_collect:.1f}s")

# Free memory
del seats_l, hands_l, seats_p1, seats_p2, seats_p1_pairs, pair_hand_lazy
gc.collect()

# =========================================================================
# Build evidence heaps per pair (per score type)
# =========================================================================
print()
print("="*80)
print("Building top-5 evidence per pair per score")
print("="*80)

pair_evidence = defaultdict(lambda: {"transfer": [], "passive": [], "aggressive": []})

# Convert to pandas for iteration (this should fit in memory now)
print("Converting to pandas...")
pair_hand_pd = pair_hand.to_pandas()
print(f"pair_hand_pd: {pair_hand_pd.shape}")
del pair_hand
gc.collect()

# Iterate
t_heap = time.time()
for r in pair_hand_pd.itertuples():
    pid = r.pair_id
    hand_id = r.hand_id
    ev = pair_evidence[pid]
    # Transfer
    if len(ev["transfer"]) < 5:
        heapq.heappush(ev["transfer"], (r.transfer_score, hand_id))
    else:
        heapq.heappushpop(ev["transfer"], (r.transfer_score, hand_id))
    # Passive
    if len(ev["passive"]) < 5:
        heapq.heappush(ev["passive"], (r.passive_score, hand_id))
    else:
        heapq.heappushpop(ev["passive"], (r.passive_score, hand_id))
    # Aggressive
    if len(ev["aggressive"]) < 5:
        heapq.heappush(ev["aggressive"], (r.aggressive_score, hand_id))
    else:
        heapq.heappushpop(ev["aggressive"], (r.aggressive_score, hand_id))

print(f"Heaps built in {time.time()-t_heap:.1f}s. {len(pair_evidence):,} pairs with evidence.")
del pair_hand_pd
gc.collect()

# =========================================================================
# Build SUBMISSION A — Behavior-agnostic chip-transfer evidence
# =========================================================================
print()
print("="*80)
print("SUBMISSION A — Chip-transfer evidence (top 5 by asymmetry)")
print("="*80)

sub_a = user_sub.copy()
new_evidence_a = {}
for pid, ev in pair_evidence.items():
    sorted_hands = sorted(ev["transfer"], key=lambda x: -x[0])
    top5 = [h for s, h in sorted_hands[:5]]
    new_evidence_a[pid] = top5

# Apply
for i, row in sub_a.iterrows():
    pid = row.pair_id
    if pid in new_evidence_a:
        top5 = new_evidence_a[pid]
        for k in range(5):
            col = f'evidence_hand_{k+1}'
            if k < len(top5):
                sub_a.at[i, col] = top5[k]
            else:
                sub_a.at[i, col] = 'NO_EVIDENCE'

sub_a.to_csv(OUT / "submission_A_evidence_chip_transfer.csv", index=False)
print(f"Saved submission_A_evidence_chip_transfer.csv: {sub_a.shape}")

evidence_cols = ['evidence_hand_1','evidence_hand_2','evidence_hand_3','evidence_hand_4','evidence_hand_5']
for c in evidence_cols:
    no_ev = (sub_a[c] == 'NO_EVIDENCE').sum()
    print(f"  {c}: {no_ev} NO_EVIDENCE ({100*no_ev/len(sub_a):.1f}%)")

# =========================================================================
# Build SUBMISSION B — Behavior-specific evidence
# =========================================================================
print()
print("="*80)
print("SUBMISSION B — Behavior-specific evidence")
print("="*80)

sub_b = user_sub.copy()
new_evidence_b = {}
pred_beh_map = user_sub.set_index('pair_id')['predicted_behavior'].to_dict()

for pid, ev in pair_evidence.items():
    beh = pred_beh_map.get(pid, 'directed_transfer')
    if beh == 'soft_play':
        sorted_hands = sorted(ev["passive"], key=lambda x: -x[0])
    elif beh == 'coordinated_isolation':
        sorted_hands = sorted(ev["aggressive"], key=lambda x: -x[0])
        if all(s == 0 for s, _ in sorted_hands[:5]):
            sorted_hands = sorted(ev["transfer"], key=lambda x: -x[0])
    else:
        sorted_hands = sorted(ev["transfer"], key=lambda x: -x[0])
    top5 = [h for s, h in sorted_hands[:5]]
    new_evidence_b[pid] = top5

# Apply
for i, row in sub_b.iterrows():
    pid = row.pair_id
    if pid in new_evidence_b:
        top5 = new_evidence_b[pid]
        for k in range(5):
            col = f'evidence_hand_{k+1}'
            if k < len(top5):
                sub_b.at[i, col] = top5[k]
            else:
                sub_b.at[i, col] = 'NO_EVIDENCE'

sub_b.to_csv(OUT / "submission_B_evidence_behavior_specific.csv", index=False)
print(f"Saved submission_B_evidence_behavior_specific.csv: {sub_b.shape}")

for c in evidence_cols:
    no_ev = (sub_b[c] == 'NO_EVIDENCE').sum()
    print(f"  {c}: {no_ev} NO_EVIDENCE ({100*no_ev/len(sub_b):.1f}%)")

# =========================================================================
# Compare to original
# =========================================================================
print()
print("="*80)
print("COMPARISON")
print("="*80)

overlap_a = 0
overlap_b = 0
for i, row in user_sub.iterrows():
    pid = row.pair_id
    orig_set = set([row.evidence_hand_1, row.evidence_hand_2, row.evidence_hand_3, row.evidence_hand_4, row.evidence_hand_5])
    if pid in new_evidence_a:
        new_a_set = set(new_evidence_a[pid])
        overlap_a += len(orig_set & new_a_set)
    if pid in new_evidence_b:
        new_b_set = set(new_evidence_b[pid])
        overlap_b += len(orig_set & new_b_set)

print(f"Total pairs: {len(user_sub)}")
print(f"Average overlap with original (per pair):")
print(f"  A (chip transfer): {overlap_a/len(user_sub):.2f} / 5.0 ({100*overlap_a/len(user_sub)/5:.1f}%)")
print(f"  B (behavior-specific): {overlap_b/len(user_sub):.2f} / 5.0 ({100*overlap_b/len(user_sub)/5:.1f}%)")

print()
print("Files saved:")
print(f"  - submission_A_evidence_chip_transfer.csv")
print(f"  - submission_B_evidence_behavior_specific.csv")
print(f"\nDone in {time.time()-t0:.1f}s")
