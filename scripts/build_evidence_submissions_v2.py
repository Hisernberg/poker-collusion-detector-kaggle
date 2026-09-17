"""
Build 2 evidence-focused submissions — memory-optimized.
Process seats row-group by row-group, score pairs on the spot.
Don't buffer incomplete hands (just process whatever we have for each hand
within the row group — most hands don't span row groups in parquet).
"""
import os, gc, time, json, heapq
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import polars as pl

DATA = Path("/home/z/my-project/kaggle/comp-data")
OUT  = Path("/home/z/my-project/kaggle/runs")
USER_SUB = Path("/home/z/my-project/kaggle/user-submissions/pu-aware-output/submission.csv")

print("="*80)
print("Building 2 evidence-focused submissions (memory-optimized)")
print("="*80)
t0 = time.time()

user_sub = pd.read_csv(USER_SUB)
print(f"User submission: {user_sub.shape}")

eval_pairs = pd.read_csv(DATA / "evaluation_pairs.csv")
print(f"Eval pairs: {eval_pairs.shape}")

# Build pair_lookup for eval pairs only
pair_lookup_eval = {}
pair_to_players_eval = {}
for r in eval_pairs.itertuples():
    p1, p2 = sorted([r.player_1, r.player_2])
    pair_lookup_eval[(p1, p2)] = r.pair_id
    pair_to_players_eval[r.pair_id] = (p1, p2)
print(f"Eval pair_lookup: {len(pair_lookup_eval):,}")

needed_players = set()
for p1, p2 in pair_to_players_eval.values():
    needed_players.add(p1); needed_players.add(p2)
print(f"Needed players (eval only): {len(needed_players):,}")

# =========================================================================
# Load hands.parquet
# =========================================================================
print()
print("Loading hands.parquet...")
hands_pl = pl.read_parquet(DATA / "hands.parquet", columns=[
    "hand_id", "phase", "big_blind", "final_pot", "players_dealt"
])
hands_dict = {}
for row in hands_pl.iter_rows(named=True):
    hands_dict[row["hand_id"]] = (
        row["phase"], int(row["big_blind"]), int(row["final_pot"]), int(row["players_dealt"])
    )
del hands_pl
gc.collect()
print(f"hands_dict: {len(hands_dict):,}")

# =========================================================================
# Stream seats.parquet — DON'T buffer incomplete hands
# =========================================================================
print()
print("="*80)
print("Streaming seats.parquet (no buffering)")
print("="*80)

def new_evidence_heap():
    return {"transfer": [], "passive": [], "aggressive": []}

pair_evidence = defaultdict(new_evidence_heap)

pf_seats = pq.ParquetFile(DATA / "seats.parquet")
print(f"seats.parquet: {pf_seats.metadata.num_rows:,} rows, {pf_seats.num_row_groups} row groups")

t_stream = time.time()
hands_scored = 0
pairs_scored = 0

for rg_idx in range(pf_seats.num_row_groups):
    table = pf_seats.read_row_group(rg_idx, columns=[
        "hand_id", "player_id", "starting_stack", "total_contribution",
        "net_chips", "folded", "went_to_showdown", "won_share"
    ])
    df = table.to_pandas()
    df = df[df.player_id.isin(needed_players)]

    # Group by hand_id within this row group
    for hand_id, group in df.groupby("hand_id"):
        if hand_id not in hands_dict:
            continue
        phase, big_blind, final_pot, players_dealt = hands_dict[hand_id]
        bb = max(big_blind, 1)

        players_dict = {}
        for r in group.itertuples():
            players_dict[r.player_id] = (
                int(r.starting_stack), int(r.total_contribution), int(r.net_chips),
                bool(r.folded), bool(r.went_to_showdown),
                float(r.won_share) if pd.notna(r.won_share) else 0.0
            )

        if len(players_dict) < 2:
            continue

        # Score all pairs in this hand (even if not all players present, still score)
        sorted_players = sorted(players_dict.keys())
        for i in range(len(sorted_players)):
            p1 = sorted_players[i]
            s1 = players_dict[p1]
            for j in range(i+1, len(sorted_players)):
                p2 = sorted_players[j]
                key = (p1, p2)
                pid = pair_lookup_eval.get(key)
                if pid is None:
                    continue
                s2 = players_dict[p2]
                stack1, contrib1, net1, folded1, showdown1, won1 = s1
                stack2, contrib2, net2, folded2, showdown2, won2 = s2

                # Signal 1: chip transfer (asymmetric outcome)
                asymmetry = abs(net1 - net2) / bb
                one_wins_one_loses = (net1 > 0) != (net2 > 0)
                transfer_score = asymmetry * (2.0 if one_wins_one_loses else 0.3)

                # Signal 2: passive outcome
                passive_score = 0.0
                if showdown1 and showdown2:
                    if final_pot < bb * 4:
                        passive_score = 3.0 - (final_pot / bb / 4.0)
                    elif final_pot < bb * 10:
                        passive_score = 1.0 - (final_pot / bb / 10.0)
                if contrib1 < bb * 2 and contrib2 < bb * 2:
                    passive_score += 0.5

                # Signal 3: aggressive (coordinated_isolation)
                aggressive_score = 0.0
                if final_pot > bb * 20:
                    if one_wins_one_loses and asymmetry > bb * 10:
                        aggressive_score = min(5.0, asymmetry / bb / 5.0)

                # Update heaps
                ev = pair_evidence[pid]
                if len(ev["transfer"]) < 5:
                    heapq.heappush(ev["transfer"], (transfer_score, hand_id))
                else:
                    heapq.heappushpop(ev["transfer"], (transfer_score, hand_id))
                if len(ev["passive"]) < 5:
                    heapq.heappush(ev["passive"], (passive_score, hand_id))
                else:
                    heapq.heappushpop(ev["passive"], (passive_score, hand_id))
                if len(ev["aggressive"]) < 5:
                    heapq.heappush(ev["aggressive"], (aggressive_score, hand_id))
                else:
                    heapq.heappushpop(ev["aggressive"], (aggressive_score, hand_id))

                pairs_scored += 1
        hands_scored += 1

    if (rg_idx + 1) % 20 == 0 or rg_idx == pf_seats.num_row_groups - 1:
        elapsed = time.time() - t_stream
        print(f"  row group {rg_idx+1}/{pf_seats.num_row_groups}: hands={hands_scored:,}, pairs_scored={pairs_scored:,}, elapsed={elapsed:.1f}s", flush=True)

print(f"Done in {time.time()-t_stream:.1f}s. Pairs with evidence: {len(pair_evidence):,}")
del hands_dict
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

# Per-pair, choose evidence based on user's predicted_behavior
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
    else:  # directed_transfer or other
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
