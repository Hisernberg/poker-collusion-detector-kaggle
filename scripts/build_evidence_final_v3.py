"""
Final evidence submission builder — pure numpy, no Python loops over rows.
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
print("Building evidence submissions (numpy vectorized)")
print("="*80)
t0 = time.time()

user_sub = pd.read_csv(USER_SUB)
eval_pairs = pd.read_csv(DATA / "evaluation_pairs.csv")

pair_lookup = {}
for r in eval_pairs.itertuples():
    p1, p2 = sorted([r.player_1, r.player_2])
    pair_lookup[(p1, p2)] = r.pair_id

needed_players = set()
for k in pair_lookup.keys():
    needed_players.add(k[0]); needed_players.add(k[1])
print(f"Needed players: {len(needed_players):,}")

# Load hands.parquet
hands_pl = pl.read_parquet(DATA / "hands.parquet", columns=["hand_id", "big_blind", "final_pot"])
hands_dict = {row["hand_id"]: (int(row["big_blind"]), int(row["final_pot"]))
              for row in hands_pl.iter_rows(named=True)}
del hands_pl
gc.collect()
print(f"hands_dict: {len(hands_dict):,}")

# Per-pair evidence heaps
pair_evidence = defaultdict(lambda: {"transfer": [], "passive": [], "aggressive": []})

# Stream using row-group reading (we already showed 60K rows = 8MB is fine)
print("Streaming seats.parquet via row groups...")
pf = pq.ParquetFile(DATA / "seats.parquet")
t_stream = time.time()
hands_scored = 0

for rg_idx in range(pf.num_row_groups):
    # Read row group as a pandas DataFrame
    table = pf.read_row_group(rg_idx, columns=[
        "hand_id", "player_id", "starting_stack", "total_contribution",
        "net_chips", "folded", "went_to_showdown", "won_share"
    ])
    df = table.to_pandas()
    del table

    # Vectorized filter to needed players (using isin on a 11K-element set)
    # This creates a boolean mask of length 60K — efficient
    needed_mask = df.player_id.isin(needed_players)
    if not needed_mask.any():
        continue
    df = df[needed_mask]
    del needed_mask
    gc.collect()

    # Group by hand_id using pandas (efficient for small DataFrames)
    for hand_id, group in df.groupby("hand_id", sort=False):
        if hand_id not in hands_dict:
            continue
        big_blind, final_pot = hands_dict[hand_id]
        bb = max(big_blind, 1)

        # Get arrays (lightweight)
        players = group.player_id.values
        stacks = group.starting_stack.values
        contribs = group.total_contribution.values
        nets = group.net_chips.values
        foldeds = group.folded.values
        showdowns = group.went_to_showdown.values
        won_shares = group.won_share.values

        n_players = len(players)
        if n_players < 2:
            continue

        # Build players_dict for fast lookup (small, just 6 players max)
        # Use index-based access instead of dict to save memory
        sorted_idx = np.argsort(players)
        sorted_players = players[sorted_idx]

        # For each pair (i, j) with i < j, check if they're a known pair
        for ii in range(n_players):
            i = sorted_idx[ii]
            p1 = players[i]
            for jj in range(ii + 1, n_players):
                j = sorted_idx[jj]
                p2 = players[j]
                pid = pair_lookup.get((p1, p2))
                if pid is None:
                    continue

                net1, net2 = nets[i], nets[j]
                contrib1, contrib2 = contribs[i], contribs[j]
                showdown1, showdown2 = bool(showdowns[i]), bool(showdowns[j])

                asymmetry = abs(net1 - net2) / bb
                one_wins = (net1 > 0) != (net2 > 0)
                transfer_score = asymmetry * (2.0 if one_wins else 0.3)

                passive_score = 0.0
                if showdown1 and showdown2:
                    if final_pot < bb * 4:
                        passive_score = 3.0 - (final_pot / bb / 4.0)
                    elif final_pot < bb * 10:
                        passive_score = 1.0 - (final_pot / bb / 10.0)
                if contrib1 < bb * 2 and contrib2 < bb * 2:
                    passive_score += 0.5

                aggressive_score = 0.0
                if final_pot > bb * 20:
                    if one_wins and asymmetry > bb * 10:
                        aggressive_score = min(5.0, asymmetry / bb / 5.0)

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

        hands_scored += 1

    if (rg_idx + 1) % 20 == 0 or rg_idx == pf.num_row_groups - 1:
        elapsed = time.time() - t_stream
        print(f"  rg {rg_idx+1}/{pf.num_row_groups}: hands={hands_scored:,}, pairs_w_ev={len(pair_evidence):,}, elapsed={elapsed:.1f}s", flush=True)

print(f"Done in {time.time()-t_stream:.1f}s. Pairs with evidence: {len(pair_evidence):,}, hands: {hands_scored:,}")
del hands_dict
gc.collect()

# Build SUBMISSION A — Chip-transfer evidence
print("Building SUBMISSION A — Chip-transfer evidence")
sub_a = user_sub.copy()
evidence_a = {pid: [h for s, h in sorted(ev["transfer"], key=lambda x: -x[0])[:5]]
              for pid, ev in pair_evidence.items()}

for i, row in sub_a.iterrows():
    pid = row.pair_id
    top5 = evidence_a.get(pid, [])
    for k in range(5):
        col = f'evidence_hand_{k+1}'
        sub_a.at[i, col] = top5[k] if k < len(top5) else 'NO_EVIDENCE'

sub_a.to_csv(OUT / "submission_A_evidence_chip_transfer.csv", index=False)
print(f"Saved submission_A_evidence_chip_transfer.csv: {sub_a.shape}")

# Build SUBMISSION B — Behavior-specific evidence
print("Building SUBMISSION B — Behavior-specific evidence")
sub_b = user_sub.copy()
pred_beh_map = user_sub.set_index('pair_id')['predicted_behavior'].to_dict()
evidence_passive = {pid: [h for s, h in sorted(ev["passive"], key=lambda x: -x[0])[:5]]
                    for pid, ev in pair_evidence.items()}
evidence_aggressive = {pid: [h for s, h in sorted(ev["aggressive"], key=lambda x: -x[0])[:5]]
                       for pid, ev in pair_evidence.items()}

for i, row in sub_b.iterrows():
    pid = row.pair_id
    beh = pred_beh_map.get(pid, 'directed_transfer')
    if beh == 'soft_play':
        top5 = evidence_passive.get(pid, [])
    elif beh == 'coordinated_isolation':
        top5 = evidence_aggressive.get(pid, []) or evidence_a.get(pid, [])
    else:
        top5 = evidence_a.get(pid, [])
    for k in range(5):
        col = f'evidence_hand_{k+1}'
        sub_b.at[i, col] = top5[k] if k < len(top5) else 'NO_EVIDENCE'

sub_b.to_csv(OUT / "submission_B_evidence_behavior_specific.csv", index=False)
print(f"Saved submission_B_evidence_behavior_specific.csv: {sub_b.shape}")

# Compare
overlap_a = 0
overlap_b = 0
for i, row in user_sub.iterrows():
    pid = row.pair_id
    orig_set = {row.evidence_hand_1, row.evidence_hand_2, row.evidence_hand_3, row.evidence_hand_4, row.evidence_hand_5}
    overlap_a += len(orig_set & set(evidence_a.get(pid, [])))
    beh = pred_beh_map.get(pid, 'directed_transfer')
    if beh == 'soft_play':
        b_ev = evidence_passive.get(pid, [])
    elif beh == 'coordinated_isolation':
        b_ev = evidence_aggressive.get(pid, []) or evidence_a.get(pid, [])
    else:
        b_ev = evidence_a.get(pid, [])
    overlap_b += len(orig_set & set(b_ev))

print(f"\nAverage overlap with original (per pair):")
print(f"  A (chip transfer): {overlap_a/len(user_sub):.2f} / 5.0 ({100*overlap_a/len(user_sub)/5:.1f}%)")
print(f"  B (behavior-specific): {overlap_b/len(user_sub):.2f} / 5.0 ({100*overlap_b/len(user_sub)/5:.1f}%)")
print(f"\nDone in {time.time()-t0:.1f}s")
