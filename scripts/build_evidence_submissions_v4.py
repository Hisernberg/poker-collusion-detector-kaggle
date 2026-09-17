"""
Build evidence-focused submissions — CSV chunked approach.
Process seats row-group by row-group, write (pair_id, hand_id, score) to CSV
as we go. Then group_by pair_id at the end.
"""
import os, gc, time, json, csv
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
print("Building evidence-focused submissions (chunked CSV)")
print("="*80)
t0 = time.time()

user_sub = pd.read_csv(USER_SUB)
print(f"User submission: {user_sub.shape}")

eval_pairs = pd.read_csv(DATA / "evaluation_pairs.csv")
print(f"Eval pairs: {eval_pairs.shape}")

# Build pair_lookup
pair_lookup = {}
for r in eval_pairs.itertuples():
    p1, p2 = sorted([r.player_1, r.player_2])
    pair_lookup[(p1, p2)] = r.pair_id

# Build pairs_df for fast lookup
pairs_df = pd.DataFrame({
    "p1": [k[0] for k in pair_lookup.keys()],
    "p2": [k[1] for k in pair_lookup.keys()],
    "pair_id": list(pair_lookup.values()),
})

needed_players = set(pairs_df.p1) | set(pairs_df.p2)
print(f"Needed players: {len(needed_players):,}")

# =========================================================================
# Load hands.parquet
# =========================================================================
print()
print("Loading hands.parquet...")
hands_pl = pl.read_parquet(DATA / "hands.parquet", columns=[
    "hand_id", "big_blind", "final_pot", "players_dealt"
])
hands_dict = {}
for row in hands_pl.iter_rows(named=True):
    hands_dict[row["hand_id"]] = (
        int(row["big_blind"]), int(row["final_pot"]), int(row["players_dealt"])
    )
del hands_pl
gc.collect()
print(f"hands_dict: {len(hands_dict):,}")

# =========================================================================
# Open output CSV file for streaming writes
# =========================================================================
out_csv_path = OUT / "pair_hand_scores.csv"
out_csv_file = open(out_csv_path, "w", newline="")
out_csv_writer = csv.writer(out_csv_file)
out_csv_writer.writerow(["pair_id", "hand_id", "transfer_score", "passive_score", "aggressive_score"])

# =========================================================================
# Stream seats, write scores directly
# =========================================================================
print()
print("="*80)
print("Streaming seats.parquet, writing scores to CSV")
print("="*80)

pf_seats = pq.ParquetFile(DATA / "seats.parquet")
print(f"seats.parquet: {pf_seats.metadata.num_rows:,} rows, {pf_seats.num_row_groups} row groups")

t_stream = time.time()
rows_written = 0

for rg_idx in range(pf_seats.num_row_groups):
    table = pf_seats.read_row_group(rg_idx, columns=[
        "hand_id", "player_id", "starting_stack", "total_contribution",
        "net_chips", "folded", "went_to_showdown", "won_share"
    ])
    df = table.to_pandas()
    df = df[df.player_id.isin(needed_players)]

    for hand_id, group in df.groupby("hand_id"):
        if hand_id not in hands_dict:
            continue
        big_blind, final_pot, players_dealt = hands_dict[hand_id]
        bb = max(big_blind, 1)

        # Build players_dict
        players_dict = {}
        for r in group.itertuples():
            players_dict[r.player_id] = (
                int(r.starting_stack), int(r.total_contribution), int(r.net_chips),
                bool(r.folded), bool(r.went_to_showdown),
                float(r.won_share) if pd.notna(r.won_share) else 0.0
            )

        if len(players_dict) < 2:
            continue

        sorted_players = sorted(players_dict.keys())
        for i in range(len(sorted_players)):
            p1 = sorted_players[i]
            s1 = players_dict[p1]
            for j in range(i+1, len(sorted_players)):
                p2 = sorted_players[j]
                key = (p1, p2)
                pid = pair_lookup.get(key)
                if pid is None:
                    continue
                s2 = players_dict[p2]
                stack1, contrib1, net1, folded1, showdown1, won1 = s1
                stack2, contrib2, net2, folded2, showdown2, won2 = s2

                # Compute scores
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

                out_csv_writer.writerow([pid, hand_id, transfer_score, passive_score, aggressive_score])
                rows_written += 1

    if (rg_idx + 1) % 20 == 0 or rg_idx == pf_seats.num_row_groups - 1:
        elapsed = time.time() - t_stream
        print(f"  row group {rg_idx+1}/{pf_seats.num_row_groups}: rows={rows_written:,}, elapsed={elapsed:.1f}s", flush=True)

out_csv_file.close()
print(f"Done in {time.time()-t_stream:.1f}s. Wrote {rows_written:,} rows to {out_csv_path}")
del hands_dict
gc.collect()

# Check file size
file_size_mb = out_csv_path.stat().st_size / 1e6
print(f"CSV file size: {file_size_mb:.1f} MB")

# =========================================================================
# Read back and find top-5 per pair per score
# =========================================================================
print()
print("="*80)
print("Reading scores and finding top-5 per pair per score")
print("="*80)

# Use polars to read the CSV efficiently
print("Loading pair_hand_scores.csv with polars...")
scores_pl = pl.read_csv(out_csv_path)
print(f"scores_pl: {scores_pl.shape}, mem={scores_pl.estimated_size('mb'):.1f}MB")

# Sort and take top 5 per pair per score
print("Finding top-5 per pair per score...")

# Top 5 by transfer_score
top_transfer = scores_pl.sort(["pair_id", "transfer_score"], descending=[False, True]).group_by("pair_id", maintain_order=True).head(5)
print(f"top_transfer: {top_transfer.shape}")

# Top 5 by passive_score
top_passive = scores_pl.sort(["pair_id", "passive_score"], descending=[False, True]).group_by("pair_id", maintain_order=True).head(5)
print(f"top_passive: {top_passive.shape}")

# Top 5 by aggressive_score
top_aggressive = scores_pl.sort(["pair_id", "aggressive_score"], descending=[False, True]).group_by("pair_id", maintain_order=True).head(5)
print(f"top_aggressive: {top_aggressive.shape}")

# Convert to dicts
top_transfer_pd = top_transfer.to_pandas()
top_passive_pd = top_passive.to_pandas()
top_aggressive_pd = top_aggressive.to_pandas()

# Per pair: list of hand_ids (in order of score desc)
evidence_a = {}  # by transfer
evidence_passive = {}  # by passive
evidence_aggressive = {}  # by aggressive

for r in top_transfer_pd.itertuples():
    evidence_a.setdefault(r.pair_id, []).append(r.hand_id)
for r in top_passive_pd.itertuples():
    evidence_passive.setdefault(r.pair_id, []).append(r.hand_id)
for r in top_aggressive_pd.itertuples():
    evidence_aggressive.setdefault(r.pair_id, []).append(r.hand_id)

print(f"\nPairs with evidence_a (transfer): {len(evidence_a):,}")
print(f"Pairs with evidence_passive: {len(evidence_passive):,}")
print(f"Pairs with evidence_aggressive: {len(evidence_aggressive):,}")

del scores_pl, top_transfer, top_passive, top_aggressive, top_transfer_pd, top_passive_pd, top_aggressive_pd
gc.collect()

# =========================================================================
# Build SUBMISSION A — Chip-transfer evidence (behavior-agnostic)
# =========================================================================
print()
print("="*80)
print("SUBMISSION A — Chip-transfer evidence")
print("="*80)

sub_a = user_sub.copy()
for i, row in sub_a.iterrows():
    pid = row.pair_id
    top5 = evidence_a.get(pid, [])
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
pred_beh_map = user_sub.set_index('pair_id')['predicted_behavior'].to_dict()

for i, row in sub_b.iterrows():
    pid = row.pair_id
    beh = pred_beh_map.get(pid, 'directed_transfer')
    if beh == 'soft_play':
        top5 = evidence_passive.get(pid, [])
    elif beh == 'coordinated_isolation':
        top5 = evidence_aggressive.get(pid, [])
        if not top5:
            top5 = evidence_a.get(pid, [])
    else:  # directed_transfer or other
        top5 = evidence_a.get(pid, [])

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
    orig_set = {row.evidence_hand_1, row.evidence_hand_2, row.evidence_hand_3, row.evidence_hand_4, row.evidence_hand_5}
    if pid in evidence_a:
        overlap_a += len(orig_set & set(evidence_a[pid]))
    # Compare to B
    beh = pred_beh_map.get(pid, 'directed_transfer')
    if beh == 'soft_play':
        b_ev = evidence_passive.get(pid, [])
    elif beh == 'coordinated_isolation':
        b_ev = evidence_aggressive.get(pid, []) or evidence_a.get(pid, [])
    else:
        b_ev = evidence_a.get(pid, [])
    overlap_b += len(orig_set & set(b_ev))

print(f"Total pairs: {len(user_sub)}")
print(f"Average overlap with original (per pair):")
print(f"  A (chip transfer): {overlap_a/len(user_sub):.2f} / 5.0 ({100*overlap_a/len(user_sub)/5:.1f}%)")
print(f"  B (behavior-specific): {overlap_b/len(user_sub):.2f} / 5.0 ({100*overlap_b/len(user_sub)/5:.1f}%)")

print()
print("Files saved:")
print(f"  - submission_A_evidence_chip_transfer.csv")
print(f"  - submission_B_evidence_behavior_specific.csv")
print(f"\nDone in {time.time()-t0:.1f}s")
