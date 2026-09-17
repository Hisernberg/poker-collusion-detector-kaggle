"""
Poker Collusion Detector v2 — Memory-Efficient Streaming Version
================================================================
Memory budget: 4GB RAM, 2 CPUs.

Strategy:
  PASS 1: Stream seats.parquet row-group by row-group, filter to needed players,
          record (hand_id, player_id, seat_data) only for relevant hands.
  PASS 2: Stream actions.parquet similarly, filter to relevant hand_ids.
  PASS 3: For each relevant hand, compute per-pair-per-hand features in-memory.
  PASS 4: Aggregate to pair-level features, train LightGBM, predict, build submission.

Uses Polars for efficient streaming and filtering.
"""
import os, gc, time, json, sys, warnings, pickle
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import polars as pl
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

# Configure polars for low memory (polars 1.x uses streaming chunk size via env var)
os.environ["POLARS_STREAMING_CHUNK_SIZE"] = "50000"

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

DATA = Path("/home/z/my-project/kaggle/comp-data")
OUT  = Path("/home/z/my-project/kaggle/runs")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260917
N_FOLDS = 5
np.random.seed(SEED)

print("="*80)
print("STAGE 0 — Loading small CSVs + building pair indices")
print("="*80)
t0 = time.time()

dev_labels = pd.read_csv(DATA / "development_labels.csv")
dev_evidence = pd.read_csv(DATA / "development_evidence.csv")
eval_pairs  = pd.read_csv(DATA / "evaluation_pairs.csv")
sample_sub  = pd.read_csv(DATA / "sample_submission.csv")
players_df  = pd.read_parquet(DATA / "players.parquet")

print(f"dev_labels: {dev_labels.shape}  ({dev_labels.label.sum()} positives)")
print(f"eval_pairs: {eval_pairs.shape}")

# Build pair_lookup: frozenset({p1, p2}) -> pair_id
pair_lookup = {}
pair_to_players = {}
for r in dev_labels.itertuples():
    pair_lookup[frozenset((r.player_1, r.player_2))] = r.pair_id
    pair_to_players[r.pair_id] = (r.player_1, r.player_2)
for r in eval_pairs.itertuples():
    pair_lookup[frozenset((r.player_1, r.player_2))] = r.pair_id
    pair_to_players[r.pair_id] = (r.player_1, r.player_2)
print(f"Total pairs: {len(pair_lookup):,}")

# Players needed
needed_players = set()
for p1, p2 in pair_to_players.values():
    needed_players.add(p1); needed_players.add(p2)
print(f"Players needed: {len(needed_players):,}")

# Gold evidence
gold_evidence = defaultdict(set)
for r in dev_evidence.itertuples():
    gold_evidence[r.pair_id].add(r.hand_id)

# Gold family
gold_family = {r.pair_id: r.behavior_family for r in dev_labels.itertuples() if r.label == 1}

print(f"Setup time: {time.time()-t0:.1f}s")

# =========================================================================
# PASS 1 — Stream seats, filter by player, accumulate per-pair hand records
# =========================================================================
print()
print("="*80)
print("PASS 1 — Stream seats.parquet, filter by needed players")
print("="*80)

# Use Polars lazy scan for efficient streaming
seats_scan = pl.scan_parquet(DATA / "seats.parquet")
print("Schema:", seats_scan.collect_schema())

# Filter to needed players only, collect
needed_players_list = list(needed_players)
t_seats = time.time()
seats_filtered = seats_scan.filter(
    pl.col("player_id").is_in(needed_players_list)
).select([
    "hand_id", "player_id", "seat_no", "starting_stack",
    "total_contribution", "net_chips", "folded", "went_to_showdown", "won_share"
]).collect(streaming=True)

print(f"seats filtered: {seats_filtered.shape}, time={time.time()-t_seats:.1f}s, mem={seats_filtered.estimated_size('mb'):.1f}MB")

# Free seats_filtered (we only need seats_full)
del seats_filtered
gc.collect()

# We need hole_card_1 and hole_card_2 too for hand-strength features
seats_scan_full = pl.scan_parquet(DATA / "seats.parquet")
seats_full = seats_scan_full.filter(
    pl.col("player_id").is_in(needed_players_list)
).select([
    "hand_id", "player_id", "hole_card_1", "hole_card_2",
    "starting_stack", "total_contribution", "net_chips", "folded",
    "went_to_showdown", "won_share"
]).collect(streaming=True)
print(f"seats full (with hole cards): {seats_full.shape}, mem={seats_full.estimated_size('mb'):.1f}MB")

# Now we know needed hand_ids (from seats_full)
needed_hand_ids = set(seats_full["hand_id"].unique().to_list())
print(f"Needed hand IDs: {len(needed_hand_ids):,}")

# =========================================================================
# Load hands.parquet (full — only 48MB, 2M rows, fits easily)
# =========================================================================
print()
print("Loading hands.parquet...")
hands = pl.read_parquet(DATA / "hands.parquet", columns=[
    "hand_id", "phase", "small_blind", "big_blind",
    "board_cards", "final_pot", "players_at_showdown"
])
print(f"hands: {hands.shape}, mem={hands.estimated_size('mb'):.1f}MB")

# =========================================================================
# PASS 2 — Stream actions.parquet, filter by needed hand_id
# =========================================================================
print()
print("="*80)
print("PASS 2 — Stream actions.parquet, filter by needed hand_ids")
print("="*80)

# Use Polars lazy scan with predicate pushdown
needed_hand_ids_list = list(needed_hand_ids)
t_actions = time.time()
actions = pl.scan_parquet(DATA / "actions.parquet").filter(
    pl.col("hand_id").is_in(needed_hand_ids_list)
).select([
    "hand_id", "action_no", "street", "player_id", "action",
    "amount", "amount_to", "pot_before", "stack_before", "to_call", "players_active"
]).collect(streaming=True)
print(f"actions filtered: {actions.shape}, time={time.time()-t_actions:.1f}s, mem={actions.estimated_size('mb'):.1f}MB")

# =========================================================================
# PASS 3 — Per-pair-per-hand feature extraction
# =========================================================================
print()
print("="*80)
print("PASS 3 — Per-pair-per-hand feature extraction")
print("="*80)

# Convert to pandas (filtered datasets should fit)
seats_pd = seats_full.to_pandas()
hands_pd = hands.to_pandas()
actions_pd = actions.to_pandas()

# Free polars memory
del seats_full, hands, actions
gc.collect()

print(f"Conversions done. seats_pd={seats_pd.shape}, hands_pd={hands_pd.shape}, actions_pd={actions_pd.shape}")

# Group by hand_id for fast iteration
seats_by_hand = {hand_id: g for hand_id, g in seats_pd.groupby("hand_id")}
actions_by_hand = {hand_id: g for hand_id, g in actions_pd.groupby("hand_id")}
print(f"Built per-hand groups: {len(seats_by_hand)} hand-seats, {len(actions_by_hand)} hand-actions")

# Free the ungrouped DataFrames (keep only grouped dicts)
del seats_pd, actions_pd
gc.collect()

# Pre-index hands
hands_indexed = hands_pd.set_index("hand_id")
del hands_pd
gc.collect()

# For each hand, enumerate pairs of players and look up pair_id
# Build per-hand -> set of pair_ids
t_pair_hand = time.time()
print("Building pair -> hands index...")
pair_hands = defaultdict(list)
pair_hand_player = {}  # (pair_id, hand_id) -> (p1, p2)

for hand_id, seat_df in seats_by_hand.items():
    players_in_hand = set(seat_df.player_id)
    if len(players_in_hand) < 2:
        continue
    players_list = list(players_in_hand)
    for i in range(len(players_list)):
        for j in range(i+1, len(players_list)):
            key = frozenset((players_list[i], players_list[j]))
            pid = pair_lookup.get(key)
            if pid is not None:
                pair_hands[pid].append(hand_id)
                pair_hand_player[(pid, hand_id)] = (players_list[i], players_list[j])

print(f"pair-hands index: {len(pair_hands):,} pairs have shared hands, time={time.time()-t_pair_hand:.1f}s")
total_pair_hand_records = sum(len(v) for v in pair_hands.values())
print(f"Total (pair, hand) records to process: {total_pair_hand_records:,}")

# Save the index
pair_hand_index = []
for pid, hand_list in pair_hands.items():
    for h in hand_list:
        pair_hand_index.append({"pair_id": pid, "hand_id": h, "p1": pair_hand_player[(pid,h)][0], "p2": pair_hand_player[(pid,h)][1]})
pair_hand_index_df = pd.DataFrame(pair_hand_index)
pair_hand_index_df.to_parquet(OUT / "pair_hand_index.parquet")
print(f"Saved pair_hand_index: {pair_hand_index_df.shape}")
del pair_hand_index, pair_hand_index_df
gc.collect()

# Card rank dict
RANK_DICT = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}

def eval_hand_7(cards):
    """Return integer score (higher = better) for 5-7 cards."""
    if not cards: return 0
    ranks = sorted((RANK_DICT.get(c[0], 2) for c in cards), reverse=True)
    suits = {}
    for c in cards:
        suits.setdefault(c[1], []).append(RANK_DICT.get(c[0], 2))

    flush_suit = None
    for s, rs in suits.items():
        if len(rs) >= 5:
            flush_suit = s
            break

    def straight_high(rankset):
        s = set(rankset)
        if 14 in s: s.add(1)
        for hi in range(14, 4, -1):
            if all(hi - k in s for k in range(5)):
                return hi
        return 0

    if flush_suit:
        sh = straight_high(suits[flush_suit])
        if sh:
            return 8 * 15**5 + sh
        f = sorted(suits[flush_suit], reverse=True)[:5]
        return 5 * 15**5 + sum(v * 15**(4 - i) for i, v in enumerate(f))

    from collections import Counter
    cnt = Counter(ranks)
    groups = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)

    if groups[0][1] == 4:
        k = max(r for r in ranks if r != groups[0][0])
        return 7 * 15**5 + groups[0][0] * 15 + k
    if groups[0][1] == 3 and len(groups) > 1 and groups[1][1] >= 2:
        return 6 * 15**5 + groups[0][0] * 15 + groups[1][0]
    sh = straight_high(ranks)
    if sh:
        return 4 * 15**5 + sh
    if groups[0][1] == 3:
        ks = [r for r in ranks if r != groups[0][0]][:2]
        return 3 * 15**5 + groups[0][0] * 225 + ks[0] * 15 + ks[1]
    if groups[0][1] == 2 and len(groups) > 1 and groups[1][1] == 2:
        k = max(r for r in ranks if r not in (groups[0][0], groups[1][0]))
        return 2 * 15**5 + groups[0][0] * 225 + groups[1][0] * 15 + k
    if groups[0][1] == 2:
        ks = [r for r in ranks if r != groups[0][0]][:3]
        return 1 * 15**5 + groups[0][0] * 3375 + sum(v * 15**(2 - i) for i, v in enumerate(ks))
    return sum(v * 15**(4 - i) for i, v in enumerate(ranks[:5]))

def preflop_score(c1, c2):
    """Quick preflop strength score."""
    if not c1 or not c2: return 0
    a, b = sorted((RANK_DICT.get(c1[0], 2), RANK_DICT.get(c2[0], 2)), reverse=True)
    s = a * 2 + b + (20 if a == b else 0) + (3 if c1[1] == c2[1] else 0) + (2 if a - b == 1 else 0)
    return s

# =========================================================================
# Per-pair-per-hand feature extraction (vectorized where possible)
# =========================================================================
# Process in chunks of pairs (each chunk = 1000 pairs)
print()
print("Computing per-pair-per-hand features...")
t_feat = time.time()
pair_hand_features = []

CHUNK_SIZE = 5000
all_pair_ids = list(pair_hands.keys())
n_chunks = (len(all_pair_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE
processed_pairs = 0

for chunk_idx in range(n_chunks):
    chunk_pairs = all_pair_ids[chunk_idx*CHUNK_SIZE:(chunk_idx+1)*CHUNK_SIZE]
    for pid in chunk_pairs:
        p1, p2 = pair_to_players[pid]
        for hand_id in pair_hands[pid]:
            key = (pid, hand_id)
            if key not in pair_hand_player:
                continue
            hp1, hp2 = pair_hand_player[key]
            # Get hand info
            if hand_id not in hands_index: continue
            hand_row = hands_index.loc[hand_id]
            # Get seats
            seat_df = seats_by_hand.get(hand_id)
            if seat_df is None: continue
            if hp1 not in set(seat_df.player_id) or hp2 not in set(seat_df.player_id):
                continue
            s1 = seat_df[seat_df.player_id == hp1].iloc[0]
            s2 = seat_df[seat_df.player_id == hp2].iloc[0]
            # Get actions
            action_df = actions_by_hand.get(hand_id)
            if action_df is None: action_df = pd.DataFrame()

            big_blind = max(int(hand_row.big_blind), 1)
            board_cards = hand_row.board_cards.split() if isinstance(hand_row.board_cards, str) else []
            phase = hand_row.phase

            # Pre-compute hole-card strengths (if available)
            hole1 = (s1.hole_card_1, s1.hole_card_2) if isinstance(s1.hole_card_1, str) else (None, None)
            hole2 = (s2.hole_card_1, s2.hole_card_2) if isinstance(s2.hole_card_1, str) else (None, None)

            # Per-hand features
            f = {
                "pair_id": pid,
                "hand_id": hand_id,
                "phase": phase,
                "n_hands_in_pair": len(pair_hands[pid]),
                "p1_vpip": 0, "p2_vpip": 0, "p1_pfr": 0, "p2_pfr": 0,
                "p1_fold_to_p2": 0, "p2_fold_to_p1": 0,
                "p1_calls_p2": 0, "p2_calls_p1": 0,
                "p1_raises_p2": 0, "p2_raises_p1": 0,
                "hu_checkdowns": 0, "both_postflop": 0,
                "sandwich_pf": 0, "third_folds_after_pq_raises": 0,
                "p1_aggression": 0, "p2_aggression": 0,
                "p1_passivity_vs_p2": 0, "p2_passivity_vs_p1": 0,
                "n_p1_actions": 0, "n_p2_actions": 0,
                "p1_fold_with_strong_vs_p2": 0, "p2_fold_with_strong_vs_p1": 0,
                "p1_check_strong_hu": 0, "p2_check_strong_hu": 0,
                "net1_bb": float(s1.net_chips) / big_blind,
                "net2_bb": float(s2.net_chips) / big_blind,
                "abs_net_diff_bb": abs(float(s1.net_chips) - float(s2.net_chips)) / big_blind,
                "pot_bb": float(hand_row.final_pot) / big_blind,
                "both_went_showdown": int(bool(s1.went_to_showdown) and bool(s2.went_to_showdown)),
                "p1_contribution_bb": float(s1.total_contribution) / big_blind,
                "p2_contribution_bb": float(s2.total_contribution) / big_blind,
                "p1_won_share": float(s1.won_share or 0.0),
                "p2_won_share": float(s2.won_share or 0.0),
                "preflop_strength_p1": preflop_score(*hole1) if hole1[0] else 0,
                "preflop_strength_p2": preflop_score(*hole2) if hole2[0] else 0,
                "preflop_strength_diff": abs(preflop_score(*hole1) - preflop_score(*hole2)) if (hole1[0] and hole2[0]) else 0,
            }

            # Process actions
            last_agg = None
            street = None
            folded = set()
            invest = defaultdict(int)
            raisers_pf = []
            postflop_players = set()
            actions_seq = action_df.values.tolist() if len(action_df) else []
            # columns: hand_id, action_no, street, player_id, action, amount, amount_to, pot_before, stack_before, to_call, players_active

            for row in actions_seq:
                _, _, r_street, r_pid, r_act, r_amt, _, _, _, _, r_pact = row
                if r_street != street:
                    street = r_street
                    last_agg = None
                if r_street != "preflop":
                    postflop_players.add(r_pid)

                is_p1 = (r_pid == hp1)
                is_p2 = (r_pid == hp2)
                other = hp2 if is_p1 else (hp1 if is_p2 else None)

                if r_street == "preflop":
                    if r_act in ("call","bet","raise","all_in"):
                        if is_p1: f["p1_vpip"] = 1
                        elif is_p2: f["p2_vpip"] = 1
                    if r_act in ("bet","raise","all_in"):
                        raisers_pf.append(r_pid)
                        if is_p1: f["p1_pfr"] = 1
                        elif is_p2: f["p2_pfr"] = 1

                if is_p1:
                    f["n_p1_actions"] += 1
                    if r_act in ("bet","raise","all_in"): f["p1_aggression"] += 1
                elif is_p2:
                    f["n_p2_actions"] += 1
                    if r_act in ("bet","raise","all_in"): f["p2_aggression"] += 1

                if r_act == "fold":
                    folded.add(r_pid)
                    if other is not None and last_agg == other:
                        # Detect strong-hand fold
                        invest_bb = invest[r_pid] / big_blind
                        if r_street == "preflop":
                            str_score = preflop_score(*hole1) if is_p1 and hole1[0] else (preflop_score(*hole2) if is_p2 and hole2[0] else 0)
                            if str_score >= 18:  # strong hand threshold
                                if is_p1: f["p1_fold_with_strong_vs_p2"] += 1; f["p1_passivity_vs_p2"] += 1
                                elif is_p2: f["p2_fold_with_strong_vs_p1"] += 1; f["p2_passivity_vs_p1"] += 1
                            else:
                                if is_p1: f["p1_passivity_vs_p2"] += 1
                                elif is_p2: f["p2_passivity_vs_p1"] += 1
                        else:
                            if is_p1: f["p1_passivity_vs_p2"] += 1
                            elif is_p2: f["p2_passivity_vs_p1"] += 1

                if r_act == "call" and last_agg == other:
                    if is_p1: f["p1_calls_p2"] += 1
                    elif is_p2: f["p2_calls_p1"] += 1
                if r_act in ("bet","raise","all_in") and last_agg == other:
                    if is_p1: f["p1_raises_p2"] += 1
                    elif is_p2: f["p2_raises_p1"] += 1

                if r_act == "check" and r_street != "preflop" and r_pact == 2:
                    if hp1 not in folded and hp2 not in folded:
                        # HU check — check if strong
                        if is_p1:
                            nb = {"flop":3,"turn":4,"river":5}.get(r_street, 0)
                            if nb and hole1[0] and len(board_cards) >= nb:
                                hs = eval_hand_7([hole1[0], hole1[1]] + board_cards[:nb])
                                if hs // 15**5 >= 2:  # pair or better
                                    f["p1_check_strong_hu"] += 1
                        elif is_p2:
                            nb = {"flop":3,"turn":4,"river":5}.get(r_street, 0)
                            if nb and hole2[0] and len(board_cards) >= nb:
                                hs = eval_hand_7([hole2[0], hole2[1]] + board_cards[:nb])
                                if hs // 15**5 >= 2:
                                    f["p2_check_strong_hu"] += 1

                if r_act == "fold" and (not is_p1) and (not is_p2):
                    if hp1 in raisers_pf and hp2 in raisers_pf and r_street == "preflop":
                        f["third_folds_after_pq_raises"] += 1

                invest[r_pid] += r_amt
                if r_act in ("bet","raise","all_in"):
                    last_agg = r_pid

            if hp1 in raisers_pf and hp2 in raisers_pf and len(set(raisers_pf)) >= 3:
                f["sandwich_pf"] = 1
            f["both_postflop"] = int(hp1 in postflop_players and hp2 in postflop_players)

            # HU checkdown detection: limped pot + both reached showdown with no aggression
            if s1.went_to_showdown and s2.went_to_showdown and len(raisers_pf) == 0 and f["p1_aggression"] == 0 and f["p2_aggression"] == 0:
                f["hu_checkdowns"] = 1
            # Also: passive vs each other (no raises between them postflop)
            if f["p1_raises_p2"] == 0 and f["p2_raises_p1"] == 0 and f["both_postflop"] == 1:
                f["hu_checkdowns"] = max(f["hu_checkdowns"], 1)

            pair_hand_features.append(f)

    processed_pairs += len(chunk_pairs)
    if (chunk_idx + 1) % 5 == 0 or processed_pairs >= len(all_pair_ids):
        elapsed = time.time() - t_feat
        rate = processed_pairs / max(elapsed, 1)
        eta = (len(all_pair_ids) - processed_pairs) / max(rate, 1)
        print(f"  processed {processed_pairs:,}/{len(all_pair_ids):,} pairs ({len(pair_hand_features):,} features), elapsed={elapsed:.0f}s, ETA={eta:.0f}s", flush=True)

print(f"Total: {len(pair_hand_features):,} pair-hand features in {time.time()-t_feat:.1f}s")

phf = pd.DataFrame(pair_hand_features)
del pair_hand_features
gc.collect()
print(f"phf shape: {phf.shape}, mem={phf.memory_usage(deep=True).sum()/1e6:.1f}MB")
phf.to_parquet(OUT / "pair_hand_features.parquet")
print("Saved pair_hand_features.parquet")

# =========================================================================
# STAGE A2 — Aggregate to pair-level features
# =========================================================================
print()
print("="*80)
print("STAGE A2 — Aggregating pair-level features")
print("="*80)

# Numeric aggregations
agg_dict = {
    "n_hands_in_pair": ["first"],
    "p1_vpip": ["sum", "mean"],
    "p2_vpip": ["sum", "mean"],
    "p1_pfr": ["sum", "mean"],
    "p2_pfr": ["sum", "mean"],
    "p1_fold_to_p2": ["sum", "mean"],
    "p2_fold_to_p1": ["sum", "mean"],
    "p1_calls_p2": ["sum", "mean"],
    "p2_calls_p1": ["sum", "mean"],
    "p1_raises_p2": ["sum", "mean"],
    "p2_raises_p1": ["sum", "mean"],
    "hu_checkdowns": ["sum", "mean"],
    "sandwich_pf": ["sum", "mean"],
    "third_folds_after_pq_raises": ["sum", "mean"],
    "both_postflop": ["sum", "mean"],
    "p1_aggression": ["sum", "mean"],
    "p2_aggression": ["sum", "mean"],
    "p1_passivity_vs_p2": ["sum", "mean"],
    "p2_passivity_vs_p1": ["sum", "mean"],
    "n_p1_actions": ["sum", "mean"],
    "n_p2_actions": ["sum", "mean"],
    "p1_fold_with_strong_vs_p2": ["sum", "mean"],
    "p2_fold_with_strong_vs_p1": ["sum", "mean"],
    "p1_check_strong_hu": ["sum", "mean"],
    "p2_check_strong_hu": ["sum", "mean"],
    "net1_bb": ["sum", "mean", "max", "min", "std"],
    "net2_bb": ["sum", "mean", "max", "min", "std"],
    "abs_net_diff_bb": ["mean", "max", "std"],
    "pot_bb": ["mean", "max", "sum"],
    "both_went_showdown": ["sum", "mean"],
    "p1_contribution_bb": ["sum", "mean"],
    "p2_contribution_bb": ["sum", "mean"],
    "p1_won_share": ["sum", "mean"],
    "p2_won_share": ["sum", "mean"],
    "preflop_strength_p1": ["mean"],
    "preflop_strength_p2": ["mean"],
    "preflop_strength_diff": ["mean"],
}

pair_features = phf.groupby("pair_id").agg(agg_dict)
# Flatten multi-index columns
pair_features.columns = ["_".join(c) if isinstance(c, tuple) else c for c in pair_features.columns]
pair_features = pair_features.reset_index()
print(f"pair_features (raw): {pair_features.shape}")

# Derived features
pair_features["fold_to_partner_total"] = pair_features["p1_fold_to_p2_sum"] + pair_features["p2_fold_to_p1_sum"]
pair_features["fold_to_partner_rate"] = pair_features["fold_to_partner_total"] / pair_features["n_hands_in_pair_first"].clip(lower=1)
pair_features["hu_checkdown_rate"] = pair_features["hu_checkdowns_sum"] / pair_features["n_hands_in_pair_first"].clip(lower=1)
pair_features["sandwich_rate"] = pair_features["sandwich_pf_sum"] / pair_features["n_hands_in_pair_first"].clip(lower=1)
pair_features["third_fold_rate"] = pair_features["third_folds_after_pq_raises_sum"] / pair_features["n_hands_in_pair_first"].clip(lower=1)
pair_features["net_asymmetry_sum"] = (pair_features["net1_bb_sum"] - pair_features["net2_bb_sum"]).abs()
pair_features["net_asymmetry_mean"] = (pair_features["net1_bb_mean"] - pair_features["net2_bb_mean"]).abs()
pair_features["chips_transferred_bb"] = pair_features["net_asymmetry_sum"]
pair_features["contribution_asymmetry"] = (pair_features["p1_contribution_bb_sum"] - pair_features["p2_contribution_bb_sum"]).abs()
pair_features["aggression_asymmetry"] = (pair_features["p1_aggression_sum"] - pair_features["p2_aggression_sum"]).abs()
pair_features["passivity_total"] = pair_features["p1_passivity_vs_p2_sum"] + pair_features["p2_passivity_vs_p1_sum"]
pair_features["passivity_rate"] = pair_features["passivity_total"] / (pair_features["n_p1_actions_sum"] + pair_features["n_p2_actions_sum"]).clip(lower=1)
pair_features["strong_fold_total"] = pair_features["p1_fold_with_strong_vs_p2_sum"] + pair_features["p2_fold_with_strong_vs_p1_sum"]
pair_features["strong_fold_rate"] = pair_features["strong_fold_total"] / pair_features["n_hands_in_pair_first"].clip(lower=1)
pair_features["check_strong_total"] = pair_features["p1_check_strong_hu_sum"] + pair_features["p2_check_strong_hu_sum"]
pair_features["check_strong_rate"] = pair_features["check_strong_total"] / pair_features["n_hands_in_pair_first"].clip(lower=1)
pair_features["raise_vs_total"] = pair_features["p1_raises_p2_sum"] + pair_features["p2_raises_p1_sum"]
pair_features["calls_vs_total"] = pair_features["p1_calls_p2_sum"] + pair_features["p2_calls_p1_sum"]
pair_features["action_ratio_calls_raises"] = pair_features["calls_vs_total"] / (pair_features["raise_vs_total"] + 1)

# Add player profile features
pair_features = pair_features.merge(
    dev_labels[["pair_id","player_1","player_2"]].rename(columns={"player_1":"p1","player_2":"p2"}),
    on="pair_id", how="left"
)
# For eval pairs not in dev, fill from eval_pairs
eval_player_map = eval_pairs[["pair_id","player_1","player_2"]].rename(columns={"player_1":"p1","player_2":"p2"})
mask = pair_features["p1"].isna()
if mask.any():
    pair_features.loc[mask, ["p1","p2"]] = pair_features.loc[mask, "pair_id"].map(eval_player_map.set_index("pair_id")["p1"]).values, pair_features.loc[mask, "pair_id"].map(eval_player_map.set_index("pair_id")["p2"]).values
# Actually let me do it more carefully
pair_features = pair_features.drop(columns=["p1","p2"], errors="ignore")
all_player_map = pd.concat([
    dev_labels[["pair_id","player_1","player_2"]].rename(columns={"player_1":"p1","player_2":"p2"}),
    eval_pairs[["pair_id","player_1","player_2"]].rename(columns={"player_1":"p1","player_2":"p2"})
]).drop_duplicates(subset=["pair_id"])
pair_features = pair_features.merge(all_player_map, on="pair_id", how="left")
print(f"pair_features (with player ids): {pair_features.shape}")

# Player profile features (encode categoricals as codes)
players_df_p = players_df.copy()
for c in ["experience_hands_bucket","preferred_stake","region_bucket","client_family"]:
    players_df_p[c+"_code"] = players_df_p[c].astype("category").cat.codes

p1_prof = players_df_p.rename(columns={"player_id":"p1","account_age_days":"p1_account_age_days","experience_hands_bucket_code":"p1_exp_code","preferred_stake_code":"p1_stake_code","region_bucket_code":"p1_region_code","client_family_code":"p1_client_code"})[["p1","p1_account_age_days","p1_exp_code","p1_stake_code","p1_region_code","p1_client_code"]]
p2_prof = players_df_p.rename(columns={"player_id":"p2","account_age_days":"p2_account_age_days","experience_hands_bucket_code":"p2_exp_code","preferred_stake_code":"p2_stake_code","region_bucket_code":"p2_region_code","client_family_code":"p2_client_code"})[["p2","p2_account_age_days","p2_exp_code","p2_stake_code","p2_region_code","p2_client_code"]]

pair_features = pair_features.merge(p1_prof, on="p1", how="left")
pair_features = pair_features.merge(p2_prof, on="p2", how="left")

# Same-region, same-stake indicators
pair_features["same_region"] = (pair_features["p1_region_code"] == pair_features["p2_region_code"]).astype(int)
pair_features["same_stake"] = (pair_features["p1_stake_code"] == pair_features["p2_stake_code"]).astype(int)
pair_features["same_client"] = (pair_features["p1_client_code"] == pair_features["p2_client_code"]).astype(int)
pair_features["account_age_diff"] = (pair_features["p1_account_age_days"] - pair_features["p2_account_age_days"]).abs()

# Add shared_hands from eval_pairs (already in evaluation_pairs.csv)
pair_features = pair_features.merge(eval_pairs[["pair_id","shared_hands"]], on="pair_id", how="left")
pair_features["shared_hands"] = pair_features["shared_hands"].fillna(pair_features["n_hands_in_pair_first"])

# Save features
pair_features.to_parquet(OUT / "pair_features.parquet")
print(f"Saved pair_features: {pair_features.shape}")
print(f"Total time so far: {time.time()-t0:.1f}s")

# =========================================================================
# STAGE B — Train models
# =========================================================================
print()
print("="*80)
print("STAGE B — Train LightGBM models")
print("="*80)

# Merge label info
train_df = pair_features.merge(dev_labels[["pair_id","label","behavior_family"]], on="pair_id", how="inner")
print(f"Train pairs: {train_df.shape}, positives: {int(train_df.label.sum())}")

# Feature columns
drop_cols = {"pair_id","p1","p2","label","behavior_family"}
feat_cols = [c for c in train_df.columns if c not in drop_cols and train_df[c].dtype != object and train_df[c].dtype != "string"]
print(f"# feature columns: {len(feat_cols)}")

# Impute NaN
for c in feat_cols:
    if train_df[c].isna().any():
        train_df[c] = train_df[c].fillna(train_df[c].median() if not train_df[c].isna().all() else 0)

X = train_df[feat_cols].values.astype(np.float32)
y = train_df.label.values.astype(int)
families = train_df.behavior_family.values

y_dt = (families == "directed_transfer").astype(int)
y_sp = (families == "soft_play").astype(int)
y_ci = (families == "coordinated_isolation").astype(int)

# 5-fold CV
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def lgb_params(num_leaves=31, n_estimators=300):
    return dict(
        objective="binary",
        num_leaves=num_leaves,
        learning_rate=0.05,
        n_estimators=n_estimators,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=SEED,
        n_jobs=2,
        verbose=-1,
    )

def train_family_models(y_family, family_name):
    print(f"\n--- Training {family_name} models ---")
    oof = np.zeros(len(train_df))
    fold_scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y_family)):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y_family[tr], y_family[va]
        model = lgb.LGBMClassifier(**lgb_params())
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
        oof[va] = model.predict_proba(X_va)[:, 1]
        if y_va.sum() > 0:
            score = average_precision_score(y_va, oof[va])
        else:
            score = 0.0
        fold_scores.append(score)
        print(f"  fold {fold}: AP={score:.4f}, best_iter={model.best_iteration_}")
    print(f"  Mean AP: {np.mean(fold_scores):.4f}")
    return oof

oof_dt = train_family_models(y_dt, "directed_transfer")
oof_sp = train_family_models(y_sp, "soft_play")
oof_ci = train_family_models(y_ci, "coordinated_isolation")
oof_risk = train_family_models(y, "any_suspicious")

# Composite risk OOF
oof_combined = np.maximum.reduce([oof_dt, oof_sp, oof_ci, oof_risk * 0.7])
overall_ap = average_precision_score(y, oof_combined)
print(f"\nOverall OOF AP (composite risk): {overall_ap:.4f}")

# Per-family AP
for fam, oof_arr in [("directed_transfer", oof_dt), ("soft_play", oof_sp), ("coordinated_isolation", oof_ci)]:
    y_fam = (families == fam).astype(int)
    if y_fam.sum() > 0:
        fam_ap = average_precision_score(y_fam, oof_arr)
        print(f"  {fam} OOF AP: {fam_ap:.4f}")

# Train final models on full data
print("\n--- Training final models on full data ---")
final_dt = lgb.LGBMClassifier(**lgb_params())
final_dt.fit(X, y_dt)
final_sp = lgb.LGBMClassifier(**lgb_params())
final_sp.fit(X, y_sp)
final_ci = lgb.LGBMClassifier(**lgb_params())
final_ci.fit(X, y_ci)
final_risk = lgb.LGBMClassifier(**lgb_params())
final_risk.fit(X, y)

# Save models
with open(OUT / "models.pkl", "wb") as f:
    pickle.dump({"dt":final_dt, "sp":final_sp, "ci":final_ci, "risk":final_risk, "feat_cols":feat_cols}, f)
print("Saved models")

# =========================================================================
# STAGE C — Predict on evaluation pairs + evidence selection
# =========================================================================
print()
print("="*80)
print("STAGE C — Predict on evaluation pairs")
print("="*80)

eval_df = pair_features[pair_features.pair_id.isin(set(eval_pairs.pair_id))].copy()
print(f"Eval pairs in features: {eval_df.shape}")
assert len(eval_df) == len(eval_pairs), f"Mismatch: {len(eval_df)} vs {len(eval_pairs)}"

# Impute
for c in feat_cols:
    if eval_df[c].isna().any():
        eval_df[c] = eval_df[c].fillna(train_df[c].median() if not train_df[c].isna().all() else 0)

X_eval = eval_df[feat_cols].values.astype(np.float32)

p_dt = final_dt.predict_proba(X_eval)[:, 1]
p_sp = final_sp.predict_proba(X_eval)[:, 1]
p_ci = final_ci.predict_proba(X_eval)[:, 1]
p_risk = final_risk.predict_proba(X_eval)[:, 1]

# Combined risk score
risk_combined = np.maximum.reduce([p_dt, p_sp, p_ci, p_risk * 0.7])

# Predicted behavior (argmax across families, with threshold)
behaviors = []
fam_names = ["directed_transfer", "soft_play", "coordinated_isolation"]
for i in range(len(eval_df)):
    probs = [p_dt[i], p_sp[i], p_ci[i]]
    fam_idx = int(np.argmax(probs))
    if max(probs) < 0.4 and p_risk[i] < 0.4:
        behaviors.append("none")
    else:
        behaviors.append(fam_names[fam_idx])
print(f"Predicted behavior distribution: {Counter(behaviors)}")

# Evidence selection: for each eval pair, score hands and pick top-5
print("\n--- Computing per-hand evidence scores ---")
# Use pair_hand_features for evidence selection
# Score each hand by combining behavioral signals
phf_for_eval = phf[phf.pair_id.isin(set(eval_pairs.pair_id))].copy()

# Per-hand suspicion score (weighted combination of behavioral signals)
phf_for_eval["suspicion_score"] = (
    (phf_for_eval["p1_fold_to_p2"] + phf_for_eval["p2_fold_to_p1"]) * 1.0 +
    phf_for_eval["hu_checkdowns"] * 2.0 +
    (phf_for_eval["sandwich_pf"] + phf_for_eval["third_folds_after_pq_raises"]) * 1.5 +
    phf_for_eval["abs_net_diff_bb"] * 0.3 +
    (phf_for_eval["p1_passivity_vs_p2"] + phf_for_eval["p2_passivity_vs_p1"]) * 0.5 +
    (phf_for_eval["p1_fold_with_strong_vs_p2"] + phf_for_eval["p2_fold_with_strong_vs_p1"]) * 2.0 +
    (phf_for_eval["p1_check_strong_hu"] + phf_for_eval["p2_check_strong_hu"]) * 1.5
).values

# For each eval pair, sort hands by suspicion score, take top 5
pair_to_hand_scores = defaultdict(list)
for r in phf_for_eval.itertuples():
    pair_to_hand_scores[r.pair_id].append((r.hand_id, r.suspicion_score))

# Build submission
sub_rows = []
eval_pair_id_to_idx = {pid: i for i, pid in enumerate(eval_df.pair_id.values)}

for r in eval_pairs.itertuples():
    pid = r.pair_id
    if pid not in eval_pair_id_to_idx:
        # Pair not in our features — use defaults
        sub_rows.append({
            "pair_id": pid,
            "risk_score": 0.0,
            "predicted_behavior": "none",
            "evidence_hand_1": "NO_EVIDENCE",
            "evidence_hand_2": "NO_EVIDENCE",
            "evidence_hand_3": "NO_EVIDENCE",
            "evidence_hand_4": "NO_EVIDENCE",
            "evidence_hand_5": "NO_EVIDENCE",
        })
        continue
    idx = eval_pair_id_to_idx[pid]
    hand_scores = pair_to_hand_scores.get(pid, [])
    hand_scores.sort(key=lambda x: -x[1])
    top5 = [h for h, s in hand_scores[:5]]
    while len(top5) < 5:
        top5.append("NO_EVIDENCE")
    sub_rows.append({
        "pair_id": pid,
        "risk_score": float(risk_combined[idx]),
        "predicted_behavior": behaviors[idx],
        "evidence_hand_1": top5[0],
        "evidence_hand_2": top5[1],
        "evidence_hand_3": top5[2],
        "evidence_hand_4": top5[3],
        "evidence_hand_5": top5[4],
    })

sub_df = pd.DataFrame(sub_rows)
sub_df.to_csv(OUT / "submission.csv", index=False)
print(f"Saved submission: {sub_df.shape}")
print(sub_df.head(10))
print(f"\nRisk score stats:")
print(sub_df.risk_score.describe())
print(f"\nPredicted behaviors:")
print(sub_df.predicted_behavior.value_counts())

# Save CV results
cv_results = {
    "overall_oof_ap": float(overall_ap),
    "n_train": int(len(train_df)),
    "n_positives": int(y.sum()),
    "n_features": int(len(feat_cols)),
    "n_eval": int(len(eval_df)),
    "feature_cols": feat_cols,
}
with open(OUT / "cv_results.json", "w") as f:
    json.dump(cv_results, f, indent=2, default=str)

print()
print("="*80)
print(f"DONE in {time.time()-t0:.1f}s")
print(f"Submission: {OUT / 'submission.csv'}")
print(f"OOF AP: {overall_ap:.4f}")
print("="*80)
