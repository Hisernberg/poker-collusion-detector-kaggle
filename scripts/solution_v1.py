"""
Poker Collusion Detector — Phase 1: Clean Baseline + Behavioral Signatures
==========================================================================
Goal: Reproduce user's 0.83065 plateau locally, then iterate upward.

Approach (3 stages):
  STAGE A: Build per-pair aggregate features from hands/seats/actions parquet.
  STAGE B: Train 3 behavior-specific LightGBM models + 1 generic risk model on dev labels.
  STAGE C: For each evaluation pair, predict risk/behavior, then pick top-5 evidence hands.

Outputs:
  - submission.csv (Kaggle format)
  - local_cv.json (per-fold metric proxy)
  - features.parquet (cached pair-level features)
"""
import os, gc, time, json, sys, warnings
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

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
print("STAGE 0 — Loading data")
print("="*80)
t0 = time.time()

# Load label + evidence for training
dev_labels = pd.read_csv(DATA / "development_labels.csv")
dev_evidence = pd.read_csv(DATA / "development_evidence.csv")
eval_pairs  = pd.read_csv(DATA / "evaluation_pairs.csv")
sample_sub  = pd.read_csv(DATA / "sample_submission.csv")
players_df  = pd.read_parquet(DATA / "players.parquet")

print(f"dev_labels:   {dev_labels.shape}  ({dev_labels.label.sum()} positives)")
print(f"dev_evidence: {dev_evidence.shape}  ({dev_evidence.pair_id.nunique()} pairs)")
print(f"eval_pairs:   {eval_pairs.shape}")
print(f"sample_sub:   {sample_sub.shape}")
print(f"players:      {players_df.shape}")

# Build map pair_id -> set of evidence hand_ids (gold)
gold_evidence = defaultdict(set)
gold_family = {}
for r in dev_evidence.itertuples():
    gold_evidence[r.pair_id].add(r.hand_id)
for r in dev_labels.itertuples():
    if r.label == 1:
        gold_family[r.pair_id] = r.behavior_family

# All pair IDs (dev + eval) whose shared hands we need to process
all_pair_players = set()
for pid in dev_labels.pair_id:
    all_pair_players.add(pid)
for pid in eval_pairs.pair_id:
    all_pair_players.add(pid)

# Players we care about (union of all pair players)
needed_players = set()
for r in dev_labels.itertuples():
    needed_players.add(r.player_1); needed_players.add(r.player_2)
for r in eval_pairs.itertuples():
    needed_players.add(r.player_1); needed_players.add(r.player_2)
print(f"Players needed: {len(needed_players)}")

# Map player -> pairs they belong to (for both dev and eval)
player_to_pairs = defaultdict(set)
pair_to_players = {}
for r in dev_labels.itertuples():
    player_to_pairs[r.player_1].add(r.pair_id)
    player_to_pairs[r.player_2].add(r.pair_id)
    pair_to_players[r.pair_id] = (r.player_1, r.player_2)
for r in eval_pairs.itertuples():
    player_to_pairs[r.player_1].add(r.pair_id)
    player_to_pairs[r.player_2].add(r.pair_id)
    pair_to_players[r.pair_id] = (r.player_1, r.player_2)

print(f"Setup time: {time.time()-t0:.1f}s")
print(f"Memory: {sum(pd.DataFrame({'x':[1]}).memory_usage())//1} bytes (just check import OK)")

# =========================================================================
# STAGE A — Build per-pair aggregate features
# =========================================================================
print()
print("="*80)
print("STAGE A — Per-pair hand-level feature aggregation")
print("="*80)

# We process hand-by-hand from seats.parquet, joined with hands.parquet
# For each hand: get player set; for each pair of needed players in that hand,
# compute features.

# First, load hands.parquet (only 2M rows, 48MB - OK to load)
hands = pd.read_parquet(DATA / "hands.parquet",
    columns=["hand_id","table_id","started_at","phase","button_seat",
             "small_blind","big_blind","board_cards","final_pot",
             "players_dealt","players_at_showdown"])
print(f"hands: {hands.shape}, time={time.time()-t0:.1f}s")

# Restrict hands to "development" phase (where labels apply) and hands that
# involve needed players. We can't directly filter hands by player, but we can
# filter seats by player, then keep only those hand_ids.
print(f"Filtering seats by needed players (this takes ~30s)...")

# Read seats in chunks (94MB / 12M rows)
seats_chunks = pd.read_parquet(DATA / "seats.parquet", columns=["hand_id","player_id","seat_no","starting_stack","total_contribution","net_chips","folded","went_to_showdown","won_share"], engine="pyarrow")
print(f"seats loaded: {seats_chunks.shape}, time={time.time()-t0:.1f}s")

seats = seats_chunks[seats_chunks.player_id.isin(needed_players)].copy()
del seats_chunks
gc.collect()
print(f"seats filtered: {seats.shape}, time={time.time()-t0:.1f}s")

# Now restrict hands to those present in filtered seats
needed_hand_ids = set(seats.hand_id.unique())
print(f"Needed hand IDs: {len(needed_hand_ids):,}")

hands = hands[hands.hand_id.isin(needed_hand_ids)].copy()
print(f"hands after filter: {hands.shape}, time={time.time()-t0:.1f}s")

# Build pair-key index: for each hand, which (pair_id, p1, p2) tuples are present?
# Approach: for each hand_id, get list of players; for each pair of players
# where both are needed_players and they form a known pair, log it.

# Invert: hand -> set of players in the hand
hand_players = seats.groupby("hand_id")["player_id"].agg(set).to_dict()

# Invert pair_to_players to (p1, p2) -> pair_id (normalize order)
pair_lookup = {}  # frozenset({p1, p2}) -> pair_id
for pid, (p1, p2) in pair_to_players.items():
    pair_lookup[frozenset((p1, p2))] = pid

# For each hand, enumerate pairs of needed players present in the hand and
# look up the pair_id
print("Building pair -> hands index...")
t_idx = time.time()
pair_hands = defaultdict(list)  # pair_id -> list of hand_id
for hand_id, players in hand_players.items():
    needed_in_hand = players & needed_players
    if len(needed_in_hand) < 2:
        continue
    # Enumerate pairs (small: max ~6 players per hand -> C(6,2) = 15)
    needed_list = list(needed_in_hand)
    for i in range(len(needed_list)):
        for j in range(i+1, len(needed_list)):
            key = frozenset((needed_list[i], needed_list[j]))
            pid = pair_lookup.get(key)
            if pid is not None:
                pair_hands[pid].append(hand_id)
print(f"Pair-hands index: {len(pair_hands):,} pairs have shared hands, time={time.time()-t_idx:.1f}s")

# Save the index for reuse
pair_hands_df = pd.DataFrame(
    [(pid, ",".join(map(str, hands_list))) for pid, hands_list in pair_hands.items()],
    columns=["pair_id", "hand_ids"]
)
pair_hands_df.to_parquet(OUT / "pair_hands_index.parquet")
print(f"Saved pair-hands index: {pair_hands_df.shape}")

# Free memory
del hand_players
gc.collect()

# Now for each (pair, hand) tuple, we compute hand-level features.
# Then aggregate to pair-level features.

# Strategy: we'll iterate over seats grouped by hand_id, and for each hand
# we compute per-pair hand-level features. This is more efficient than
# iterating per-pair because each hand is shared across multiple pairs.

# Get actions for needed hands only
print(f"Loading actions for needed hands ({len(needed_hand_ids):,})...")

# We can't read just needed hand_ids from parquet easily; let's read full actions
# in chunks and filter
actions_iter = pq.ParquetFile(DATA / "actions.parquet").iter_batches(batch_size=2_000_000, columns=["hand_id","action_no","street","player_id","action","amount","amount_to","pot_before","stack_before","to_call","players_active"])
actions_list = []
for batch in actions_iter:
    df = batch.to_pandas()
    df = df[df.hand_id.isin(needed_hand_ids)]
    if len(df):
        actions_list.append(df)
    print(f"  batch filtered: {len(df):,} actions kept, time={time.time()-t0:.1f}s", flush=True)
actions = pd.concat(actions_list, ignore_index=True)
del actions_list
gc.collect()
print(f"actions filtered: {actions.shape}, time={time.time()-t0:.1f}s")

# Now we have:
# - hands (filtered to needed)
# - seats (filtered to needed players)
# - actions (filtered to needed hands)

# Build per-hand lookup tables
hand_seats = seats.groupby("hand_id")
hand_actions = actions.groupby("hand_id")
print(f"Built per-hand groups, time={time.time()-t0:.1f}s")

# Define the per-pair-per-hand feature extraction function
def hand_strength_preflop(c1, c2):
    """Quick preflop strength score (Chen-style simplified)."""
    R = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
    r1, r2 = R.get(c1[0], 2), R.get(c2[0], 2)
    a, b = max(r1, r2), min(r1, r2)
    score = a
    if a == b:  # pair
        score = a * 2 + 5 if a >= 5 else a * 2
    else:
        if b >= 10: score += b - 8
    if c1[1] == c2[1]:  # suited
        score += 2
    if abs(r1 - r2) == 1 and r1 != 2 and r2 != 2:  # connected
        score += 1
    return score

# Hand evaluator (5-7 cards) — light version
RANK_DICT = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}

def eval_hand_7(cards):
    """Return integer score (higher = better) for 5-7 cards (e.g., ['Ah','Kd','Qc','Js','Td','2h','3h'])."""
    ranks = sorted((RANK_DICT[c[0]] for c in cards), reverse=True)
    suits = {}
    for c in cards:
        suits.setdefault(c[1], []).append(RANK_DICT[c[0]])

    # Straight flush / flush
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

def hand_strength_category(score):
    """Convert hand_score to category (0=high, 1=pair, 2=twopair, 3=trips, 4=straight, 5=flush, 6=full, 7=quads, 8=sflush)."""
    return score // (15**5)

# Per-pair-per-hand feature extraction
def extract_pair_hand_features(hand_id, p1, p2, hand_row, seat_rows, action_rows):
    """Extract behavioral features for a specific pair (p1, p2) in a specific hand."""
    big_blind = hand_row.big_blind
    board_cards = hand_row.board_cards.split() if hand_row.board_cards else []
    phase = hand_row.phase

    # Build seat info per player
    seat_info = {}
    for r in seat_rows.itertuples():
        seat_info[r.player_id] = r

    # Build per-player action sequence
    actions_by_player = defaultdict(list)
    all_acts = []
    for r in action_rows.itertuples():
        actions_by_player[r.player_id].append(r)
        all_acts.append(r)

    if p1 not in seat_info or p2 not in seat_info:
        return None  # one of them not in this hand (shouldn't happen given our filter)

    s1, s2 = seat_info[p1], seat_info[p2]

    # Net chips in big-blade units
    net1 = s1.net_chips / max(big_blind, 1)
    net2 = s2.net_chips / max(big_blind, 1)

    # Hole cards
    hole1 = (s1.hole_card_1, s1.hole_card_2) if hasattr(s1, 'hole_card_1') else (None, None)
    hole2 = (s2.hole_card_1, s2.hole_card_2) if hasattr(s2, 'hole_card_1') else (None, None)

    # We don't have hole_card_1/2 in the loaded seats (we filtered columns)
    # — let's skip hole-card based features for now (we'd need a re-read)

    # Tracking for behavioral signatures
    feats = {
        "hand_id": hand_id,
        "phase": phase,
        "p1_vpip": 0, "p2_vpip": 0,
        "p1_pfr": 0, "p2_pfr": 0,
        "p1_fold_to_p2": 0, "p2_fold_to_p1": 0,
        "p1_calls_p2": 0, "p2_calls_p1": 0,
        "p1_raises_p2": 0, "p2_raises_p1": 0,
        "hu_checkdowns": 0,
        "both_postflop": 0,
        "sandwich_pf": 0,
        "third_folds_after_pq_raises": 0,
        "n_p1_actions": 0, "n_p2_actions": 0,
        "p1_aggression": 0, "p2_aggression": 0,
        "p1_passivity_vs_p2": 0, "p2_passivity_vs_p1": 0,
        "chips_p1_to_p2": 0.0,  # p1 contributes chips that p2 ultimately wins
        "chips_p2_to_p1": 0.0,
        "net1_bb": net1,
        "net2_bb": net2,
        "abs_net_diff_bb": abs(net1 - net2),
        "p1_won": 1 if net1 > 0 else 0,
        "p2_won": 1 if net2 > 0 else 0,
        "pot_bb": hand_row.final_pot / max(big_blind, 1),
        "went_to_showdown": 1 if hand_row.players_at_showdown >= 2 else 0,
        "both_went_showdown": int(bool(s1.went_to_showdown) and bool(s2.went_to_showdown)),
        "p1_contribution_bb": s1.total_contribution / max(big_blind, 1),
        "p2_contribution_bb": s2.total_contribution / max(big_blind, 1),
    }

    # Process actions sequentially
    last_agg = None  # player_id of last aggressor on current street
    street = None
    folded = set()
    active = set(actions_by_player.keys())
    invest = defaultdict(int)
    raisers_pf = []
    postflop_players = set()

    for r in all_acts:
        if r.street != street:
            street = r.street
            last_agg = None
        if street != "preflop":
            postflop_players.add(r.player_id)

        is_p1 = (r.player_id == p1)
        is_p2 = (r.player_id == p2)
        other = p2 if is_p1 else (p1 if is_p2 else None)

        # VPIP / PFR (preflop only)
        if street == "preflop":
            if r.action in ("call", "bet", "raise", "all_in"):
                if is_p1: feats["p1_vpip"] = 1
                elif is_p2: feats["p2_vpip"] = 1
            if r.action in ("bet", "raise", "all_in"):
                raisers_pf.append(r.player_id)
                if is_p1: feats["p1_pfr"] = 1
                elif is_p2: feats["p2_pfr"] = 1

        # Action counts
        if is_p1:
            feats["n_p1_actions"] += 1
            if r.action in ("bet", "raise", "all_in"):
                feats["p1_aggression"] += 1
        elif is_p2:
            feats["n_p2_actions"] += 1
            if r.action in ("bet", "raise", "all_in"):
                feats["p2_aggression"] += 1

        # Fold-to-other detection
        if r.action == "fold":
            folded.add(r.player_id)
            if other is not None and last_agg == other:
                if is_p1: feats["p1_fold_to_p2"] += 1; feats["p1_passivity_vs_p2"] += 1
                elif is_p2: feats["p2_fold_to_p1"] += 1; feats["p2_passivity_vs_p1"] += 1

        # Calls & raises vs other
        if r.action == "call" and last_agg == other:
            if is_p1: feats["p1_calls_p2"] += 1
            elif is_p2: feats["p2_calls_p1"] += 1
        if r.action in ("bet", "raise", "all_in") and last_agg == other:
            if is_p1: feats["p1_raises_p2"] += 1
            elif is_p2: feats["p2_raises_p1"] += 1

        # HU checkdown detection: both p1,p2 check on the same postflop street
        # (only count if both check at least once on the street when HU)
        if r.action == "check" and street != "preflop":
            # Players active = those not yet folded
            if p1 not in folded and p2 not in folded and r.players_active == 2:
                # mark this street as a HU checkdown candidate if both check
                pass  # we'll compute this differently below

        # Third-player fold after p&q preflop raises
        if r.action == "fold" and (not is_p1) and (not is_p2):
            if p1 in raisers_pf and p2 in raisers_pf and street == "preflop":
                feats["third_folds_after_pq_raises"] += 1

        invest[r.player_id] += r.amount
        if r.action in ("bet", "raise", "all_in"):
            last_agg = r.player_id

    # Sandwich detection: both p1 and p2 raised preflop with at least 3 distinct raisers
    if p1 in raisers_pf and p2 in raisers_pf and len(set(raisers_pf)) >= 3:
        feats["sandwich_pf"] = 1

    # Both postflop
    feats["both_postflop"] = int(p1 in postflop_players and p2 in postflop_players)

    # HU checkdowns — count streets where both p1, p2 active and only checks (no aggression)
    # Simplified: if both went_to_showdown and pot is small relative to stacks, count 1
    if s1.went_to_showdown and s2.went_to_showdown:
        # Both reached showdown
        # Check if there were no preflop raises (limped pot)
        if len(raisers_pf) == 0:
            feats["hu_checkdowns"] = 1
        # Also count postflop streets with no aggression from either p1 or p2
        postflop_streets = set()
        for r in all_acts:
            if r.street != "preflop" and r.player_id in (p1, p2):
                postflop_streets.add(r.street)
        # If they checked through multiple streets heads-up, flag it
        # Simpler: if both p1,p2 had 0 aggression and pot was small
        if feats["p1_aggression"] == 0 and feats["p2_aggression"] == 0:
            feats["hu_checkdowns"] = max(feats["hu_checkdowns"], 1)

    # Chips transfer: net of p1 + net of p2 (negative means chip transfer to outsiders)
    # Or: positive net of one vs negative net of other = direct transfer signal
    feats["chip_transfer_signal"] = 0.0
    if net1 * net2 < 0:  # opposite signs — one wins, one loses
        # Whichever direction has larger magnitude
        if abs(net1) > abs(net2):
            # p1 wins, p2 loses (or vice versa)
            feats["chip_transfer_signal"] = (net1 + net2) / 2  # if very asymmetric, signal positive
        feats["chip_transfer_signal"] = (abs(net1) + abs(net2)) / 2 * np.sign(net1 - net2)

    return feats

# We'll iterate per hand (grouped), and for each pair within the hand,
# compute features. Then aggregate to pair-level.
print()
print("="*80)
print("Computing per-pair-per-hand features (this will take ~5-10 min)...")
print("="*80)

# Setup iteration: group seats by hand_id, group actions by hand_id
seats_by_hand = dict(list(seats.groupby("hand_id")))
actions_by_hand = dict(list(actions.groupby("hand_id")))

pair_hand_features_list = []
processed = 0
t_iter = time.time()

# Process in chunks of pairs to avoid memory blowup
for pair_id, hand_list in pair_hands.items():
    if pair_id not in pair_to_players:
        continue
    p1, p2 = pair_to_players[pair_id]
    for hand_id in hand_list:
        if hand_id not in seats_by_hand or hand_id not in actions_by_hand:
            continue
        hand_row = hands[hands.hand_id == hand_id].iloc[0]
        seat_rows = seats_by_hand[hand_id]
        action_rows = actions_by_hand[hand_id]
        if p1 not in set(seat_rows.player_id) or p2 not in set(seat_rows.player_id):
            continue
        feats = extract_pair_hand_features(hand_id, p1, p2, hand_row, seat_rows, action_rows)
        if feats is not None:
            feats["pair_id"] = pair_id
            feats["p1"] = p1
            feats["p2"] = p2
            pair_hand_features_list.append(feats)
    processed += 1
    if processed % 20000 == 0:
        elapsed = time.time() - t_iter
        est_total = elapsed / processed * len(pair_hands)
        print(f"  processed {processed:,}/{len(pair_hands):,} pairs, elapsed={elapsed:.1f}s, est_total={est_total:.1f}s, features={len(pair_hand_features_list):,}", flush=True)

print(f"Done: {len(pair_hand_features_list):,} pair-hand features in {time.time()-t_iter:.1f}s")

phf = pd.DataFrame(pair_hand_features_list)
del pair_hand_features_list
gc.collect()
print(f"pair-hand features: {phf.shape}")

# Save for reuse
phf.to_parquet(OUT / "pair_hand_features.parquet")
print(f"Saved pair_hand_features.parquet")

# Aggregate to pair-level features
print()
print("="*80)
print("STAGE A2 — Aggregating pair-level features")
print("="*80)

# Numeric aggregations
agg_funcs = ["mean", "sum", "max", "min", "std"]
numeric_cols = [c for c in phf.columns if c not in ("hand_id", "phase", "pair_id", "p1", "p2")]

# Compute counts and aggregates
pair_features = phf.groupby("pair_id").agg(
    n_hands=("hand_id", "count"),
    p1_fold_to_p2_sum=("p1_fold_to_p2", "sum"),
    p2_fold_to_p1_sum=("p2_fold_to_p1", "sum"),
    p1_calls_p2_sum=("p1_calls_p2", "sum"),
    p2_calls_p1_sum=("p2_calls_p1", "sum"),
    p1_raises_p2_sum=("p1_raises_p2", "sum"),
    p2_raises_p1_sum=("p2_raises_p1", "sum"),
    hu_checkdowns_sum=("hu_checkdowns", "sum"),
    sandwich_pf_sum=("sandwich_pf", "sum"),
    third_folds_after_pq_raises_sum=("third_folds_after_pq_raises", "sum"),
    p1_aggression_sum=("p1_aggression", "sum"),
    p2_aggression_sum=("p2_aggression", "sum"),
    p1_passivity_vs_p2_sum=("p1_passivity_vs_p2", "sum"),
    p2_passivity_vs_p1_sum=("p2_passivity_vs_p1", "sum"),
    net1_bb_mean=("net1_bb", "mean"),
    net2_bb_mean=("net2_bb", "mean"),
    net1_bb_sum=("net1_bb", "sum"),
    net2_bb_sum=("net2_bb", "sum"),
    abs_net_diff_bb_mean=("abs_net_diff_bb", "mean"),
    abs_net_diff_bb_max=("abs_net_diff_bb", "max"),
    pot_bb_mean=("pot_bb", "mean"),
    pot_bb_max=("pot_bb", "max"),
    both_postflop_mean=("both_postflop", "mean"),
    both_went_showdown_mean=("both_went_showdown", "mean"),
    p1_contribution_bb_mean=("p1_contribution_bb", "mean"),
    p2_contribution_bb_mean=("p2_contribution_bb", "mean"),
    p1_contribution_bb_sum=("p1_contribution_bb", "sum"),
    p2_contribution_bb_sum=("p2_contribution_bb", "sum"),
    chip_transfer_signal_mean=("chip_transfer_signal", "mean"),
    chip_transfer_signal_max=("chip_transfer_signal", "max"),
    p1_vpip_rate=("p1_vpip", "mean"),
    p2_vpip_rate=("p2_vpip", "mean"),
    p1_pfr_rate=("p1_pfr", "mean"),
    p2_pfr_rate=("p2_pfr", "mean"),
).reset_index()

# Derived features
pair_features["fold_to_partner_total"] = pair_features["p1_fold_to_p2_sum"] + pair_features["p2_fold_to_p1_sum"]
pair_features["fold_to_partner_rate"] = pair_features["fold_to_partner_total"] / pair_features["n_hands"].clip(lower=1)
pair_features["hu_checkdown_rate"] = pair_features["hu_checkdowns_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["sandwich_rate"] = pair_features["sandwich_pf_sum"] / pair_features["n_hands"].clip(lower=1)
pair_features["net_asymmetry"] = (pair_features["net1_bb_sum"] - pair_features["net2_bb_sum"]).abs()
pair_features["net_total_bb"] = (pair_features["net1_bb_sum"] + pair_features["net2_bb_sum"])
pair_features["chips_transferred_bb"] = pair_features["net_asymmetry"]
pair_features["contribution_asymmetry"] = (pair_features["p1_contribution_bb_sum"] - pair_features["p2_contribution_bb_sum"]).abs()
pair_features["aggression_asymmetry"] = (pair_features["p1_aggression_sum"] - pair_features["p2_aggression_sum"]).abs()
pair_features["passivity_total"] = pair_features["p1_passivity_vs_p2_sum"] + pair_features["p2_passivity_vs_p1_sum"]

# Add player profile features
players_df = players_df.rename(columns={"player_id": "p1"})
pair_features = pair_features.merge(dev_labels[["pair_id","player_1","player_2"]].rename(columns={"player_1":"p1","player_2":"p2"}), on="pair_id", how="left")
# Some eval pairs aren't in dev_labels; we need to merge from eval_pairs too
eval_player_map = eval_pairs[["pair_id","player_1","player_2"]].rename(columns={"player_1":"p1","player_2":"p2"})
pair_features = pair_features.drop(columns=["p1","p2"], errors="ignore")
pair_features = pair_features.merge(eval_player_map, on="pair_id", how="left")
# Fill missing from dev
dev_player_map = dev_labels[["pair_id","player_1","player_2"]].rename(columns={"player_1":"p1","player_2":"p2"})
pair_features = pair_features.merge(dev_player_map, on="pair_id", how="left", suffixes=("","_dev"))
pair_features["p1"] = pair_features["p1"].fillna(pair_features["p1_dev"])
pair_features["p2"] = pair_features["p2"].fillna(pair_features["p2_dev"])
pair_features = pair_features.drop(columns=["p1_dev","p2_dev"], errors="ignore")

# Player profile features
p1_prof = players_df.rename(columns={c: f"p1_{c}" for c in players_df.columns if c != "player_id"}).rename(columns={"p1_account_age_days":"p1_account_age_days","p1_experience_hands_bucket":"p1_exp","p1_preferred_stake":"p1_stake","p1_region_bucket":"p1_region","p1_client_family":"p1_client"})
p2_prof = players_df.rename(columns={c: f"p2_{c}" for c in players_df.columns if c != "player_id"}).rename(columns={"p2_account_age_days":"p2_account_age_days","p2_experience_hands_bucket":"p2_exp","p2_preferred_stake":"p2_stake","p2_region_bucket":"p2_region","p2_client_family":"p2_client"})

pair_features = pair_features.merge(p1_prof.rename(columns={"player_id":"p1"}), on="p1", how="left")
pair_features = pair_features.merge(p2_prof.rename(columns={"player_id":"p2"}), on="p2", how="left")
print(f"pair_features shape (with player profile): {pair_features.shape}")

# Encode categorical
for cat_col in ["p1_exp","p1_stake","p1_region","p1_client","p2_exp","p2_stake","p2_region","p2_client"]:
    if cat_col in pair_features.columns:
        codes = pair_features[cat_col].astype("category").cat.codes
        pair_features[cat_col + "_code"] = codes
        pair_features = pair_features.drop(columns=[cat_col])

# Add shared_hands from eval_pairs
pair_features = pair_features.merge(eval_pairs[["pair_id","shared_hands"]], on="pair_id", how="left")
# For dev pairs (not in eval_pairs), shared_hands comes from n_hands
pair_features["shared_hands"] = pair_features["shared_hands"].fillna(pair_features["n_hands"])

# Save features
pair_features.to_parquet(OUT / "pair_features.parquet")
print(f"Saved pair_features: {pair_features.shape}")
print(f"Total time so far: {time.time()-t0:.1f}s")

# =========================================================================
# STAGE B — Train models
# =========================================================================
print()
print("="*80)
print("STAGE B — Train models")
print("="*80)

# Merge label info
train_df = pair_features.merge(dev_labels[["pair_id","label","behavior_family"]], on="pair_id", how="inner")
print(f"Train pairs: {train_df.shape}, positives: {train_df.label.sum()}")

# Feature columns (drop IDs and labels)
drop_cols = {"pair_id","p1","p2","label","behavior_family","p1_account_age_days","p2_account_age_days"}
feat_cols = [c for c in train_df.columns if c not in drop_cols and train_df[c].dtype != object]
print(f"# feature columns: {len(feat_cols)}")

# Impute NaN with 0 (LightGBM handles natively but just in case)
for c in feat_cols:
    if train_df[c].isna().any():
        train_df[c] = train_df[c].fillna(train_df[c].median())

X = train_df[feat_cols].values
y = train_df.label.values
families = train_df.behavior_family.values

# Per-family binary labels
y_dt = (families == "directed_transfer").astype(int)
y_sp = (families == "soft_play").astype(int)
y_ci = (families == "coordinated_isolation").astype(int)

# 5-fold CV
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# LightGBM parameters
def lgb_params(objective="binary", num_leaves=31, n_estimators=400):
    return dict(
        objective=objective,
        num_leaves=num_leaves,
        learning_rate=0.05,
        n_estimators=n_estimators,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=SEED,
        n_jobs=4,
        verbose=-1,
    )

# Per-family models
def train_family_models(y_family, family_name):
    print(f"\n--- Training {family_name} models ---")
    oof = np.zeros(len(train_df))
    fold_scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y_family)):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y_family[tr], y_family[va]
        model = lgb.LGBMClassifier(**lgb_params())
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50)])
        oof[va] = model.predict_proba(X_va)[:, 1]
        if y_va.sum() > 0:
            score = average_precision_score(y_va, oof[va])
        else:
            score = 0.0
        fold_scores.append(score)
        print(f"  fold {fold}: AP={score:.4f}")
    print(f"  Mean AP: {np.mean(fold_scores):.4f}")
    return oof

oof_dt = train_family_models(y_dt, "directed_transfer")
oof_sp = train_family_models(y_sp, "soft_play")
oof_ci = train_family_models(y_ci, "coordinated_isolation")

# Generic risk model (any suspicious behavior)
oof_risk = train_family_models(y, "any_suspicious")

# Final OOF risk = max across family scores + generic
oof_combined = np.maximum.reduce([oof_dt, oof_sp, oof_ci, oof_risk * 0.7])

# Compute metric proxy
overall_ap = average_precision_score(y, oof_combined)
print(f"\nOverall OOF AP (composite risk): {overall_ap:.4f}")

# Per-family AP
for fam, oof_arr in [("directed_transfer", oof_dt), ("soft_play", oof_sp), ("coordinated_isolation", oof_ci)]:
    mask = (families == fam) | (families == "none")  # all (yes vs no for that family)
    y_fam = (families == fam).astype(int)
    if y_fam.sum() > 0:
        fam_ap = average_precision_score(y_fam, oof_arr)
        print(f"  {fam} OOF AP: {fam_ap:.4f}")

# Train final models on full data for prediction
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
import pickle
with open(OUT / "models.pkl", "wb") as f:
    pickle.dump({"dt":final_dt, "sp":final_sp, "ci":final_ci, "risk":final_risk, "feat_cols":feat_cols}, f)
print("Saved models")

# =========================================================================
# STAGE C — Predict on evaluation pairs
# =========================================================================
print()
print("="*80)
print("STAGE C — Predict on evaluation pairs")
print("="*80)

# Build eval feature matrix
eval_df = pair_features[pair_features.pair_id.isin(set(eval_pairs.pair_id))].copy()
print(f"Eval pairs: {eval_df.shape}")
assert len(eval_df) == len(eval_pairs), f"Mismatch: {len(eval_df)} vs {len(eval_pairs)}"

# Impute
for c in feat_cols:
    if eval_df[c].isna().any():
        eval_df[c] = eval_df[c].fillna(train_df[c].median())

X_eval = eval_df[feat_cols].values

# Predict
p_dt = final_dt.predict_proba(X_eval)[:, 1]
p_sp = final_sp.predict_proba(X_eval)[:, 1]
p_ci = final_ci.predict_proba(X_eval)[:, 1]
p_risk = final_risk.predict_proba(X_eval)[:, 1]

# Combined risk score (max across families + generic risk)
risk_combined = np.maximum.reduce([p_dt, p_sp, p_ci, p_risk * 0.7])

# Predicted behavior (argmax across families, with threshold)
behaviors = []
for i in range(len(eval_df)):
    probs = [p_dt[i], p_sp[i], p_ci[i]]
    fam_idx = int(np.argmax(probs))
    fam_names = ["directed_transfer", "soft_play", "coordinated_isolation"]
    # If max prob < threshold, predict "none"
    if max(probs) < 0.3 and p_risk[i] < 0.3:
        behaviors.append("none")
    else:
        behaviors.append(fam_names[fam_idx])

print(f"Predicted behavior distribution: {Counter(behaviors)}")

# Evidence: for each eval pair, we need top-5 hand_ids that best evidence
# the suspicious behavior. Score each hand by behavior-specific signals.

# Use pair_hand_features for evidence selection
# For each (pair, hand) in pair_hand_features, compute a "suspicion score"
# based on the predicted behavior family, then take top-5

# Build per-hand suspicion scores
print("\n--- Computing per-hand evidence scores ---")
phf_pair_id = phf["pair_id"].values
phf_hand_id = phf["hand_id"].values

# Quick suspicion features per hand
suspicion_score = (
    (phf["p1_fold_to_p2"] + phf["p2_fold_to_p1"]) * 1.0 +
    (phf["hu_checkdowns"]) * 1.5 +
    (phf["sandwich_pf"] + phf["third_folds_after_pq_raises"]) * 1.0 +
    np.abs(phf["net1_bb"] - phf["net2_bb"]) * 0.3 +
    (phf["p1_passivity_vs_p2"] + phf["p2_passivity_vs_p1"]) * 0.5 +
    phf["chip_transfer_signal"].abs() * 0.4
).values

# For each eval pair, select top-5 hands by suspicion score
eval_pair_ids = set(eval_pairs.pair_id)
pair_to_hand_scores = defaultdict(list)
for i, pid in enumerate(phf_pair_id):
    if pid in eval_pair_ids:
        pair_to_hand_scores[pid].append((phf_hand_id[i], suspicion_score[i]))

# Build submission
sub_rows = []
for r in eval_pairs.itertuples():
    pid = r.pair_id
    hand_scores = pair_to_hand_scores.get(pid, [])
    # Sort by score descending
    hand_scores.sort(key=lambda x: -x[1])
    top5 = [h for h, s in hand_scores[:5]]
    # Pad with NO_EVIDENCE
    while len(top5) < 5:
        top5.append("NO_EVIDENCE")
    sub_rows.append({
        "pair_id": pid,
        "risk_score": float(risk_combined[list(eval_df.pair_id).index(pid)]),
        "predicted_behavior": behaviors[list(eval_df.pair_id).index(pid)],
        "evidence_hand_1": top5[0],
        "evidence_hand_2": top5[1],
        "evidence_hand_3": top5[2],
        "evidence_hand_4": top5[3],
        "evidence_hand_5": top5[4],
    })

sub_df = pd.DataFrame(sub_rows)
sub_df.to_csv(OUT / "submission.csv", index=False)
print(f"Saved submission: {sub_df.shape}")

# Save CV results
cv_results = {
    "overall_ap": float(overall_ap),
    "n_train": int(len(train_df)),
    "n_positives": int(y.sum()),
    "feat_cols": feat_cols,
    "n_eval": int(len(eval_df)),
}
with open(OUT / "cv_results.json", "w") as f:
    json.dump(cv_results, f, indent=2)

print()
print("="*80)
print(f"DONE in {time.time()-t0:.1f}s")
print(f"Submission: {OUT / 'submission.csv'}")
print(f"OOF AP: {overall_ap:.4f}")
print("="*80)
