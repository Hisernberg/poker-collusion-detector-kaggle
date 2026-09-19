"""Shared library: constants, card math, host metric, small helpers.
Ported from the 0.81 public notebook (fixing transcript mangling) + local adaptations.
"""
from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from numba import njit

warnings_ok = True

SEED = 42
N_FOLDS = 5
SHRINK_N = 25.0
PU_WEIGHT = 0.1
POS_WEIGHT = 5.0
NEG_WEIGHT = 1.0
PSEUDO_POS_WEIGHT = 1.0
POLICY_PSEUDOCOUNTS = 20.0
HAND_ROUNDS = 400
PAIR_ROUNDS = 750
FAMILY_ROUNDS = 400
HAND_SEEDS = [SEED, SEED + 7]
PAIR_SEEDS = [SEED, SEED + 11, SEED + 23]
EVIDENCE_SPEC_W = 0.60
NONE_GRID = [1.0, 0.10, 0.05, 0.03, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.004, 0.003]

TARGET_BEHAVIORS = ("directed_transfer", "soft_play", "coordinated_isolation")
ALLOWED_BEHAVIORS = {"none", "other_coordination", *TARGET_BEHAVIORS}
SPEC_COLS = tuple(f"hand_spec_{fam}" for fam in TARGET_BEHAVIORS)
FAM_INDEX = {fam: i for i, fam in enumerate(TARGET_BEHAVIORS)}
EVIDENCE_COLUMNS = tuple(f"evidence_hand_{rank}" for rank in range(1, 6))
NO_EVIDENCE = "NO_EVIDENCE"
REQUIRED_COLUMNS = {"pair_id", "risk_score", "predicted_behavior", *EVIDENCE_COLUMNS}

RANKS = {r: i for i, r in enumerate("23456789TJQKA", start=2)}
RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"
CARD_MAP = {f"{r}{su}": i * 4 + j for i, r in enumerate(RANK_CHARS) for j, su in enumerate(SUIT_CHARS)}
B5 = 15**5
AGGR = ["bet", "raise"]
ENTRY_ACTIONS = ["call", "raise", "bet", "all_in"]

BASELINE_COLS = [
    "net_bb", "vpip", "n_aggr", "n_raise", "n_call", "n_fold_facing",
    "went_to_showdown", "contrib_bb", "vpip_a", "loose", "junk_vpip", "vpip_resid", "pfr",
    "post_loose", "junk_continue",
]
P_COLS = [
    "seat_no", "contrib_bb", "net_bb", "vpip", "folded", "went_to_showdown", "won_share", "chen",
    "n_actions", "n_aggr", "n_raise", "n_call", "n_check", "n_allin", "n_fold_facing",
    "n_postflop_check", "n_hu_actions", "last_street", "max_amount_bb", "max_to_call_bb",
    "pos", "vpip_a", "pfr", "pf_entry_no", "pf_faced_raise", "p_vpip", "loose", "tight",
    "loose_pfr", "junk_vpip", "junk_pfr", "vpip_resid",
    "post_loose", "post_tight", "post_resid", "junk_continue", "fold_made",
]
POST_PLAYER_COLS = ["post_loose", "post_tight", "post_resid", "junk_continue", "fold_made"]
HAND_CACHE_COLS = (
    "post_loose_sum", "post_loose_partner", "junk_call_after_partner_raise",
    "both_passive_made", "fold_better_surprise",
    "iso_street", "iso_street_pf", "iso_multi", "iso_multi_pf",
    "fold_to_other_pf", "fold_to_other_post", "call_partner_pf",
)
PAIR_CACHE_COLS = (
    "post_loose_sum_mean", "junk_call_after_partner_raise_sum",
    "post_loose_pf_sum", "both_passive_made_sum",
    "fold_contrast", "call_contrast", "raise_contrast", "hu_check_rate",
    "fold_contrast_pf", "fold_contrast_post", "iso_street_mean", "iso_multi_sum",
)
S_COLS = ["v_flop", "v_turn", "v_river", "v_final", "cat_final", "rk_flop", "rk_turn", "rk_river", "rk_final"]
FILL0_PLAYER = [
    "n_actions", "n_aggr", "n_raise", "n_call", "n_check", "n_allin", "n_fold_facing",
    "n_postflop_check", "n_hu_actions", "last_street", "max_amount_bb", "max_to_call_bb",
]
STRENGTH_CMP = ["chen", "contrib_bb", "folded", "last_street", "v_flop", "v_turn", "v_final", "cat_final", "rk_flop", "rk_turn", "rk_final"]

HAND_RAW = [
    "pot_bb", "players_at_showdown", "both_vpip", "both_showdown", "one_folded", "partner_won_other_lost",
    "net_gap", "transfer_any", "transfer_pot_ratio", "loser_chen", "winner_chen", "chen_gap",
    "loser_stronger_preflop", "loser_contrib_bb", "pair_contrib_bb", "pair_aggr", "pair_raise",
    "pair_call", "pair_check", "pair_postflop_check", "n_partners_aggr", "max_amount_bb", "last_street",
    "pair_overbets", "fold_to_partner", "call_partner", "raise_partner", "fold_to_other", "call_other",
    "raise_other", "hu_actions", "hu_check", "hu_call", "hu_aggr", "hu_late_check", "hu_fold",
    "outsider_fold_to_pair", "outsider_call_to_pair", "outsider_raise_to_pair",
    "loser_cat_final", "winner_cat_final", "loser_rk_final", "winner_rk_final", "loser_rk_flop",
    "winner_rk_flop", "cat_gap_final", "max_cat_final", "loser_beats_winner_flop", "loser_beats_winner_turn",
    "loser_beats_winner_final", "loser_best_final", "winner_best_final", "fold_better_hand",
    "both_made_no_aggr", "winner_weak_won", "fold_to_partner_pf", "fold_to_partner_post",
    "call_partner_pf", "call_partner_post", "raise_partner_pf", "raise_partner_post",
    "outsider_fold_to_pair_pf", "fold_to_other_pf", "fold_to_other_post",
    "call_other_pf", "call_other_post", "raise_other_pf", "raise_other_post",
    "both_vpip_a", "n_vpip_a", "n_pfr", "loose_sum", "loose_min", "loose_max", "tight_max",
    "loose_pfr_sum", "junk_vpip_sum", "junk_vpip_min", "junk_pfr_sum", "vpip_resid_sum",
    "chen_max_pair", "chen_min_pair", "pf_faced_raise_any", "n_in_blinds", "second_entrant_loose",
    "both_vpip_junk", "both_vpip_one_junk", "second_entrant_junk", "both_surprising", "junk_raise",
    "post_loose_sum", "post_loose_min", "post_tight_max", "junk_continue_sum", "junk_continue_min",
    "post_resid_sum", "fold_made_sum", "post_loose_partner", "junk_call_partner_post",
    "fold_made_partner", "fold_better_surprise", "made_check_sum", "made_check_hu",
    # -- additions (evidence-targeted) --
    "call_partner_bb", "call_partner_max_bb", "call_partner_potfrac_sum", "fold_partner_potfrac_max",
]
HAND_FLAGS = [
    "fold_stronger_to_partner", "fold_better_to_partner", "both_strong_no_raise", "multi_call_partner",
    "squeeze", "squeeze_then_fold", "pf_squeeze", "squeeze_then_fold_pf", "dump", "dump_better_hand",
    "checkdown", "hu_checkdown_strong", "hu_check_two_pair_plus", "transfer_x_call",
    "junk_call_after_partner_raise", "both_passive_made", "fold_made_to_partner",
    "iso_multi", "iso_multi_pf", "iso_street", "iso_street_pf",
]
HAND_TOMAX = ["transfer_to_max", "net_gap_to_max", "pot_to_max", "outsider_fold_to_max"]
PRANK_BASE = HAND_RAW + HAND_FLAGS
HAND_FEATS = HAND_RAW + HAND_FLAGS + HAND_TOMAX + [f"{c}_prank" for c in PRANK_BASE]
ACTION_FILL = [
    "fold_to_partner", "call_partner", "raise_partner", "fold_to_other", "call_other", "raise_other",
    "p1_fold_to_partner", "p1_call_partner", "fold_to_partner_pf", "fold_to_partner_post",
    "call_partner_post", "call_partner_pf", "raise_partner_pf", "raise_partner_post", "p1_aggr_pf", "p2_aggr_pf",
    "hu_actions", "hu_check", "hu_call", "hu_aggr", "hu_late_check", "hu_fold", "pair_overbets",
    "outsider_fold_to_pair", "outsider_call_to_pair", "outsider_raise_to_pair", "outsider_fold_to_pair_pf",
    "fold_to_other_pf", "fold_to_other_post", "call_other_pf", "call_other_post", "raise_other_pf", "raise_other_post",
    "iso_street", "iso_street_pf",
    "post_loose_partner", "junk_call_partner_post", "fold_made_partner", "fold_better_surprise",
    "made_check_sum", "made_check_hu",
]
AGG_MEAN = [
    "both_vpip", "both_showdown", "partner_won_other_lost", "transfer_any", "net_gap", "pot_bb",
    "loser_stronger_preflop", "chen_gap", "fold_to_partner", "call_partner", "raise_partner",
    "fold_to_other", "call_other", "raise_other", "hu_actions", "hu_check", "hu_call", "hu_aggr",
    "hu_late_check", "outsider_fold_to_pair", "outsider_call_to_pair", "outsider_raise_to_pair",
    "n_partners_aggr", "pair_aggr", "pair_raise", "pair_overbets", "loser_beats_winner_final",
    "loser_beats_winner_flop", "loser_best_final", "winner_best_final", "fold_better_hand",
    "fold_better_to_partner", "both_made_no_aggr", "winner_weak_won", "cat_gap_final",
    "fold_to_partner_pf", "fold_to_partner_post", "raise_partner_pf", "outsider_fold_to_pair_pf",
    "pf_squeeze", "squeeze_then_fold_pf", "dump_better_hand", "hu_check_two_pair_plus",
    "both_vpip_a", "n_vpip_a", "n_pfr", "loose_sum", "loose_min", "loose_max", "tight_max",
    "loose_pfr_sum", "junk_vpip_sum", "junk_vpip_min", "junk_pfr_sum", "vpip_resid_sum",
    "second_entrant_loose", "both_vpip_junk", "both_vpip_one_junk", "second_entrant_junk",
    "both_surprising", "junk_raise",
    "post_loose_sum", "post_loose_min", "post_loose_partner", "junk_continue_sum",
    "junk_call_partner_post", "fold_better_surprise", "made_check_sum", "made_check_hu",
    "fold_made_sum",
    "iso_multi", "iso_multi_pf", "iso_street", "iso_street_pf",
    # additions
    "call_partner_bb", "call_partner_max_bb", "call_partner_potfrac_sum", "fold_partner_potfrac_max",
]
AGG_MAX = [
    "transfer_any", "net_gap", "pot_bb", "outsider_fold_to_pair", "call_partner", "hu_check",
    "max_amount_bb", "loose_sum", "loose_min", "junk_vpip_sum",
    "post_loose_sum", "post_loose_partner", "junk_call_partner_post",
    "call_partner_bb", "call_partner_max_bb",
]
AGG_TOP3 = [
    "transfer_any", "net_gap", "outsider_fold_to_pair", "hu_check", "call_partner", "loose_sum",
    "loose_min", "junk_vpip_sum", "junk_pfr_sum", "second_entrant_loose",
    "post_loose_sum", "post_loose_min", "post_loose_partner", "junk_continue_sum", "junk_call_partner_post",
    "call_partner_bb",
]
AGG_SUM = [
    "fold_better_to_partner", "dump_better_hand", "squeeze_then_fold_pf", "loser_beats_winner_final",
    "both_made_no_aggr", "both_vpip_a", "both_vpip_junk", "both_vpip_one_junk", "second_entrant_junk",
    "both_surprising", "junk_raise",
    "junk_call_after_partner_raise", "both_passive_made", "fold_made_to_partner", "junk_call_partner_post",
    "iso_multi", "iso_multi_pf", "iso_street", "iso_street_pf",
]
CONTRAST = [
    "net_bb", "n_aggr", "n_raise", "n_call", "n_fold_facing", "vpip", "vpip_a", "loose",
    "junk_vpip", "vpip_resid", "pfr", "post_loose", "junk_continue",
]
TRANK_COLS = [
    "vpip_pf_sum", "p1_vpip_pf", "p2_vpip_pf", "vpip_pf_min", "n_call_pf_sum", "p1_n_call_pf",
    "p2_n_call_pf", "n_call_pf_min", "n_raise_pf_sum", "n_fold_facing_pf_min", "n_aggr_pf_sum",
    "net_bb_pf_gap", "hs_top5", "hs_top3", "hs_mean", "hs_n95", "transfer_rate", "transfer_dominant",
    "fold_better_to_partner_sum", "call_partner_asym", "hu_aggr_mean", "pair_aggr_mean", "call_other_mean",
    "loser_beats_winner_final_sum", "both_made_no_aggr_sum", "squeeze_then_fold_pf_sum", "pf_squeeze_mean",
    "outsider_fold_to_pair_pf_mean", "raise_partner_pf_mean", "hu_check_mean", "both_showdown_mean",
    "both_vpip_a_mean", "both_vpip_a_sum", "loose_sum_mean", "loose_sum_top3", "loose_min_top3",
    "junk_vpip_sum_mean", "junk_vpip_sum_top3", "second_entrant_junk_sum", "second_entrant_loose_top3",
    "both_vpip_junk_sum", "both_surprising_sum", "junk_raise_sum", "vpip_a_pf_sum", "loose_pf_sum",
    "junk_vpip_pf_sum", "vpip_resid_pf_sum", "vpip_a_pf_min", "loose_pf_min",
    "post_loose_sum_mean", "post_loose_sum_top3", "post_loose_min_top3",
    "post_loose_partner_mean", "post_loose_partner_top3",
    "junk_continue_sum_mean", "junk_call_partner_post_mean", "junk_call_partner_post_sum",
    "junk_call_after_partner_raise_sum", "both_passive_made_sum", "fold_better_surprise_mean",
    "fold_made_to_partner_sum", "post_loose_pf_sum", "post_loose_pf_min", "junk_continue_pf_sum",
    "fold_contrast", "call_contrast", "raise_contrast", "hu_check_rate", "winner_not_best_mean",
    "fold_contrast_pf", "fold_contrast_post", "iso_street_mean", "iso_street_sum",
    "iso_street_pf_mean", "iso_street_pf_sum", "iso_multi_mean", "iso_multi_sum", "iso_multi_pf_sum",
    # additions
    "call_partner_bb_mean", "call_partner_bb_top3", "call_partner_max_bb_max",
    "transfer_dir_align_top5", "transfer_top5_share",
]
PAIR_META = {"pair_id", "table_id", "player_1", "player_2", "label", "behavior_family", "is_labeled", "fold", "shared_hands", "chunk", "pred_family"}

HAND_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
    verbose=-1, seed=SEED, num_threads=os.cpu_count() or 2,
)
SUS_PARAMS = {**HAND_PARAMS, "scale_pos_weight": 8.0}
PAIR_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100,
    feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
    verbose=-1, seed=SEED, num_threads=os.cpu_count() or 2,
)

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{(time.time() - T0) / 60:6.1f}m] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------
def chen(c1: str, c2: str) -> float:
    r1, s1 = RANKS[c1[0]], c1[1]
    r2, s2 = RANKS[c2[0]], c2[1]
    hi, lo = max(r1, r2), min(r1, r2)
    base = {14: 10, 13: 8, 12: 7, 11: 6}.get(hi, hi / 2)
    if r1 == r2:
        return max(5.0, base * 2)
    score = base + (2 if s1 == s2 else 0)
    gap = hi - lo - 1
    score -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and hi < 12:
        score += 1
    return float(score)


def card_to_int(card: str) -> int:
    return RANK_CHARS.index(card[0]) * 4 + SUIT_CHARS.index(card[1])


@njit(cache=True)
def eval5(c0, c1, c2, c3, c4):
    ranks = np.zeros(5, np.int64)
    ranks[0] = c0 >> 2
    ranks[1] = c1 >> 2
    ranks[2] = c2 >> 2
    ranks[3] = c3 >> 2
    ranks[4] = c4 >> 2
    flush = ((c0 & 3) == (c1 & 3)) and ((c1 & 3) == (c2 & 3)) and ((c2 & 3) == (c3 & 3)) and ((c3 & 3) == (c4 & 3))
    cnt = np.zeros(13, np.int64)
    for i in range(5):
        cnt[ranks[i]] += 1
    mask = 0
    for i in range(5):
        mask |= 1 << ranks[i]
    straight_high = -1
    for hi in range(12, 3, -1):
        if (mask >> (hi - 4)) & 31 == 31:
            straight_high = hi
            break
    if straight_high < 0 and (mask & 0b1000000001111) == 0b1000000001111:
        straight_high = 3
    four = -1
    three = -1
    pair_hi = -1
    pair_lo = -1
    for r in range(12, -1, -1):
        if cnt[r] == 4:
            four = r
        elif cnt[r] == 3:
            three = r
        elif cnt[r] == 2:
            if pair_hi < 0:
                pair_hi = r
            else:
                pair_lo = r
    kick = np.zeros(5, np.int64)
    k = 0
    for r in range(12, -1, -1):
        if cnt[r] == 1:
            kick[k] = r
            k += 1
    B = 15
    if straight_high >= 0 and flush:
        return 8 * B**5 + straight_high
    if four >= 0:
        return 7 * B**5 + four * B + kick[0]
    if three >= 0 and pair_hi >= 0:
        return 6 * B**5 + three * B + pair_hi
    if flush:
        v = 5 * B**5
        for i in range(5):
            v += kick[i] * B ** (4 - i)
        return v
    if straight_high >= 0:
        return 4 * B**5 + straight_high
    if three >= 0:
        return 3 * B**5 + three * B**2 + kick[0] * B + kick[1]
    if pair_hi >= 0 and pair_lo >= 0:
        return 2 * B**5 + pair_hi * B**2 + pair_lo * B + kick[0]
    if pair_hi >= 0:
        return 1 * B**5 + pair_hi * B**3 + kick[0] * B**2 + kick[1] * B + kick[2]
    v = 0
    for i in range(5):
        v += kick[i] * B ** (4 - i)
    return v


@njit(cache=True)
def eval_best(cards, n):
    best = -1
    if n == 5:
        return eval5(cards[0], cards[1], cards[2], cards[3], cards[4])
    if n == 6:
        for skip in range(6):
            idx = np.empty(5, np.int64)
            k = 0
            for i in range(6):
                if i != skip:
                    idx[k] = cards[i]
                    k += 1
            v = eval5(idx[0], idx[1], idx[2], idx[3], idx[4])
            if v > best:
                best = v
        return best
    for s1 in range(7):
        for s2 in range(s1 + 1, 7):
            idx = np.empty(5, np.int64)
            k = 0
            for i in range(7):
                if i != s1 and i != s2:
                    idx[k] = cards[i]
                    k += 1
            v = eval5(idx[0], idx[1], idx[2], idx[3], idx[4])
            if v > best:
                best = v
    return best


@njit(cache=True)
def eval_streets(h1, h2, board, nboard):
    cards = np.empty(7, np.int64)
    cards[0] = h1
    cards[1] = h2
    for i in range(nboard):
        cards[2 + i] = board[i]
    vf = eval_best(cards, 5) if nboard >= 3 else -1
    vt = eval_best(cards, 6) if nboard >= 4 else -1
    vr = eval_best(cards, 7) if nboard >= 5 else -1
    return vf, vt, vr


@njit(cache=True, parallel=False)
def eval_table(h1, h2, board, nboard, out):
    for i in range(h1.shape[0]):
        vf, vt, vr = eval_streets(h1[i], h2[i], board[i], nboard[i])
        out[i, 0] = vf
        out[i, 1] = vt
        out[i, 2] = vr


def _assert_evaluator() -> None:
    def value(cards):
        return eval_best(np.array([card_to_int(c) for c in cards], np.int64), len(cards))

    order = [
        ["As", "Ks", "Qs", "Js", "Ts"],
        ["Ah", "Ad", "Ac", "As", "Kd"],
        ["Kh", "Kd", "Kc", "Qs", "Qd"],
        ["Ah", "Th", "7h", "4h", "2h"],
        ["Ah", "Kd", "Qc", "Js", "Td"],
        ["5h", "4d", "3c", "2s", "Ad"],
        ["Ah", "Ad", "Ac", "Ks", "Qd"],
        ["Ah", "Ad", "Kc", "Ks", "Qd"],
        ["Ah", "Ad", "Kc", "Js", "Qd"],
        ["Ah", "Kd", "Qc", "Js", "9d"],
    ]
    scores = [value(c) for c in order]
    assert scores == sorted(scores, reverse=True), scores


# ---------------------------------------------------------------------------
# Host metric (verbatim rules; local diagnostics only)
# ---------------------------------------------------------------------------
def _average_precision(y_true, scores) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    ranked = y_true[np.argsort(-scores, kind="mergesort")]
    hits = np.cumsum(ranked)
    ranks = np.arange(1, len(ranked) + 1)
    return float(np.sum((hits / ranks) * ranked) / positives)


def _clean_evidence(values) -> list[str]:
    out = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text != NO_EVIDENCE:
            out.append(text)
    return out


def host_score(solution, submission, row_id_column_name="pair_id", return_components=False):
    if row_id_column_name != "pair_id":
        raise ValueError("The row ID column must be pair_id.")
    if not REQUIRED_COLUMNS.issubset(submission.columns):
        raise ValueError(f"submission.csv is missing columns: {sorted(REQUIRED_COLUMNS - set(submission.columns))}")
    if submission["pair_id"].duplicated().any():
        raise ValueError("pair_id values must be unique.")
    if set(solution["pair_id"].astype(str)) != set(submission["pair_id"].astype(str)):
        raise ValueError("pair_id coverage mismatch.")
    truth = solution.set_index("pair_id").sort_index()
    predictions = submission.set_index("pair_id").loc[truth.index]
    risk = pd.to_numeric(predictions["risk_score"], errors="coerce")
    if risk.isna().any() or not risk.between(0, 1).all():
        raise ValueError("risk_score must be numeric and between 0 and 1.")
    predicted_behavior = predictions["predicted_behavior"].astype(str)
    invalid = set(predicted_behavior) - ALLOWED_BEHAVIORS
    if invalid:
        raise ValueError(f"Invalid predicted_behavior values: {sorted(invalid)}")
    for row in predictions.loc[:, EVIDENCE_COLUMNS].itertuples(index=False, name=None):
        evidence = _clean_evidence(list(row))
        if len(evidence) != len(set(evidence)):
            raise ValueError("Evidence hand IDs must not repeat within a pair.")
    y_true = pd.to_numeric(truth["risk_score"], errors="raise").to_numpy(dtype=int)
    risk_values = risk.to_numpy(dtype=float)
    pair_ap = _average_precision(y_true, risk_values)
    true_behavior = truth["predicted_behavior"].astype(str).to_numpy()
    pb = predicted_behavior.to_numpy()
    behavior_scores = []
    for behavior in TARGET_BEHAVIORS:
        bt = (true_behavior == behavior).astype(int)
        behavior_scores.append(0.0 if bt.sum() == 0 else _average_precision(bt, np.where(pb == behavior, risk_values, 0.0)))
    behavior_map = float(np.mean(behavior_scores))
    evidence_scores = []
    for position in np.flatnonzero(y_true == 1):
        relevant = set(_clean_evidence(truth.iloc[position].loc[list(EVIDENCE_COLUMNS)].tolist()))
        submitted = _clean_evidence(predictions.iloc[position].loc[list(EVIDENCE_COLUMNS)].tolist())
        if not relevant:
            evidence_scores.append(0.0)
            continue
        hits, precision_sum = 0, 0.0
        for rank, hand_id in enumerate(submitted[:5], start=1):
            if hand_id in relevant:
                hits += 1
                precision_sum += hits / rank
        evidence_scores.append(precision_sum / min(len(relevant), 5))
    evidence_map = float(np.mean(evidence_scores)) if evidence_scores else 0.0
    final = 0.70 * pair_ap + 0.20 * evidence_map + 0.10 * behavior_map
    if return_components:
        return {
            "final": final, "pair_ap": pair_ap, "evidence_map5": evidence_map, "behavior_map": behavior_map,
            "behavior_class_ap": dict(zip(TARGET_BEHAVIORS, behavior_scores)),
        }
    return float(final)


def map5_within_pairs(df, score_col="hand_score", target_col="is_evidence", pair_col="pair_id") -> float:
    vals = []
    ordered = df.sort_values([pair_col, score_col, "pot_bb", "hand_id"], ascending=[True, False, False, True], kind="mergesort")
    for _, grp in ordered.groupby(pair_col, sort=False):
        rel = grp[target_col].to_numpy()
        n_rel = int(rel.sum())
        if n_rel == 0:
            continue
        top = rel[:5]
        hits = np.cumsum(top)
        vals.append(float(np.sum((hits / np.arange(1, len(top) + 1)) * top) / min(n_rel, 5)))
    return float(np.mean(vals)) if vals else 0.0


def _spec_matrix(df) -> np.ndarray:
    return np.column_stack([np.asarray(df[c].to_numpy(), dtype=np.float64) for c in SPEC_COLS])


def mix_evidence_scores(global_s, spec_mat, family) -> np.ndarray:
    global_s = np.asarray(global_s, dtype=np.float64)
    if isinstance(family, pd.Series):
        idx = family.map(FAM_INDEX).fillna(-1).to_numpy(dtype=np.int16)
    else:
        idx = pd.Series(family).map(FAM_INDEX).fillna(-1).to_numpy(dtype=np.int16)
    out = global_s.copy()
    ok = idx >= 0
    if ok.any():
        rows = np.flatnonzero(ok)
        out[ok] = EVIDENCE_SPEC_W * spec_mat[rows, idx[ok]] + (1.0 - EVIDENCE_SPEC_W) * global_s[ok]
    return out.astype(np.float32)


def family_weight_matrix(fam_probs: dict, fam_prior: dict) -> np.ndarray:
    mat = np.column_stack([
        np.asarray(fam_probs[fam], dtype=np.float64) / max(float(fam_prior[fam]), 1e-9)
        for fam in TARGET_BEHAVIORS
    ])
    mat = np.clip(mat, 0.0, None)
    z = mat.sum(axis=1, keepdims=True)
    return np.divide(mat, z, out=np.zeros_like(mat), where=z > 0)


def mix_evidence_soft(global_s, spec_mat, weights, active=None) -> np.ndarray:
    global_s = np.asarray(global_s, dtype=np.float64)
    spec_mat = np.asarray(spec_mat, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    blended = (spec_mat * weights).sum(axis=1)
    out = global_s.copy()
    if active is None:
        active = weights.sum(axis=1) > 1e-12
    else:
        active = np.asarray(active, dtype=bool)
    if active.any():
        out[active] = EVIDENCE_SPEC_W * blended[active] + (1.0 - EVIDENCE_SPEC_W) * global_s[active]
    return out.astype(np.float32)


def report_map5(df, score_col: str, tag: str) -> tuple[float, dict]:
    pos = df[df["label"] == 1] if "label" in df.columns else df
    overall = map5_within_pairs(pos, score_col=score_col)
    per = {fam: map5_within_pairs(pos[pos["behavior_family"] == fam], score_col=score_col) for fam in TARGET_BEHAVIORS}
    log(f"{tag} MAP@5 = {overall:.4f}  per family: { {k: round(v, 4) for k, v in per.items()} }")
    return overall, per


def _lgb_bag(X, y, seeds=HAND_SEEDS, params=HAND_PARAMS, rounds=HAND_ROUNDS):
    return [lgb.train({**params, "seed": sd}, lgb.Dataset(X, y), num_boost_round=rounds) for sd in seeds]


def _predict_bag(models, X) -> np.ndarray:
    return np.mean([m.predict(X) for m in models], axis=0)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@dataclass
class Paths:
    data: Path
    work: Path

    @property
    def action_context(self):
        """Glob string for sharded action context (OOM-safe)."""
        return str(self.work / "action_context_shards" / "part_*.parquet")

    @property
    def player_hands(self) -> Path:
        return self.work / "player_hands.parquet"

    @property
    def player_hands_full(self):
        """Glob string for sharded player_hands_full (policy+postflop merged)."""
        return str(self.work / "player_hands_full_shards" / "part_*.parquet")

    @property
    def player_strength(self):
        """Glob string for sharded player strength (OOM-safe)."""
        return str(self.work / "player_strength_shards" / "part_*.parquet")

    @property
    def postflop_policy(self) -> Path:
        return self.work / "postflop_policy.parquet"

    @property
    def dev_pairs(self) -> Path:
        return self.work / "dev_pairs.parquet"

    @property
    def eval_pairs(self) -> Path:
        return self.work / "eval_pairs.parquet"

    @property
    def dev_pair_hands(self) -> Path:
        return self.work / "dev_pair_hands.parquet"

    @property
    def eval_pair_hands(self) -> Path:
        return self.work / "eval_pair_hands.parquet"

    @property
    def labelled_hand_feats(self) -> Path:
        return self.work / "labelled_hand_feats.parquet"

    @property
    def hand_models(self) -> Path:
        return self.work / "hand_models.pkl"

    @property
    def dev_pair_features(self) -> Path:
        return self.work / "dev_pair_features.parquet"

    @property
    def pair_models(self) -> Path:
        return self.work / "pair_models.pkl"

    @property
    def eval_pair_features(self) -> Path:
        return self.work / "eval_pair_features.parquet"

    @property
    def eval_candidates(self) -> Path:
        return self.work / "eval_candidates.parquet"


def resolve_paths() -> Paths:
    data = Path(os.environ.get("POKER_DATA_DIR", "/home/z/my-project/data"))
    work = Path(os.environ.get("POKER_WORK_DIR", "/home/z/my-project/work"))
    work.mkdir(parents=True, exist_ok=True)
    return Paths(data=data, work=work)
