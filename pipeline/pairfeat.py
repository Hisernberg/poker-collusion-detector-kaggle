"""Pair-hand feature builder (ported from 0.81 notebook + evidence-targeted additions).
One row per (pair, shared hand). Used by s2 (labelled), s3 (dev chunks), s5 (eval).
"""
import sys
import numpy as np
import polars as pl

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa

PLAYER_HANDS_FULL = None  # set by caller


def _side(player_hands, strength, prefix, key, hand_ids=None, table_ids=None):
    ph = pl.scan_parquet(player_hands)
    st = pl.scan_parquet(strength)
    if table_ids is not None:
        ph = ph.filter(pl.col("table_id").is_in(list(table_ids)))
        st = st.filter(pl.col("table_id").is_in(list(table_ids)))
    if hand_ids is not None:
        ph = ph.filter(pl.col("hand_id").is_in(list(hand_ids)))
        st = st.filter(pl.col("hand_id").is_in(list(hand_ids)))
    ph = ph.select(["hand_id", "player_id", "pot_bb", "players_at_showdown", "big_blind", *P_COLS])
    st = st.select(["hand_id", "player_id", *S_COLS])
    return ph.join(st, on=["hand_id", "player_id"]).select(
        ["hand_id", pl.col("player_id").alias(key), "pot_bb", "players_at_showdown", "big_blind",
         *[pl.col(c).alias(f"{prefix}_{c}") for c in P_COLS + S_COLS]]
    )


def _street_value(prefix, street_expr):
    return (
        pl.when(street_expr == 0).then(pl.col(f"{prefix}_chen") * 1000.0)
        .when(street_expr == 1).then(pl.col(f"{prefix}_v_flop").cast(pl.Float64))
        .when(street_expr == 2).then(pl.col(f"{prefix}_v_turn").cast(pl.Float64))
        .otherwise(pl.col(f"{prefix}_v_river").cast(pl.Float64))
    )


def _street_cat_expr():
    v = (
        pl.when(pl.col("street_no") == 1).then(pl.col("v_flop"))
        .when(pl.col("street_no") == 2).then(pl.col("v_turn"))
        .when(pl.col("street_no") == 3).then(pl.col("v_river"))
        .otherwise(-1)
    )
    return (
        pl.when(v.is_not_null() & (v >= 0))
        .then((v // B5).cast(pl.Int8))
        .otherwise(pl.lit(-1, dtype=pl.Int8))
        .alias("cat")
    )


def build_pair_hand_features(ph: pl.LazyFrame, paths, player_hands, out_path=None, return_df=False, hand_ids=None, table_ids=None):
    """ph: LF of pair-hand rows (pair_id, hand_id, table_id, hand_idx, player_1, player_2).
    Returns wide LF; caller collects (streaming) and either writes or consumes."""
    base = (
        ph.join(_side(player_hands, paths.player_strength, "p1", "player_1", hand_ids, table_ids), on=["hand_id", "player_1"])
        .join(_side(player_hands, paths.player_strength, "p2", "player_2", hand_ids, table_ids).drop(["pot_bb", "players_at_showdown", "big_blind"]), on=["hand_id", "player_2"])
        .with_columns(
            (pl.col("p1_net_bb") - pl.col("p2_net_bb")).alias("signed_net_diff"),
            (pl.col("p1_net_bb") - pl.col("p2_net_bb")).abs().alias("net_gap"),
            pl.min_horizontal((-pl.col("p1_net_bb")).clip(lower_bound=0), pl.col("p2_net_bb").clip(lower_bound=0)).alias("transfer_1_to_2"),
            pl.min_horizontal((-pl.col("p2_net_bb")).clip(lower_bound=0), pl.col("p1_net_bb").clip(lower_bound=0)).alias("transfer_2_to_1"),
            (pl.col("p1_vpip") & pl.col("p2_vpip")).cast(pl.Int8).alias("both_vpip"),
            (pl.col("p1_went_to_showdown") & pl.col("p2_went_to_showdown")).cast(pl.Int8).alias("both_showdown"),
            (pl.col("p1_folded") ^ pl.col("p2_folded")).cast(pl.Int8).alias("one_folded"),
            (((pl.col("p1_won_share") > 0) & (pl.col("p2_net_bb") < 0)) | ((pl.col("p2_won_share") > 0) & (pl.col("p1_net_bb") < 0))).cast(pl.Int8).alias("partner_won_other_lost"),
            (pl.col("p1_net_bb") < pl.col("p2_net_bb")).cast(pl.Int8).alias("loser_is_p1"),
            (pl.col("p1_n_aggr") + pl.col("p2_n_aggr")).alias("pair_aggr"),
            (pl.col("p1_n_raise") + pl.col("p2_n_raise")).alias("pair_raise"),
            (pl.col("p1_n_call") + pl.col("p2_n_call")).alias("pair_call"),
            (pl.col("p1_n_check") + pl.col("p2_n_check")).alias("pair_check"),
            (pl.col("p1_n_postflop_check") + pl.col("p2_n_postflop_check")).alias("pair_postflop_check"),
            ((pl.col("p1_n_aggr") > 0).cast(pl.Int8) + (pl.col("p2_n_aggr") > 0).cast(pl.Int8)).alias("n_partners_aggr"),
            pl.max_horizontal("p1_max_amount_bb", "p2_max_amount_bb").alias("max_amount_bb"),
            pl.max_horizontal("p1_last_street", "p2_last_street").alias("last_street"),
            (pl.col("p1_contrib_bb") + pl.col("p2_contrib_bb")).alias("pair_contrib_bb"),
            (pl.col("p1_vpip_a") & pl.col("p2_vpip_a")).cast(pl.Int8).alias("both_vpip_a"),
            (pl.col("p1_vpip_a").cast(pl.Int8) + pl.col("p2_vpip_a").cast(pl.Int8)).alias("n_vpip_a"),
            (pl.col("p1_pfr").cast(pl.Int8) + pl.col("p2_pfr").cast(pl.Int8)).alias("n_pfr"),
            (pl.col("p1_loose") + pl.col("p2_loose")).alias("loose_sum"),
            pl.min_horizontal("p1_loose", "p2_loose").alias("loose_min"),
            pl.max_horizontal("p1_loose", "p2_loose").alias("loose_max"),
            pl.max_horizontal("p1_tight", "p2_tight").alias("tight_max"),
            (pl.col("p1_loose_pfr") + pl.col("p2_loose_pfr")).alias("loose_pfr_sum"),
            (pl.col("p1_junk_vpip") + pl.col("p2_junk_vpip")).alias("junk_vpip_sum"),
            pl.min_horizontal("p1_junk_vpip", "p2_junk_vpip").alias("junk_vpip_min"),
            (pl.col("p1_junk_pfr") + pl.col("p2_junk_pfr")).alias("junk_pfr_sum"),
            (pl.col("p1_vpip_resid") + pl.col("p2_vpip_resid")).alias("vpip_resid_sum"),
            (pl.col("p1_post_loose") + pl.col("p2_post_loose")).alias("post_loose_sum"),
            pl.min_horizontal("p1_post_loose", "p2_post_loose").alias("post_loose_min"),
            pl.max_horizontal("p1_post_tight", "p2_post_tight").alias("post_tight_max"),
            (pl.col("p1_junk_continue") + pl.col("p2_junk_continue")).alias("junk_continue_sum"),
            pl.min_horizontal("p1_junk_continue", "p2_junk_continue").alias("junk_continue_min"),
            (pl.col("p1_post_resid") + pl.col("p2_post_resid")).alias("post_resid_sum"),
            (pl.col("p1_fold_made") + pl.col("p2_fold_made")).alias("fold_made_sum"),
            pl.max_horizontal("p1_chen", "p2_chen").alias("chen_max_pair"),
            pl.min_horizontal("p1_chen", "p2_chen").alias("chen_min_pair"),
            (pl.col("p1_pf_faced_raise") | pl.col("p2_pf_faced_raise")).cast(pl.Int8).alias("pf_faced_raise_any"),
            (pl.col("p1_pos").is_in([1, 2]).cast(pl.Int8) + pl.col("p2_pos").is_in([1, 2]).cast(pl.Int8)).alias("n_in_blinds"),
            pl.when(pl.col("p1_vpip_a") & pl.col("p2_vpip_a")).then(
                pl.when(pl.col("p1_pf_entry_no") > pl.col("p2_pf_entry_no")).then(pl.col("p1_chen")).otherwise(pl.col("p2_chen"))
            ).otherwise(99.0).alias("_second_chen"),
            pl.when(pl.col("p1_vpip_a") & pl.col("p2_vpip_a")).then(
                pl.when(pl.col("p1_pf_entry_no") > pl.col("p2_pf_entry_no")).then(pl.col("p1_loose")).otherwise(pl.col("p2_loose"))
            ).otherwise(0.0).alias("second_entrant_loose"),
        )
        .with_columns(
            ((pl.col("both_vpip_a") == 1) & (pl.col("chen_max_pair") <= 5)).cast(pl.Int8).alias("both_vpip_junk"),
            ((pl.col("both_vpip_a") == 1) & (pl.col("chen_min_pair") <= 4)).cast(pl.Int8).alias("both_vpip_one_junk"),
            (pl.col("_second_chen") <= 4).cast(pl.Int8).alias("second_entrant_junk"),
            ((pl.col("both_vpip_a") == 1) & (pl.col("loose_min") >= 1.5)).cast(pl.Int8).alias("both_surprising"),
            ((pl.col("n_pfr") >= 1) & (pl.col("junk_pfr_sum") >= 4)).cast(pl.Int8).alias("junk_raise"),
            pl.max_horizontal("transfer_1_to_2", "transfer_2_to_1").alias("transfer_any"),
            *[pl.when(pl.col("loser_is_p1") == 1).then(pl.col(f"p1_{c}")).otherwise(pl.col(f"p2_{c}")).alias(f"loser_{c}") for c in STRENGTH_CMP],
            *[pl.when(pl.col("loser_is_p1") == 1).then(pl.col(f"p2_{c}")).otherwise(pl.col(f"p1_{c}")).alias(f"winner_{c}") for c in STRENGTH_CMP],
            pl.when(pl.col("loser_is_p1") == 1).then(_street_value("p1", pl.col("p1_last_street"))).otherwise(_street_value("p2", pl.col("p2_last_street"))).alias("_loser_v_fold"),
            pl.when(pl.col("loser_is_p1") == 1).then(_street_value("p2", pl.col("p1_last_street"))).otherwise(_street_value("p1", pl.col("p2_last_street"))).alias("_winner_v_fold"),
        )
        .with_columns(
            (pl.col("loser_chen") - pl.col("winner_chen")).alias("chen_gap"),
            (pl.col("loser_chen") > pl.col("winner_chen")).cast(pl.Int8).alias("loser_stronger_preflop"),
            (pl.col("transfer_any") / (pl.col("pot_bb") + 1e-3)).clip(0, 1).alias("transfer_pot_ratio"),
            ((pl.col("loser_v_flop") >= 0) & (pl.col("loser_v_flop") > pl.col("winner_v_flop"))).cast(pl.Int8).alias("loser_beats_winner_flop"),
            ((pl.col("loser_v_turn") >= 0) & (pl.col("loser_v_turn") > pl.col("winner_v_turn"))).cast(pl.Int8).alias("loser_beats_winner_turn"),
            ((pl.col("loser_v_final") >= 0) & (pl.col("loser_v_final") > pl.col("winner_v_final"))).cast(pl.Int8).alias("loser_beats_winner_final"),
            (pl.col("loser_rk_final") == 1).cast(pl.Int8).alias("loser_best_final"),
            (pl.col("winner_rk_final") == 1).cast(pl.Int8).alias("winner_best_final"),
            (pl.col("winner_cat_final") - pl.col("loser_cat_final")).alias("cat_gap_final"),
            pl.max_horizontal("loser_cat_final", "winner_cat_final").alias("max_cat_final"),
            (pl.col("loser_folded") & (pl.col("_loser_v_fold") > pl.col("_winner_v_fold"))).cast(pl.Int8).alias("fold_better_hand"),
            ((pl.col("loser_cat_final") >= 1) & (pl.col("winner_cat_final") >= 1) & (pl.col("pair_aggr") == 0) & (pl.col("both_showdown") == 1)).cast(pl.Int8).alias("both_made_no_aggr"),
            ((pl.col("winner_rk_final") >= 3) & (pl.col("partner_won_other_lost") == 1)).cast(pl.Int8).alias("winner_weak_won"),
        )
        .drop(["_loser_v_fold", "_winner_v_fold", "_second_chen"])
    )

    members = pl.concat([
        ph.select(["pair_id", "hand_id", pl.col("player_1").alias("member"), pl.col("player_2").alias("partner"), pl.lit(1, dtype=pl.Int8).alias("is_p1")]),
        ph.select(["pair_id", "hand_id", pl.col("player_2").alias("member"), pl.col("player_1").alias("partner"), pl.lit(0, dtype=pl.Int8).alias("is_p1")]),
    ])
    ac = pl.scan_parquet(paths.action_context)
    if table_ids is not None:
        ac = ac.filter(pl.col("table_id").is_in(list(table_ids)))
    if hand_ids is not None:
        ac = ac.filter(pl.col("hand_id").is_in(list(hand_ids)))
    ac = ac.select(
        ["hand_id", "action_no", "street_no", "player_id", "action", "to_call", "players_active", "is_aggr", "last_aggr", "amount_pot_ratio", "to_call_bb"]
    )
    fold_at = ac.filter(pl.col("action") == "fold").group_by(["hand_id", "player_id"]).agg(
        pl.col("action_no").min().alias("partner_fold_no")
    ).rename({"player_id": "partner"})
    st = pl.scan_parquet(paths.player_strength)
    if table_ids is not None:
        st = st.filter(pl.col("table_id").is_in(list(table_ids)))
    if hand_ids is not None:
        st = st.filter(pl.col("hand_id").is_in(list(hand_ids)))
    st = st.select(["hand_id", "player_id", "v_flop", "v_turn", "v_river"])
    post_p = pl.scan_parquet(paths.postflop_policy).select(["cat", "street_no", "p_continue"])
    ma = (
        members.join(ac, left_on=["hand_id", "member"], right_on=["hand_id", "player_id"], how="inner")
        .join(fold_at, on=["hand_id", "partner"], how="left")
        .join(st, left_on=["hand_id", "member"], right_on=["hand_id", "player_id"], how="left")
        .with_columns(
            ((pl.col("to_call") > 0) & (pl.col("last_aggr") == pl.col("partner"))).alias("facing_partner"),
            ((pl.col("to_call") > 0) & pl.col("last_aggr").is_not_null() & (pl.col("last_aggr") != pl.col("partner"))).alias("facing_other"),
            ((pl.col("players_active") == 2) & (pl.col("partner_fold_no").is_null() | (pl.col("partner_fold_no") > pl.col("action_no")))).alias("true_hu"),
            (pl.col("street_no") == 0).alias("pf"),
            _street_cat_expr(),
            (pl.col("action") != "fold").alias("continued"),
        )
        .join(post_p, on=["cat", "street_no"], how="left")
        .with_columns(pl.col("p_continue").fill_null(0.5))
        .with_columns(
            pl.when(pl.col("facing_partner") & (pl.col("street_no") > 0) & (pl.col("to_call") > 0) & pl.col("continued") & (pl.col("cat") >= 0))
            .then(-pl.col("p_continue").log()).otherwise(0.0).cast(pl.Float32).alias("_post_loose_p"),
            pl.when(pl.col("facing_partner") & (pl.col("street_no") > 0) & (pl.col("to_call") > 0) & (pl.col("action") == "fold") & (pl.col("cat") >= 2))
            .then(-(1 - pl.col("p_continue")).log()).otherwise(0.0).cast(pl.Float32).alias("_fold_better_surp"),
            (pl.col("facing_partner") & (pl.col("street_no") > 0) & (pl.col("action") == "call") & (pl.col("cat") <= 1) & (pl.col("cat") >= 0)).alias("_junk_call_p"),
            (pl.col("facing_partner") & (pl.col("street_no") > 0) & (pl.col("action") == "fold") & (pl.col("cat") >= 2)).alias("_fold_made_p"),
            ((pl.col("street_no") > 0) & (pl.col("action") == "check") & (pl.col("cat") >= 2)).alias("_made_check"),
            # additions: size of calls/folds vs partner
            (pl.col("facing_partner") & (pl.col("action") == "call")).cast(pl.Float32).mul(pl.col("to_call_bb").fill_null(0.0)).alias("_call_p_bb"),
            pl.when(pl.col("facing_partner") & (pl.col("action") == "call")).then(pl.col("to_call_bb")).otherwise(None).alias("_call_p_bb_max"),
            pl.when(pl.col("facing_partner") & (pl.col("action") == "call")).then(pl.col("amount_pot_ratio")).otherwise(None).alias("_call_p_potfrac"),
            pl.when(pl.col("facing_partner") & (pl.col("action") == "fold")).then(pl.col("to_call_bb")).otherwise(None).alias("_fold_p_tocall"),
        )
    )
    inter = ma.group_by(["pair_id", "hand_id"]).agg(
        (pl.col("facing_partner") & (pl.col("action") == "fold")).sum().cast(pl.Int8).alias("fold_to_partner"),
        (pl.col("facing_partner") & (pl.col("action") == "call")).sum().cast(pl.Int8).alias("call_partner"),
        (pl.col("facing_partner") & pl.col("is_aggr")).sum().cast(pl.Int8).alias("raise_partner"),
        (pl.col("facing_other") & (pl.col("action") == "fold")).sum().cast(pl.Int8).alias("fold_to_other"),
        (pl.col("facing_other") & (pl.col("action") == "call")).sum().cast(pl.Int8).alias("call_other"),
        (pl.col("facing_other") & pl.col("is_aggr")).sum().cast(pl.Int8).alias("raise_other"),
        (pl.col("facing_partner") & (pl.col("action") == "fold") & (pl.col("is_p1") == 1)).sum().cast(pl.Int8).alias("p1_fold_to_partner"),
        (pl.col("facing_partner") & (pl.col("action") == "call") & (pl.col("is_p1") == 1)).sum().cast(pl.Int8).alias("p1_call_partner"),
        (pl.col("facing_partner") & (pl.col("action") == "fold") & pl.col("pf")).sum().cast(pl.Int8).alias("fold_to_partner_pf"),
        (pl.col("facing_partner") & (pl.col("action") == "fold") & ~pl.col("pf")).sum().cast(pl.Int8).alias("fold_to_partner_post"),
        (pl.col("facing_partner") & (pl.col("action") == "call") & ~pl.col("pf")).sum().cast(pl.Int8).alias("call_partner_post"),
        (pl.col("facing_partner") & (pl.col("action") == "call") & pl.col("pf")).sum().cast(pl.Int8).alias("call_partner_pf"),
        (pl.col("facing_partner") & pl.col("is_aggr") & pl.col("pf")).sum().cast(pl.Int8).alias("raise_partner_pf"),
        (pl.col("facing_partner") & pl.col("is_aggr") & ~pl.col("pf")).sum().cast(pl.Int8).alias("raise_partner_post"),
        (pl.col("facing_other") & (pl.col("action") == "fold") & pl.col("pf")).sum().cast(pl.Int8).alias("fold_to_other_pf"),
        (pl.col("facing_other") & (pl.col("action") == "fold") & ~pl.col("pf")).sum().cast(pl.Int8).alias("fold_to_other_post"),
        (pl.col("facing_other") & (pl.col("action") == "call") & pl.col("pf")).sum().cast(pl.Int8).alias("call_other_pf"),
        (pl.col("facing_other") & (pl.col("action") == "call") & ~pl.col("pf")).sum().cast(pl.Int8).alias("call_other_post"),
        (pl.col("facing_other") & pl.col("is_aggr") & pl.col("pf")).sum().cast(pl.Int8).alias("raise_other_pf"),
        (pl.col("facing_other") & pl.col("is_aggr") & ~pl.col("pf")).sum().cast(pl.Int8).alias("raise_other_post"),
        (pl.col("is_aggr") & pl.col("pf") & (pl.col("is_p1") == 1)).sum().cast(pl.Int8).alias("p1_aggr_pf"),
        (pl.col("is_aggr") & pl.col("pf") & (pl.col("is_p1") == 0)).sum().cast(pl.Int8).alias("p2_aggr_pf"),
        pl.col("true_hu").sum().cast(pl.Int8).alias("hu_actions"),
        (pl.col("true_hu") & (pl.col("action") == "check")).sum().cast(pl.Int8).alias("hu_check"),
        (pl.col("true_hu") & (pl.col("action") == "call")).sum().cast(pl.Int8).alias("hu_call"),
        (pl.col("true_hu") & pl.col("is_aggr")).sum().cast(pl.Int8).alias("hu_aggr"),
        (pl.col("true_hu") & (pl.col("street_no") >= 2) & (pl.col("action") == "check")).sum().cast(pl.Int8).alias("hu_late_check"),
        (pl.col("true_hu") & (pl.col("action") == "fold")).sum().cast(pl.Int8).alias("hu_fold"),
        (pl.col("is_aggr") & (pl.col("amount_pot_ratio") >= 1.0)).sum().cast(pl.Int8).alias("pair_overbets"),
        pl.col("_post_loose_p").sum().cast(pl.Float32).alias("post_loose_partner"),
        pl.col("_fold_better_surp").sum().cast(pl.Float32).alias("fold_better_surprise"),
        pl.col("_junk_call_p").sum().cast(pl.Int8).alias("junk_call_partner_post"),
        pl.col("_fold_made_p").sum().cast(pl.Int8).alias("fold_made_partner"),
        pl.col("_made_check").sum().cast(pl.Int8).alias("made_check_sum"),
        (pl.col("true_hu") & pl.col("_made_check")).sum().cast(pl.Int8).alias("made_check_hu"),
        pl.col("_call_p_bb").sum().cast(pl.Float32).alias("call_partner_bb"),
        pl.col("_call_p_bb_max").max().fill_null(0.0).cast(pl.Float32).alias("call_partner_max_bb"),
        pl.col("_call_p_potfrac").sum().fill_null(0.0).cast(pl.Float32).alias("call_partner_potfrac_sum"),
        pl.col("_fold_p_tocall").max().fill_null(0.0).cast(pl.Float32).alias("fold_partner_potfrac_max"),
    )
    responses = ac.filter((pl.col("to_call") > 0) & pl.col("last_aggr").is_not_null()).select(
        ["hand_id", pl.col("last_aggr").alias("member"), pl.col("player_id").alias("responder"), "action", "is_aggr", "street_no", (pl.col("street_no") == 0).alias("pf")]
    )
    press = (
        members.join(responses, on=["hand_id", "member"], how="inner")
        .filter(pl.col("responder") != pl.col("partner"))
        .group_by(["pair_id", "hand_id"]).agg(
            (pl.col("action") == "fold").sum().cast(pl.Int8).alias("outsider_fold_to_pair"),
            (pl.col("action") == "call").sum().cast(pl.Int8).alias("outsider_call_to_pair"),
            pl.col("is_aggr").sum().cast(pl.Int8).alias("outsider_raise_to_pair"),
            ((pl.col("action") == "fold") & pl.col("pf")).sum().cast(pl.Int8).alias("outsider_fold_to_pair_pf"),
        )
    )
    iso_st = (
        members.join(responses, on=["hand_id", "member"], how="inner")
        .filter(pl.col("responder") != pl.col("partner"))
        .group_by(["pair_id", "hand_id", "street_no"])
        .agg((pl.col("action") == "fold").sum().cast(pl.Int8).alias("_out_fold_st"))
    )
    pair_st = (
        ma.filter(pl.col("is_aggr"))
        .group_by(["pair_id", "hand_id", "street_no"])
        .agg(pl.col("member").n_unique().cast(pl.Int8).alias("_n_aggr_st"))
    )
    iso_flags = (
        pair_st.join(iso_st, on=["pair_id", "hand_id", "street_no"], how="left")
        .with_columns(pl.col("_out_fold_st").fill_null(0))
        .group_by(["pair_id", "hand_id"])
        .agg(
            ((pl.col("_n_aggr_st") >= 2) & (pl.col("_out_fold_st") >= 2)).max().cast(pl.Int8).alias("iso_street"),
            ((pl.col("street_no") == 0) & (pl.col("_n_aggr_st") >= 2) & (pl.col("_out_fold_st") >= 2)).max().cast(pl.Int8).alias("iso_street_pf"),
        )
    )
    out = (
        base.join(inter, on=["pair_id", "hand_id"], how="left").join(press, on=["pair_id", "hand_id"], how="left")
        .join(iso_flags, on=["pair_id", "hand_id"], how="left")
        .with_columns([pl.col(c).fill_null(0) for c in ACTION_FILL])
        .with_columns(
            ((pl.col("fold_to_partner") > 0) & (pl.col("loser_stronger_preflop") == 1)).cast(pl.Int8).alias("fold_stronger_to_partner"),
            ((pl.col("fold_to_partner") > 0) & (pl.col("fold_better_hand") == 1)).cast(pl.Int8).alias("fold_better_to_partner"),
            ((pl.col("loser_chen") >= 6) & (pl.col("winner_chen") >= 6) & (pl.col("raise_partner") == 0) & (pl.col("both_vpip") == 1)).cast(pl.Int8).alias("both_strong_no_raise"),
            (pl.col("call_partner") >= 2).cast(pl.Int8).alias("multi_call_partner"),
            ((pl.col("n_partners_aggr") == 2) & (pl.col("outsider_fold_to_pair") >= 1)).cast(pl.Int8).alias("squeeze"),
            ((pl.col("n_partners_aggr") == 2) & (pl.col("outsider_fold_to_pair") >= 1) & (pl.col("fold_to_partner") >= 1)).cast(pl.Int8).alias("squeeze_then_fold"),
            ((pl.col("n_partners_aggr") == 2) & (pl.col("outsider_fold_to_pair") >= 2)).cast(pl.Int8).alias("iso_multi"),
            ((pl.col("p1_aggr_pf") >= 1) & (pl.col("p2_aggr_pf") >= 1) & (pl.col("outsider_fold_to_pair_pf") >= 2)).cast(pl.Int8).alias("iso_multi_pf"),
            ((pl.col("raise_partner_pf") >= 1) & (pl.col("outsider_fold_to_pair_pf") >= 1)).cast(pl.Int8).alias("pf_squeeze"),
            ((pl.col("raise_partner_pf") >= 1) & (pl.col("outsider_fold_to_pair_pf") >= 1) & (pl.col("fold_to_partner") >= 1)).cast(pl.Int8).alias("squeeze_then_fold_pf"),
            ((pl.col("partner_won_other_lost") == 1) & (pl.col("call_partner") >= 1) & (pl.col("transfer_any") >= 10)).cast(pl.Int8).alias("dump"),
            ((pl.col("partner_won_other_lost") == 1) & (pl.col("call_partner") >= 1) & (pl.col("loser_beats_winner_final") == 1)).cast(pl.Int8).alias("dump_better_hand"),
            ((pl.col("both_showdown") == 1) & (pl.col("hu_check") >= 2) & (pl.col("pair_aggr") == 0)).cast(pl.Int8).alias("checkdown"),
            ((pl.col("both_showdown") == 1) & (pl.col("hu_late_check") >= 1) & (pl.max_horizontal("loser_chen", "winner_chen") >= 7)).cast(pl.Int8).alias("hu_checkdown_strong"),
            ((pl.col("hu_check") >= 1) & (pl.col("max_cat_final") >= 2) & (pl.col("pair_aggr") == 0)).cast(pl.Int8).alias("hu_check_two_pair_plus"),
            (pl.col("transfer_any") * pl.col("call_partner")).cast(pl.Float32).alias("transfer_x_call"),
            (pl.col("junk_call_partner_post") >= 1).cast(pl.Int8).alias("junk_call_after_partner_raise"),
            ((pl.col("p1_cat_final") >= 2) & (pl.col("p2_cat_final") >= 2) & (pl.col("pair_aggr") == 0) & (pl.col("both_showdown") == 1)).cast(pl.Int8).alias("both_passive_made"),
            (pl.col("fold_made_partner") >= 1).cast(pl.Int8).alias("fold_made_to_partner"),
        )
        .with_columns(
            (pl.col("transfer_any") / (pl.col("transfer_any").max().over("pair_id") + 1e-3)).cast(pl.Float32).alias("transfer_to_max"),
            (pl.col("net_gap") / (pl.col("net_gap").max().over("pair_id") + 1e-3)).cast(pl.Float32).alias("net_gap_to_max"),
            (pl.col("pot_bb") / (pl.col("pot_bb").max().over("pair_id") + 1e-3)).cast(pl.Float32).alias("pot_to_max"),
            (pl.col("outsider_fold_to_pair") / (pl.col("outsider_fold_to_pair").max().over("pair_id") + 1e-3)).cast(pl.Float32).alias("outsider_fold_to_max"),
            *[(pl.col(c).rank(method="average").over("pair_id") / pl.len().over("pair_id")).cast(pl.Float32).alias(f"{c}_prank") for c in PRANK_BASE],
        )
    )
    return out
