"""Worker: build ONE shard of a given stage. Memory is fully released on exit.
Usage: python3 s1_worker.py <stage> <shard_idx>   stages: ac|ph|pol|st|phf
"""
import sys
import numpy as np
import polars as pl

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa
from poker_lib import _assert_evaluator

stage, si = sys.argv[1], int(sys.argv[2])
paths = resolve_paths()
WORK = paths.work
N_SH = 24
DATA = paths.data

hands = pl.read_parquet(WORK / "hands_meta.parquet")
hands_small = hands.select(["hand_id", "table_id", "phase", "hand_idx", "big_blind", "final_pot", "players_at_showdown", "button_seat"]).lazy()
AC_GLOB = str(WORK / "action_context_shards" / "part_*.parquet")
PH_GLOB = str(WORK / "player_hands_shards" / "part_*.parquet")
ST_GLOB = str(WORK / "player_strength_shards" / "part_*.parquet")
seats_lf = pl.scan_parquet(DATA / "seats.parquet")

if stage == "ac":
    out = WORK / "action_context_shards" / f"part_{si:02d}.parquet"
    if out.exists():
        sys.exit(0)
    (
        pl.scan_parquet(DATA / "actions.parquet")
        .filter((pl.col("hand_id").hash(seed=7) % N_SH) == si)
        .join(hands.lazy().select(["hand_id", "big_blind", "phase", "table_id"]), on="hand_id")
        .sort(["hand_id", "action_no"])
        .with_columns(
            (pl.col("action").is_in(AGGR) | ((pl.col("action") == "all_in") & (pl.col("amount") > pl.col("to_call")))).alias("is_aggr"),
            (pl.col("amount") / pl.col("big_blind")).cast(pl.Float32).alias("amount_bb"),
            (pl.col("to_call") / pl.col("big_blind")).cast(pl.Float32).alias("to_call_bb"),
            (pl.col("amount") / pl.max_horizontal("pot_before", "big_blind")).clip(0, 20).cast(pl.Float32).alias("amount_pot_ratio"),
            pl.col("street").replace_strict({"preflop": 0, "flop": 1, "turn": 2, "river": 3}, return_dtype=pl.Int8).alias("street_no"),
        )
        .with_columns(pl.when(pl.col("is_aggr")).then(pl.col("player_id")).otherwise(None).alias("_ag"))
        .with_columns(pl.col("_ag").shift(1).forward_fill().over(["hand_id", "street"]).alias("last_aggr"))
        .select(["hand_id", "table_id", "phase", "action_no", "street_no", "player_id", "action", "to_call", "players_active", "is_aggr", "last_aggr", "amount_bb", "to_call_bb", "amount_pot_ratio"])
        .sink_parquet(out)
    )

elif stage == "ph":
    out = WORK / "player_hands_shards" / f"part_{si:02d}.parquet"
    if out.exists():
        sys.exit(0)
    cards = seats_lf.select(["hole_card_1", "hole_card_2"]).unique().collect()
    cards = cards.with_columns(pl.Series("chen", [chen(a, b) for a, b in zip(cards["hole_card_1"], cards["hole_card_2"])], dtype=pl.Float32))
    ac_agg = (
        pl.scan_parquet(AC_GLOB)
        .filter((pl.col("hand_id").hash(seed=7) % N_SH) == si)
        .group_by(["hand_id", "player_id"])
        .agg(
            pl.len().cast(pl.Int16).alias("n_actions"),
            pl.col("is_aggr").sum().cast(pl.Int16).alias("n_aggr"),
            (pl.col("action") == "raise").sum().cast(pl.Int16).alias("n_raise"),
            (pl.col("action") == "call").sum().cast(pl.Int16).alias("n_call"),
            (pl.col("action") == "check").sum().cast(pl.Int16).alias("n_check"),
            (pl.col("action") == "all_in").sum().cast(pl.Int16).alias("n_allin"),
            ((pl.col("action") == "fold") & (pl.col("to_call") > 0)).sum().cast(pl.Int16).alias("n_fold_facing"),
            ((pl.col("street_no") > 0) & (pl.col("action") == "check")).sum().cast(pl.Int16).alias("n_postflop_check"),
            (pl.col("players_active") == 2).sum().cast(pl.Int16).alias("n_hu_actions"),
            pl.col("street_no").max().cast(pl.Int8).alias("last_street"),
            pl.col("amount_bb").max().fill_null(0).cast(pl.Float32).alias("max_amount_bb"),
            pl.col("to_call_bb").max().fill_null(0).cast(pl.Float32).alias("max_to_call_bb"),
            ((pl.col("street_no") == 0) & pl.col("action").is_in(ENTRY_ACTIONS)).any().alias("vpip_a"),
            ((pl.col("street_no") == 0) & pl.col("is_aggr")).any().alias("pfr"),
            pl.col("action_no").filter((pl.col("street_no") == 0) & pl.col("action").is_in(ENTRY_ACTIONS)).min().alias("pf_entry_no"),
            (pl.col("to_call_bb").filter(pl.col("street_no") == 0).first() > 1.0).alias("pf_faced_raise"),
            (pl.col("street_no") == 0).any().alias("acted_pf"),
        )
    )
    (
        seats_lf.filter((pl.col("hand_id").hash(seed=7) % N_SH) == si)
        .join(hands_small, on="hand_id")
        .join(cards.lazy(), on=["hole_card_1", "hole_card_2"])
        .with_columns(
            ((pl.col("seat_no") - pl.col("button_seat") + 6) % 6).cast(pl.Int8).alias("pos"),
            (
                pl.max_horizontal(pl.col("hole_card_1").str.slice(0, 1).replace_strict(RANKS, return_dtype=pl.Int16), pl.col("hole_card_2").str.slice(0, 1).replace_strict(RANKS, return_dtype=pl.Int16)) * 15
                + pl.min_horizontal(pl.col("hole_card_1").str.slice(0, 1).replace_strict(RANKS, return_dtype=pl.Int16), pl.col("hole_card_2").str.slice(0, 1).replace_strict(RANKS, return_dtype=pl.Int16))
                + 300 * (pl.col("hole_card_1").str.slice(1, 1) == pl.col("hole_card_2").str.slice(1, 1)).cast(pl.Int16)
            ).cast(pl.Int16).alias("hole_class"),
        )
        .with_columns(
            (pl.col("final_pot") / pl.col("big_blind")).cast(pl.Float32).alias("pot_bb"),
            (pl.col("total_contribution") / pl.col("big_blind")).cast(pl.Float32).alias("contrib_bb"),
            (pl.col("net_chips") / pl.col("big_blind")).cast(pl.Float32).alias("net_bb"),
            (pl.col("total_contribution") > pl.col("big_blind")).alias("vpip"),
        )
        .join(ac_agg, on=["hand_id", "player_id"], how="left")
        .with_columns([pl.col(c).fill_null(0) for c in FILL0_PLAYER])
        .with_columns(
            pl.col("vpip_a").fill_null(False), pl.col("pfr").fill_null(False), pl.col("acted_pf").fill_null(False),
            pl.col("pf_entry_no").fill_null(9999).cast(pl.Int32), pl.col("pf_faced_raise").fill_null(False),
        )
        .select(["hand_id", "player_id", "table_id", "phase", "hand_idx", "big_blind", "pot_bb", "players_at_showdown", "seat_no", "pos", "hole_class", "contrib_bb", "net_bb", "vpip", "vpip_a", "pfr", "pf_entry_no", "pf_faced_raise", "acted_pf", "folded", "went_to_showdown", "won_share", "chen", *FILL0_PLAYER])
        .sink_parquet(out)
    )

elif stage == "pol":
    out = WORK / "player_hands_policy_shards" / f"part_{si:02d}.parquet"
    if out.exists():
        sys.exit(0)
    pol = pl.scan_parquet(WORK / "preflop_policy.parquet")
    (
        pl.scan_parquet(PH_GLOB.replace("*", f"{si:02d}"))
        .join(pol, on=["hole_class", "pos", "pf_faced_raise"], how="left")
        .with_columns(pl.col("p_vpip").fill_null(0.3), pl.col("p_pfr").fill_null(0.1))
        .with_columns(
            (pl.col("vpip_a").cast(pl.Float32) * (-pl.col("p_vpip").log())).cast(pl.Float32).alias("loose"),
            ((~pl.col("vpip_a")).cast(pl.Float32) * (-(1 - pl.col("p_vpip")).log())).cast(pl.Float32).alias("tight"),
            (pl.col("pfr").cast(pl.Float32) * (-pl.col("p_pfr").log())).cast(pl.Float32).alias("loose_pfr"),
            (pl.col("vpip_a").cast(pl.Float32) * (8.0 - pl.col("chen")).clip(lower_bound=0)).cast(pl.Float32).alias("junk_vpip"),
            (pl.col("pfr").cast(pl.Float32) * (8.0 - pl.col("chen")).clip(lower_bound=0)).cast(pl.Float32).alias("junk_pfr"),
            (pl.col("vpip_a").cast(pl.Float32) - pl.col("p_vpip")).cast(pl.Float32).alias("vpip_resid"),
        )
        .sink_parquet(out)
    )

elif stage == "st":
    out = WORK / "player_strength_shards" / f"part_{si:02d}.parquet"
    if out.exists():
        sys.exit(0)
    _assert_evaluator()
    hb = pl.read_parquet(DATA / "hands.parquet", columns=["hand_id", "board_cards"]).with_row_index("hrow")
    bl = hb["board_cards"].fill_null("").str.split(" ").list.eval(pl.element().replace_strict(CARD_MAP, default=-1, return_dtype=pl.Int64))
    board = np.column_stack([bl.list.get(i, null_on_oob=True).fill_null(-1).to_numpy() for i in range(5)]).astype(np.int64)
    nboard = (board >= 0).sum(axis=1).astype(np.int64)
    hb_small = hb.lazy().select(["hand_id", "hrow"]).join(hands.select(["hand_id", "table_id"]).lazy(), on="hand_id").select(["hand_id", "table_id", "hrow"])
    ps = (
        seats_lf.filter((pl.col("hand_id").hash(seed=7) % N_SH) == si)
        .select(["hand_id", "player_id", "hole_card_1", "hole_card_2"])
        .with_columns(
            pl.col("hole_card_1").replace_strict(CARD_MAP, return_dtype=pl.Int64).alias("h1"),
            pl.col("hole_card_2").replace_strict(CARD_MAP, return_dtype=pl.Int64).alias("h2"),
        )
        .join(hb_small, on="hand_id")
        .select(["hand_id", "table_id", "player_id", "h1", "h2", "hrow"])
        .collect(engine="streaming")
    )
    hrow = ps["hrow"].to_numpy().astype(np.int64)
    out_v = np.empty((ps.height, 3), np.int64)
    eval_table(ps["h1"].to_numpy().astype(np.int64), ps["h2"].to_numpy().astype(np.int64), board[hrow], nboard[hrow], out_v)
    nb_p = nboard[hrow]
    v_final = np.where(nb_p >= 5, out_v[:, 2], np.where(nb_p == 4, out_v[:, 1], np.where(nb_p == 3, out_v[:, 0], -1)))
    ps = ps.with_columns(
        pl.Series("v_flop", out_v[:, 0]), pl.Series("v_turn", out_v[:, 1]), pl.Series("v_river", out_v[:, 2]), pl.Series("v_final", v_final),
    ).drop(["h1", "h2", "hrow"])
    ps = ps.with_columns(
        pl.when(pl.col("v_final") >= 0).then(pl.col("v_final") // B5).otherwise(-1).cast(pl.Int8).alias("cat_final"),
        *[
            pl.when(pl.col(c) >= 0).then(pl.col(c).rank(descending=True, method="min").over("hand_id")).otherwise(None).cast(pl.Int8).alias(c.replace("v_", "rk_"))
            for c in ["v_flop", "v_turn", "v_river", "v_final"]
        ],
    )
    ps.write_parquet(out)

elif stage == "phf":
    out = WORK / "player_hands_full_shards" / f"part_{si:02d}.parquet"
    if out.exists():
        sys.exit(0)
    pol2 = pl.scan_parquet(WORK / "postflop_policy.parquet")
    st_lf = pl.scan_parquet(ST_GLOB).select(["hand_id", "player_id", "v_flop", "v_turn", "v_river"])
    ac = pl.scan_parquet(AC_GLOB).filter(((pl.col("hand_id").hash(seed=7) % N_SH) == si) & (pl.col("street_no") > 0) & (pl.col("to_call") > 0))
    st = st_lf.filter((pl.col("hand_id").hash(seed=7) % N_SH) == si)
    faced = (
        ac.join(st, on=["hand_id", "player_id"])
        .with_columns(
            pl.when(pl.col("street_no") == 1).then(pl.col("v_flop"))
            .when(pl.col("street_no") == 2).then(pl.col("v_turn"))
            .when(pl.col("street_no") == 3).then(pl.col("v_river"))
            .otherwise(-1).alias("_v")
        )
        .with_columns(
            pl.when(pl.col("_v").is_not_null() & (pl.col("_v") >= 0)).then((pl.col("_v") // B5).cast(pl.Int8)).otherwise(pl.lit(-1, dtype=pl.Int8)).alias("cat")
        )
        .join(pol2, on=["cat", "street_no"], how="left")
        .with_columns(pl.col("p_continue").fill_null(0.5), (pl.col("action") != "fold").alias("continued"))
        .with_columns(
            pl.when((pl.col("cat") >= 0) & pl.col("continued")).then(-pl.col("p_continue").log()).otherwise(0.0).cast(pl.Float32).alias("_loose"),
            pl.when((pl.col("cat") >= 0) & ~pl.col("continued")).then(-(1 - pl.col("p_continue")).log()).otherwise(0.0).cast(pl.Float32).alias("_tight"),
            pl.when(pl.col("cat") >= 0).then(pl.col("continued").cast(pl.Float32) - pl.col("p_continue")).otherwise(0.0).cast(pl.Float32).alias("_resid"),
            (pl.col("continued") & (pl.col("cat") <= 1)).cast(pl.Float32).alias("_junk"),
            ((~pl.col("continued")) & (pl.col("cat") >= 2)).cast(pl.Float32).alias("_fold_made"),
        )
        .group_by(["hand_id", "player_id"])
        .agg(
            pl.col("_loose").sum().cast(pl.Float32).alias("post_loose"),
            pl.col("_tight").sum().cast(pl.Float32).alias("post_tight"),
            pl.col("_resid").sum().cast(pl.Float32).alias("post_resid"),
            pl.col("_junk").sum().cast(pl.Float32).alias("junk_continue"),
            pl.col("_fold_made").sum().cast(pl.Float32).alias("fold_made"),
        )
    )
    (
        pl.scan_parquet(WORK / "player_hands_policy_shards" / f"part_{si:02d}.parquet")
        .join(faced, on=["hand_id", "player_id"], how="left")
        .with_columns([pl.col(c).fill_null(0).cast(pl.Float32) for c in POST_PLAYER_COLS])
        .sink_parquet(out)
    )
print(f"worker {stage} {si} done")
