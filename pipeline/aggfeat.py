"""Pair-level aggregation: collapse pair-hand rows to one row per pair (ported + additions)."""
import sys
import numpy as np
import polars as pl
import pandas as pd

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa


def aggregate_pairs(hf: pl.DataFrame, baselines_phase: pl.DataFrame, pairs: pl.DataFrame, thr: dict) -> pd.DataFrame:
    cols = (
        ["pair_id", "hand_id", "table_id", "hand_idx", "player_1", "player_2", "hand_score", "hand_sus",
         "loser_is_p1", "signed_net_diff", "transfer_1_to_2", "transfer_2_to_1", "p1_call_partner", "p1_fold_to_partner",
         "winner_rk_final", "fold_to_partner_pf", "fold_to_partner_post", "call_partner_pf", "call_partner_post",
         "raise_partner_pf", "raise_partner_post",
         "fold_to_other_pf", "fold_to_other_post", "call_other_pf", "call_other_post", "raise_other_pf", "raise_other_post"]
        + [f"p{i}_{c}" for i in (1, 2) for c in CONTRAST]
        + sorted(set(AGG_MEAN + AGG_MAX + AGG_TOP3 + AGG_SUM))
    )
    cols = list(dict.fromkeys(cols))
    lf = hf.select(cols).lazy()
    aggs = [pl.len().alias("n_shared"), pl.col("table_id").first(), pl.col("player_1").first(), pl.col("player_2").first()]
    aggs += [pl.col(c).mean().alias(f"{c}_mean") for c in AGG_MEAN]
    aggs += [pl.col(c).max().alias(f"{c}_max") for c in AGG_MAX]
    aggs += [pl.col(c).top_k(3).mean().alias(f"{c}_top3") for c in AGG_TOP3]
    aggs += [pl.col(c).sum().alias(f"{c}_sum") for c in AGG_SUM]
    aggs += [pl.col(f"p{i}_{c}").cast(pl.Float32).mean().alias(f"_p{i}_{c}") for i in (1, 2) for c in CONTRAST]
    aggs += [
        pl.col("hand_score").mean().alias("hs_mean"), pl.col("hand_score").max().alias("hs_max"),
        pl.col("hand_score").top_k(3).mean().alias("hs_top3"), pl.col("hand_score").top_k(5).mean().alias("hs_top5"),
        pl.col("hand_score").quantile(0.9).alias("hs_p90"),
        (pl.col("hand_score") > thr["thr95"]).sum().alias("hs_n95"),
        (pl.col("hand_score") > thr["thr99"]).sum().alias("hs_n99"),
        (pl.col("hand_score") > thr["thr95"]).mean().alias("hs_f95"),
        pl.col("hand_sus").mean().alias("sus_mean"), pl.col("hand_sus").max().alias("sus_max"),
        pl.col("hand_sus").top_k(3).mean().alias("sus_top3"), pl.col("hand_sus").top_k(5).mean().alias("sus_top5"),
        pl.col("hand_sus").quantile(0.9).alias("sus_p90"),
        (pl.col("hand_sus") > thr["sus95"]).sum().alias("sus_n95"),
        (pl.col("hand_sus") > thr["sus99"]).sum().alias("sus_n99"),
        (pl.col("hand_sus") > thr["sus95"]).mean().alias("sus_f95"),
        pl.col("transfer_1_to_2").sum().alias("_t12"), pl.col("transfer_2_to_1").sum().alias("_t21"),
        pl.col("signed_net_diff").sum().alias("_snd"), pl.col("signed_net_diff").abs().sum().alias("_asnd"),
        pl.col("p1_call_partner").sum().alias("_p1_cp"), pl.col("call_partner").sum().alias("_cp"),
        pl.col("p1_fold_to_partner").sum().alias("_p1_fp"), pl.col("fold_to_partner").sum().alias("_fp"),
        pl.col("raise_partner").sum().alias("_rp"),
        pl.col("fold_to_other").sum().alias("_fto"), pl.col("call_other").sum().alias("_co"),
        pl.col("raise_other").sum().alias("_ro"),
        pl.col("fold_to_partner_pf").sum().alias("_fp_pf"), pl.col("call_partner_pf").sum().alias("_cp_pf"),
        pl.col("raise_partner_pf").sum().alias("_rp_pf"),
        pl.col("fold_to_partner_post").sum().alias("_fp_post"), pl.col("call_partner_post").sum().alias("_cp_post"),
        pl.col("raise_partner_post").sum().alias("_rp_post"),
        pl.col("fold_to_other_pf").sum().alias("_fto_pf"), pl.col("call_other_pf").sum().alias("_co_pf"),
        pl.col("raise_other_pf").sum().alias("_ro_pf"),
        pl.col("fold_to_other_post").sum().alias("_fto_post"), pl.col("call_other_post").sum().alias("_co_post"),
        pl.col("raise_other_post").sum().alias("_ro_post"),
        pl.col("hu_check").sum().alias("_huc"), pl.col("hu_aggr").sum().alias("_hua"),
        (pl.col("winner_rk_final") > 1).cast(pl.Float32).mean().alias("winner_not_best_mean"),
        pl.col("loser_is_p1").sort_by("hand_score", descending=True).head(5).mean().alias("_top5_loser_p1"),
        pl.col("loser_is_p1").sort_by("transfer_any", descending=True).head(5).mean().alias("_top5t_loser_p1"),
        pl.col("transfer_any").sort_by("transfer_any", descending=True).head(5).sum().alias("_top5_tsum"),
    ]
    pf = lf.group_by("pair_id").agg(aggs)
    burst = (
        lf.with_columns(
            ((pl.col("hand_idx").cast(pl.Float32) - pl.col("hand_idx").min().over("pair_id"))
             / (pl.col("hand_idx").max().over("pair_id") - pl.col("hand_idx").min().over("pair_id") + 1) * 8)
            .floor().clip(0, 7).cast(pl.Int8).alias("_bin")
        )
        .group_by(["pair_id", "_bin"]).agg(
            pl.col("hand_score").mean().alias("_b"),
            pl.col("hand_sus").mean().alias("_bs"),
            pl.col("transfer_any").mean().alias("_bt"),
        )
        .group_by("pair_id").agg(
            pl.col("_b").max().alias("hs_burst_max"), (pl.col("_b").max() - pl.col("_b").mean()).alias("hs_burst_excess"),
            pl.col("_bs").max().alias("sus_burst_max"), (pl.col("_bs").max() - pl.col("_bs").mean()).alias("sus_burst_excess"),
            pl.col("_bt").max().alias("transfer_burst_max"),
        )
    )
    pf = pf.join(burst, on="pair_id").collect(engine="streaming")
    for pfx, key in (("p1", "player_1"), ("p2", "player_2")):
        pf = pf.join(
            baselines_phase.rename({c: f"{pfx}_{c}" for c in baselines_phase.columns if c != "player_id"}).rename({"player_id": key}),
            on=key, how="left",
        )
    exprs = []
    for c in CONTRAST:
        for i in (1, 2):
            field = (pl.col(f"p{i}_base_{c}") * pl.col(f"p{i}_base_hands") - pl.col(f"_p{i}_{c}") * pl.col("n_shared")) / (
                pl.col(f"p{i}_base_hands") - pl.col("n_shared")
            ).clip(lower_bound=1)
            exprs.append(((pl.col(f"_p{i}_{c}") - field) * pl.col("n_shared") / (pl.col("n_shared") + SHRINK_N)).cast(pl.Float32).alias(f"p{i}_{c}_pf"))
    pf = pf.with_columns(exprs).with_columns(
        *[(pl.col(f"p1_{c}_pf") + pl.col(f"p2_{c}_pf")).alias(f"{c}_pf_sum") for c in CONTRAST],
        *[(pl.col(f"p1_{c}_pf") - pl.col(f"p2_{c}_pf")).abs().alias(f"{c}_pf_gap") for c in CONTRAST],
        *[pl.min_horizontal(f"p1_{c}_pf", f"p2_{c}_pf").alias(f"{c}_pf_min") for c in CONTRAST],
        ((pl.col("_t12") - pl.col("_t21")).abs() / (pl.col("_t12") + pl.col("_t21") + 1.0)).alias("transfer_imbalance"),
        ((pl.col("_t12") + pl.col("_t21")) / pl.col("n_shared")).alias("transfer_rate"),
        pl.max_horizontal("_t12", "_t21").alias("transfer_dominant"),
        (pl.col("_snd").abs() / (pl.col("_asnd") + 1.0)).alias("direction_consistency"),
        pl.max_horizontal(pl.col("_top5_loser_p1"), 1 - pl.col("_top5_loser_p1")).alias("top5_same_loser"),
        pl.max_horizontal(pl.col("_top5t_loser_p1"), 1 - pl.col("_top5t_loser_p1")).alias("top5t_same_loser"),
        (pl.max_horizontal(pl.col("_p1_cp"), pl.col("_cp") - pl.col("_p1_cp")) / (pl.col("_cp") + 1.0)).alias("call_partner_asym"),
        (pl.max_horizontal(pl.col("_p1_fp"), pl.col("_fp") - pl.col("_p1_fp")) / (pl.col("_fp") + 1.0)).alias("fold_partner_asym"),
        (pl.col("_fp") / (pl.col("_fp") + pl.col("_cp") + pl.col("_rp")).clip(lower_bound=1)).alias("fold_rate_partner"),
        (pl.col("_cp") / (pl.col("_fp") + pl.col("_cp") + pl.col("_rp")).clip(lower_bound=1)).alias("call_rate_partner"),
        (pl.col("_rp") / (pl.col("_fp") + pl.col("_cp") + pl.col("_rp")).clip(lower_bound=1)).alias("raise_rate_partner"),
        (pl.col("_fto") / (pl.col("_fto") + pl.col("_co") + pl.col("_ro")).clip(lower_bound=1)).alias("fold_rate_field"),
        (pl.col("_co") / (pl.col("_fto") + pl.col("_co") + pl.col("_ro")).clip(lower_bound=1)).alias("call_rate_field"),
        (pl.col("_ro") / (pl.col("_fto") + pl.col("_co") + pl.col("_ro")).clip(lower_bound=1)).alias("raise_rate_field"),
        ((pl.col("_fp") + pl.col("_cp") + pl.col("_rp")) / pl.col("n_shared")).alias("partner_priced_per_hand"),
        (pl.col("_huc") / (pl.col("_huc") + pl.col("_hua")).clip(lower_bound=1)).alias("hu_check_rate"),
        (pl.col("_p1_net_bb") - pl.col("p1_base_net_bb")).alias("p1_net_resid"),
        (pl.col("_p2_net_bb") - pl.col("p2_base_net_bb")).alias("p2_net_resid"),
        # additions: transfer direction alignment of top hands + top5 concentration
        (1.0 - pl.col("_top5t_loser_p1")).mul(pl.col("_top5t_loser_p1")).mul(4.0).alias("transfer_dir_align_top5"),
        (pl.col("_top5_tsum") / (pl.col("_t12") + pl.col("_t21") + 1e-3)).clip(0, 1).alias("transfer_top5_share"),
    ).with_columns(
        (pl.col("p1_net_resid") - pl.col("p2_net_resid")).abs().alias("net_resid_gap"),
        pl.min_horizontal("p1_net_resid", "p2_net_resid").alias("net_resid_min"),
        pl.max_horizontal("p1_net_resid", "p2_net_resid").alias("net_resid_max"),
        (pl.col("fold_rate_partner") - pl.col("fold_rate_field")).alias("fold_contrast"),
        (pl.col("call_rate_partner") - pl.col("call_rate_field")).alias("call_contrast"),
        (pl.col("raise_rate_partner") - pl.col("raise_rate_field")).alias("raise_contrast"),
        (pl.col("_fp_pf") / (pl.col("_fp_pf") + pl.col("_cp_pf") + pl.col("_rp_pf")).clip(lower_bound=1)
         - pl.col("_fto_pf") / (pl.col("_fto_pf") + pl.col("_co_pf") + pl.col("_ro_pf")).clip(lower_bound=1)
         ).alias("fold_contrast_pf"),
        (pl.col("_fp_post") / (pl.col("_fp_post") + pl.col("_cp_post") + pl.col("_rp_post")).clip(lower_bound=1)
         - pl.col("_fto_post") / (pl.col("_fto_post") + pl.col("_co_post") + pl.col("_ro_post")).clip(lower_bound=1)
         ).alias("fold_contrast_post"),
    )
    drop = [c for c in pf.columns if c.startswith("_") or c.startswith("p1_base_") or c.startswith("p2_base_")]
    pf = pf.drop(drop)
    pf = pf.with_columns([
        (pl.col(c).rank(method="average").over("table_id") / pl.len().over("table_id")).cast(pl.Float32).alias(f"{c}_trank")
        for c in TRANK_COLS if c in pf.columns
    ])
    keep = [c for c in pairs.columns if c in ("pair_id", "label", "behavior_family", "is_labeled", "fold", "shared_hands", "chunk")]
    return pf.join(pairs.select(keep), on="pair_id", how="left").to_pandas()
