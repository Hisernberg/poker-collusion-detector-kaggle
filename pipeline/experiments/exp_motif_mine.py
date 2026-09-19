"""MOTIF MINER: grid-search precise planted-action templates per family.
Template space: (facing_partner, street, action, to_call_bb threshold, strength bucket chen/cat).
Metric: evidence-hand coverage (recall) vs control-hand contamination (1 - precision proxy).
"""
import sys
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa

WORK = "/home/z/my-project/work"
php = pl.read_parquet(f"{WORK}/forensics_pair_hands2.parquet").unique(subset=["pair_id", "hand_id"])
acts = pl.read_parquet(f"{WORK}/forensics_actions.parquet")
seats = pl.read_parquet(f"{WORK}/forensics_seats_all.parquet")
# strength for cat at action street
hand_ids_all = php["hand_id"].unique().to_list()
st = (pl.scan_parquet("/home/z/my-project/work/player_strength_shards/part_*.parquet")
      .filter(pl.col("hand_id").is_in(hand_ids_all))
      .select(["hand_id", "player_id", "v_flop", "v_turn", "v_river"]).collect(engine="streaming"))

acts = acts.join(st, on=["hand_id", "player_id"], how="left")
acts = acts.with_columns(
    pl.when(pl.col("street_no") == 1).then(pl.col("v_flop"))
    .when(pl.col("street_no") == 2).then(pl.col("v_turn"))
    .when(pl.col("street_no") == 3).then(pl.col("v_river"))
    .otherwise(-1).alias("_v")
)
acts = acts.with_columns(
    pl.when(pl.col("_v") >= 0).then((pl.col("_v") // B5).cast(pl.Int8)).otherwise(pl.lit(-1, dtype=pl.Int8)).alias("cat")
)
# member context: is this action by a pair member facing their partner?
mem = pl.concat([
    php.select(["pair_id", "hand_id", pl.col("p_low").alias("member"), pl.col("p_high").alias("partner"), "is_ev", "behavior_family"]),
    php.select(["pair_id", "hand_id", pl.col("p_high").alias("member"), pl.col("p_low").alias("partner"), "is_ev", "behavior_family"]),
])
mchen = seats.select(["hand_id", pl.col("player_id").alias("member"), "chen"])
ma = (
    mem.join(mchen, on=["hand_id", "member"], how="left")
    .join(acts, left_on=["hand_id", "member"], right_on=["hand_id", "player_id"], how="inner")
    .with_columns(
        ((pl.col("to_call") > 0) & (pl.col("last_aggr") == pl.col("partner"))).alias("vs_partner"),
        ((pl.col("to_call") > 0) & pl.col("last_aggr").is_not_null() & (pl.col("last_aggr") != pl.col("partner"))).alias("vs_other"),
    )
)
A = ma.to_pandas()
A.to_parquet(f"{WORK}/motif_actions.parquet")
print("member-action rows:", len(A), "in ev hands:", A["is_ev"].sum())

# per-hand coverage helper
hand_is_ev = A.groupby("hand_id")["is_ev"].first()
fam_of_hand = A.groupby("hand_id")["behavior_family"].first()
ev_hands = set(hand_is_ev[hand_is_ev].index)
ctl_hands = set(hand_is_ev[~hand_is_ev].index)
print("ev hands:", len(ev_hands), "ctl hands:", len(ctl_hands))

def eval_template(mask):
    """mask: boolean Series over A. Return (recall per fam, ctl contamination)."""
    h = A.loc[mask, "hand_id"].unique()
    hset = set(h)
    rec = {f: len(hset & {h_ for h_ in ev_hands if fam_of_hand.get(h_) == f}) / max(len([h_ for h_ in ev_hands if fam_of_hand.get(h_) == f]), 1) for f in ["directed_transfer", "soft_play", "coordinated_isolation"]}
    n_ev = len(hset & ev_hands)
    n_ct = len(hset & ctl_hands)
    prec = n_ev / max(n_ev + n_ct, 1)
    return rec, prec

grids = []
for vs in ["vs_partner", "vs_other", "any"]:
    for street in [0, 1, 2, 3, "any"]:
        for action in ["call", "fold", "raise", "bet", "all_in", "any"]:
            for tc in [0, 1, 2, 3, 5, 8, 12, "any"]:
                for catmax in [0, 1, 2, "any"]:
                    for chenmax in [4, 6, 8, "any"]:
                        grids.append((vs, street, action, tc, catmax, chenmax))
print("grid size:", len(grids))

mask_cache = {
    "vs_partner": A["vs_partner"], "vs_other": A["vs_other"], "any": pd.Series(True, index=A.index),
}
rows = []
for (vs, street, action, tc, catmax, chenmax) in grids:
    m = mask_cache[vs]
    if street != "any":
        m = m & (A["street_no"] == street) if vs == "any" else mask_cache[vs] & (A["street_no"] == street)
    if action != "any":
        m = m & (A["action"] == action)
    if tc != "any":
        m = m & (A["to_call_bb"] >= tc)
    if catmax != "any":
        m = m & (A["cat"] >= 0) & (A["cat"] <= catmax)
    if chenmax != "any":
        m = m & (A["chen"] <= chenmax)
    m = m.fillna(False)
    if m.sum() < 50:
        continue
    rec, prec = eval_template(m)
    best_fam = max(rec, key=rec.get)
    rows.append((vs, street, action, tc, catmax, chenmax, rec["directed_transfer"], rec["soft_play"], rec["coordinated_isolation"], prec, int(m.sum())))

R = pd.DataFrame(rows, columns=["vs", "street", "action", "tc", "catmax", "chenmax", "rec_dt", "rec_sp", "rec_ci", "prec", "n"])
R["score"] = R[["rec_dt", "rec_sp", "rec_ci"]].max(axis=1)
R = R.sort_values("score", ascending=False)
print("\nTOP 30 templates by best-family recall:")
print(R.head(30).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
R.to_csv(f"{WORK}/motif_results.csv", index=False)
