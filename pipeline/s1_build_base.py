"""Stage 1 driver: runs shard workers as subprocesses (memory-safe), then global steps.
Idempotent: skips completed shards/artifacts.
"""
import sys, gc, subprocess, json
import numpy as np
import polars as pl
from itertools import combinations

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa

paths = resolve_paths()
WORK = paths.work
N_SH = 24
PY = sys.executable
log(f"S1 driver start; DATA={paths.data}")

# hands meta
if not (WORK / "hands_meta.parquet").exists():
    hands = (
        pl.read_parquet(paths.data / "hands.parquet",
                        columns=["hand_id", "table_id", "started_at", "phase", "big_blind", "final_pot", "players_at_showdown", "button_seat"])
        .sort(["table_id", "started_at", "hand_id"])
        .with_columns(pl.int_range(0, pl.len()).over("table_id").cast(pl.Int16).alias("hand_idx"))
    )
    hands.write_parquet(WORK / "hands_meta.parquet")
    del hands
hands = pl.read_parquet(WORK / "hands_meta.parquet")
n_dev = hands.filter(pl.col("phase") == "development").group_by("table_id").len()["len"].max()
labels = pl.read_csv(paths.data / "development_labels.csv")
eval_pairs_raw = pl.read_csv(paths.data / "evaluation_pairs.csv")
positives = set(pl.concat([labels.filter(pl.col("label") == 1)["player_1"], labels.filter(pl.col("label") == 1)["player_2"]]).to_list())
log(f"hands {hands.height:,}; dev/table {n_dev}; positives {len(positives)}")


def run_stage(stage: str):
    import pathlib
    d = {
        "ac": WORK / "action_context_shards", "ph": WORK / "player_hands_shards",
        "pol": WORK / "player_hands_policy_shards", "st": WORK / "player_strength_shards",
        "phf": WORK / "player_hands_full_shards",
    }[stage]
    d.mkdir(exist_ok=True)
    for si in range(N_SH):
        out = d / f"part_{si:02d}.parquet"
        if out.exists():
            continue
        r = subprocess.run([PY, "/home/z/my-project/scripts/s1_worker.py", stage, str(si)], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-3000:])
            raise RuntimeError(f"worker {stage} {si} failed")
        log(f"{stage} shard {si + 1}/{N_SH} ok")


run_stage("ac")
log(f"ac rows: {pl.scan_parquet(str(WORK / 'action_context_shards' / 'part_*.parquet')).select(pl.len()).collect().item():,}")
run_stage("ph")

# preflop policy (global, small)
if not (WORK / "preflop_policy.parquet").exists():
    ph_lf = pl.scan_parquet(str(WORK / "player_hands_shards" / "part_*.parquet"))
    cls = ph_lf.group_by("hole_class").agg(pl.col("vpip_a").cast(pl.Float32).mean().alias("_cls_vpip"), pl.col("pfr").cast(pl.Float32).mean().alias("_cls_pfr")).collect(engine="streaming")
    ctx = ph_lf.group_by(["hole_class", "pos", "pf_faced_raise"]).agg(
        pl.len().alias("_n"), pl.col("vpip_a").cast(pl.Float32).sum().alias("_sv"), pl.col("pfr").cast(pl.Float32).sum().alias("_sp")
    ).join(cls.lazy(), on="hole_class").collect(engine="streaming")
    k = POLICY_PSEUDOCOUNTS
    policy = ctx.with_columns(
        ((pl.col("_sv") + k * pl.col("_cls_vpip")) / (pl.col("_n") + k)).clip(1e-4, 1 - 1e-4).cast(pl.Float32).alias("p_vpip"),
        ((pl.col("_sp") + k * pl.col("_cls_pfr")) / (pl.col("_n") + k)).clip(1e-4, 1 - 1e-4).cast(pl.Float32).alias("p_pfr"),
    ).select(["hole_class", "pos", "pf_faced_raise", "p_vpip", "p_pfr"])
    policy.write_parquet(WORK / "preflop_policy.parquet")
    log(f"preflop policy rows: {policy.height:,}")
    del cls, ctx, policy
    gc.collect()
run_stage("pol")

# strength
run_stage("st")
log(f"strength rows: {pl.scan_parquet(str(WORK / 'player_strength_shards' / 'part_*.parquet')).select(pl.len()).collect().item():,}")

# postflop policy (global, small)
if not (WORK / "postflop_policy.parquet").exists():
    AC_GLOB = str(WORK / "action_context_shards" / "part_*.parquet")
    ST_GLOB = str(WORK / "player_strength_shards" / "part_*.parquet")
    ac = pl.scan_parquet(AC_GLOB).filter((pl.col("street_no") > 0) & (pl.col("to_call") > 0))
    st = pl.scan_parquet(ST_GLOB).select(["hand_id", "player_id", "v_flop", "v_turn", "v_river"])
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
        .filter(pl.col("cat") >= 0)
        .with_columns((pl.col("action") != "fold").alias("continued"))
    )
    k = POLICY_PSEUDOCOUNTS
    cls2 = faced.group_by("cat").agg(pl.col("continued").cast(pl.Float32).mean().alias("_cls_p")).collect(engine="streaming")
    ctx2 = faced.group_by(["cat", "street_no"]).agg(pl.len().alias("_n"), pl.col("continued").cast(pl.Float32).sum().alias("_sc")).join(cls2.lazy(), on="cat").collect(engine="streaming")
    policy2 = ctx2.with_columns(
        ((pl.col("_sc") + k * pl.col("_cls_p")) / (pl.col("_n") + k)).clip(1e-4, 1 - 1e-4).cast(pl.Float32).alias("p_continue")
    ).select(["cat", "street_no", "p_continue"])
    policy2.write_parquet(WORK / "postflop_policy.parquet")
    print("postflop P(continue):\n", policy2.sort(["street_no", "cat"]).to_pandas().to_string(index=False))
    del ac, st, faced, cls2, ctx2, policy2
    gc.collect()
run_stage("phf")
log(f"phf rows: {pl.scan_parquet(str(WORK / 'player_hands_full_shards' / 'part_*.parquet')).select(pl.len()).collect().item():,}")

# baselines
if not (WORK / "baselines.parquet").exists():
    baselines = (
        pl.scan_parquet(str(WORK / "player_hands_full_shards" / "part_*.parquet"))
        .group_by(["player_id", "phase"])
        .agg(pl.len().alias("base_hands"), *[pl.col(c).cast(pl.Float32).mean().alias(f"base_{c}") for c in BASELINE_COLS])
        .collect(engine="streaming")
    )
    baselines.write_parquet(WORK / "baselines.parquet")
log(f"baselines: {pl.read_parquet(WORK / 'baselines.parquet').height:,}")

# pair index
phf_glob = str(WORK / "player_hands_full_shards" / "part_*.parquet")

def pair_hand_map(player_hands_glob: str, phase: str) -> pl.LazyFrame:
    hp = (
        pl.scan_parquet(player_hands_glob)
        .filter(pl.col("phase") == phase)
        .group_by(["hand_id", "table_id", "hand_idx"])
        .agg(pl.col("player_id").sort_by("seat_no").alias("players"))
    )
    frames = [
        hp.select(["hand_id", "table_id", "hand_idx", pl.col("players").list.get(i).alias("a"), pl.col("players").list.get(j).alias("b")])
        for i, j in combinations(range(6), 2)
    ]
    return pl.concat(frames).with_columns(
        pl.min_horizontal("a", "b").alias("p_low"),
        pl.max_horizontal("a", "b").alias("p_high"),
    ).drop(["a", "b"])

if not (paths.dev_pairs.exists() and paths.eval_pairs.exists() and paths.dev_pair_hands.exists() and paths.eval_pair_hands.exists()):
    lab = labels.with_columns(pl.min_horizontal("player_1", "player_2").alias("p_low"), pl.max_horizontal("player_1", "player_2").alias("p_high"))
    dev_all = pair_hand_map(phf_glob, "development")
    dev_counts = (
        dev_all.group_by(["p_low", "p_high"])
        .agg(pl.col("table_id").first(), pl.len().cast(pl.Int32).alias("shared_hands"))
        .collect(engine="streaming")
        .sort(["table_id", "p_low", "p_high"])
        .join(lab.select(["pair_id", "p_low", "p_high", "label", "behavior_family"]), on=["p_low", "p_high"], how="left")
    )
    min_shared = int(np.ceil(eval_pairs_raw["shared_hands"].min() * n_dev / (hands.height / 400 - n_dev)))
    log(f"min_shared for dev eligibility: {min_shared}")
    known = dev_counts.filter(pl.col("pair_id").is_not_null()).with_columns(pl.lit(True).alias("is_labeled"))
    unknown = (
        dev_counts.filter(
            pl.col("pair_id").is_null()
            & (pl.col("shared_hands") >= min_shared)
            & ~pl.col("p_low").is_in(list(positives))
            & ~pl.col("p_high").is_in(list(positives))
        )
        .with_columns(
            pl.concat_str([pl.lit("U"), "p_low", "p_high"], separator="_").alias("pair_id"),
            pl.lit(0, dtype=pl.Int64).alias("label"),
            pl.lit("unknown").alias("behavior_family"),
            pl.lit(False).alias("is_labeled"),
        )
    )
    dev_pairs = pl.concat([known, unknown], how="diagonal_relaxed").sort(["table_id", "pair_id"])
    dev_pairs.write_parquet(paths.dev_pairs)
    del dev_counts, known, unknown
    gc.collect()
    (
        dev_all.join(pl.scan_parquet(paths.dev_pairs).select(["pair_id", "p_low", "p_high"]), on=["p_low", "p_high"], how="inner")
        .select(["pair_id", "hand_id", "table_id", "hand_idx", pl.col("p_low").alias("player_1"), pl.col("p_high").alias("player_2")])
        .sink_parquet(paths.dev_pair_hands)
    )
    del dev_all
    gc.collect()

    ev = eval_pairs_raw.with_columns(pl.min_horizontal("player_1", "player_2").alias("p_low"), pl.max_horizontal("player_1", "player_2").alias("p_high"))
    eph = (
        pair_hand_map(phf_glob, "evaluation")
        .join(ev.lazy().select(["pair_id", "p_low", "p_high"]), on=["p_low", "p_high"], how="inner")
        .select(["pair_id", "hand_id", "table_id", "hand_idx", pl.col("p_low").alias("player_1"), pl.col("p_high").alias("player_2")])
    )
    n_eph = eph.select(pl.len()).collect(engine="streaming").item()
    assert n_eph == int(eval_pairs_raw["shared_hands"].sum()), f"eval pair-hands {n_eph} != sum {eval_pairs_raw['shared_hands'].sum()}"
    eph.sink_parquet(paths.eval_pair_hands)
    log(f"eval pair-hands written: {n_eph:,}")
    ev_prep = (
        ev.lazy().select(["pair_id", pl.col("p_low").alias("player_1"), pl.col("p_high").alias("player_2"), "shared_hands"])
        .join(pl.scan_parquet(paths.eval_pair_hands).group_by("pair_id").agg(pl.len().cast(pl.Int32).alias("shared_calc")), on="pair_id", how="left")
        .collect(engine="streaming")
    )
    mism = ev_prep.filter(pl.col("shared_hands") != pl.col("shared_calc")).select(pl.len()).item()
    assert mism == 0, f"shared_hands mismatch rows: {mism}"
    ev_prep.drop("shared_calc").write_parquet(paths.eval_pairs)
    del eph, ev_prep
    gc.collect()

dev_pairs = pl.read_parquet(paths.dev_pairs)
eval_prep = pl.read_parquet(paths.eval_pairs)
log(
    f"dev pairs: {dev_pairs.height:,} (labelled {dev_pairs['is_labeled'].sum():,}); "
    f"dev pair-hands {pl.scan_parquet(paths.dev_pair_hands).select(pl.len()).collect().item():,}; "
    f"eval pairs {eval_prep.height:,}; eval pair-hands {pl.scan_parquet(paths.eval_pair_hands).select(pl.len()).collect().item():,}"
)
log("STAGE 1 DONE")
