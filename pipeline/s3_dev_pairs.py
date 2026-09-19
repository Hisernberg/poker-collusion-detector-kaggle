"""Stage 3: stream dev pair-hands in table chunks -> OOF hand scores -> aggregate -> dev pair features."""
import sys, gc, pickle, json
import numpy as np
import polars as pl
import pandas as pd

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa
from poker_lib import _lgb_bag, _predict_bag, _spec_matrix, mix_evidence_scores, mix_evidence_soft, report_map5, map5_within_pairs, family_weight_matrix
from pairfeat import build_pair_hand_features
from aggfeat import aggregate_pairs

paths = resolve_paths()
log("S3 start: dev pair features (streaming)")

dev_pairs = pl.read_parquet(paths.dev_pairs)
with open(paths.hand_models, "rb") as f:
    art = pickle.load(f)
table_fold = art["table_fold"]
thr = art["thr"]
rank_models, sus_models, spec_models = art["rank_models"], art["sus_models"], art["spec_models"]

tables = sorted(dev_pairs["table_id"].unique().to_list())
N_CHUNK = 24
chunks = [tables[i::N_CHUNK] for i in range(N_CHUNK)]

baselines = pl.read_parquet(paths.work / "baselines.parquet")
base_dev = baselines.filter(pl.col("phase") == "development").drop("phase")
phf = paths.player_hands_full

import pathlib
PART_DIR = paths.work / "s3_parts"
PART_DIR.mkdir(exist_ok=True)
parts = []
for ci, chunk_tables in enumerate(chunks):
    part_path = PART_DIR / f"part_{ci:02d}.parquet"
    if part_path.exists():
        parts.append(pl.read_parquet(part_path))
        log(f"chunk {ci + 1}/{N_CHUNK}: loaded from cache")
        continue
    ph = pl.scan_parquet(paths.dev_pair_hands).filter(pl.col("table_id").is_in(chunk_tables))
    wide_lf = build_pair_hand_features(ph, paths, phf, table_ids=chunk_tables)
    hf = wide_lf.collect(engine="streaming")
    del wide_lf
    gc.collect()
    fold_of_row = hf["table_id"].replace_strict(table_fold, return_dtype=pl.Int8).to_numpy()
    n = hf.height
    hs = np.zeros(n, np.float32)
    su = np.zeros(n, np.float32)
    SLICE = 250_000
    for start in range(0, n, SLICE):
        end = min(start + SLICE, n)
        X = hf.slice(start, end - start).select(HAND_FEATS).to_pandas().astype("float32")
        fl = fold_of_row[start:end]
        for fold in range(N_FOLDS):
            mask = fl == fold
            if mask.any():
                hs[start:end][mask] = np.mean([m.predict(X.loc[mask]) for m in rank_models[fold]], axis=0)
                su[start:end][mask] = sus_models[fold].predict(X.loc[mask])
        del X
        gc.collect()
    hf = hf.with_columns(pl.Series("hand_score", hs), pl.Series("hand_sus", su))
    dp_chunk = dev_pairs.filter(pl.col("table_id").is_in(chunk_tables))
    pf = aggregate_pairs(hf, base_dev, dp_chunk, thr)
    pl.from_pandas(pf).write_parquet(part_path)
    parts.append(pl.from_pandas(pf))
    log(f"chunk {ci + 1}/{N_CHUNK}: {hf.height:,} pair-hands -> {len(pf):,} pairs")
    del hf, hs, su, pf
    gc.collect()

dev_pf_pl = pl.concat(parts)
dev_pf_pl.write_parquet(paths.dev_pair_features)
dev_pf = dev_pf_pl.to_pandas()
log(f"S3 DONE: dev pair features {dev_pf.shape}")
