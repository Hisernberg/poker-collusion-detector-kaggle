"""Test: are evidence hands inside locally 'hot' interaction windows?
Compute rolling behavior intensity (±10 hands) within pair on the forensics battery (has _hrow order).
If AUC strong -> add rolling features to pipeline.
"""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

b = pd.read_parquet("/home/z/my-project/work/forensics_battery.parquet")
b = b.merge(pd.read_parquet("/home/z/my-project/work/forensics_pair_hands2.parquet")
            .query('pair_id in @b.pair_id.unique()')[["pair_id", "hand_id"]].drop_duplicates(),
            on=["pair_id", "hand_id"], how="left") if False else b
# battery already has per-pair rows; need temporal order: re-merge _hrow
php = pd.read_parquet("/home/z/my-project/work/forensics_pair_hands2.parquet")
import polars as pl
hands = pl.read_parquet("/home/z/my-project/data/hands.parquet", columns=["hand_id"]).with_row_index("_hrow").to_pandas()
b = b.merge(hands, on="hand_id", how="left").sort_values(["pair_id", "_hrow"]).reset_index(drop=True)
b["is_ev"] = b["is_ev"].astype(bool)

W = 10
for col in ["call_partner", "fold_to_partner", "raise_partner", "hu_check", "pair_aggr_total", "t12", "t21"]:
    roll = (b.groupby("pair_id", sort=False)[col]
             .transform(lambda s: s.rolling(2 * W + 1, center=True, min_periods=1).sum()))
    b[f"roll_{col}"] = roll - b[col]  # exclude self

fam_frames = {}
for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    d = b[b["behavior_family"] == fam]
    ev, ct = d[d["is_ev"]], d[~d["is_ev"]]
    print(f"\n== {fam} (ev {len(ev)} / ctl {len(ct)}) ==")
    for c in [f"roll_{x}" for x in ["call_partner", "fold_to_partner", "raise_partner", "hu_check", "pair_aggr_total", "t12", "t21"]]:
        auc = roc_auc_score(np.r_[np.ones(len(ev)), np.zeros(len(ct))], np.r_[ev[c].fillna(-1), ct[c].fillna(-1)])
        print(f"  {c:24s} AUC {max(auc, 1 - auc):.3f}")
