"""Error analysis: why does OOF evidence MAP@5 stall at 0.52?
For each positive pair: ranks of evidence hands; characterize top-1 false positives vs evidence hands.
"""
import sys
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, "/home/z/my-project/scripts")
from poker_lib import *  # noqa

paths = resolve_paths()
dh = pl.read_parquet(paths.labelled_hand_feats).to_pandas()
dh = dh[dh["label"] == 1].copy()
dh = dh.sort_values(["pair_id", "hand_score"], ascending=[True, False])
dh["rank_in_pair"] = dh.groupby("pair_id").cumcount() + 1

ev = dh[dh["is_evidence"]]
print("evidence hand rank distribution (under OOF global score):")
print(ev["rank_in_pair"].describe())
print("top-1 rate:", (ev["rank_in_pair"] == 1).mean().round(3), "| top-3:", (ev["rank_in_pair"] <= 3).mean().round(3), "| top-5:", (ev["rank_in_pair"] <= 5).mean().round(3))

# how many evidence hands in top-5 per pair?
tp5 = dh[dh["rank_in_pair"] <= 5].groupby("pair_id")["is_evidence"].sum()
print("\nevidence hands captured in top-5 per pair: mean", tp5.mean().round(3), "dist:", tp5.value_counts().sort_index().to_dict())

# per family top-1 rate
for fam in TARGET_BEHAVIORS:
    e = ev[ev["behavior_family"] == fam]
    print(f"{fam}: top1 {(e['rank_in_pair'] == 1).mean():.3f} top3 {(e['rank_in_pair'] <= 3).mean():.3f} top5 {(e['rank_in_pair'] <= 5).mean():.3f}")

# characterize FALSE top-1 vs TRUE evidence rows (same pairs)
top1 = dh[dh["rank_in_pair"] == 1]
fp = top1[~top1["is_evidence"]]
tp = top1[top1["is_evidence"]]
cols = ["pot_bb", "pair_aggr", "call_partner", "fold_to_partner", "raise_partner", "hu_check",
        "transfer_any", "outsider_fold_to_pair", "fold_better_surprise", "junk_call_partner_post",
        "post_loose_partner", "call_partner_bb", "call_partner_max_bb", "both_showdown", "max_cat_final"]
print("\nfeature | false-top1 | true-top1")
for c in cols:
    print(f"{c:26s} {fp[c].mean():8.2f} {tp[c].mean():8.2f}")

# evidence hands ranked >5: what do they look like?
late = ev[ev["rank_in_pair"] > 10]
print(f"\nevidence ranked >10: {len(late)} ({len(late)/len(ev):.2%})")
for c in cols:
    print(f"{c:26s} late-ev {late[c].mean():8.2f} | all-ev {ev[c].mean():8.2f}")
