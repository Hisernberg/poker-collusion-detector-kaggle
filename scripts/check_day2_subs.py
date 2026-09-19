"""Cross-check day-2 submissions: risk-recipe correlations, evidence overlap vs v4, top-set stability."""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

SUBS = {
    "v4_old": "/home/z/my-project/subs/sub_v4_catheavy.csv",
    "s1": "/home/z/my-project/subs/sub_day2_s1.csv",
    "s2": "/home/z/my-project/subs/sub_day2_s2.csv",
    "s3": "/home/z/my-project/subs/sub_day2_s3.csv",
    "s4": "/home/z/my-project/subs/sub_day2_s4.csv",
}
data = {}
for k, p in SUBS.items():
    df = pd.read_csv(p, dtype={"predicted_behavior": str})
    data[k] = df
    print(f"{k}: rows={len(df)} risk[{df.risk_score.min():.6f},{df.risk_score.max():.6f}] "
          f"ev_nonnull={int((df[[f'evidence_hand_{i}' for i in range(1,6)]] != 'NO_EVIDENCE').all(axis=1).sum())} "
          f"active={int((df.predicted_behavior != 'none').sum())}")

base = data["v4_old"]
pairs = base["pair_id"].to_numpy()
for k in ["s1", "s2", "s3", "s4"]:
    d = data[k].set_index("pair_id").loc[pairs].reset_index()
    b = base.set_index("pair_id").loc[pairs].reset_index()
    rho, _ = spearmanr(d["risk_score"], b["risk_score"])
    top_old = set(b.nlargest(5627, "risk_score")["pair_id"])
    top_new = set(d.nlargest(5627, "risk_score")["pair_id"])
    j = len(top_old & top_new) / len(top_old)
    evc = [f"evidence_hand_{i}" for i in range(1, 6)]
    old_ev = b[evc].values.tolist()
    new_ev = d[evc].values.tolist()
    overlap = np.mean([len(set(a) & set(c)) / 5 for a, c in zip(old_ev, new_ev)])
    print(f"{k}: risk_spearman_vs_v4={rho:.4f} top5627_jaccard={j:.3f} evidence_overlap5={overlap:.3f}")

# evidence diff between s1 and v4 (same risk): how many pairs changed their top-5?
d = data["s1"].set_index("pair_id").loc[pairs]
b = base.set_index("pair_id").loc[pairs]
evc = [f"evidence_hand_{i}" for i in range(1, 6)]
same = sum(set(a) == set(c) for a, c in zip(b[evc].values, d[evc].values))
print(f"s1 vs v4 identical top-5 sets: {same}/{len(pairs)} ({same/len(pairs):.1%})")
# behavior agreement
for k in ["s1", "s2", "s3", "s4"]:
    d = data[k].set_index("pair_id").loc[pairs]
    agree = (d["predicted_behavior"].to_numpy() == b["predicted_behavior"].to_numpy()).mean()
    print(f"{k}: behavior agreement vs v4 = {agree:.4f}")
