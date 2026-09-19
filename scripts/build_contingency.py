"""Contingency files: new risk recipes (s2/s3/s4) + OLD v4 evidence columns.
Used only if day-2 probe shows the new evidence regressed on LB."""
import pandas as pd

V4 = "/home/z/my-project/subs/sub_v4_catheavy.csv"
evc = [f"evidence_hand_{i}" for i in range(1, 6)]
v4 = pd.read_csv(V4)
ev = v4[["pair_id"] + evc]
for r in ["s2", "s3", "s4"]:
    d = pd.read_csv(f"/home/z/my-project/subs/sub_day2_{r}.csv")
    assert (d["pair_id"].to_numpy() == v4["pair_id"].to_numpy()).all()
    d = d.drop(columns=evc).merge(ev, on="pair_id", how="left")
    d = d[["pair_id", "risk_score", "predicted_behavior"] + evc]
    out = f"/home/z/my-project/subs/sub_day2_{r}v4ev.csv"
    d.to_csv(out, index=False)
    chk = pd.read_csv(out, dtype={c: str for c in evc})
    assert chk.shape == (112_540, 8) and not chk.isna().sum().sum()
    print(f"{out}: OK ({len(d)} rows)")
