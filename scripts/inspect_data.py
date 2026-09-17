"""
Inspect competition data: schema, sizes, distributions, and key relationships.
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os

DATA_DIR = "/home/z/my-project/kaggle/comp-data"
os.chdir(DATA_DIR)

print("="*80)
print("1) sample_submission.csv (target format)")
print("="*80)
ss = pd.read_csv("sample_submission.csv")
print(f"Shape: {ss.shape}")
print(f"Columns: {list(ss.columns)}")
print(ss.head(10))
print(f"\nScore distribution:\n{ss.describe()}")
print(f"\nNon-zero rows: {(ss.iloc[:,1] != 0).sum() if ss.shape[1] > 1 else 'N/A'}")

print("\n" + "="*80)
print("2) evaluation_pairs.csv (test pairs we need to score)")
print("="*80)
ep = pd.read_csv("evaluation_pairs.csv")
print(f"Shape: {ep.shape}")
print(f"Columns: {list(ep.columns)}")
print(ep.head(10))
print(f"\nUnique values per column:\n{ep.nunique()}")

print("\n" + "="*80)
print("3) development_labels.csv (training labels)")
print("="*80)
dl = pd.read_csv("development_labels.csv")
print(f"Shape: {dl.shape}")
print(f"Columns: {list(dl.columns)}")
print(dl.head(10))
print(f"\nLabel distribution:\n{dl.iloc[:,-1].value_counts() if dl.shape[1] > 1 else 'N/A'}")

print("\n" + "="*80)
print("4) development_evidence.csv (training evidence)")
print("="*80)
de = pd.read_csv("development_evidence.csv")
print(f"Shape: {de.shape}")
print(f"Columns: {list(de.columns)}")
print(de.head(10))
print(f"\nUnique values per column:\n{de.nunique()}")

# Join labels + evidence
if "id" in dl.columns and "id" in de.columns:
    merged = de.merge(dl, on="id", how="left")
    print(f"\nMerged shape: {merged.shape}")
    print(f"Label balance (after merge):")
    label_col = [c for c in merged.columns if c not in de.columns][0]
    print(merged[label_col].value_counts())
    print(f"\nPositive rate: {merged[label_col].mean():.4f}")

print("\n" + "="*80)
print("5) players.parquet")
print("="*80)
pl = pd.read_parquet("players.parquet")
print(f"Shape: {pl.shape}")
print(f"Columns: {list(pl.columns)}")
print(pl.head(5))
print(f"\nDtypes:\n{pl.dtypes}")

print("\n" + "="*80)
print("6) hands.parquet (just schema, sample)")
print("="*80)
hands_pf = pq.ParquetFile("hands.parquet")
print(f"Schema:\n{hands_pf.schema_arrow}")
print(f"Num row groups: {hands_pf.num_row_groups}")
print(f"Total rows: {hands_pf.metadata.num_rows}")
hands_sample = pd.read_parquet("hands.parquet").head(5)
print(f"\nSample:\n{hands_sample}")

print("\n" + "="*80)
print("7) seats.parquet (just schema, sample)")
print("="*80)
seats_pf = pq.ParquetFile("seats.parquet")
print(f"Schema:\n{seats_pf.schema_arrow}")
print(f"Total rows: {seats_pf.metadata.num_rows}")
seats_sample = pd.read_parquet("seats.parquet").head(5)
print(f"\nSample:\n{seats_sample}")

print("\n" + "="*80)
print("8) actions.parquet (just schema, sample)")
print("="*80)
actions_pf = pq.ParquetFile("actions.parquet")
print(f"Schema:\n{actions_pf.schema_arrow}")
print(f"Total rows: {actions_pf.metadata.num_rows}")
actions_sample = pd.read_parquet("actions.parquet").head(10)
print(f"\nSample:\n{actions_sample}")
print(f"\nAction types:")
if "action_type" in actions_sample.columns or "action" in actions_sample.columns:
    act_col = "action_type" if "action_type" in actions_sample.columns else "action"
    # Read full column - but limit to first row group for performance
    rg = actions_pf.read_row_group(0, columns=[act_col]).to_pandas()
    print(rg[act_col].value_counts())
