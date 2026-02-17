"""Quick diagnostic: compare GT IDs vs Parquet IDs to find the format mismatch."""
import json
import pandas as pd
import os

GT_FILE = "data/raw/dev_gt.jsonl"
CRF_DEV_EN = "data/raw/dyspnea-crf-development/dyspnea-crf-development/data/en-00000-of-00001.parquet"
NOTES_EN = "data/raw/dyspnea-clinical-notes/dyspnea-clinical-notes/data/en-00000-of-00001.parquet"

# 1. GT sample
print("=== GT SAMPLE (first 3 rows) ===")
with open(GT_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        rec = json.loads(line)
        print(f"  Row {i} keys: {list(rec.keys())[:10]}")
        print(f"    document_id = {repr(rec.get('document_id'))}")
        print(f"    patient_id  = {repr(rec.get('patient_id'))}")
        print(f"    hadm_id     = {repr(rec.get('hadm_id'))}")

# 2. CRF development parquet
print("\n=== CRF-DEV PARQUET (en) ===")
if os.path.exists(CRF_DEV_EN):
    df = pd.read_parquet(CRF_DEV_EN)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Shape: {df.shape}")
    print(f"  First 3 rows (all cols):")
    for i, (_, row) in enumerate(df.head(3).iterrows()):
        print(f"    Row {i}: { {k: repr(v)[:80] for k,v in row.items()} }")

# 3. Clinical notes parquet
print("\n=== CLINICAL-NOTES PARQUET (en) ===")
if os.path.exists(NOTES_EN):
    df2 = pd.read_parquet(NOTES_EN)
    print(f"  Columns: {list(df2.columns)}")
    print(f"  Shape: {df2.shape}")
    print(f"  First 3 rows (ID cols only):")
    for i, (_, row) in enumerate(df2.head(3).iterrows()):
        print(f"    Row {i}: { {k: repr(v)[:80] for k,v in row.items() if k.lower() in ('document_id','patient_id','hadm_id','id')} }")
