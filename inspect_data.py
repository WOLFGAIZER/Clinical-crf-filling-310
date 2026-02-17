import pandas as pd
import json
import os
import glob

# --- CONFIGURATION ---
GT_FILE = r"data/raw/dev_gt.jsonl"
DATA_FOLDERS = [
    r"data/raw/dyspnea-clinical-notes",
    r"data/raw/dyspnea-crf-development",
    r"data/raw/dyspnea-crf-train",
    r"data/raw/synthetic-crf-train",
]

def inspect_ground_truth():
    print(f"\n=== 1. INSPECTING GROUND TRUTH ({GT_FILE}) ===")
    if not os.path.exists(GT_FILE):
        print("ERROR: GT File not found!")
        return set()

    ids = set()
    with open(GT_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            uid = data.get('document_id') or data.get('patient_id')
            if uid: ids.add(str(uid).strip())

            # Print details for the first 5 rows only
            if i < 5:
                print(f"\n[Row {i}] Keys found: {list(data.keys())}")
                print(f"  -> ID Value: {repr(uid)}")
                print(f"  -> ID Type:  {type(uid)}")

    print(f"\nTotal GT patients: {len(ids)}")
    return ids

def inspect_all_parquets():
    print(f"\n=== 2. INSPECTING ALL PARQUET SHARDS ===")

    all_ids = set()
    for folder in DATA_FOLDERS:
        parquet_files = glob.glob(os.path.join(folder, "**", "*.parquet"), recursive=True)
        if not parquet_files:
            print(f"  [{folder}] No parquet files found.")
            continue

        for pf in parquet_files:
            df = pd.read_parquet(pf)
            df.columns = [c.lower() for c in df.columns]
            id_col = next((c for c in ['document_id', 'doc_id', 'id', 'patient_id'] if c in df.columns), None)

            if id_col:
                shard_ids = set(df[id_col].astype(str).str.strip())
                all_ids.update(shard_ids)
                print(f"  [{os.path.basename(pf)}] {len(shard_ids)} IDs  (columns: {list(df.columns)[:5]}...)")
            else:
                print(f"  [{os.path.basename(pf)}] WARNING: no ID column found. Columns: {list(df.columns)}")

    print(f"\nTotal unique parquet IDs (all shards): {len(all_ids)}")
    return all_ids

if __name__ == "__main__":
    gt_ids = inspect_ground_truth()
    parquet_ids = inspect_all_parquets()

    print("\n=== 3. COMPATIBILITY CHECK ===")
    common = gt_ids.intersection(parquet_ids)
    print(f"Total IDs in GT:        {len(gt_ids)}")
    print(f"Total IDs in Parquets:  {len(parquet_ids)}")
    print(f"INTERSECTION (Matches): {len(common)}")

    if len(common) == 0:
        print("\n>>> DIAGNOSIS: NO MATCHES FOUND.")
        print("The GT IDs do not appear in any downloaded parquet shard.")
        print("Check that you have the correct dataset split (dev vs train).")
    else:
        print(f"\n>>> DIAGNOSIS: {len(common)} MATCHES FOUND! The pipeline should work.")
        print(f"Sample matched IDs: {list(common)[:10]}")