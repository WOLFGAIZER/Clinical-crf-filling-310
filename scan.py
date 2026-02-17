import pandas as pd
import glob
import os

target_id = '1014081'

# Search ALL data folders, not just one
data_folders = [
    r"data/raw/dyspnea-clinical-notes",
    r"data/raw/dyspnea-crf-development",
    r"data/raw/dyspnea-crf-train",
    r"data/raw/synthetic-crf-train",
]

# Collect every parquet shard across all folders
all_shards = []
for folder in data_folders:
    found = glob.glob(os.path.join(folder, "**", "*.parquet"), recursive=True)
    all_shards.extend(found)

print(f"Scanning {len(all_shards)} parquet shards across {len(data_folders)} folders...\n")

for shard in all_shards:
    df = pd.read_parquet(shard)
    df.columns = [c.lower() for c in df.columns]
    id_col = next((c for c in ['document_id', 'doc_id', 'id', 'patient_id'] if c in df.columns), None)
    if id_col and target_id in df[id_col].astype(str).values:
        print(f"MATCH FOUND: Patient {target_id} is in {shard}")
        print(f"  Columns: {list(df.columns)}")
        break
else:
    print(f"Patient {target_id} not found in any of the {len(all_shards)} shards.")