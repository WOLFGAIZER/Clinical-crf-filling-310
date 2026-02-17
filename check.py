import json
import os

GT_PATH = "data/raw/dev_gt.jsonl"
NOTES_DIR = "data/raw/dyspnea-clinical-notes"

with open(GT_PATH, 'r') as f:
    first_row = json.loads(f.readline())
    print("Keys found in Ground Truth:", first_row.keys())
    print("Sample ID value:", first_row.get('patient_id') or first_row.get('hadm_id'))

print("\nFirst 5 files in notes directory:")
print(os.listdir(NOTES_DIR)[:5])