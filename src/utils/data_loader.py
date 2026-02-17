import pandas as pd
import json
import os
import glob
from typing import List


class DataLoader:
    def __init__(self, data_folders: List[str], gt_path: str):
        """
        Args:
            data_folders: List of directories containing .parquet shards.
                          All shards (en, it, etc.) from ALL folders are read
                          into a single unified lookup before matching.
            gt_path:      Path to the ground-truth JSONL file (e.g. dev_gt.jsonl).
        """
        self.data_folders = data_folders
        self.gt_path = gt_path

        # Keys to exclude from schema generation
        self.ignore_keys = {
            'patient_id', 'hadm_id', 'doc_id', 'document_id', 'admission_time',
            'discharge_time', 'split', 'text', 'notes', 'ground_truth', 'annotations'
        }

    # ------------------------------------------------------------------
    # Backward-compatible property so old code referencing self.notes_folder
    # still works (points to the first folder in the list).
    # ------------------------------------------------------------------
    @property
    def notes_folder(self):
        return self.data_folders[0] if self.data_folders else ""

    def get_target_schema(self) -> list:
        """
        Extracts target items from the nested 'annotations' list in dev_gt.jsonl.
        """
        target_keys = set()
        print(f"Extracting target schema from {self.gt_path}...")
        try:
            with open(self.gt_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i > 100: break
                    try:
                        record = json.loads(line)
                        if 'annotations' in record:
                            for ann in record['annotations']:
                                if 'item' in ann: target_keys.add(ann['item'])
                    except: continue
        except Exception as e:
            print(f"Error reading schema: {e}")
        return sorted(list(target_keys))

    def load_valid_options(self, options_folder: str) -> dict:
        """
        Loads schema constraints from the dyspnea-valid-options folder.
        """
        options_map = {}

        # 1. Safety Check
        if not os.path.exists(options_folder):
            print(f"Warning: Options folder {options_folder} not found.")
            return {}

        print(f"Loading Valid Options from {options_folder}...")

        # 2. Iterate through ALL files
        for filename in os.listdir(options_folder):
            if filename.endswith(".json"):
                field_name = filename.replace(".json", "")
                try:
                    with open(os.path.join(options_folder, filename), 'r') as f:
                        data = json.load(f)
                        options_map[field_name] = data
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        print(f"Loaded options for {len(options_map)} fields.")
        return options_map

    # ------------------------------------------------------------------
    # Helper: collect ALL parquet files across every data folder
    # ------------------------------------------------------------------
    def _collect_all_parquet_files(self) -> list:
        """Recursively find every .parquet file inside each data folder."""
        all_files = []
        for folder in self.data_folders:
            if not os.path.isdir(folder):
                print(f"Warning: data folder not found – {folder}")
                continue
            found = glob.glob(os.path.join(folder, "**", "*.parquet"), recursive=True)
            print(f"  [{os.path.basename(os.path.normpath(folder))}] {len(found)} shard(s)")
            all_files.extend(found)
        return all_files

    # ------------------------------------------------------------------
    # Helper: read all shards into a single DataFrame
    # ------------------------------------------------------------------
    def _build_unified_notes_table(self, parquet_files: list) -> pd.DataFrame:
        """
        Reads ALL parquet shards into one DataFrame with normalised
        column names and a string-typed document-id column.
        """
        frames = []
        for p_file in parquet_files:
            try:
                df = pd.read_parquet(p_file)
                df.columns = [c.lower() for c in df.columns]

                # Tag rows with their source shard (useful for debugging)
                df['_source_shard'] = os.path.basename(p_file)

                frames.append(df)
                print(f"  Read {len(df):>5} rows from {os.path.basename(p_file)}")
            except Exception as e:
                print(f"  Error reading {os.path.basename(p_file)}: {e}")

        if not frames:
            return pd.DataFrame()

        unified = pd.concat(frames, ignore_index=True)

        # Normalise the ID column to string
        id_col = next(
            (c for c in ['document_id', 'doc_id', 'id', 'patient_id'] if c in unified.columns),
            None
        )
        if id_col:
            unified[id_col] = unified[id_col].astype(str).str.strip()

            # Create a _base_id column that strips language suffixes like
            # "_en", "_it" so that IDs from language-specific shards
            # (e.g. "1014081_en") can still match GT IDs ("1014081").
            unified['_base_id'] = (
                unified[id_col]
                .str.replace(r'_(en|it|de|fr|es)$', '', regex=True)
            )
        else:
            unified['_base_id'] = None

        n_unique = unified['_base_id'].nunique() if id_col else '?'
        print(f"\n  ✓ Unified table: {len(unified)} total rows, "
              f"{n_unique} unique base IDs")
        return unified

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------
    def load_and_merge(self):
        """
        1. Loads GT IDs from dev_gt.jsonl.
        2. Reads ALL parquet shards from ALL data_folders into a single
           unified DataFrame.
        3. Matches rows by 'document_id' and returns merged patient dicts.
        """
        # --- Step 1: Ground Truth ---
        print(f"Loading Ground Truth from {self.gt_path}...")
        gt_lookup = {}
        with open(self.gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                uid = str(rec.get('document_id') or rec.get('patient_id') or rec.get('hadm_id'))
                gt_lookup[uid] = rec
        print(f"  GT contains {len(gt_lookup)} patients.\n")

        # --- Step 2: Collect & read ALL shards ---
        print(f"Scanning {len(self.data_folders)} data folder(s) for parquet shards...")
        parquet_files = self._collect_all_parquet_files()

        if not parquet_files:
            print("ERROR: No .parquet files found in any data folder.")
            return []

        print(f"\nReading {len(parquet_files)} shard(s) into unified table...")
        unified_df = self._build_unified_notes_table(parquet_files)

        if unified_df.empty:
            print("ERROR: All shards were empty or unreadable.")
            return []

        # --- Step 3: Match GT IDs against the unified table ---
        id_col = next(
            (c for c in ['document_id', 'doc_id', 'id', 'patient_id'] if c in unified_df.columns),
            None
        )
        text_col = next(
            (c for c in ['clinical_note', 'text', 'body'] if c in unified_df.columns),
            None
        )

        if not id_col:
            print(f"ERROR: No ID column found. Available columns: {list(unified_df.columns)}")
            return []

        # Match using _base_id (which has language suffixes stripped)
        matched_df = unified_df[unified_df['_base_id'].isin(gt_lookup.keys())]
        print(f"\n  Matched {len(matched_df)} rows across all shards "
              f"({matched_df['_base_id'].nunique()} unique patients).")

        # --- Step 4: Build patient objects ---
        merged_patients = []
        found_ids = set()  # tracks _base_id to avoid duplicates

        for _, row in matched_df.iterrows():
            base_id = row['_base_id']
            if base_id in found_ids:
                continue

            patient_obj = gt_lookup[base_id].copy()

            # Attach clinical note text if available
            if text_col and text_col in row.index and pd.notna(row[text_col]):
                anchor_time = patient_obj.get('admission_time', '2026-01-01')
                patient_obj['notes'] = [{
                    'timestamp': anchor_time,
                    'text': row[text_col]
                }]

            merged_patients.append(patient_obj)
            found_ids.add(base_id)

        unmatched = set(gt_lookup.keys()) - found_ids
        if unmatched:
            print(f"\n  ⚠  {len(unmatched)} GT patients had no matching parquet row.")
            print(f"     Sample unmatched GT IDs: {list(unmatched)[:5]}")

        print(f"\nSuccessfully merged {len(merged_patients)} documents.")
        return merged_patients