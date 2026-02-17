import asyncio
import json
import argparse
import os
import math

# Imports from your specific file structure
from src.preprocess.wtts_builder import WTTSBuilder
from src.utils.data_loader import DataLoader
from src.model.predictor import generate_async

import vertexai
from vertexai.generative_models import GenerativeModel

# --- PROMPTS ---

# PASS 1: Create the Clinical Skeleton
# Goal: Compress the WTTS string into a readable timeline, retaining IDs.
SKELETON_PROMPT = """
You are a Clinical Data Specialist.
Convert the Weighted Time Series (WTTS) below into a "Clinical Chronology Skeleton".

INPUT (WTTS):
{wtts_string}

INSTRUCTIONS:
1. Create a strict chronological timeline (Admission to Discharge).
2. IMPORTANT: You MUST retain the [S_xx] ID for every event you list.
3. Filter out "Routine" (Weight 0.1) events unless they indicate a status change.
4. Keep exact values (e.g., "BP 90/60", "Temp 102.5").

OUTPUT FORMAT:
[Date] [S_xx]: Event details
[Date] [S_xx]: Event details
...
"""

# PASS 2: Evidence-Linked Extraction with Constraints
# Goal: Answer specific CRF items using the Skeleton + Valid Options.
EXTRACTION_PROMPT = """
You are a Clinical Coding Expert.
Review the Patient Skeleton and the Valid Options for the requested items.

PATIENT SKELETON:
{skeleton}

TASK:
For each Clinical Item listed below, determine the value AND the supporting Sentence ID.
1. **Value**: Must come strictly from the "Valid Options" provided.
2. **Evidence**: Must be the specific [S_xx] ID from the skeleton that proves the value.

ITEMS TO EXTRACT & THEIR OPTIONS:
{chunk_schema_json}

OUTPUT FORMAT (JSON Object):
{{
  "item_name": {{
    "value": "Selected Option",
    "evidence": "S_xx",
    "reasoning": "Brief explanation"
  }},
  ...
}}
"""

def chunk_data(data, size):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(data), size):
        yield data[i:i + size]

async def process_patient(model, builder, patient_data, target_items, valid_options, semaphore):
    """
    Executes the Two-Pass Pipeline for a single patient.
    """
    async with semaphore:
        pid = str(patient_data.get('patient_id') or patient_data.get('hadm_id') or 'unknown')
        
        try:
            # --- PHASE 1: WTTS Construction ---
            # Converts raw notes -> [S_01] ("Date", "Event", P, W)...
            wtts_string = builder.build_wtts_string(patient_data)
            
            # --- PHASE 2: Skeleton Generation (Pass 1) ---
            skeleton_input = SKELETON_PROMPT.format(wtts_string=wtts_string)
            skeleton_resp = await generate_async(skeleton_input, model)
            
            # Handle cases where predictor returns dict vs string
            skeleton_text = str(skeleton_resp)
            if isinstance(skeleton_resp, dict):
                # Try to extract text if buried in a dict, otherwise dump it
                skeleton_text = json.dumps(skeleton_resp)

            # --- PHASE 3: Extraction (Pass 2) ---
            # We process items in chunks of 10 to prevent hallucinations
            final_predictions = {}
            
            # Create chunks of the target items list
            item_chunks = list(chunk_data(target_items, 10))
            
            for chunk_items in item_chunks:
                # Prepare a mini-schema for just these 10 items
                # { "dyspnea": ["Grade 0", ...], "seizure": ["Yes", "No"] }
                chunk_schema = {
                    item: valid_options.get(item, ["Yes", "No", "Unknown"]) 
                    for item in chunk_items
                }
                
                extract_input = EXTRACTION_PROMPT.format(
                    skeleton=skeleton_text,
                    chunk_schema_json=json.dumps(chunk_schema)
                )
                
                # Call LLM
                chunk_resp = await generate_async(extract_input, model)
                
                # Merge results
                if isinstance(chunk_resp, dict):
                    final_predictions.update(chunk_resp)
            
            return {
                "patient_id": pid,
                "skeleton_debug": skeleton_text[:500] + "...", # Log first 500 chars
                "predictions": final_predictions
            }

        except Exception as e:
            print(f"Error processing {pid}: {e}")
            return None

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", default="clinical-crf-filling", help="GCP Project ID")
    parser.add_argument("--data_folders", nargs="+",
                        default=[
                            r"C:\Users\sai78\Desktop\Clinical_CRF_filling\data\raw\dyspnea-clinical-notes",
                            r"C:\Users\sai78\Desktop\Clinical_CRF_filling\data\raw\dyspnea-crf-development",
                        ],
                        help="Directories containing .parquet shards (searched recursively)")
    parser.add_argument("--gt_file", 
                        default=r"C:\Users\sai78\Desktop\Clinical_CRF_filling\data\raw\dev_gt.jsonl")
    parser.add_argument("--options_folder", 
                        default=r"C:\Users\sai78\Desktop\Clinical_CRF_filling\data\raw\dyspnea-valid-options\dyspnea-valid-options\data")
    parser.add_argument("--output_file", 
                        default="data/processed/materialized_ehr/submission.json")
    args = parser.parse_args()

    # 1. Setup
    vertexai.init(project=args.project_id, location="us-central1")
    model = GenerativeModel("gemini-1.5-pro-preview-0409")
    
    # Limit concurrency to avoid hitting Vertex AI rate limits
    semaphore = asyncio.Semaphore(10) 

    # 2. Load Data
    loader = DataLoader(data_folders=args.data_folders, gt_path=args.gt_file)
    
    # Load Schema & Options
    target_items = loader.get_target_schema() # List of 100+ items
    valid_options = loader.load_valid_options(args.options_folder) # Dict of valid values
    
    # Merge Notes
    merged_data = loader.load_and_merge()

    if not merged_data:
        print("No data found. Exiting.")
        return

    # 3. Process
    builder = WTTSBuilder()
    print(f"Starting pipeline for {len(merged_data)} patients...")
    print(f"Schema: {len(target_items)} items per patient.")
    
    tasks = [
        process_patient(model, builder, p, target_items, valid_options, semaphore) 
        for p in merged_data
    ]
    
    results = await asyncio.gather(*tasks)
    results = [r for r in results if r is not None]

    # 4. Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Done! Results saved to {args.output_file}")

if __name__ == "__main__":
    asyncio.run(main())