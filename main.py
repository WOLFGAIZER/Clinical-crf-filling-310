import asyncio
import json
import argparse
import os
import math

# Imports from your specific file structure
from src.preprocess.wtts_builder import WTTSBuilder
from src.utils.data_loader import DataLoader

# RAG pipeline imports (lazy — only used when --use_rag is set)
RAG_AVAILABLE = False
try:
    from src.rag.embedder import WTTSEmbedder
    from src.rag.rag_pipeline import RAGCRFExtractor
    RAG_AVAILABLE = True
except ImportError:
    pass

import google.generativeai as genai

# --- CONFIG ---
API_KEY = "AIzaSyAkgKha4IxsCjRXbeirhyoygT9Qmr4qYzU"

# --- PROMPTS ---

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


async def generate_async(prompt, model, max_retries=5, initial_delay=2):
    """Call Gemini via google-generativeai SDK (async-safe)."""
    loop = asyncio.get_event_loop()
    for attempt in range(max_retries):
        try:
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    contents=prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json"
                    ),
                )
            )
            try:
                # Basic sanitization for potential markdown blocks in response
                text = response.text.strip()
                if text.startswith("```json"):
                    text = text.split("```json")[1].split("```")[0].strip()
                elif text.startswith("```"):
                    text = text.split("```")[1].split("```")[0].strip()

                json_response = json.loads(text)
                return json_response
            except json.JSONDecodeError:
                print(f"Generated content is not valid JSON. Retrying (Attempt {attempt+1}/{max_retries})...")
                continue

        except Exception as e:
            error_message = str(e).lower()
            # Catch 429 (ResourceExhausted), 500, 503, etc.
            if any(code in error_message for code in ["429", "resource_exhausted", "quota", "500", "503", "504"]):
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    print(f"Rate limit or server error ({error_message[:50]}...). Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    print(f"Max retries reached.")
                    return {"error": f"Max retries reached - {error_message}"}
            else:
                print(f"Error in generate_async: {error_message}")
                return {"error": error_message}

    return {"error": "Failed to generate valid JSON after multiple attempts"}


async def process_patient(model, builder, patient_data, target_items, valid_options, semaphore):
    """Executes the Two-Pass Pipeline for a single patient."""
    async with semaphore:
        pid = str(patient_data.get('document_id') or patient_data.get('patient_id')
                  or patient_data.get('hadm_id') or 'unknown')

        try:
            # --- PHASE 1: WTTS Construction ---
            wtts_string = builder.build_wtts_string(patient_data)

            # --- PHASE 2: Skeleton Generation (Pass 1) ---
            skeleton_input = SKELETON_PROMPT.format(wtts_string=wtts_string)
            skeleton_resp = await generate_async(skeleton_input, model)

            skeleton_text = str(skeleton_resp)
            if isinstance(skeleton_resp, dict):
                skeleton_text = json.dumps(skeleton_resp)

            # --- PHASE 3: Extraction (Pass 2) ---
            final_predictions = {}
            item_chunks = list(chunk_data(target_items, 10))

            for chunk_items in item_chunks:
                chunk_schema = {
                    item: valid_options.get(item, ["Yes", "No", "Unknown"])
                    for item in chunk_items
                }

                extract_input = EXTRACTION_PROMPT.format(
                    skeleton=skeleton_text,
                    chunk_schema_json=json.dumps(chunk_schema)
                )

                chunk_resp = await generate_async(extract_input, model)

                if isinstance(chunk_resp, dict):
                    if 'error' in chunk_resp:
                        print(f"  [WARN] LLM error for {pid}, chunk {chunk_items[:3]}...: {chunk_resp['error']}")
                    else:
                        final_predictions.update(chunk_resp)

            return {
                "patient_id": pid,
                "skeleton_debug": skeleton_text[:500] + "...",
                "predictions": final_predictions
            }

        except Exception as e:
            print(f"Error processing {pid}: {e}")
            return None


# ---------------------------------------------------------------------------
#  SUBMISSION FORMATTING
# ---------------------------------------------------------------------------

def format_as_official_submission(results, target_items, language="en"):
    """
    Converts internal results into official Codabench JSONL format.
    Ensures ALL target_items are present and raw text is excluded.
    """
    submission_records = []

    # Build lookup for results - use stripped IDs
    results_lookup = {str(r['patient_id']).strip(): r.get('predictions', {}) for r in results}

    for pid, predictions in results_lookup.items():
        # Remove language suffix if already present in pid to avoid e.g. 123_en_en
        base_pid = pid.split('_')[0]
        doc_id = f"{base_pid}_{language}"

        # Case-insensitive lookup for predictions
        norm_predictions = {str(k).lower().strip(): v for k, v in predictions.items()}

        pred_list = []
        for item in target_items:
            item_lower = item.lower().strip()
            # Look up prediction for this item (case-insensitive)
            val_obj = norm_predictions.get(item_lower, "unknown")

            # Extract value if it's a dict (our internal format)
            if isinstance(val_obj, dict):
                val = val_obj.get("value", "unknown")
            else:
                val = str(val_obj)

            # Normalize predicted value
            val = val.strip().lower() if val else "unknown"
            if not val:
                val = "unknown"

            pred_list.append({
                "item": item,
                "prediction": val
            })

        submission_records.append({
            "document_id": doc_id,
            "predictions": pred_list
        })

    return submission_records


# ---------------------------------------------------------------------------
#  EVALUATION -- Accuracy & F1 Scoring
# ---------------------------------------------------------------------------

def _normalise(value):
    """Lowercase + strip for fair comparison."""
    if value is None:
        return ""
    return str(value).strip().lower()


def evaluate_predictions(results, gt_path):
    """
    Compare pipeline results against dev_gt.jsonl.
    Prints accuracy, macro-F1, per-item breakdown, and sample errors.
    Returns (overall_dict, per_item_dict).
    """
    # --- Load GT ---
    gt = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Use base ID for matching
            doc_id = str(rec['document_id']).strip().split('_')[0]
            gt[doc_id] = {a['item']: a['ground_truth'] for a in rec.get('annotations', [])}

    # --- Build prediction lookup ---
    preds = {}
    for r in results:
        # Use base ID for matching
        doc_id = str(r.get('patient_id', 'unknown')).strip().split('_')[0]
        items = {}
        for item_name, item_val in r.get('predictions', {}).items():
            # Store keys in lowercase for case-insensitive matching
            key = str(item_name).lower().strip()
            if isinstance(item_val, dict):
                items[key] = item_val.get('value', str(item_val))
            else:
                items[key] = str(item_val)
        preds[doc_id] = items

    # --- Collect all unique items ---
    all_items = set()
    for doc_items in gt.values():
        all_items.update(doc_items.keys())

    # --- Score ---
    item_stats = {item: {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0, 'correct': 0}
                  for item in all_items}
    total_comparisons = 0
    total_correct = 0
    matched_patients = 0
    errors = []

    for doc_id, gt_items in gt.items():
        pred_items = preds.get(doc_id, {})
        if pred_items:
            matched_patients += 1

        for item_name, gt_val in gt_items.items():
            gt_norm = _normalise(gt_val)
            # Match item name in lowercase
            pred_val = pred_items.get(item_name.lower().strip())
            pred_norm = _normalise(pred_val) if pred_val is not None else ""

            total_comparisons += 1
            item_stats[item_name]['total'] += 1

            if gt_norm == pred_norm:
                total_correct += 1
                item_stats[item_name]['correct'] += 1
                item_stats[item_name]['tp'] += 1
            else:
                item_stats[item_name]['fn'] += 1
                if pred_norm:
                    item_stats[item_name]['fp'] += 1
                errors.append((doc_id, item_name, gt_val,
                               pred_val if pred_val is not None else '<MISSING>'))

    accuracy = total_correct / total_comparisons if total_comparisons > 0 else 0.0

    # --- Per-item P/R/F1 ---
    f1s = []
    per_item = {}
    for item_name in sorted(all_items):
        s = item_stats[item_name]
        tp, fp, fn = s['tp'], s['fp'], s['fn']
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        item_acc = s['correct'] / s['total'] if s['total'] > 0 else 0.0
        per_item[item_name] = {'accuracy': item_acc, 'precision': prec,
                               'recall': rec, 'f1': f1, 'total': s['total']}
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    # --- Print report ---
    print("\n" + "=" * 70)
    print("  CL4Health CRF Filling -- Evaluation Report")
    print("=" * 70)
    print(f"\n  GT Patients:       {len(gt)}")
    print(f"  Pred Patients:     {len(preds)}")
    print(f"  Matched Patients:  {matched_patients}")
    print(f"\n  Total Comparisons: {total_comparisons}")
    print(f"  Total Correct:     {total_correct}")
    print(f"\n  {'Accuracy':>20s}: {accuracy:.4f}")
    print(f"  {'Macro F1':>20s}: {macro_f1:.4f}")

    # Top / bottom items
    sorted_items = sorted(per_item.items(), key=lambda x: x[1]['f1'], reverse=True)
    n_show = min(15, len(sorted_items))

    print(f"\n  Top {n_show} Items by F1:")
    print(f"  {'Item':<45s} {'Acc':>6s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    print(f"  {'-'*45} {'---':>6s} {'---':>6s} {'---':>6s} {'---':>6s}")
    for name, s in sorted_items[:n_show]:
        print(f"  {name:<45s} {s['accuracy']:>6.2f} {s['precision']:>6.2f} {s['recall']:>6.2f} {s['f1']:>6.2f}")

    print(f"\n  Bottom {n_show} Items by F1:")
    print(f"  {'Item':<45s} {'Acc':>6s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    print(f"  {'-'*45} {'---':>6s} {'---':>6s} {'---':>6s} {'---':>6s}")
    for name, s in sorted_items[-n_show:]:
        print(f"  {name:<45s} {s['accuracy']:>6.2f} {s['precision']:>6.2f} {s['recall']:>6.2f} {s['f1']:>6.2f}")

    # Sample errors
    if errors:
        n_err = min(15, len(errors))
        print(f"\n  Sample Mismatches ({n_err} of {len(errors)}):")
        print(f"  {'DocID':<12s} {'Item':<40s} {'GT':<20s} {'Pred':<20s}")
        print(f"  {'-'*12} {'-'*40} {'-'*20} {'-'*20}")
        for doc_id, item, gt_v, pred_v in errors[:n_err]:
            print(f"  {doc_id:<12s} {item:<40s} {str(gt_v):<20s} {str(pred_v):<20s}")

    print("=" * 70)

    return {'accuracy': round(accuracy, 4), 'macro_f1': round(macro_f1, 4)}, per_item


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", default=API_KEY,
                        help="Google AI Studio API key")
    parser.add_argument("--model_name", default="gemini-1.5-pro",
                        help="Gemini model name")
    parser.add_argument("--data_folders", nargs="+",
                        default=[
                            "data/raw/dyspnea-clinical-notes",
                            "data/raw/dyspnea-crf-development",
                        ],
                        help="Directories containing .parquet shards (searched recursively)")
    parser.add_argument("--gt_file",
                        default="data/raw/dev_gt.jsonl")
    parser.add_argument("--options_folder",
                        default="data/raw/dyspnea-valid-options")
    parser.add_argument("--output_file",
                        default="data/processed/materialized_ehr/pipeline_results.json",
                        help="Path to save internal pipeline results (with debug info)")
    parser.add_argument("--submission_file",
                        default="submission/mock_data_dev_codabench.jsonl",
                        help="Path to save official Codabench submission JSONL")
    parser.add_argument("--language", default="en", choices=["en", "it"],
                        help="Language suffix for submission document IDs")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation after generating predictions")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="Max concurrent patient processing (free tier: keep low)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of patients to process (for testing)")
    # --- RAG options ---
    parser.add_argument("--use_rag", action="store_true",
                        help="Use RAG-guided extraction (retrieves relevant tuples per CRF item)")
    parser.add_argument("--rag_top_k", type=int, default=15,
                        help="Number of WTTS tuples to retrieve per CRF item group (RAG mode)")
    parser.add_argument("--rag_model", type=str, default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model for embeddings (swap to clinical model on GPU)")
    parser.add_argument("--rag_device", type=str, default="cpu",
                        help="Device for embedding model: 'cpu' or 'cuda'")
    args = parser.parse_args()

    # 1. Setup — Configure Gemini API
    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel(args.model_name)
    print(f"Using model: {args.model_name} (Google AI Studio)")

    # Limit concurrency (free tier = 15 RPM, so keep low)
    semaphore = asyncio.Semaphore(args.concurrency)

    # 2. Load Data
    loader = DataLoader(data_folders=args.data_folders, gt_path=args.gt_file)

    target_items = loader.get_target_schema()
    valid_options = loader.load_valid_options(args.options_folder)

    merged_data = loader.load_and_merge()

    if args.limit:
        merged_data = merged_data[:args.limit]

    if not merged_data:
        print("No data found. Exiting.")
        return

    # 3. Process
    builder = WTTSBuilder()
    print(f"Starting pipeline for {len(merged_data)} patients...")
    print(f"Schema: {len(target_items)} items per patient.")

    if args.use_rag:
        # --- RAG Pipeline ---
        if not RAG_AVAILABLE:
            print("ERROR: RAG dependencies not installed. Run:")
            print("  pip install sentence-transformers faiss-cpu")
            return

        print(f"\n  [RAG MODE] Embedding model: {args.rag_model}")
        print(f"  [RAG MODE] Device: {args.rag_device}")
        print(f"  [RAG MODE] Top-k: {args.rag_top_k}\n")

        embedder = WTTSEmbedder(model_name=args.rag_model, device=args.rag_device)
        extractor = RAGCRFExtractor(
            embedder=embedder,
            generate_fn=generate_async,
            top_k=args.rag_top_k,
        )

        tasks = [
            extractor.extract_patient(
                p, builder, target_items, valid_options, semaphore, model
            )
            for p in merged_data
        ]
    else:
        # --- Original Two-Pass Pipeline ---
        tasks = [
            process_patient(model, builder, p, target_items, valid_options, semaphore)
            for p in merged_data
        ]

    results = await asyncio.gather(*tasks)
    results = [r for r in results if r is not None]

    # 4. Save Internal Results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nPipeline results saved to {args.output_file}")

    # 5. Generate and Save Official Submission
    if args.submission_file:
        print(f"Generating official submission format ({args.language})...")
        submission_data = format_as_official_submission(results, target_items, args.language)

        os.makedirs(os.path.dirname(args.submission_file) or ".", exist_ok=True)
        with open(args.submission_file, 'w', encoding='utf-8') as f:
            for rec in submission_data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Official submission saved to {args.submission_file}")

    # 6. Evaluate against GT
    if not args.skip_eval:
        print("\nRunning evaluation against ground truth...")
        overall, _ = evaluate_predictions(results, args.gt_file)
        print(f"\n  >>> Final Accuracy: {overall['accuracy']:.4f}  |  Macro F1: {overall['macro_f1']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())