"""
Convert pipeline output to official CL4Health Codabench submission format.

Our pipeline outputs:
  [{"patient_id": "1014081", "predictions": {"item_name": {"value": "y", ...}}, ...}, ...]

Codabench expects (one JSON per line in JSONL):
  {"document_id": "1014081_en", "predictions": [{"item": "item_name", "prediction": "y"}, ...]}

This script:
  1. Reads pipeline output (JSON) and ground truth (JSONL)
  2. Maps predictions to match GT annotation order exactly
  3. Writes a Codabench-compatible submission JSONL
  4. Optionally runs local scoring and creates the upload ZIP

Usage:
  python convert_to_submission.py
  python convert_to_submission.py --pipeline_output results.json --language en --score --zip
"""

import json
import os
import argparse
import subprocess
import sys


def load_ground_truth(gt_path: str) -> list:
    """Load dev_gt.jsonl to get document IDs and annotation item order."""
    records = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_pipeline_output(output_path: str) -> dict:
    """
    Load our pipeline's output JSON.
    Returns a dict mapping patient_id → predictions dict.
    """
    with open(output_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    lookup = {}
    for r in results:
        pid = str(r.get("patient_id", "unknown"))
        preds = r.get("predictions", {})
        # Flatten: handle both {"item": {"value": "y"}} and {"item": "y"}
        flat = {}
        for item_name, item_val in preds.items():
            if isinstance(item_val, dict):
                flat[item_name] = str(item_val.get("value", "unknown"))
            else:
                flat[item_name] = str(item_val)
        lookup[pid] = flat
    return lookup


def convert(gt_records: list, pipeline_lookup: dict, language: str) -> list:
    """
    Convert pipeline predictions to Codabench format.

    For each GT patient:
      - Create a submission record with document_id = "{id}_{language}"
      - For each annotation item (in GT order), look up the prediction
      - Default to "unknown" if no prediction exists
    """
    submission = []
    matched = 0
    unmatched = 0

    for gt_rec in gt_records:
        doc_id = str(gt_rec["document_id"])
        annotations = gt_rec.get("annotations", [])

        # Look up our predictions for this patient
        preds = pipeline_lookup.get(doc_id, {})

        if preds:
            matched += 1
        else:
            unmatched += 1

        # Build predictions list in the SAME ORDER as GT annotations
        pred_list = []
        for ann in annotations:
            item_name = ann["item"]
            predicted_value = preds.get(item_name, "unknown")

            # Normalize: strip whitespace, lowercase
            predicted_value = predicted_value.strip().lower() if predicted_value else "unknown"
            if not predicted_value:
                predicted_value = "unknown"

            pred_list.append({
                "item": item_name,
                "prediction": predicted_value,
            })

        submission.append({
            "document_id": f"{doc_id}_{language}",
            "predictions": pred_list,
        })

    print(f"  Matched: {matched} patients")
    print(f"  Unmatched (defaulting to 'unknown'): {unmatched} patients")
    return submission


def write_submission_jsonl(submission: list, output_path: str):
    """Write one JSON object per line."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in submission:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert pipeline output to Codabench submission format"
    )
    parser.add_argument(
        "--pipeline_output",
        default="data/processed/materialized_ehr/submission.json",
        help="Path to pipeline output JSON",
    )
    parser.add_argument(
        "--gt_file",
        default="data/raw/dev_gt.jsonl",
        help="Path to ground truth JSONL",
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=["en", "it"],
        help="Language suffix for document IDs",
    )
    parser.add_argument(
        "--output",
        default="submission/mock_data_dev_codabench.jsonl",
        help="Output submission JSONL path",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Run local scoring after conversion",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create Codabench ZIP after conversion",
    )
    args = parser.parse_args()

    print("\n=== Converting Pipeline Output to Codabench Format ===")

    # 1. Load data
    print(f"\n1. Loading ground truth from: {args.gt_file}")
    gt_records = load_ground_truth(args.gt_file)
    print(f"   {len(gt_records)} patients in GT")

    print(f"\n2. Loading pipeline output from: {args.pipeline_output}")
    pipeline_lookup = load_pipeline_output(args.pipeline_output)
    print(f"   {len(pipeline_lookup)} patients in pipeline output")

    # 2. Convert
    print(f"\n3. Converting to Codabench format (language={args.language})...")
    submission = convert(gt_records, pipeline_lookup, args.language)

    # 3. Write
    print(f"\n4. Writing submission JSONL...")
    write_submission_jsonl(submission, args.output)

    # 4. Optionally score locally
    if args.score:
        print(f"\n5. Running local scoring...")
        # The official scorer expects dev_gt.jsonl at development_data/dev_gt.jsonl
        # Create symlink/copy if needed
        dev_data_dir = "development_data"
        dev_gt_target = os.path.join(dev_data_dir, "dev_gt.jsonl")
        if not os.path.exists(dev_gt_target):
            os.makedirs(dev_data_dir, exist_ok=True)
            import shutil
            shutil.copy2(args.gt_file, dev_gt_target)
            print(f"   Copied GT to {dev_gt_target}")

        cmd = [
            sys.executable, "scoring.py",
            "--submission_path", args.output,
            "--language", args.language,
        ]
        print(f"   Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)

    # 5. Optionally create ZIP
    if args.zip:
        zip_output = os.path.join("submission", "submission_clean.zip")
        print(f"\n6. Creating Codabench ZIP...")
        cmd = [
            sys.executable, "check_submission_format.py",
            args.output,
            "--out", zip_output,
        ]
        print(f"   Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)
        print(f"\n   Upload {zip_output} to Codabench:")
        print(f"   https://www.codabench.org/competitions/11984/#/participate-tab")

    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
