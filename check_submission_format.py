#!/usr/bin/env python3
"""
Validate a Codabench result-only submission JSONL and produce a clean ZIP.

This script checks that the JSONL file follows the expected structure:
- Each line is a JSON object
- Required keys: "document_id" (string), "predictions" (list)
- Each element in "predictions" is a dict with keys: "item" (string), "prediction" (string)

If valid, it creates a ZIP with the JSONL file at the archive root, ready to upload.

Usage:
  python check_submission_format.py <path/to/mock_data_dev_codabench.jsonl> [--out submission_clean.zip]

Exit codes:
  0 on success, non-zero on validation failure or I/O error
"""

import argparse
import json
import os
import sys
import zipfile
from typing import Any, Dict, List

REQUIRED_FILE_NAME = "mock_data_dev_codabench.jsonl"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {idx}: invalid JSON: {e}")
            if not isinstance(obj, dict):
                raise ValueError(f"Line {idx}: JSON object must be a dict")
            data.append(obj)
    if not data:
        raise ValueError("File contains no valid JSONL records")
    return data


def validate_record(rec: Dict[str, Any], index: int) -> None:
    if "document_id" not in rec:
        raise ValueError(f"Record {index}: missing 'document_id'")
    if not isinstance(rec["document_id"], str) or not rec["document_id"].strip():
        raise ValueError(f"Record {index}: 'document_id' must be a non-empty string")

    if "predictions" not in rec:
        raise ValueError(f"Record {index}: missing 'predictions'")
    preds = rec["predictions"]
    if not isinstance(preds, list):
        raise ValueError(f"Record {index}: 'predictions' must be a list")
    if len(preds) == 0:
        raise ValueError(f"Record {index}: 'predictions' list must not be empty")

    for j, p in enumerate(preds):
        if not isinstance(p, dict):
            raise ValueError(f"Record {index} prediction {j}: must be a dict")
        if "item" not in p:
            raise ValueError(f"Record {index} prediction {j}: missing 'item'")
        if "prediction" not in p:
            raise ValueError(f"Record {index} prediction {j}: missing 'prediction'")
        if not isinstance(p["item"], str) or not p["item"].strip():
            raise ValueError(f"Record {index} prediction {j}: 'item' must be a non-empty string")
        if not isinstance(p["prediction"], str):
            raise ValueError(f"Record {index} prediction {j}: 'prediction' must be a string")


def validate_jsonl_structure(records: List[Dict[str, Any]]) -> None:
    for i, rec in enumerate(records):
        validate_record(rec, i)


def make_clean_zip(jsonl_path: str, zip_out: str) -> str:
    os.makedirs(os.path.dirname(zip_out) or ".", exist_ok=True)
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(jsonl_path, arcname=REQUIRED_FILE_NAME)
    return zip_out


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and zip Codabench submission JSONL")
    parser.add_argument("jsonl", help="Path to mock_data_dev_codabench.jsonl (or similarly structured file)")
    parser.add_argument("--out", dest="out_zip", default="submission_clean.zip", help="Output ZIP file path")
    args = parser.parse_args()

    jsonl_path = os.path.abspath(args.jsonl)
    if not os.path.exists(jsonl_path):
        print(f"Error: file not found: {jsonl_path}", file=sys.stderr)
        return 2

    # Warn if the filename differs from REQUIRED_FILE_NAME
    valid_name = True
    base = os.path.basename(jsonl_path)
    if base != REQUIRED_FILE_NAME:
        valid_name = False
        print(f"Warning: input filename is '{base}'. Codabench expects '{REQUIRED_FILE_NAME}'. Converting it for submission.\n", file=sys.stderr)

    try:
        records = read_jsonl(jsonl_path)
        validate_jsonl_structure(records)
    except Exception as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        return 3

    try:
        out_zip = os.path.abspath(args.out_zip)
        if valid_name:
            make_clean_zip(jsonl_path, out_zip)
        else:
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_jsonl_path = os.path.join(tmpdir, REQUIRED_FILE_NAME)
                with open(temp_jsonl_path, "w", encoding="utf-8") as f_out, open(jsonl_path, "r", encoding="utf-8") as f_in:
                    f_out.write(f_in.read())
                make_clean_zip(temp_jsonl_path, out_zip)
    except Exception as e:
        print(f"Failed to create ZIP: {e}", file=sys.stderr)
        return 4

    print("Validation passed. ZIP created:", out_zip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
