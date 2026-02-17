python -m src.preprocess.wtts_builder ^
    --input_dir data/raw/dyspnea-clinical-notes ^
    --gt_file data/raw/dev_gt.jsonl ^
    --output_dir data/processed/materialized_ehr
pause