import os
import sys
import json
from sklearn.metrics import f1_score
import argparse


def load_jsonl(path):
    """Load a JSONL file as a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


class Scorer:
    def __init__(self, not_available_string: str, language:str):
        self.not_available_string = not_available_string
        self.return_value_for_zero_division = 0
        if language not in ["en", "it"]:
            raise ValueError(f"Unsupported language: {language}. Supported languages are 'en' and 'it'.")
        self.language = language

    def calculate_score(self, reference, submission):
        scores = []
        self.TP = 0
        self.FP = 0
        self.FN = 0        
        for ref_one_patient, sub_one_patient in zip(reference, submission):
            sub_one_patient_id, lang = sub_one_patient["document_id"].split("_", 1)
            if ref_one_patient["document_id"] != sub_one_patient_id:
                raise ValueError(
                    f"Document ID mismatch: reference {ref_one_patient['document_id']} vs submission {sub_one_patient['document_id']}"
                )
            if lang != self.language:
                raise ValueError(
                    f"Language mismatch: expected {self.language} but got {lang} in submission"
                )
            score_one_patient = self.calculate_score_one_patient(
                ref_one_patient,
                sub_one_patient,
            )
            scores.append(score_one_patient)

        if not scores:
            return 0.0
        print(f"TP: {self.TP}, FP: {self.FP}, FN: {self.FN}")   

        return sum(scores) / len(scores)

    def calculate_score_one_patient(self, reference_one_patient, submission_one_patient):
        # Expected structure:
        # reference_one_patient["annotations"] = [{"ground_truth": ...}, ...]
        # submission_one_patient["predictions"] = [{"prediction": ...}, ...]
        y_true = [item["ground_truth"] for item in reference_one_patient["annotations"]]
        y_pred = [item["prediction"] for item in submission_one_patient["predictions"]]

        for i, t, p in zip(range(len(y_true)), y_true, y_pred):
            if t != self.not_available_string or p != self.not_available_string:
                if t == p:
                    self.TP += 1
                elif t == self.not_available_string and p != self.not_available_string:
                    self.FP += 1
                elif t != p and p == self.not_available_string:
                    self.FN += 1
        f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
        )
        return f1


def main(your_submission_path: str, language: str, test_or_dev: str) -> None:
    print("\n=== Scoring program starting ===")
    output_dir = "your_sumbmission_scores"

    if test_or_dev == "test":
        ref_path = 'development_data/dev_gt.jsonl'
    elif test_or_dev == "development":
        ref_path = 'development_data/dev_gt.jsonl'
    else:
        raise ValueError("test_or_dev must be either 'test' or 'development'")

    sub_path = your_submission_path

    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference file not found at {ref_path}")
    if not os.path.exists(sub_path):
        raise FileNotFoundError(f"Submission predictions not found at {sub_path}")

    print(f"Loading reference from {ref_path}")
    try:
        reference = load_jsonl(ref_path)
    except:
        if test_or_dev == "test":
            raise ValueError(f"Test data has not been released yet.")

    print(f"Loading submission from {sub_path}")
    submission = load_jsonl(sub_path)

    scorer = Scorer(not_available_string="unknown", language=language)
    score = scorer.calculate_score(reference, submission)

    print(f"Final macro-F1 = {score}")

    os.makedirs(output_dir, exist_ok=True)

    # Codabench reads scores.json (or scores.txt). Let's use JSON:
    scores_path = os.path.join(output_dir, "scores.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump({"f1_macro": float(score)}, f)

    print(f"Scores written to {scores_path}")
    print("=== Scoring program finished successfully ===\n")


if __name__ == "__main__":
    # get from argparse the agumetns pred, ref, output, language
    argparse = argparse.ArgumentParser(description="Score submission")
    argparse.add_argument("--submission_path", type=str, help="Path to the submission JSONL")
    argparse.add_argument("--language", type=str, help="Language of the submission (en or it)")
    args = argparse.parse_args()

    your_submission_path = args.submission_path
    language = args.language
    main(your_submission_path, language, test_or_dev="development")
