
import pandas as pd
import re

from pathlib import Path


# ==========================================================
# FILE PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

EVALUATION_DIR = (
    PROJECT_ROOT / "data/evaluation"
)

GOLD_FILE = EVALUATION_DIR / "gold_standard.csv"

MACHINE_FILE = EVALUATION_DIR / "machine_triples.csv"

RESULTS_FILE = EVALUATION_DIR / "evaluation_metrics.csv"

COMPARISON_FILE = EVALUATION_DIR / "evaluation_comparison.csv"


# ==========================================================
# STEP 1: NORMALISE TEXT
# ==========================================================

def normalize(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = re.sub(r"[^\w\s]", "", value)

    value = re.sub(r"\s+", " ", value)

    return value


# ==========================================================
# STEP 2: LOAD DATASETS
# ==========================================================

def load_datasets():

    print("\nLoading evaluation datasets...")

    gold_df = pd.read_csv(GOLD_FILE)

    machine_df = pd.read_csv(MACHINE_FILE)

    required_columns = [
        "subject",
        "relationship",
        "object"
    ]

    for column in required_columns:

        if column not in gold_df.columns:
            raise ValueError(
                f"Gold standard missing column: {column}"
            )

        if column not in machine_df.columns:
            raise ValueError(
                f"Machine dataset missing column: {column}"
            )

    # Preserve sentence IDs for sentence-level comparison.
    if "sentence_id" not in gold_df.columns:
        raise ValueError("Gold standard requires sentence_id.")

    if "sentence_id" not in machine_df.columns:
        raise ValueError("Machine triples require sentence_id.")

    print(f"Gold-standard triples: {len(gold_df)}")

    print(f"Machine triples: {len(machine_df)}")

    return gold_df, machine_df


# ==========================================================
# STEP 3: CREATE TRIPLE REPRESENTATIONS
# ==========================================================

def get_triples(df):

    return [
        (
            str(row["sentence_id"]).strip(),
            normalize(row["subject"]),
            normalize(row["relationship"]),
            normalize(row["object"])
        )
        for _, row in df.iterrows()
    ]


# ==========================================================
# STEP 4: EVALUATE TRIPLES
# ==========================================================

def evaluate_triples(gold_df, machine_df):

    print("\nComparing triples...")

    gold_triples = get_triples(gold_df)

    machine_triples = get_triples(machine_df)

    # Convert to sets to remove duplicates.
    gold_set = set(gold_triples)

    machine_set = set(machine_triples)

    # Calculate TP, FP and FN.
    matched = gold_set & machine_set

    false_positives = machine_set - gold_set

    false_negatives = gold_set - machine_set

    true_positives = len(matched)

    false_positives_count = len(false_positives)

    false_negatives_count = len(false_negatives)

    # Precision
    precision = (
        true_positives /
        (true_positives + false_positives_count)
        if true_positives + false_positives_count > 0
        else 0
    )

    # Recall
    recall = (
        true_positives /
        (true_positives + false_negatives_count)
        if true_positives + false_negatives_count > 0
        else 0
    )

    # F1-score
    f1_score = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    results = {
        "Gold Standard Triples": len(gold_set),
        "Machine Triples": len(machine_set),
        "TP": true_positives,
        "FP": false_positives_count,
        "FN": false_negatives_count,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score
    }

    # ======================================================
    # STEP 5: CREATE COMPARISON REPORT
    # ======================================================

    comparison_records = []

    for triple in matched:

        comparison_records.append({
            "sentence_id": triple[0],
            "subject": triple[1],
            "relationship": triple[2],
            "object": triple[3],
            "evaluation": "TP"
        })

    for triple in false_positives:

        comparison_records.append({
            "sentence_id": triple[0],
            "subject": triple[1],
            "relationship": triple[2],
            "object": triple[3],
            "evaluation": "FP"
        })

    for triple in false_negatives:

        comparison_records.append({
            "sentence_id": triple[0],
            "subject": triple[1],
            "relationship": triple[2],
            "object": triple[3],
            "evaluation": "FN"
        })

    comparison_df = pd.DataFrame(comparison_records)

    comparison_df.to_csv(
        COMPARISON_FILE,
        index=False
    )

    # ======================================================
    # STEP 6: SAVE METRICS
    # ======================================================

    results_df = pd.DataFrame([results])

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    # ======================================================
    # STEP 7: DISPLAY RESULTS
    # ======================================================

    print("\n" + "=" * 55)
    print("OPENIE EVALUATION RESULTS")
    print("=" * 55)

    print(f"Gold-standard triples: {len(gold_set)}")

    print(f"Machine triples:      {len(machine_set)}")

    print(f"\nTrue Positives:       {true_positives}")

    print(f"False Positives:      {false_positives_count}")

    print(f"False Negatives:      {false_negatives_count}")

    print(f"\nPrecision:            {precision:.4f}")

    print(f"Recall:               {recall:.4f}")

    print(f"F1-score:             {f1_score:.4f}")

    print("\nFiles saved:")

    print(RESULTS_FILE)

    print(COMPARISON_FILE)

    return results


# ==========================================================
# MAIN
# ==========================================================

def main():

    gold_df, machine_df = load_datasets()

    evaluate_triples(
        gold_df,
        machine_df
    )


if __name__ == "__main__":

    main()
