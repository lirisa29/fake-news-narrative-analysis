
import os

# Stanford OpenIE Protocol Buffers compatibility
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
from pathlib import Path

from narrative_analysis.triple_extraction import extract_dataset_triples

from narrative_analysis.postprocessing.entity_resolver import EntityResolver

from narrative_analysis.postprocessing.tuple_quality import filter_tuple_quality

from narrative_analysis.postprocessing.duplicate_tuples import remove_duplicate_tuples

from narrative_analysis.postprocessing.semantic_equivalence import SemanticEquivalenceDetector


# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT / "data/evaluation/random_sample.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data/evaluation"
)

RAW_OUTPUT = OUTPUT_DIR / "machine_triples_raw.csv"

FINAL_OUTPUT = OUTPUT_DIR / "machine_triples.csv"

FUZZY_THRESHOLD = 80

SEMANTIC_THRESHOLD = 0.88


# ==========================================================
# STEP 1: LOAD SELECTED SENTENCES
# ==========================================================

def load_evaluation_sentences():

    print("\nLoading evaluation sentences...")

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "sentence_id",
        "article_id",
        "source",
        "sentence"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    df = df.dropna(
        subset=["sentence"]
    ).copy()

    # Preserve sentence IDs and article metadata.
    # Stanford OpenIE expects a list of sentences per article.
    openie_input = df[
        [
            "sentence_id",
            "article_id",
            "source",
            "sentence"
        ]
    ].copy()

    # Add publication date if available.
    if "date" in df.columns:
        openie_input["published_at"] = df["date"]
    elif "published_at" in df.columns:
        openie_input["published_at"] = df["published_at"]
    else:
        openie_input["published_at"] = None

    openie_input["sentences"] = openie_input["sentence"].apply(
        lambda sentence: [str(sentence).strip()]
    )

    print(f"Sentences loaded: {len(openie_input)}")

    return openie_input


# ==========================================================
# STEP 2: RUN STANFORD OPENIE
# ==========================================================

def run_openie(openie_input, install_path=None):

    print("\nRunning Stanford OpenIE...")

    extracted_triples = extract_dataset_triples(
        articles=openie_input,
        install_path=install_path
    )

    # Reattach sentence IDs for later evaluation.
    sentence_lookup = {}

    for _, row in openie_input.iterrows():

        key = (
            str(row["article_id"]),
            str(row["sentence"]).strip()
        )

        sentence_lookup[key] = row["sentence_id"]

    for triple in extracted_triples:

        key = (
            str(triple["article_id"]),
            str(triple["sentence"]).strip()
        )

        triple["sentence_id"] = sentence_lookup.get(
            key
        )

    if not extracted_triples:
        raise ValueError("Stanford OpenIE produced no triples.")

    raw_df = pd.DataFrame(extracted_triples)

    raw_df.to_csv(
        RAW_OUTPUT,
        index=False
    )

    print(f"Raw triples extracted: {len(raw_df)}")

    print(f"Raw output saved: {RAW_OUTPUT}")

    return extracted_triples


# ==========================================================
# STEP 3: ENTITY RESOLUTION
# ==========================================================

def apply_entity_resolution(triples):

    print("\nApplying entity resolution...")

    resolver = EntityResolver(
        threshold=FUZZY_THRESHOLD
    )

    resolved_triples = resolver.resolve_dataset(
        triples
    )

    print(f"Triples after entity resolution: {len(resolved_triples)}")

    return resolved_triples


# ==========================================================
# STEP 4: TUPLE QUALITY FILTERING
# ==========================================================

def apply_quality_filtering(triples):

    print("\nApplying tuple quality filtering...")

    filtered_triples = filter_tuple_quality(
        triples
    )

    print(f"Triples after quality filtering: {len(filtered_triples)}")

    return filtered_triples


# ==========================================================
# STEP 5: DUPLICATE TUPLE REMOVAL
# ==========================================================

def apply_duplicate_removal(triples):

    print("\nRemoving duplicate tuples...")

    unique_triples = remove_duplicate_tuples(
        triples
    )

    print(f"Triples after duplicate removal: {len(unique_triples)}")

    return unique_triples


# ==========================================================
# STEP 6: SEMANTIC EQUIVALENCE DETECTION
# ==========================================================

def apply_semantic_equivalence(triples):

    print("\nApplying semantic equivalence detection...")

    detector = SemanticEquivalenceDetector(
        threshold=SEMANTIC_THRESHOLD
    )

    final_triples = detector.remove_semantic_duplicates(
        triples
    )

    print(f"Final triples: {len(final_triples)}")

    return final_triples


# ==========================================================
# STEP 7: SAVE FINAL MACHINE TRIPLES
# ==========================================================

def save_final_triples(triples):

    print("\nSaving final machine-generated triples...")

    output_df = pd.DataFrame(triples)

    if output_df.empty:
        raise ValueError("No final triples available to save.")

    output_columns = [
        "tuple_id",
        "sentence_id",
        "subject",
        "relationship",
        "object",
        "source",
        "article_id",
        "sentence"
    ]

    # Retain available columns and create tuple IDs.
    output_df.insert(
        0,
        "tuple_id",
        [
            f"T{i:06d}"
            for i in range(1, len(output_df) + 1)
        ]
    )

    output_df = output_df.reindex(
        columns=output_columns
    )

    output_df.to_csv(
        FINAL_OUTPUT,
        index=False
    )

    print(f"Final triples saved: {FINAL_OUTPUT}")

    return output_df


# ==========================================================
# MAIN
# ==========================================================

def run_pipeline(openie_install_path=None):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("OPENIE EVALUATION EXTRACTION")
    print("=" * 60)

    # Load selected sentences
    openie_input = load_evaluation_sentences()

    # Run Stanford OpenIE
    extracted_triples = run_openie(
        openie_input,
        install_path=openie_install_path
    )

    # Apply postprocessing stages
    resolved_triples = apply_entity_resolution(
        extracted_triples
    )

    quality_filtered_triples = apply_quality_filtering(
        resolved_triples
    )

    unique_triples = apply_duplicate_removal(
        quality_filtered_triples
    )

    final_triples = apply_semantic_equivalence(
        unique_triples
    )

    # Save final results
    output_df = save_final_triples(
        final_triples
    )

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETED")
    print("=" * 60)

    print(f"Sentences processed: {len(openie_input)}")

    print(f"Raw triples: {len(extracted_triples)}")

    print(f"Final triples: {len(output_df)}")

    return output_df


if __name__ == "__main__":

    run_pipeline(
        openie_install_path="C:/Users/liris/Desktop/PDDA/fake-news-narrative-analysis/openie_resources"
    )