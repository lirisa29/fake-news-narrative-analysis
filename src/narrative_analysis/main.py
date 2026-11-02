import os

# Stanford OpenIE Protocol Buffers compatibility.
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import argparse
from pathlib import Path

import pandas as pd

from narrative_analysis.loader import load_dataset
from narrative_analysis.preprocessing.text_cleaner import clean_text
from narrative_analysis.preprocessing.duplicate_remover import remove_duplicate_articles
from narrative_analysis.preprocessing.summarizer import lexrank_summarize
from narrative_analysis.preprocessing.sentence_segmenter import segment_sentences
from narrative_analysis.preprocessing.coreference import run_coreference_preprocessor
from narrative_analysis.triple_extraction import extract_dataset_triples
from narrative_analysis.postprocessing.entity_resolver import EntityResolver
from narrative_analysis.postprocessing.tuple_quality import filter_tuple_quality
from narrative_analysis.postprocessing.duplicate_tuples import remove_duplicate_tuples
from narrative_analysis.postprocessing.semantic_equivalence import SemanticEquivalenceDetector


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_pipeline(
    input_file: str | Path,
    output_file: str | Path,
    openie_install_path: str | None = None,
    fuzzy_threshold: int = 80,
    summary_sentences: int = 5
):

    input_file = Path(input_file)
    output_file = Path(output_file)

    print("=" * 60)
    print("Starting processing pipeline")
    print("=" * 60)

    # --------------------------------------------------
    # STEP 1: DATA INGESTION
    # --------------------------------------------------

    print("\nSTEP 1: Loading dataset")

    articles = load_dataset(input_file)

    # --------------------------------------------------
    # STEP 2: TEXT CLEANING
    # --------------------------------------------------

    print("\nSTEP 2: Cleaning article text")

    if "text" not in articles.columns:
        raise ValueError(
            "Dataset must contain a 'text' column."
        )

    articles["cleaned_text"] = articles["text"].apply(
        clean_text
    )

    articles = remove_duplicate_articles(
        articles,
        text_column="cleaned_text",
        source_column="source"
    )

    # --------------------------------------------------
    # STEP 3: COREFERENCE RESOLUTION
    # --------------------------------------------------

    print("\nSTEP 3: Coreference resolution")

    articles["coreference_text"] = articles["cleaned_text"].apply(
        run_coreference_preprocessor
    )

    if articles["coreference_text"].isna().all():
        raise ValueError(
            "Coreference preprocessing produced no usable text."
        )

    # --------------------------------------------------
    # STEP 4: ARTICLE-LEVEL SUMMARIZATION
    # --------------------------------------------------

    print("\nSTEP 4: LexRank article summarization")

    articles["summary"] = articles["coreference_text"].apply(
        lambda text: lexrank_summarize(
            text,
            sentence_count=summary_sentences
        )
    )

    # --------------------------------------------------
    # STEP 5: SENTENCE SEGMENTATION
    # --------------------------------------------------

    print("\nSTEP 5: Sentence segmentation")

    articles["sentences"] = articles["summary"].apply(
        segment_sentences
    )

    # --------------------------------------------------
    # STEP 6: OPENIE TRIPLE EXTRACTION
    # --------------------------------------------------

    print("\nSTEP 6: OpenIE triple extraction")

    # Pass the processed article text to OpenIE.
    articles["processed_text"] = articles["sentences"].apply(
        lambda sentences: " ".join(sentences)
    )

    extracted_triples = extract_dataset_triples(
        articles=articles,
        install_path=openie_install_path
    )

    # --------------------------------------------------
    # STEP 7: ENTITY RESOLUTION
    # --------------------------------------------------

    print("\nSTEP 7: Resolving entities")

    resolver = EntityResolver(
        threshold=fuzzy_threshold
    )

    resolved_triples = resolver.resolve_dataset(
        extracted_triples
    )

    print(f"Resolved tuples: {len(resolved_triples)}")

    for triple in resolved_triples[:10]:
        print(triple)

    
    # --------------------------------------------------
    # STEP 9: TUPLE QUALITY FILTERING
    # --------------------------------------------------

    print("\nSTEP 9: Tuple quality filtering")

    quality_filtered_triples = filter_tuple_quality(
        resolved_triples
    )

    # --------------------------------------------------
    # STEP 10: DUPLICATE TUPLE REMOVAL
    # --------------------------------------------------

    print("\nSTEP 10: Duplicate tuple removal")

    unique_triples = remove_duplicate_tuples(
        quality_filtered_triples
    )

    # --------------------------------------------------
    # STEP 11: SEMANTIC EQUIVALENCE DETECTION
    # --------------------------------------------------

    print("\nSTEP 11: Semantic equivalence detection")

    semantic_detector = SemanticEquivalenceDetector(
        threshold=0.88
    )

    final_triples = semantic_detector.remove_semantic_duplicates(
        unique_triples
    )

    # --------------------------------------------------
    # STEP 12: SAVE FINAL PROCESSED DATASET
    # --------------------------------------------------

    print("\nSTEP 12: Saving final processed dataset")

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_df = pd.DataFrame(final_triples)

    # Generate tuple IDs after all filtering.
    output_df.insert(
        0,
        "tuple_id",
        [
            f"T{i:06d}"
            for i in range(1, len(output_df) + 1)
        ]
    )

    output_columns = [
        "tuple_id",
        "subject",
        "relationship",
        "object",
        "source",
        "date",
        "article_id",
        "sentence"
    ]

    output_df = output_df.reindex(
        columns=output_columns
    )

    output_df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------
    # PIPELINE SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)

    print(f"Articles processed: {len(articles)}")
    print(f"Extracted triples: {len(extracted_triples)}")
    print(f"Final triples: {len(final_triples)}")

    print(f"\nOutput: {output_file}")

    return output_df


def main():

    parser = argparse.ArgumentParser(
        description="Fake News Narrative Processing Pipeline"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/interim/fake_news_combined.csv"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/tuples.csv"
    )

    parser.add_argument(
        "--openie-install-path",
        type=str,
        default=None
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=80
    )

    parser.add_argument(
        "--summary-sentences",
        type=int,
        default=10,
        help="Number of sentences in the LexRank summary"
    )

    args = parser.parse_args()

    run_pipeline(
        input_file=args.input,
        output_file=args.output,
        openie_install_path=args.openie_install_path,
        fuzzy_threshold=args.threshold,
        summary_sentences=args.summary_sentences
    )


if __name__ == "__main__":
    main()