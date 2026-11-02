import re
import pandas as pd


def normalize_for_comparison(text: str) -> str:
    """
    Normalize article text for duplicate detection.
    """

    if not isinstance(text, str):
        return ""

    # Convert to lowercase.
    text = text.lower()

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing whitespace.
    return text.strip()


def remove_duplicate_articles(
    articles: pd.DataFrame,
    text_column: str = "cleaned_text",
    source_column: str = "source"
) -> pd.DataFrame:
    """
    Remove duplicate articles published by the same source.

    Articles are considered duplicates when their normalized
    content and source are identical.

    The first occurrence is retained.
    """

    articles = articles.copy()

    # Normalize source names.
    articles["_normalized_source"] = (
        articles[source_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Normalize article content.
    articles["_normalized_content"] = (
        articles[text_column]
        .apply(normalize_for_comparison)
    )

    # Remove articles without usable content.
    articles = articles[
        articles["_normalized_content"] != ""
    ]

    original_count = len(articles)

    # Remove duplicates within the same source.
    articles = articles.drop_duplicates(
        subset=[
            "_normalized_source",
            "_normalized_content"
        ],
        keep="first"
    )

    duplicate_count = original_count - len(articles)

    # Remove temporary columns.
    articles = articles.drop(
        columns=[
            "_normalized_source",
            "_normalized_content"
        ]
    )

    articles = articles.reset_index(drop=True)

    print(f"Duplicate articles removed: {duplicate_count}")
    print(f"Articles remaining: {len(articles)}")

    return articles