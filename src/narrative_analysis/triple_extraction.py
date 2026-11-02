
import os
import pandas as pd

def extract_dataset_triples(
    articles: pd.DataFrame,
    install_path: str | None = None
) -> list[dict]:
    """
    Run Stanford OpenIE across all processed articles.

    Extract triples sentence by sentence and preserve
    the sentence associated with each extracted triple.
    """

    if install_path:
        os.environ["OPENIE_INSTALL_PATH"] = install_path

    from openie import StanfordOpenIE

    extracted_triples = []

    print("\nStarting Stanford OpenIE...")

    with StanfordOpenIE() as client:

        for index, article in articles.iterrows():

            article_id = article["article_id"]

            print(
                f"Processing article "
                f"{index + 1}/{len(articles)}: {article_id}"
            )

             # Get processed article sentences.
            sentences = article["sentences"]

            # Preserve publication date.
            published_at = article.get(
                "published_at",
                article.get("date")
            )

            if pd.isna(published_at):
                published_at = None

            elif hasattr(published_at, "isoformat"):
                published_at = published_at.isoformat()

            else:
                published_at = str(published_at)

            # Extract triples from individual sentences.
            for sentence in sentences:

                if not sentence.strip():
                    continue

                try:

                    triples = client.annotate(sentence)

                    for triple in triples:

                        extracted_triples.append({

                            "subject": triple["subject"],

                            "relationship": triple["relation"],

                            "object": triple["object"],

                            "article_id": article_id,

                            "source": article["source"],

                            "date": published_at,

                            "sentence": sentence

                        })

                except Exception as e:

                    print(
                        f"Error processing sentence "
                        f"in article {article_id}: {e}"
                    )

    print(
        f"\nTotal extracted triples: "
        f"{len(extracted_triples)}"
    )

    return extracted_triples