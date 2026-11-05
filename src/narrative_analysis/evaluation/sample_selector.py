import pandas as pd
import random

from pathlib import Path

from narrative_analysis.loader import load_dataset
from narrative_analysis.preprocessing.text_cleaner import clean_text
from narrative_analysis.preprocessing.summarizer import lexrank_summarize
from narrative_analysis.preprocessing.coreference import run_coreference_preprocessor
from narrative_analysis.preprocessing.sentence_segmenter import segment_sentences


SEED = 42
NUMBER_OF_ARTICLES = 10
NUMBER_OF_SENTENCES = 10


def select_random_sentences(input_file):

    random.seed(SEED)

    articles = load_dataset(input_file)

    articles["cleaned_text"] = articles["text"].apply(clean_text)

    articles["coreference_text"] = articles["cleaned_text"].apply(
        run_coreference_preprocessor
    )

    articles["summary"] = articles["coreference_text"].apply(
        lambda text: lexrank_summarize(
            text,
            sentence_count=10
        )
    )

    articles["sentences"] = articles["summary"].apply(
        segment_sentences
    )

    # Randomly select 10 articles
    sampled_articles = articles.sample(
        n=min(NUMBER_OF_ARTICLES, len(articles)),
        random_state=SEED
    )

    # Collect sentences and their article information
    sentence_records = []

    for _, article in sampled_articles.iterrows():

        for sentence in article["sentences"]:

            sentence_records.append({
                "article_id": article.get("article_id"),
                "source": article.get("source"),
                "sentence": sentence
            })

    # Randomly select 10 sentences
    sampled_sentences = random.sample(
        sentence_records,
        min(NUMBER_OF_SENTENCES, len(sentence_records))
    )

    annotation_df = pd.DataFrame(sampled_sentences)

    annotation_df.insert(
        0,
        "sentence_id",
        [
            f"S{i:03d}"
            for i in range(1, len(annotation_df) + 1)
        ]
    )

    output_path = Path("data/evaluation/random_sample.csv")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    annotation_df.to_csv(
        output_path,
        index=False
    )

    print(annotation_df)

    print(f"\nSelected {len(annotation_df)} sentences.")

    return annotation_df


if __name__ == "__main__":

    select_random_sentences(
        "data/interim/fake_news_combined.csv"
    )