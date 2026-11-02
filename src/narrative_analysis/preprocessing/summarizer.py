import nltk

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def lexrank_summarize(
    text: str,
    sentence_count: int = 5
) -> str:
    """
    Generate an extractive article-level summary
    using LexRank.
    """

    if not isinstance(text, str) or not text.strip():
        return ""

    try:
        parser = PlaintextParser.from_string(
            text,
            Tokenizer("english")
        )

        summarizer = LexRankSummarizer()

        summary_sentences = summarizer(
            parser.document,
            sentence_count
        )

        return " ".join(
            str(sentence)
            for sentence in summary_sentences
        )

    except (ValueError, LookupError):
        return text