import nltk


def segment_sentences(text: str) -> list[str]:
    """
    Split processed article text into sentences.
    """

    if not isinstance(text, str) or not text.strip():
        return []

    sentences = nltk.sent_tokenize(text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]