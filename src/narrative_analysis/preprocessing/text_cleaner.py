import re
import unicodedata

from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    
    # Clean article text by removing HTML, formatting artefacts and symbols.

    if not isinstance(text, str):
        return ""

    # Remove HTML content.
    soup = BeautifulSoup(text, "lxml")

    # Remove irrelevant HTML elements.
    for element in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "iframe",
        "form"
    ]):
        element.decompose()

    text = soup.get_text(separator=" ", strip=True)

    # Normalize Unicode characters.
    text = unicodedata.normalize("NFKC", text)

    # Remove URLs.
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove excessive whitespace.
    text = re.sub(r"\s+", " ", text)

    # Remove irrelevant symbols while retaining
    # punctuation useful for NLP.
    text = re.sub(
        r"[^\w\s.,!?;:'\"()\-]",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()