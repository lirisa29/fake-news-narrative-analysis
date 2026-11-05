import spacy

# Load the model only once.
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("coreferee")

def run_coreference_preprocessor(raw_text: str) -> str:

    if not isinstance(raw_text, str) or not raw_text.strip():
        return ""

    doc = nlp(raw_text)

    resolved_tokens = []

    for token in doc:

        replacement = token.text

        if token.pos_ == "PRON":

            try:
                antecedents = doc._.coref_chains.resolve(token)

                if antecedents and len(antecedents) == 1:
                    replacement = antecedents[0].text

                    for chunk in doc.noun_chunks:

                        if (
                            chunk.start <= antecedents[0].i
                            < chunk.end
                        ):

                            replacement = chunk.text
                            break

            except (IndexError, ValueError, TypeError):
                pass

        resolved_tokens.append(
            replacement + token.whitespace_
        )

    return "".join(resolved_tokens).strip()