def normalize_tuple_value(value):
    """
    Normalize tuple components for comparison.
    """

    if not isinstance(value, str):
        return ""

    return " ".join(value.lower().strip().split())


def remove_duplicate_tuples(
    triples: list[dict]
) -> list[dict]:
    """
    Remove exact duplicate tuples within the same article.

    Preserve the first occurrence and its sentence.
    """

    unique_triples = []
    seen = set()

    for triple in triples:

        key = (
            normalize_tuple_value(triple.get("article_id")),
            normalize_tuple_value(triple.get("subject")),
            normalize_tuple_value(triple.get("relationship")),
            normalize_tuple_value(triple.get("object"))
        )

        if key in seen:
            continue

        seen.add(key)
        unique_triples.append(triple)

    removed_count = len(triples) - len(unique_triples)

    print(
        f"Duplicate tuples removed: {removed_count}"
    )

    print(
        f"Unique tuples remaining: {len(unique_triples)}"
    )

    return unique_triples