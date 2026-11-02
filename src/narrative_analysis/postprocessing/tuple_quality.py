
import re


INVALID_VALUES = {
    "",
    "none",
    "null",
    "unknown",
    "undefined",
    "unresolved",
    "nan",
    "n/a"
}

UNRESOLVED_PRONOUNS = {
    "he", "she", "it", "they",
    "him", "her", "them",
    "this", "that",
    "these", "those"
}

NAVIGATION_TERMS = {
    "click here",
    "watch here",
    "read more",
    "click to read",
    "subscribe",
    "watch video",
    "view here"
}

# Maximum permitted word counts.
MAX_SUBJECT_WORDS = 5
MAX_RELATION_WORDS = 6
MAX_OBJECT_WORDS = 8
MAX_TUPLE_WORDS = 15


def is_valid_component(value):

    if not isinstance(value, str):
        return False

    value = value.strip()

    if value.lower() in INVALID_VALUES:
        return False

    if len(value) < 2:
        return False

    if not re.search(r"[a-zA-Z0-9]", value):
        return False

    return True


def is_navigation_tuple(triple):

    proposition = " ".join([
        str(triple.get("subject", "")),
        str(triple.get("relationship", "")),
        str(triple.get("object", ""))
    ]).lower()

    return any(
        term in proposition
        for term in NAVIGATION_TERMS
    )


def word_count(value):

    return len(
        re.findall(r"\b[\w'-]+\b", value)
    )


def filter_tuple_quality(triples):

    valid_triples = []

    removal_reasons = {
        "invalid components": 0,
        "unresolved pronouns": 0,
        "navigation tuples": 0,
        "self-referential tuples": 0,
        "subject too long": 0,
        "relationship too long": 0,
        "object too long": 0,
        "complete tuple too long": 0
    }

    for triple in triples:

        subject = triple.get("subject")
        relation = triple.get("relationship")
        obj = triple.get("object")

        # Check component validity.
        if not all([
            is_valid_component(subject),
            is_valid_component(relation),
            is_valid_component(obj)
        ]):
            removal_reasons["invalid components"] += 1
            continue

        subject = subject.strip()
        relation = relation.strip()
        obj = obj.strip()

        # Reject unresolved standalone pronouns.
        if (
            subject.lower() in UNRESOLVED_PRONOUNS
            or obj.lower() in UNRESOLVED_PRONOUNS
        ):
            removal_reasons["unresolved pronouns"] += 1
            continue

        if is_navigation_tuple(triple):
            removal_reasons["navigation tuples"] += 1
            continue

        # Reject self-referential tuples.
        if subject.casefold() == obj.casefold():
            removal_reasons["self-referential tuples"] += 1
            continue

        # Component length restrictions.
        if word_count(subject) > MAX_SUBJECT_WORDS:
            removal_reasons["subject too long"] += 1
            continue

        if word_count(relation) > MAX_RELATION_WORDS:
            removal_reasons["relationship too long"] += 1
            continue

        if word_count(obj) > MAX_OBJECT_WORDS:
            removal_reasons["object too long"] += 1
            continue

        # Total tuple length restriction.
        total_words = (
            word_count(subject)
            + word_count(relation)
            + word_count(obj)
        )

        if total_words > MAX_TUPLE_WORDS:
            removal_reasons["complete tuple too long"] += 1
            continue

        valid_triples.append(triple)

    # Reporting.
    removed_count = len(triples) - len(valid_triples)

    print("\nTUPLE QUALITY FILTERING")
    print("-" * 45)

    print(f"Input tuples: {len(triples)}")
    print(f"Removed tuples: {removed_count}")
    print(f"Valid tuples: {len(valid_triples)}")

    print("\nRemoval breakdown:")

    for reason, count in removal_reasons.items():
        print(f"{reason}: {count}")

    print("-" * 45)

    return valid_triples