from rapidfuzz import fuzz

class EntityResolver:

    def __init__(self, threshold=80):

        self.canonical_map = {}
        self.threshold = threshold

    def get_canonical_name(self, name):

        name_clean = str(name).strip()

        if not name_clean:
            return name_clean

        # Reject unresolved pronouns.
        if name_clean.lower() in [
            "he", "she", "it", "they",
            "this", "that"
        ]:
            return "UNRESOLVED"

        # Substring / containment matching.
        for existing_name in self.canonical_map.values():

            if (
                name_clean.lower() in existing_name.lower()
                or existing_name.lower() in name_clean.lower()
            ):
                return existing_name

        # Fuzzy matching.
        for raw_name, canonical in self.canonical_map.items():

            if fuzz.ratio(
                name_clean.lower(),
                raw_name.lower()
            ) >= self.threshold:

                return canonical

        self.canonical_map[name_clean] = name_clean

        return name_clean

    def resolve_dataset(self, triples):

        resolved_triples = []

        for triple in triples:

            subject = self.get_canonical_name(
                triple["subject"]
            )

            obj = self.get_canonical_name(
                triple["object"]
            )

            if (
                subject != "UNRESOLVED"
                and obj != "UNRESOLVED"
            ):

                resolved_triples.append({
                    **triple,
                    "subject": subject,
                    "object": obj
                })

        return resolved_triples
