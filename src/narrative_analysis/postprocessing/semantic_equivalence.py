import re

from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticEquivalenceDetector:

    def __init__(
        self,
        threshold: float = 0.88,
        model_name: str = "all-MiniLM-L6-v2"
    ):

        self.threshold = threshold

        self.model = SentenceTransformer(model_name)

    @staticmethod
    def normalize(value):

        if not isinstance(value, str):
            return ""

        return " ".join(
            value.lower().strip().split()
        )

    @staticmethod
    def information_score(triple):
        """
        Estimate tuple information content.

        Longer, more descriptive source sentences
        receive a higher score.
        """

        subject = triple.get("subject", "")
        relationship = triple.get("relationship", "")
        obj = triple.get("object", "")
        sentence = triple.get("sentence", "")

        components = [
            subject,
            relationship,
            obj
        ]

        component_score = sum(
            len(value.split())
            for value in components
            if isinstance(value, str)
        )

        sentence_score = len(
            re.findall(r"\b\w+\b", sentence)
        ) if isinstance(sentence, str) else 0

        return (
            component_score * 2
            + sentence_score
        )

    def remove_semantic_duplicates(
        self,
        triples: list[dict]
    ) -> list[dict]:
        """
        Detect semantically equivalent relationships
        within individual articles.

        Retain the most informative tuple.
        """

        # Group tuples by article, subject and object.
        groups = defaultdict(list)

        for triple in triples:

            key = (
                self.normalize(triple.get("article_id")),
                self.normalize(triple.get("subject")),
                self.normalize(triple.get("object"))
            )

            groups[key].append(triple)

        retained_triples = []
        removed_count = 0

        for group in groups.values():

            if len(group) == 1:
                retained_triples.extend(group)
                continue

            # Compare relationship descriptions.
            relationships = [
                triple["relationship"]
                for triple in group
            ]

            embeddings = self.model.encode(
                relationships,
                convert_to_numpy=True
            )

            similarities = cosine_similarity(
                embeddings
            )

            # Greedy grouping of equivalent relationships.
            processed = set()

            for i in range(len(group)):

                if i in processed:
                    continue

                equivalent_indices = [i]

                for j in range(i + 1, len(group)):

                    if j in processed:
                        continue

                    if similarities[i][j] >= self.threshold:

                        equivalent_indices.append(j)

                processed.update(equivalent_indices)

                # Keep the tuple with the highest
                # information score.
                best_index = max(
                    equivalent_indices,
                    key=lambda idx: self.information_score(
                        group[idx]
                    )
                )

                retained_triples.append(
                    group[best_index]
                )

                removed_count += (
                    len(equivalent_indices) - 1
                )

        print(
            f"Semantically equivalent tuples removed: "
            f"{removed_count}"
        )

        print(
            f"Tuples remaining: {len(retained_triples)}"
        )

        return retained_triples