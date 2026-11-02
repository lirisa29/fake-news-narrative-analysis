
import pandas as pd
import networkx as nx

from collections import defaultdict


def build_knowledge_graph(data, max_edges=500, max_evidence=3):

    G = nx.MultiDiGraph()

    if data.empty:
        return G

    keys = [
        "subject",
        "relationship",
        "object"
    ]

    work = data[
        keys + [
            "article_id",
            "source",
            "date",
            "sentence"
        ]
    ].copy()

    # Clean missing values
    for col in keys:
        work[col] = work[col].fillna(
            "Unknown"
        ).astype(str)

    work["article_id"] = (
        work["article_id"]
        .fillna("Unknown")
        .astype(str)
    )

    work["source"] = (
        work["source"]
        .fillna("Unknown")
        .astype(str)
    )

    work["sentence"] = (
        work["sentence"]
        .fillna("")
        .astype(str)
    )

    # Aggregate duplicate relationships
    relationship_counts = (
        work.groupby(keys, sort=False)
        .agg(
            article_count=(
                "article_id",
                "nunique"
            )
        )
        .reset_index()
    )

    # Prioritise relationships with most evidence
    relationship_counts = (
        relationship_counts
        .sort_values(
            "article_count",
            ascending=False
        )
        .head(max_edges)
    )

    # Collect limited supporting evidence
    evidence = (
        work.drop_duplicates(
            subset=keys + ["article_id"]
        )
        .groupby(
            keys,
            sort=False
        )
        .head(max_evidence)
    )

    evidence_lookup = defaultdict(list)

    for row in evidence.itertuples(index=False):

        date = (
            row.date.strftime("%Y-%m-%d")
            if pd.notna(row.date)
            else "Unknown"
        )

        evidence_lookup[
            (row.subject, row.relationship, row.object)
        ].append(
            f"Article: {row.article_id}\n"
            f"Source: {row.source}\n"
            f"Date: {date}\n"
            f"Sentence: {row.sentence}"
        )

    # Create graph edges
    for row in relationship_counts.itertuples(index=False):

        subject = row.subject
        relation = row.relationship
        obj = row.object

        article_count = int(row.article_count)

        supporting_evidence = evidence_lookup[
            (subject, relation, obj)
        ]

        edge_title = (
            f"Relationship: {subject} → {obj}\n"
            f"Relation: {relation}\n"
            f"Supporting articles: {article_count}\n\n"
            + "\n\n".join(supporting_evidence)
        )

        G.add_edge(
            subject,
            obj,
            key=relation,
            label=relation,
            title=edge_title,
            article_count=article_count
        )

    return G