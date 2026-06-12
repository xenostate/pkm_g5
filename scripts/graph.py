#!/usr/bin/env python3
"""Knowledge graph layer: canonicalization, NetworkX construction, analysis, payloads."""
from collections import Counter

import numpy as np
import networkx as nx


# ── Concept canonicalization ───────────────────────────────────────────────

def _union_find_merge(labels: list[str], vectors: np.ndarray, threshold: float) -> list[list[int]]:
    """Group indices whose cosine similarity >= threshold (connected components)."""
    n = len(labels)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sims = vectors @ vectors.T
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def canonicalize_concepts(kb: dict, embed_fn, threshold: float = 0.86) -> dict:
    """Build kb["concept_index"]: raw lower-cased concept -> canonical label.

    embed_fn: list[str] -> np.ndarray of L2-normalized rows (injectable for tests;
    production passes a wrapper around the loaded e5 model).
    """
    counts = Counter()
    display = {}  # lower -> first-seen original casing
    for doc in kb.get("documents", {}).values():
        for c in doc.get("concepts", []):
            if isinstance(c, str) and c.strip():
                key = c.strip().lower()
                counts[key] += 1
                display.setdefault(key, c.strip())

    labels = sorted(counts)
    index = {}
    if labels:
        vectors = embed_fn(labels)
        for group in _union_find_merge(labels, np.asarray(vectors), threshold):
            members = [labels[i] for i in group]
            # canonical = most frequent raw form; ties -> shortest label
            canonical_key = sorted(members, key=lambda m: (-counts[m], len(m)))[0]
            canonical = display[canonical_key]
            for m in members:
                index[m] = canonical

    kb["concept_index"] = index
    return index


# ── Graph construction ─────────────────────────────────────────────────────

def build_graph(kb: dict) -> nx.Graph:
    """Document + canonical-concept bipartite graph with doc-doc similarity edges."""
    G = nx.Graph()
    index = kb.get("concept_index", {})

    for doc_id, doc in kb.get("documents", {}).items():
        G.add_node(doc_id, kind="document", label=doc.get("title", doc_id),
                   source_type=doc.get("source_type", ""), created_at=doc.get("added_at", ""))

    for doc_id, doc in kb.get("documents", {}).items():
        for raw in doc.get("concepts", []):
            if not isinstance(raw, str) or not raw.strip():
                continue
            canonical = index.get(raw.strip().lower(), raw.strip())
            node_id = f"concept::{canonical}"
            if not G.has_node(node_id):
                G.add_node(node_id, kind="concept", label=canonical, created_at=doc.get("added_at", ""))
            if G.has_edge(doc_id, node_id):
                G.edges[doc_id, node_id]["weight"] += 1.0
            else:
                G.add_edge(doc_id, node_id, kind="concept", weight=1.0, created_at=doc.get("added_at", ""))

        for conn in doc.get("connections", []):
            other = conn.get("doc_id")
            if other and G.has_node(other) and not G.has_edge(doc_id, other):
                G.add_edge(doc_id, other, kind="similarity",
                           weight=float(conn.get("similarity", 0.0)),
                           label=conn.get("description", ""), created_at=doc.get("added_at", ""))
    return G
