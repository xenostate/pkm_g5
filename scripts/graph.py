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


# ── Analysis ───────────────────────────────────────────────────────────────

def analyze_graph(G: nx.Graph, seed: int = 42) -> None:
    """Annotate nodes in-place with community id, betweenness centrality, degree."""
    if len(G) == 0:
        return
    communities = nx.community.louvain_communities(G, weight="weight", seed=seed)
    for cid, members in enumerate(communities):
        for n in members:
            G.nodes[n]["community"] = cid
    centrality = nx.betweenness_centrality(G, weight=None, normalized=True)
    for n, c in centrality.items():
        G.nodes[n]["centrality"] = round(c, 4)
        G.nodes[n]["degree"] = G.degree(n)


def find_structural_gaps(G: nx.Graph, max_density: float = 0.05, min_size: int = 2) -> list[dict]:
    """InfraNodus-style gaps: large community pairs with sparse inter-links."""
    from itertools import combinations

    comm_nodes = {}
    for n, data in G.nodes(data=True):
        if "community" in data:
            comm_nodes.setdefault(data["community"], []).append(n)

    gaps = []
    for a, b in combinations(sorted(comm_nodes), 2):
        na, nb = comm_nodes[a], comm_nodes[b]
        if len(na) < min_size or len(nb) < min_size:
            continue
        inter = sum(1 for u in na for v in nb if G.has_edge(u, v))
        density = inter / (len(na) * len(nb))
        if density <= max_density:
            bridge_a = max(na, key=lambda n: G.nodes[n].get("centrality", 0))
            bridge_b = max(nb, key=lambda n: G.nodes[n].get("centrality", 0))
            gaps.append({
                "a": a, "b": b,
                "label_a": G.nodes[bridge_a].get("label", str(a)),
                "label_b": G.nodes[bridge_b].get("label", str(b)),
                "bridge_a": bridge_a, "bridge_b": bridge_b,
                "size_product": len(na) * len(nb),
            })
    gaps.sort(key=lambda g: -g["size_product"])
    return gaps[:5]
