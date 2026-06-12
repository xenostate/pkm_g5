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


def canonicalize_concepts(kb: dict, embed_fn, threshold: float = 0.94) -> dict:
    """Build kb["concept_index"]: raw lower-cased concept -> canonical label.

    embed_fn: list[str] -> np.ndarray of L2-normalized rows (injectable for tests;
    production passes a wrapper around the loaded e5 model).

    Default threshold is calibrated for e5 embeddings, whose cosine range is
    compressed (~0.75-1.0 even for unrelated short labels): 0.94 merges true
    synonyms ("software process" ~ "software process models") without
    collapsing distinct concepts.
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


# ── Frontend payload ───────────────────────────────────────────────────────

def build_graph_payload(kb: dict) -> dict:
    """Serialize the analyzed graph for GET /api/graph (frontend contract)."""
    G = build_graph(kb)
    analyze_graph(G)
    gaps = find_structural_gaps(G)

    cached_labels = kb.get("graph_cache", {}).get("community_labels", {})
    comm_nodes = {}
    for n, d in G.nodes(data=True):
        comm_nodes.setdefault(d.get("community", -1), []).append(n)

    communities = []
    for cid, members in sorted(comm_nodes.items()):
        top = max(members, key=lambda n: G.nodes[n].get("centrality", 0))
        communities.append({
            "id": cid,
            "label": cached_labels.get(str(cid)) or G.nodes[top].get("label", str(cid)),
            "node_count": len(members),
        })

    nodes = [{
        "id": n, "label": d.get("label", n), "kind": d.get("kind", "document"),
        "entity_type": d.get("entity_type", ""), "community": d.get("community", -1),
        "centrality": d.get("centrality", 0.0), "degree": d.get("degree", 0),
        "created_at": d.get("created_at", ""), "source_type": d.get("source_type", ""),
    } for n, d in G.nodes(data=True)]

    edges = [{
        "id": f"e{i}", "source": u, "target": v, "kind": d.get("kind", ""),
        "weight": d.get("weight", 0.0), "label": d.get("label", ""),
        "category": d.get("category", ""), "created_at": d.get("created_at", ""),
    } for i, (u, v, d) in enumerate(G.edges(data=True))]

    label_by_cid = {c["id"]: c["label"] for c in communities}
    for g in gaps:
        name_a = label_by_cid.get(g["a"], g["label_a"])
        name_b = label_by_cid.get(g["b"], g["label_b"])
        g["suggestion"] = (
            f"'{name_a}' 영역과 '{name_b}' 영역이 거의 연결되어 있지 않습니다. "
            f"두 영역을 잇는 질문을 Chat에 해보세요."
        )

    return {"nodes": nodes, "edges": edges, "communities": communities, "gaps": gaps}


# ── Community labels (LLM, cached) ─────────────────────────────────────────

def label_communities(kb: dict, payload: dict, llm_fn) -> int:
    """Fill kb["graph_cache"]["community_labels"] for unlabeled communities.

    llm_fn: str prompt -> str label. One call per unlabeled community; failures
    are swallowed so labeling can never break a refresh. Returns labels added.
    """
    cache = kb.setdefault("graph_cache", {}).setdefault("community_labels", {})

    members_by_cid = {}
    for node in payload.get("nodes", []):
        members_by_cid.setdefault(node.get("community", -1), []).append(node)

    added = 0
    for comm in payload.get("communities", []):
        cid = comm["id"]
        if cid < 0 or str(cid) in cache:
            continue
        members = members_by_cid.get(cid, [])
        titles = [n["label"] for n in members if n.get("kind") == "document"][:5]
        concepts = [n["label"] for n in members if n.get("kind") != "document"][:8]
        if not titles and not concepts:
            continue
        prompt = (
            "Give a short topic label (max 4 words, same language as the titles) "
            "for a cluster of documents.\n"
            f"Document titles: {', '.join(titles)}\n"
            f"Key concepts: {', '.join(concepts)}\n"
            "Reply with the label only."
        )
        try:
            label = (llm_fn(prompt) or "").strip().strip('"')
        except Exception:
            continue
        if label:
            cache[str(cid)] = label[:60]
            added += 1
    return added
