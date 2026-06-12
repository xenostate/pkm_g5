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
    # extracted entity names share the same canonical space as concepts
    for name, entry in kb.get("entities", {}).items():
        if isinstance(name, str) and name.strip():
            key = name.strip().lower()
            counts[key] += max(len(entry.get("doc_ids", [])), 1)
            display.setdefault(key, name.strip())

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


# ── Frontend payload (concept-centric) ─────────────────────────────────────

def _concept_membership(kb: dict) -> dict[str, dict]:
    """Map canonical concept -> {doc_ids, docs (titles), created_at (earliest)}."""
    index = kb.get("concept_index", {})
    members: dict[str, dict] = {}
    for doc_id, doc in kb.get("documents", {}).items():
        for raw in doc.get("concepts", []):
            if not isinstance(raw, str) or not raw.strip():
                continue
            canonical = index.get(raw.strip().lower(), raw.strip())
            entry = members.setdefault(canonical, {"doc_ids": [], "docs": [], "created_at": ""})
            if doc_id not in entry["doc_ids"]:
                entry["doc_ids"].append(doc_id)
                entry["docs"].append(doc.get("title", doc_id))
                added = doc.get("added_at", "")
                if added and (not entry["created_at"] or added < entry["created_at"]):
                    entry["created_at"] = added
    return members


# Cross-document semantic edge floor, in rescaled-cosine units (raw e5 ~0.88).
SEMANTIC_EDGE_MIN = 0.6
# Max semantic edges kept per node at build time (strongest first).
SEMANTIC_EDGE_TOP_K = 4


def build_concept_graph(kb: dict, embed_fn=None) -> nx.Graph:
    """Concept-projection graph: canonical concepts/entities are the nodes.

    Edge kinds, by descending meaningfulness:
    - "triple":   LLM-extracted typed relation (subject -predicate-> object)
    - "semantic": embedding similarity between concepts of *different* documents
                  (same-doc relatedness is expected; cross-doc links carry insight)
    - "cooccur":  same-document co-occurrence (kept as an optional layer)
    """
    members = _concept_membership(kb)
    G = nx.Graph()
    for canonical, entry in members.items():
        G.add_node(f"concept::{canonical}", kind="concept", label=canonical,
                   doc_ids=entry["doc_ids"], docs=entry["docs"],
                   freq=len(entry["doc_ids"]), created_at=entry["created_at"])

    labels = list(members)
    sim = {}
    if embed_fn is not None and len(labels) > 1:
        vectors = np.asarray(embed_fn(labels))
        sims = vectors @ vectors.T
        pos = {lbl: i for i, lbl in enumerate(labels)}
        # e5 cosine range is compressed; rescale ~[0.7, 1.0] -> [0, 1]
        sim = {lbl: {o: max(0.0, min(1.0, (float(sims[pos[lbl], pos[o]]) - 0.7) / 0.3))
                     for o in labels if o != lbl} for lbl in labels}

    by_doc: dict[str, list[str]] = {}
    for canonical, entry in members.items():
        for doc_id in entry["doc_ids"]:
            by_doc.setdefault(doc_id, []).append(canonical)

    titles = {doc_id: doc.get("title", doc_id) for doc_id, doc in kb.get("documents", {}).items()}

    # 1) same-document co-occurrence (optional layer on the frontend)
    for doc_id, concepts in by_doc.items():
        for i, a in enumerate(concepts):
            for b in concepts[i + 1:]:
                u, v = f"concept::{a}", f"concept::{b}"
                if G.has_edge(u, v):
                    G.edges[u, v]["weight"] += 1.0
                    if titles.get(doc_id) not in G.edges[u, v]["shared_docs"]:
                        G.edges[u, v]["shared_docs"].append(titles.get(doc_id, doc_id))
                else:
                    bonus = sim.get(a, {}).get(b, 0.0)
                    G.add_edge(u, v, kind="cooccur", weight=1.0 + bonus,
                               shared_docs=[titles.get(doc_id, doc_id)],
                               created_at=min(G.nodes[u]["created_at"], G.nodes[v]["created_at"]))

    # 2) cross-document semantic edges: concepts from different docs whose
    #    embeddings are close — the non-obvious links between sources
    if sim:
        candidates: dict[str, list[tuple[float, str]]] = {}
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                if G.has_edge(f"concept::{a}", f"concept::{b}"):
                    continue  # already linked by co-occurrence
                if set(members[a]["doc_ids"]) & set(members[b]["doc_ids"]):
                    continue  # same-doc pair: relatedness is expected
                s = sim[a].get(b, 0.0)
                if s >= SEMANTIC_EDGE_MIN:
                    candidates.setdefault(a, []).append((s, b))
                    candidates.setdefault(b, []).append((s, a))
        kept: set[tuple[str, str]] = set()
        for a, pairs in candidates.items():
            for s, b in sorted(pairs, reverse=True)[:SEMANTIC_EDGE_TOP_K]:
                kept.add(tuple(sorted((a, b))))
        for a, b in kept:
            u, v = f"concept::{a}", f"concept::{b}"
            G.add_edge(u, v, kind="semantic", weight=round(sim[a][b], 3),
                       created_at=max(G.nodes[u]["created_at"], G.nodes[v]["created_at"]))

    # 3) typed relation edges from extracted SPO triples (primary layer)
    index = kb.get("concept_index", {})
    entities = kb.get("entities", {})

    def _node_for(name: str) -> str | None:
        canonical = index.get(name.strip().lower(), name.strip())
        node_id = f"concept::{canonical}"
        if not G.has_node(node_id):
            ent = entities.get(name) or entities.get(canonical)
            if ent is None:
                return None
            doc_ids = ent.get("doc_ids", [])
            G.add_node(node_id, kind="entity", label=canonical,
                       entity_type=ent.get("type", "concept"),
                       doc_ids=doc_ids, docs=[titles.get(d, d) for d in doc_ids],
                       freq=max(len(doc_ids), 1), created_at=ent.get("created_at", ""))
        return node_id

    for t in kb.get("triples", []):
        u = _node_for(t.get("subject", ""))
        v = _node_for(t.get("object", ""))
        if not u or not v or u == v:
            continue
        # typed relations override weaker edge kinds on the same pair
        G.add_edge(u, v, kind="triple", weight=2.0,
                   label=t.get("predicate", ""), category=t.get("category", "related"),
                   created_at=t.get("created_at", ""))
    return G


def build_graph_payload(kb: dict, embed_fn=None) -> dict:
    """Serialize the analyzed concept graph for GET /api/graph.

    Contract: {nodes, edges, communities, gaps, documents} where nodes are
    canonical concepts annotated with their source documents.
    """
    G = build_concept_graph(kb, embed_fn=embed_fn)
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
        "id": n, "label": d.get("label", n), "kind": d.get("kind", "concept"),
        "entity_type": d.get("entity_type", ""), "community": d.get("community", -1),
        "centrality": d.get("centrality", 0.0), "degree": d.get("degree", 0),
        "freq": d.get("freq", 1), "doc_ids": d.get("doc_ids", []), "docs": d.get("docs", []),
        "created_at": d.get("created_at", ""),
    } for n, d in G.nodes(data=True)]

    edges = [{
        "id": f"e{i}", "source": u, "target": v, "kind": d.get("kind", ""),
        "weight": round(d.get("weight", 0.0), 3), "label": d.get("label", ""),
        "shared_docs": d.get("shared_docs", []), "category": d.get("category", ""),
        "created_at": d.get("created_at", ""),
    } for i, (u, v, d) in enumerate(G.edges(data=True))]

    documents = [{
        "id": doc_id, "title": doc.get("title", doc_id),
        "source_type": doc.get("source_type", ""), "created_at": doc.get("added_at", ""),
    } for doc_id, doc in kb.get("documents", {}).items()]

    label_by_cid = {c["id"]: c["label"] for c in communities}
    for g in gaps:
        name_a = label_by_cid.get(g["a"], g["label_a"])
        name_b = label_by_cid.get(g["b"], g["label_b"])
        g["suggestion"] = (
            f"'{name_a}' 영역과 '{name_b}' 영역이 거의 연결되어 있지 않습니다. "
            f"두 영역을 잇는 질문을 Chat에 해보세요."
        )

    return {"nodes": nodes, "edges": edges, "communities": communities,
            "gaps": gaps, "documents": documents}


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
        titles = []
        for n in members:
            for title in n.get("docs", []):
                if title not in titles:
                    titles.append(title)
        titles = titles[:5]
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
