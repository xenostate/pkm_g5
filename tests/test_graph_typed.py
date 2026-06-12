from scripts.graph import build_concept_graph, build_graph_payload, canonicalize_concepts


def _kb_with_triples(sample_kb):
    sample_kb["entities"] = {
        "RAG": {"type": "tech", "doc_ids": ["d1", "d3"], "created_at": "2026-01-01T00:00:00+00:00"},
        "Embedding": {"type": "tech", "doc_ids": ["d3"], "created_at": "2026-03-01T00:00:00+00:00"},
    }
    sample_kb["triples"] = [
        {"subject": "RAG", "predicate": "uses", "object": "Embedding", "category": "uses",
         "doc_id": "d3", "created_at": "2026-03-01T00:00:00+00:00"},
        {"subject": "Agile", "predicate": "is a kind of", "object": "Software Process",
         "category": "is-a", "doc_id": "d2", "created_at": "2026-02-01T00:00:00+00:00"},
    ]
    return sample_kb


def test_triple_edges_between_canonical_nodes(sample_kb, fake_embed):
    kb = _kb_with_triples(sample_kb)
    canonicalize_concepts(kb, embed_fn=fake_embed, threshold=0.9)
    G = build_concept_graph(kb, embed_fn=fake_embed)
    # Agile and Software Process are existing concept nodes -> typed edge added
    assert G.has_edge("concept::Agile", "concept::Software Process")
    edge = G.edges["concept::Agile", "concept::Software Process"]
    assert edge["kind"] == "triple" and edge["category"] == "is-a"
    assert edge["label"] == "is a kind of"


def test_unknown_triple_endpoint_becomes_entity_node(sample_kb, fake_embed):
    kb = _kb_with_triples(sample_kb)
    canonicalize_concepts(kb, embed_fn=fake_embed, threshold=0.9)
    G = build_concept_graph(kb, embed_fn=fake_embed)
    # "Embedding" is not a doc concept; node created from kb entities with type
    node = G.nodes["concept::Embedding"]
    assert node["kind"] == "entity"
    assert node["entity_type"] == "tech"
    assert G.has_edge("concept::RAG", "concept::Embedding")


def test_entity_names_canonicalized_with_concepts(sample_kb, fake_embed):
    kb = _kb_with_triples(sample_kb)
    # entity "RAG" and concept "Retrieval Augmented Generation" must merge
    index = canonicalize_concepts(kb, embed_fn=fake_embed, threshold=0.9)
    assert index["rag"] == index["retrieval augmented generation"]


def test_semantic_edges_connect_cross_document_concepts_only(sample_kb, fake_embed):
    # "Agile" (d2) and "Agile methodology" hypothetical: use existing fixture pairs.
    # RAG (d1, merged with d3's Retrieval Augmented Generation) vs Agile (d2):
    # orthogonal vectors -> no semantic edge. Software Process co-occurs already.
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    G = build_concept_graph(sample_kb, embed_fn=fake_embed)
    semantic = [(u, v, d) for u, v, d in G.edges(data=True) if d["kind"] == "semantic"]
    for u, v, d in semantic:
        # semantic edges never duplicate a same-doc pair
        assert not set(G.nodes[u]["doc_ids"]) & set(G.nodes[v]["doc_ids"])
        assert d["weight"] >= 0.6


def test_payload_includes_triple_edges(sample_kb, fake_embed):
    kb = _kb_with_triples(sample_kb)
    canonicalize_concepts(kb, embed_fn=fake_embed, threshold=0.9)
    payload = build_graph_payload(kb, embed_fn=fake_embed)
    triple_edges = [e for e in payload["edges"] if e["kind"] == "triple"]
    assert len(triple_edges) == 2
    assert all(e["category"] in {"is-a", "uses"} for e in triple_edges)
