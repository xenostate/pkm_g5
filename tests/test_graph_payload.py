from scripts.graph import build_concept_graph, build_graph_payload, canonicalize_concepts


def test_payload_contract(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    payload = build_graph_payload(sample_kb, embed_fn=fake_embed)
    assert set(payload) == {"nodes", "edges", "communities", "gaps", "documents", "learning_questions"}
    node = next(n for n in payload["nodes"] if n["label"] == "Software Process")
    assert node["kind"] == "concept"
    assert {"community", "centrality", "freq", "doc_ids", "docs", "created_at"} <= set(node)
    # Software Process appears in d1 and d2
    assert set(node["doc_ids"]) == {"d1", "d2"} and node["freq"] == 2
    assert all({"id", "title", "source_type", "created_at"} <= set(d) for d in payload["documents"])
    assert len(payload["documents"]) == 3


def test_cooccurrence_edges_weighted_by_shared_docs(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    G = build_concept_graph(sample_kb, embed_fn=fake_embed)
    # d1 has {RAG, Software Process}; d2 has {Agile, Software Process}
    assert G.has_edge("concept::RAG", "concept::Software Process")
    assert G.has_edge("concept::Agile", "concept::Software Process")
    # no doc holds both RAG and Agile
    assert not G.has_edge("concept::RAG", "concept::Agile")
    edge = G.edges["concept::RAG", "concept::Software Process"]
    assert edge["kind"] == "cooccur"
    assert edge["weight"] >= 1.0
    assert edge["shared_docs"] == ["Intro to SE"]


def test_synonyms_collapse_into_one_concept_node(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    payload = build_graph_payload(sample_kb, embed_fn=fake_embed)
    labels = [n["label"] for n in payload["nodes"]]
    # RAG (d1) + Retrieval Augmented Generation (d3) merged into one node spanning both docs
    assert labels.count("RAG") == 1
    rag = next(n for n in payload["nodes"] if n["label"] == "RAG")
    assert set(rag["doc_ids"]) == {"d1", "d3"}


def test_cached_community_labels_used(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    base = build_graph_payload(sample_kb, embed_fn=fake_embed)
    top_label = base["communities"][0]["label"]
    sample_kb["graph_cache"] = {"community_labels": {f"top::{top_label.lower()}": "Custom Label"}}
    payload = build_graph_payload(sample_kb, embed_fn=fake_embed)
    assert any(c["label"] == "Custom Label" for c in payload["communities"])


def test_gap_suggestion_is_question_oriented(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    payload = build_graph_payload(sample_kb, embed_fn=fake_embed)
    for gap in payload["gaps"]:
        assert "질문" in gap["suggestion"]


def test_empty_kb_payload(fake_embed):
    kb = {"documents": {}}
    payload = build_graph_payload(kb, embed_fn=fake_embed)
    assert payload["nodes"] == [] and payload["edges"] == [] and payload["documents"] == []
