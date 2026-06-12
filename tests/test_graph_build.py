from scripts.graph import build_graph, canonicalize_concepts


def test_nodes_and_bipartite_edges(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    G = build_graph(sample_kb)
    assert G.nodes["d1"]["kind"] == "document"
    # canonical concept node exists and links to both docs that cite it
    assert G.nodes["concept::Software Process"]["kind"] == "concept"
    assert G.has_edge("d1", "concept::Software Process")
    assert G.has_edge("d2", "concept::Software Process")


def test_similarity_edges_deduped_with_weight(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    G = build_graph(sample_kb)
    assert G.has_edge("d1", "d2")
    assert G.edges["d1", "d2"]["weight"] == 0.8
    assert G.edges["d1", "d2"]["kind"] == "similarity"


def test_synonym_concepts_collapse_to_one_node(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    G = build_graph(sample_kb)
    concept_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "concept"]
    # RAG + Retrieval Augmented Generation merged -> 3 canonical concepts total
    assert len(concept_nodes) == 3
