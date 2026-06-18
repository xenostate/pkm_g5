from scripts.graph import canonicalize_concepts


def test_merges_synonymous_concepts(sample_kb, fake_embed):
    index = canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    # "RAG" and "Retrieval Augmented Generation" must share one canonical label
    assert index["rag"] == index["retrieval augmented generation"]
    # Unrelated concepts stay separate
    assert index["software process"] != index["agile"]


def test_canonical_label_prefers_most_frequent_then_shortest(sample_kb, fake_embed):
    index = canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    # both appear once -> shortest raw form wins
    assert index["rag"] == "RAG"


def test_index_stored_on_kb(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    assert "concept_index" in sample_kb
