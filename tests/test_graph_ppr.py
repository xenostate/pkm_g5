from scripts.graph import canonicalize_concepts, ppr_doc_scores


def test_ppr_boosts_docs_linked_via_matching_concepts(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    scores = ppr_doc_scores(sample_kb, fake_embed, "agile")
    assert scores
    # d2 cites Agile directly; d3 only has RAG (orthogonal to the query)
    assert scores["d2"] > scores["d3"]
    # normalized: max score is 1.0
    assert max(scores.values()) == 1.0


def test_ppr_empty_graph_returns_empty(fake_embed):
    kb = {"documents": {}, "qa_history": []}
    assert ppr_doc_scores(kb, fake_embed, "anything") == {}


def test_ppr_no_seed_match_returns_empty(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    # random query vector: cosine to every concept below the seed floor
    scores = ppr_doc_scores(sample_kb, fake_embed, "zzz unrelated gibberish query")
    assert scores == {}
