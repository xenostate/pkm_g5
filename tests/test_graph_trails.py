from scripts.graph import build_concept_graph, canonicalize_concepts


def test_trail_connects_concepts_mentioned_in_same_qa(sample_kb, fake_embed):
    sample_kb["qa_history"] = [{
        "question": "How does Agile relate to RAG?",
        "answer": "Agile process docs can be indexed with RAG pipelines.",
        "timestamp": "2026-04-01T00:00:00+00:00",
    }]
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    G = build_concept_graph(sample_kb, embed_fn=fake_embed)
    assert G.has_edge("concept::Agile", "concept::RAG")
    edge = G.edges["concept::Agile", "concept::RAG"]
    assert edge["kind"] == "trail"
    assert edge["weight"] == 1.0
    assert edge["created_at"] == "2026-04-01T00:00:00+00:00"


def test_trail_weight_accumulates_and_never_overrides_other_kinds(sample_kb, fake_embed):
    qa = {
        "question": "Agile and RAG again?",
        "answer": "Yes: Agile, RAG.",
        "timestamp": "2026-04-02T00:00:00+00:00",
    }
    sample_kb["qa_history"] = [qa, dict(qa)]
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    G = build_concept_graph(sample_kb, embed_fn=fake_embed)
    assert G.edges["concept::Agile", "concept::RAG"]["weight"] == 2.0
    # Software Process & RAG co-occur in d1 -> trail must not replace that edge
    sample_kb["qa_history"] = [{
        "question": "Software Process with RAG?",
        "answer": "Software Process and RAG.",
        "timestamp": "2026-04-03T00:00:00+00:00",
    }]
    G2 = build_concept_graph(sample_kb, embed_fn=fake_embed)
    assert G2.edges["concept::RAG", "concept::Software Process"]["kind"] == "cooccur"


def test_no_trail_for_single_concept_mention(sample_kb, fake_embed):
    sample_kb["qa_history"] = [{
        "question": "What is Agile?", "answer": "A methodology.",
        "timestamp": "2026-04-01T00:00:00+00:00",
    }]
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    G = build_concept_graph(sample_kb, embed_fn=fake_embed)
    assert not any(d["kind"] == "trail" for _, _, d in G.edges(data=True))
