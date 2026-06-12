from scripts.graph import build_graph_payload, canonicalize_concepts


def test_payload_contract(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    payload = build_graph_payload(sample_kb)
    assert set(payload) == {"nodes", "edges", "communities", "gaps"}
    node = next(n for n in payload["nodes"] if n["id"] == "d1")
    assert node["kind"] == "document" and "community" in node and "centrality" in node
    edge = next(e for e in payload["edges"] if e["kind"] == "similarity")
    assert {"id", "source", "target", "weight", "created_at"} <= set(edge)
    assert all({"id", "label", "node_count"} <= set(c) for c in payload["communities"])


def test_community_label_falls_back_to_top_doc_title(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    payload = build_graph_payload(sample_kb)
    labels = {c["label"] for c in payload["communities"]}
    # no cached labels -> fallback is some node label from the community
    assert all(isinstance(lbl, str) and lbl for lbl in labels)


def test_cached_community_labels_used(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    base = build_graph_payload(sample_kb)
    cid = base["communities"][0]["id"]
    sample_kb["graph_cache"] = {"community_labels": {str(cid): "Custom Label"}}
    payload = build_graph_payload(sample_kb)
    assert any(c["label"] == "Custom Label" for c in payload["communities"])


def test_gap_suggestion_is_question_oriented(sample_kb, fake_embed):
    # d3 (RAG Note) is isolated from d1/d2 -> may produce a gap if community split occurs
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    payload = build_graph_payload(sample_kb)
    for gap in payload["gaps"]:
        assert "질문" in gap["suggestion"]
