from scripts.graph import build_graph_payload, canonicalize_concepts, label_communities


def test_labels_populated_and_cached(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    payload = build_graph_payload(sample_kb)

    calls = []

    def llm_fn(prompt):
        calls.append(prompt)
        return f"Label {len(calls)}"

    label_communities(sample_kb, payload, llm_fn)
    labels = sample_kb["graph_cache"]["community_labels"]
    assert len(labels) == len(payload["communities"])
    assert all(isinstance(v, str) and v for v in labels.values())
    first_call_count = len(calls)

    # second run: everything cached -> no new LLM calls
    label_communities(sample_kb, payload, llm_fn)
    assert len(calls) == first_call_count


def test_llm_failure_does_not_raise(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    payload = build_graph_payload(sample_kb)

    def broken_llm(prompt):
        raise RuntimeError("api down")

    label_communities(sample_kb, payload, broken_llm)  # must not raise
    assert sample_kb.get("graph_cache", {}).get("community_labels", {}) == {}
