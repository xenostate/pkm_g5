import json

from scripts.graph import (build_concept_graph, build_graph_payload,
                           canonicalize_concepts, label_semantic_edges)


def _kb_with_semantic_pair(sample_kb):
    # "Agile" (d2) and "Scrum" (new doc d4) have cosine ~0.93: semantically
    # close but below the merge threshold, and never share a document
    sample_kb["documents"]["d4"] = {
        "id": "d4", "title": "Methodology Note", "source_type": "text",
        "added_at": "2026-04-01T00:00:00+00:00", "summary": "s4",
        "concepts": ["Scrum"], "connections": [],
    }
    return sample_kb


def test_unlabeled_semantic_edges_hidden_from_payload(sample_kb, fake_embed):
    kb = _kb_with_semantic_pair(sample_kb)
    canonicalize_concepts(kb, embed_fn=fake_embed, threshold=0.95)
    G = build_concept_graph(kb, embed_fn=fake_embed)
    assert any(d["kind"] == "semantic" for _, _, d in G.edges(data=True))
    payload = build_graph_payload(kb, embed_fn=fake_embed)
    assert not any(e["kind"] == "semantic" for e in payload["edges"])


def test_labeled_semantic_edges_survive_payload(sample_kb, fake_embed):
    kb = _kb_with_semantic_pair(sample_kb)
    canonicalize_concepts(kb, embed_fn=fake_embed, threshold=0.95)

    calls = []

    def llm_fn(prompt):
        calls.append(prompt)
        return json.dumps({"labels": ["같은 방법론을 가리킴"] * 10})

    added = label_semantic_edges(kb, llm_fn, embed_fn=fake_embed)
    assert added >= 1 and len(calls) == 1  # one batched call

    payload = build_graph_payload(kb, embed_fn=fake_embed)
    sems = [e for e in payload["edges"] if e["kind"] == "semantic"]
    assert sems and all(e["label"] == "같은 방법론을 가리킴" for e in sems)

    # second run: everything cached -> no new LLM call
    label_semantic_edges(kb, llm_fn, embed_fn=fake_embed)
    assert len(calls) == 1


def test_null_label_means_verified_unrelated(sample_kb, fake_embed):
    kb = _kb_with_semantic_pair(sample_kb)
    canonicalize_concepts(kb, embed_fn=fake_embed, threshold=0.95)

    def llm_fn(prompt):
        return json.dumps({"labels": [None] * 10})

    label_semantic_edges(kb, llm_fn, embed_fn=fake_embed)
    payload = build_graph_payload(kb, embed_fn=fake_embed)
    assert not any(e["kind"] == "semantic" for e in payload["edges"])
    # cached as "" so it is never re-asked
    assert "" in kb["graph_cache"]["semantic_labels"].values()
