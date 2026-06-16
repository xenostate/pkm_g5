import json

from scripts.graph import (build_graph_payload, canonicalize_concepts,
                           suggest_learning_questions)


def test_learning_questions_generated_and_cached(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    payload = build_graph_payload(sample_kb, embed_fn=fake_embed)

    def llm_fn(prompt):
        return json.dumps({"questions": [
            {"topic": "T", "question": "What is RAG?"},
            {"topic": "T", "question": "Why does Agile matter?"},
        ]})

    qs = suggest_learning_questions(sample_kb, payload, llm_fn)
    assert len(qs) == 2
    assert all(q["question"] for q in qs)
    assert sample_kb["graph_cache"]["learning_questions"] == qs


def test_learning_questions_exposed_in_payload(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    payload = build_graph_payload(sample_kb, embed_fn=fake_embed)
    assert "learning_questions" in payload

    sample_kb["graph_cache"] = {"learning_questions": [{"topic": "X", "question": "Q?"}]}
    payload2 = build_graph_payload(sample_kb, embed_fn=fake_embed)
    assert payload2["learning_questions"] == [{"topic": "X", "question": "Q?"}]


def test_malformed_llm_response_safe(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    payload = build_graph_payload(sample_kb, embed_fn=fake_embed)

    def broken(prompt):
        return "not json"

    assert suggest_learning_questions(sample_kb, payload, broken) == []
