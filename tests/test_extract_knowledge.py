from scripts.rag import parse_knowledge_response


def test_valid_json_parsed():
    raw = '''{"entities": [{"name": "GraphRAG", "type": "tech"}, {"name": "Knowledge Graph", "type": "concept"}],
"triples": [{"subject": "GraphRAG", "predicate": "builds", "object": "Knowledge Graph", "category": "uses"}]}'''
    out = parse_knowledge_response(raw)
    assert out["entities"] == [{"name": "GraphRAG", "type": "tech"},
                               {"name": "Knowledge Graph", "type": "concept"}]
    assert out["triples"][0]["category"] == "uses"


def test_invalid_category_and_type_coerced():
    raw = '''{"entities": [{"name": "X", "type": "banana"}],
"triples": [{"subject": "X", "predicate": "loves", "object": "Y", "category": "banana"}]}'''
    out = parse_knowledge_response(raw)
    assert out["entities"][0]["type"] == "concept"
    assert out["triples"][0]["category"] == "related"


def test_malformed_json_returns_empty():
    out = parse_knowledge_response("this is not json at all")
    assert out == {"entities": [], "triples": []}


def test_incomplete_triples_dropped():
    raw = '{"entities": [], "triples": [{"subject": "A", "predicate": "", "object": "B"}, {"subject": "A"}]}'
    out = parse_knowledge_response(raw)
    assert out["triples"] == []


def test_non_string_entities_dropped_and_capped():
    entities = [{"name": f"E{i}", "type": "tech"} for i in range(30)] + [{"name": 42}, "junk"]
    import json
    out = parse_knowledge_response(json.dumps({"entities": entities, "triples": []}))
    assert len(out["entities"]) == 15
    assert all(isinstance(e["name"], str) for e in out["entities"])
