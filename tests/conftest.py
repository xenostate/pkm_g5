import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def fake_embed():
    """Deterministic embedder: known synonyms map to nearly-identical vectors."""
    base = {
        "rag": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        "retrieval augmented generation": np.array([0.98, 0.02, 0.0, 0.0, 0.0]),
        "agile": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        "agile methodology": np.array([0.02, 0.98, 0.0, 0.0, 0.0]),
        "software process": np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
        "requirements": np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
        # orthogonal to every known concept: for no-seed-match tests
        "zzz unrelated gibberish query": np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
    }

    def embed(labels):
        out = []
        for label in labels:
            v = base.get(label.lower(), np.random.default_rng(abs(hash(label)) % 2**32).normal(size=5))
            out.append(v / np.linalg.norm(v))
        return np.array(out)

    return embed


@pytest.fixture
def sample_kb():
    return {
        "documents": {
            "d1": {"id": "d1", "title": "Intro to SE", "source_type": "pdf", "added_at": "2026-01-01T00:00:00+00:00",
                   "summary": "s1", "concepts": ["RAG", "Software Process"],
                   "connections": [{"doc_id": "d2", "title": "Agile", "similarity": 0.8, "description": ""}]},
            "d2": {"id": "d2", "title": "Agile", "source_type": "pdf", "added_at": "2026-02-01T00:00:00+00:00",
                   "summary": "s2", "concepts": ["Agile", "Software Process"],
                   "connections": [{"doc_id": "d1", "title": "Intro to SE", "similarity": 0.8, "description": ""}]},
            "d3": {"id": "d3", "title": "RAG Note", "source_type": "text", "added_at": "2026-03-01T00:00:00+00:00",
                   "summary": "s3", "concepts": ["Retrieval Augmented Generation"], "connections": []},
        },
        "links": [], "qa_history": [], "study_progress": {}, "stats": {},
    }
