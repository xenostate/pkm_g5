# Knowledge Graph Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the PKM knowledge map from a flat similarity/string-match graph into a typed, analyzed, temporal knowledge graph with a readable interactive Cytoscape.js frontend, and fuse the graph into RAG retrieval.

**Architecture:** A new `scripts/graph.py` module owns all graph logic (concept canonicalization via e5 embeddings, NetworkX graph construction, Louvain communities, betweenness centrality, structural-gap detection, JSON payload for the frontend, Personalized PageRank for retrieval fusion). `scripts/rag.py` gains SPO-triple/typed-entity extraction replacing flat concepts. The frontend Knowledge Map tab is rebuilt on Cytoscape.js (vendored locally) with readability-first interaction rules: no always-on edge labels, hover-neighborhood highlighting, ego-focus on click, threshold slider, temporal slider.

**Tech Stack:** Python 3.11, FastAPI, NetworkX ≥3.0 (built-in `louvain_communities`, `pagerank`), sentence-transformers e5 (already loaded), ChromaDB 1.x, OpenAI gpt-4o-mini (structured JSON), Cytoscape.js 3.x (vendored), pytest.

**Constraints:**
- Local-first: no new servers/DBs. Graph persisted inside the existing per-domain `knowledge_base.json`; NetworkX built in-memory on demand.
- LLM cost control: extraction = 1 call/document (on ingest or refresh); community labels = 1 call/community, cached; normalization/analysis = 0 LLM calls.
- KB schema changes are additive with `setdefault` migration in `load_kb` — old KBs keep working.
- All e5 encodes of short labels use the `"query: "` prefix convention only for queries; bare labels are encoded as `"passage: <label>"` to stay consistent with the indexed space where relevant; for label↔label comparison use raw label encodes on both sides (symmetric, prefix-free).

**KB schema additions (all stages):**
```json
{
  "concept_index": {"raw lower-cased concept": "Canonical Label"},
  "entities": {"Canonical Name": {"type": "concept|person|tech|org|event|other", "doc_ids": [], "created_at": "iso"}},
  "triples": [{"subject": "", "predicate": "", "object": "", "category": "is-a|part-of|uses|contrasts|causes|related", "doc_id": "", "created_at": "iso"}],
  "graph_cache": {"community_labels": {"<community_id>": "label"}}
}
```

**`GET /api/graph` response contract (frontend depends on this exactly):**
```json
{
  "nodes": [{"id": "", "label": "", "kind": "document|concept|entity", "entity_type": "", "community": 0,
             "centrality": 0.0, "degree": 0, "created_at": "iso", "source_type": "pdf|url|text|"}],
  "edges": [{"id": "", "source": "", "target": "", "kind": "similarity|concept|triple|trail",
             "weight": 0.0, "label": "", "category": "", "created_at": "iso"}],
  "communities": [{"id": 0, "label": "", "node_count": 0}],
  "gaps": [{"a": 0, "b": 0, "label_a": "", "label_b": "", "bridge_a": "", "bridge_b": "", "suggestion": ""}]
}
```

---

## Stage 1 — Normalization + Graph Analysis + Cytoscape.js

### Task 1: Test scaffold + dependency pins

**Files:**
- Modify: `scripts/requirements.txt`
- Create: `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Pin networkx and add pytest** — append to `scripts/requirements.txt`:
```
networkx>=3.0
pytest>=8.0
```
(networkx 3.6.1 is already installed transitively via torch; the pin makes it explicit.)

- [ ] **Step 2: Create `tests/conftest.py`** with a synthetic KB fixture and a fake deterministic embedder (hash-free, similarity controllable):
```python
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def fake_embed():
    """Deterministic embedder: known synonyms map to nearly-identical vectors."""
    base = {
        "rag": np.array([1.0, 0.0, 0.0, 0.0]),
        "retrieval augmented generation": np.array([0.98, 0.02, 0.0, 0.0]),
        "agile": np.array([0.0, 1.0, 0.0, 0.0]),
        "agile methodology": np.array([0.02, 0.98, 0.0, 0.0]),
        "software process": np.array([0.0, 0.0, 1.0, 0.0]),
        "requirements": np.array([0.0, 0.0, 0.0, 1.0]),
    }

    def embed(labels):
        out = []
        for label in labels:
            v = base.get(label.lower(), np.random.default_rng(abs(hash(label)) % 2**32).normal(size=4))
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
```

- [ ] **Step 3: Run `cd /Users/chris40461/workspace/pkm_g5 && .venv/bin/pip install pytest -q && .venv/bin/python -m pytest tests -q`** — expect "no tests ran" (collection OK).

- [ ] **Step 4: Commit** — `chore: add test scaffold and pin networkx/pytest`

### Task 2: Concept canonicalization (`scripts/graph.py`)

**Files:**
- Create: `scripts/graph.py`
- Test: `tests/test_graph_normalize.py`

- [ ] **Step 1: Write failing tests** — `tests/test_graph_normalize.py`:
```python
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
```

- [ ] **Step 2: Run tests, verify FAIL** (`ModuleNotFoundError: scripts.graph`).

- [ ] **Step 3: Implement** `canonicalize_concepts` in new `scripts/graph.py`:
```python
#!/usr/bin/env python3
"""Knowledge graph layer: canonicalization, NetworkX construction, analysis, payloads."""
from collections import Counter

import numpy as np


def _union_find_merge(labels: list[str], vectors: np.ndarray, threshold: float) -> list[list[int]]:
    """Group indices whose cosine similarity >= threshold (connected components)."""
    n = len(labels)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sims = vectors @ vectors.T
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def canonicalize_concepts(kb: dict, embed_fn, threshold: float = 0.86) -> dict:
    """Build kb["concept_index"]: raw lower-cased concept -> canonical label.

    embed_fn: list[str] -> np.ndarray of L2-normalized rows (injectable for tests;
    production passes a wrapper around the loaded e5 model).
    """
    counts = Counter()
    display = {}  # lower -> first-seen original casing
    for doc in kb.get("documents", {}).values():
        for c in doc.get("concepts", []):
            if isinstance(c, str) and c.strip():
                key = c.strip().lower()
                counts[key] += 1
                display.setdefault(key, c.strip())

    labels = sorted(counts)
    index = {}
    if labels:
        vectors = embed_fn(labels)
        for group in _union_find_merge(labels, np.asarray(vectors), threshold):
            members = [labels[i] for i in group]
            # canonical = most frequent raw form; ties -> shortest label
            canonical_key = sorted(members, key=lambda m: (-counts[m], len(m)))[0]
            canonical = display[canonical_key]
            for m in members:
                index[m] = canonical

    kb["concept_index"] = index
    return index
```

- [ ] **Step 4: Run tests, verify PASS.**

- [ ] **Step 5: Commit** — `feat(graph): canonicalize concepts via embedding union-find`

### Task 3: NetworkX graph construction

**Files:**
- Modify: `scripts/graph.py`
- Test: `tests/test_graph_build.py`

- [ ] **Step 1: Write failing tests**:
```python
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
```

- [ ] **Step 2: Run, verify FAIL** (ImportError build_graph).

- [ ] **Step 3: Implement** in `scripts/graph.py`:
```python
import networkx as nx


def build_graph(kb: dict) -> nx.Graph:
    """Document + canonical-concept bipartite graph with doc-doc similarity edges.

    Stage 2 extends this with typed entity/triple edges and trail edges.
    """
    G = nx.Graph()
    index = kb.get("concept_index", {})

    for doc_id, doc in kb.get("documents", {}).items():
        G.add_node(doc_id, kind="document", label=doc.get("title", doc_id),
                   source_type=doc.get("source_type", ""), created_at=doc.get("added_at", ""))

    for doc_id, doc in kb.get("documents", {}).items():
        for raw in doc.get("concepts", []):
            if not isinstance(raw, str) or not raw.strip():
                continue
            canonical = index.get(raw.strip().lower(), raw.strip())
            node_id = f"concept::{canonical}"
            if not G.has_node(node_id):
                G.add_node(node_id, kind="concept", label=canonical, created_at=doc.get("added_at", ""))
            if G.has_edge(doc_id, node_id):
                G.edges[doc_id, node_id]["weight"] += 1.0
            else:
                G.add_edge(doc_id, node_id, kind="concept", weight=1.0, created_at=doc.get("added_at", ""))

        for conn in doc.get("connections", []):
            other = conn.get("doc_id")
            if other and G.has_node(other) and not G.has_edge(doc_id, other):
                G.add_edge(doc_id, other, kind="similarity",
                           weight=float(conn.get("similarity", 0.0)),
                           label=conn.get("description", ""), created_at=doc.get("added_at", ""))
    return G
```

- [ ] **Step 4: Run tests, verify PASS.** — [ ] **Step 5: Commit** `feat(graph): build NetworkX doc-concept graph`

### Task 4: Communities, centrality, structural gaps

**Files:**
- Modify: `scripts/graph.py`
- Test: `tests/test_graph_analyze.py`

- [ ] **Step 1: Write failing tests**:
```python
import networkx as nx

from scripts.graph import analyze_graph, find_structural_gaps


def _two_cluster_graph():
    G = nx.Graph()
    # cluster A: docs a1..a3 fully connected; cluster B: b1..b3 fully connected; no inter edges
    for c in ("a", "b"):
        ids = [f"{c}{i}" for i in range(1, 4)]
        for n in ids:
            G.add_node(n, kind="document", label=n, created_at="")
        for i, u in enumerate(ids):
            for v in ids[i + 1:]:
                G.add_edge(u, v, kind="similarity", weight=0.9, created_at="")
    return G


def test_analyze_assigns_community_and_centrality():
    G = _two_cluster_graph()
    analyze_graph(G)
    assert G.nodes["a1"]["community"] != G.nodes["b1"]["community"]
    assert "centrality" in G.nodes["a1"]


def test_gap_detected_between_disconnected_communities():
    G = _two_cluster_graph()
    analyze_graph(G)
    gaps = find_structural_gaps(G, max_density=0.05)
    assert len(gaps) == 1
    assert {gaps[0]["a"], gaps[0]["b"]} == {G.nodes["a1"]["community"], G.nodes["b1"]["community"]}


def test_no_gap_when_communities_bridged():
    G = _two_cluster_graph()
    for i in range(1, 4):
        G.add_edge(f"a{i}", f"b{i}", kind="similarity", weight=0.9, created_at="")
    analyze_graph(G)
    gaps = find_structural_gaps(G, max_density=0.05)
    assert gaps == []
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement**:
```python
def analyze_graph(G: nx.Graph, seed: int = 42) -> None:
    """Annotate nodes in-place with community id, betweenness centrality, degree."""
    if len(G) == 0:
        return
    communities = nx.community.louvain_communities(G, weight="weight", seed=seed)
    for cid, members in enumerate(communities):
        for n in members:
            G.nodes[n]["community"] = cid
    centrality = nx.betweenness_centrality(G, weight=None, normalized=True)
    for n, c in centrality.items():
        G.nodes[n]["centrality"] = round(c, 4)
        G.nodes[n]["degree"] = G.degree(n)


def find_structural_gaps(G: nx.Graph, max_density: float = 0.05, min_size: int = 2) -> list[dict]:
    """InfraNodus-style gaps: large community pairs with sparse inter-links."""
    from itertools import combinations

    comm_nodes = {}
    for n, data in G.nodes(data=True):
        if "community" in data:
            comm_nodes.setdefault(data["community"], []).append(n)

    gaps = []
    for a, b in combinations(sorted(comm_nodes), 2):
        na, nb = comm_nodes[a], comm_nodes[b]
        if len(na) < min_size or len(nb) < min_size:
            continue
        inter = sum(1 for u in na for v in nb if G.has_edge(u, v))
        density = inter / (len(na) * len(nb))
        if density <= max_density:
            bridge_a = max(na, key=lambda n: G.nodes[n].get("centrality", 0))
            bridge_b = max(nb, key=lambda n: G.nodes[n].get("centrality", 0))
            gaps.append({
                "a": a, "b": b,
                "label_a": G.nodes[bridge_a].get("label", str(a)),
                "label_b": G.nodes[bridge_b].get("label", str(b)),
                "bridge_a": bridge_a, "bridge_b": bridge_b,
                "size_product": len(na) * len(nb),
            })
    gaps.sort(key=lambda g: -g["size_product"])
    return gaps[:5]
```

- [ ] **Step 4: Run tests, verify PASS.** — [ ] **Step 5: Commit** `feat(graph): louvain communities, centrality, structural gaps`

### Task 5: `/api/graph` endpoint + payload

**Files:**
- Modify: `scripts/graph.py`, `scripts/server.py` (after `/api/connections/refresh`, ~line 624)
- Test: `tests/test_graph_payload.py`

- [ ] **Step 1: Write failing tests** asserting the payload contract (nodes/edges/communities/gaps keys, community labels from `kb["graph_cache"]["community_labels"]` fall back to top-centrality doc title):
```python
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
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement `build_graph_payload(kb)`** — compose `build_graph` → `analyze_graph` → `find_structural_gaps` → serialize:
```python
def build_graph_payload(kb: dict) -> dict:
    G = build_graph(kb)
    analyze_graph(G)
    gaps = find_structural_gaps(G)

    cached_labels = kb.get("graph_cache", {}).get("community_labels", {})
    comm_nodes = {}
    for n, d in G.nodes(data=True):
        comm_nodes.setdefault(d.get("community", -1), []).append(n)

    communities = []
    for cid, members in sorted(comm_nodes.items()):
        top = max(members, key=lambda n: G.nodes[n].get("centrality", 0))
        communities.append({
            "id": cid,
            "label": cached_labels.get(str(cid)) or G.nodes[top].get("label", str(cid)),
            "node_count": len(members),
        })

    nodes = [{
        "id": n, "label": d.get("label", n), "kind": d.get("kind", "document"),
        "entity_type": d.get("entity_type", ""), "community": d.get("community", -1),
        "centrality": d.get("centrality", 0.0), "degree": d.get("degree", 0),
        "created_at": d.get("created_at", ""), "source_type": d.get("source_type", ""),
    } for n, d in G.nodes(data=True)]

    edges = [{
        "id": f"e{i}", "source": u, "target": v, "kind": d.get("kind", ""),
        "weight": d.get("weight", 0.0), "label": d.get("label", ""),
        "category": d.get("category", ""), "created_at": d.get("created_at", ""),
    } for i, (u, v, d) in enumerate(G.edges(data=True))]

    for g in gaps:
        g["suggestion"] = (
            f"'{g['label_a']}' 영역과 '{g['label_b']}' 영역이 거의 연결되어 있지 않습니다. "
            f"두 주제를 잇는 노트를 작성해 보세요."
        )

    return {"nodes": nodes, "edges": edges, "communities": communities, "gaps": gaps}
```

- [ ] **Step 4: Add endpoint** in `scripts/server.py` (import `build_graph_payload`, `canonicalize_concepts` from `scripts.graph`):
```python
@app.get("/api/graph")
async def get_graph():
    kb = _get_domain_kb()

    def _embed(labels):
        return embed_model.encode(labels, normalize_embeddings=True, show_progress_bar=False)

    def _payload():
        canonicalize_concepts(kb, embed_fn=_embed)
        return build_graph_payload(kb)

    return await asyncio.to_thread(_payload)
```

- [ ] **Step 5: Run tests, verify PASS. Start server (`./start.sh`), `curl -s localhost:8090/api/graph | python3 -m json.tool | head -40`** — expect contract-shaped JSON. Stop server.

- [ ] **Step 6: Commit** — `feat(api): add GET /api/graph with communities and gap analysis`

### Task 6: Vendor Cytoscape.js + new Knowledge Map markup

**Files:**
- Create: `frontend/vendor/cytoscape.min.js` (downloaded)
- Modify: `frontend/index.html` (replace `page-connections` inner layout, lines ~99-129), `frontend/styles.css` (append map styles)

- [ ] **Step 1: Vendor the library**:
```bash
mkdir -p frontend/vendor
curl -sL https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js -o frontend/vendor/cytoscape.min.js
```
Add `<script src="/static/vendor/cytoscape.min.js"></script>` before `/static/app.js` in `index.html`.

- [ ] **Step 2: Replace the Knowledge Map markup** — inside `<div id="page-connections">`, replace the `connections-layout` block (canvas + zoom controls + legend) with:
```html
<div class="connections-layout">
    <div class="connections-container">
        <div id="cy"></div>
        <div class="graph-toolbar">
            <input type="search" id="graph-search" placeholder="Find node...">
            <label class="graph-slider-label">similarity ≥ <span id="graph-threshold-value">0.5</span>
                <input type="range" id="graph-threshold" min="0.3" max="1" step="0.05" value="0.5">
            </label>
            <button id="graph-show-concepts" class="map-toggle active" type="button">concepts</button>
            <button id="graph-show-similarity" class="map-toggle active" type="button">similarity</button>
            <button id="graph-fit" class="btn btn-secondary" type="button">Fit</button>
        </div>
        <div id="graph-legend" class="graph-legend"></div>
        <div id="connections-empty" class="empty-state">Add at least 2 documents, then click "Refresh Connections".</div>
    </div>
    <aside class="topic-panel">
        <h3>Clusters</h3>
        <div id="community-list" class="topic-correlation-list"></div>
        <h3>Bridge Ideas</h3>
        <p class="topic-panel-copy">Disconnected areas of your knowledge — write a note that bridges them.</p>
        <div id="gap-list" class="topic-correlation-list"></div>
    </aside>
</div>
```
(Keep the existing `refresh-connections-btn` page header. The old canvas/tooltip/zoom markup is deleted; legacy canvas JS is removed in Task 7.)

- [ ] **Step 3: Append styles** to `frontend/styles.css`:
```css
#cy { width: 100%; height: 640px; border-radius: 12px; background: #101014; }
.graph-toolbar { display: flex; gap: 12px; align-items: center; padding: 10px 4px; flex-wrap: wrap; }
.graph-toolbar input[type="search"] { flex: 0 0 200px; padding: 6px 10px; border-radius: 8px;
    border: 1px solid #333; background: #1a1a20; color: #eee; }
.graph-slider-label { display: flex; gap: 8px; align-items: center; font-size: 0.85rem; color: #aaa; }
.graph-legend { display: flex; gap: 10px; flex-wrap: wrap; padding: 6px 4px; font-size: 0.8rem; color: #bbb; }
.graph-legend .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }
.gap-card { border: 1px dashed #c9a227; border-radius: 10px; padding: 10px; margin-bottom: 8px; font-size: 0.85rem; }
.gap-card button { margin-top: 6px; }
```

- [ ] **Step 4: Commit** — `feat(frontend): vendor cytoscape.js and rebuild knowledge map markup`

### Task 7: Cytoscape renderer + readability interactions

**Files:**
- Modify: `frontend/app.js` — delete legacy canvas functions (`drawConnectionsMap`, `findKnowledgeMapHoverTarget`, `showConnectionsTooltip`, `hideConnectionsTooltip`, `setKnowledgeMapScale`, `toggleKnowledgeMapLayer`, `persistKnowledgeMapView`, `syncKnowledgeMapToggleButtons`, `updateKnowledgeMapZoomLabel`, `buildOrbitTopicNodes`, `buildTopicBridges`, `buildSharedConceptLinksFromTopicBridges`, the body of `initConnections`/`renderConnections`) and replace with a Cytoscape implementation. **KEEP `openSummaryFromKnowledgeMap` (app.js:1570)** — the new `dbltap` handler depends on it; it must NOT be deleted.

**Readability rules (the point of this task):**
1. **No always-on edge labels.** Edge label visible only on `:selected` or hover.
2. **Concept nodes degree < 2 hidden by default** (toggle shows all).
3. **Hover a node → neighborhood highlight**: non-neighbors fade to opacity 0.12.
4. **Click a document → ego focus**: hide everything outside the closed neighborhood, re-run layout on the subgraph; background click restores.
5. **Similarity threshold slider** filters similarity edges live (default 0.5, so the initial view is sparse).
6. **Node size = 18 + centrality × 60; node color = community palette; concept nodes small/round-rect.**
7. **Search box**: dims non-matches, `Enter` zooms to first match.
8. **Double-click document node → open its summary** (reuse `openSummaryFromKnowledgeMap(title)`).
9. **Document scope filter (user request)**: a "Documents" panel in the side rail with one checkbox per document (+ all/none buttons). `cyElements()` keeps only checked documents, their concept/entity neighbors, and edges among them. Default: all checked. Persist checked set in `localStorage` per domain.
10. **Gap cards reframed for PDF-based knowledge (user request)**: copy is question-oriented — "이 두 영역을 잇는 질문을 해보세요" with a button that switches to the Chat tab and prefills the question input with a bridging question (`${label_a}와(과) ${label_b}는 어떤 관련이 있나요?`). No "write a note" copy. Panel sits below the document filter (secondary priority).

- [ ] **Step 1: Implement the renderer.** Core skeleton (full code goes in app.js; state object replaces `knowledgeMapState`):
```javascript
// ── Knowledge Map (Cytoscape) ──────────────────────────────────────────────
const COMMUNITY_PALETTE = ["#e15759", "#4e79a7", "#f28e2b", "#76b7b2", "#59a14f",
                           "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"];
const graphState = { cy: null, payload: null, threshold: 0.5, showConcepts: true, showSimilarity: true, egoRoot: null };

async function renderConnections() {
    const emptyEl = document.getElementById("connections-empty");
    let payload = { nodes: [], edges: [], communities: [], gaps: [] };
    try {
        const res = await apiFetch("/api/graph");
        payload = await res.json();
    } catch (err) { console.error("graph load failed", err); }
    graphState.payload = payload;

    const docCount = payload.nodes.filter(n => n.kind === "document").length;
    document.getElementById("cy").style.display = docCount ? "block" : "none";
    emptyEl.style.display = docCount ? "none" : "block";
    if (!docCount) return;

    renderCommunityPanel(payload);
    renderGapPanel(payload);
    buildCy(payload);
}

function cyElements(payload) {
    const nodes = payload.nodes
        .filter(n => graphState.showConcepts || n.kind === "document")
        .filter(n => n.kind !== "concept" || n.degree >= 2)
        .map(n => ({ data: { ...n } }));
    const ids = new Set(nodes.map(n => n.data.id));
    const edges = payload.edges
        .filter(e => ids.has(e.source) && ids.has(e.target))
        .filter(e => e.kind !== "similarity" || (graphState.showSimilarity && e.weight >= graphState.threshold))
        .map(e => ({ data: { ...e } }));
    return [...nodes, ...edges];
}

function buildCy(payload) {
    if (graphState.cy) graphState.cy.destroy();
    graphState.cy = cytoscape({
        container: document.getElementById("cy"),
        elements: cyElements(payload),
        style: [
            { selector: "node[kind='document']", style: {
                "background-color": ele => COMMUNITY_PALETTE[ele.data("community") % COMMUNITY_PALETTE.length],
                "width": ele => 18 + ele.data("centrality") * 60,
                "height": ele => 18 + ele.data("centrality") * 60,
                "label": "data(label)", "color": "#ddd", "font-size": 11,
                "text-valign": "bottom", "text-margin-y": 6, "text-max-width": 120, "text-wrap": "ellipsis",
            }},
            { selector: "node[kind='concept']", style: {
                "shape": "round-rectangle", "background-color": "#3d3d46", "width": "label", "height": 16,
                "padding": "4px", "label": "data(label)", "color": "#c9a227", "font-size": 9,
                "text-valign": "center", "text-max-width": 90, "text-wrap": "ellipsis",
            }},
            { selector: "edge", style: { "curve-style": "haystack", "line-color": "#3a3a44", "width": 1, "opacity": 0.7 }},
            { selector: "edge[kind='similarity']", style: { "line-color": "#7a7a88", "width": ele => 1 + (ele.data("weight") - 0.3) * 4 }},
            { selector: "edge[kind='concept']", style: { "line-style": "dashed", "line-color": "#6b5b1e" }},
            { selector: ".faded", style: { "opacity": 0.12, "text-opacity": 0.06 }},
            { selector: "edge:selected, node:selected", style: { "label": "data(label)", "font-size": 9, "color": "#eee" }},
        ],
        layout: { name: "cose", animate: false, nodeRepulsion: 40000, idealEdgeLength: 90, padding: 30 },
        wheelSensitivity: 0.2,
    });
    wireCyInteractions();
    renderLegend(payload);
}
```
Interactions (`wireCyInteractions`): `mouseover/mouseout node` → add/remove `.faded` on `cy.elements().difference(node.closedNeighborhood())`; `tap node[kind='document']` → ego focus (store removed elements via `cy.remove`, rerun layout, `tap` on background restores); `dbltap` → `openSummaryFromKnowledgeMap(node.data('label'))`. Toolbar listeners (`initConnections`): threshold slider / toggles re-call `buildCy(graphState.payload)`; search input → fade non-matching labels; `graph-fit` → `cy.fit()`. Side panels: `renderCommunityPanel` lists communities with color swatch + count (click → fade others); `renderGapPanel` renders `gap-card` per gap with suggestion text.

- [ ] **Step 2: Update `knowledgeMapState`/storage references** — remove `KNOWLEDGE_MAP_STORAGE_KEY` usage and the old state object; keep `refresh-connections-btn` handler calling `/api/connections/refresh` then `renderConnections()`. Check `frontend/app.js:16-33,72,152,172,323` for stale references.

- [ ] **Step 3: Manual verification** — start server, open Knowledge Map, confirm: sparse initial view, hover highlight, ego focus + restore, slider live-filter, search, gap cards. Use browser screenshot if available; otherwise verify `/api/graph` payload + JS console clean.

- [ ] **Step 4: Commit** — `feat(frontend): interactive cytoscape knowledge map with readability rules`

### Task 8: Community labels via LLM (cached)

**Files:**
- Modify: `scripts/graph.py`, `scripts/server.py` (`/api/connections/refresh`)
- Test: `tests/test_graph_labels.py`

- [ ] **Step 1: Write failing test** — `label_communities(kb, payload, llm_fn)` with a stub `llm_fn` returning `["Software Engineering", "RAG"]` populates `kb["graph_cache"]["community_labels"]` and is **not called again** when labels already cover all communities (assert stub call count).

- [ ] **Step 2: Implement** `label_communities(kb, payload, llm_fn)` in `graph.py`: for each community lacking a cached label, collect up to 5 member doc titles + concept labels, single `llm_fn(prompt)` per community returning a ≤4-word label; store under `kb["graph_cache"]["community_labels"][str(cid)]`; call `save_kb(kb)` from the caller. Production `llm_fn` wraps `get_openai_client()` chat call with `temperature=0.2, max_tokens=20`.

- [ ] **Step 3: Wire into `/api/connections/refresh`** in server.py after `refresh_all_connections`: build payload, call `label_communities`, save. `/api/graph` stays LLM-free.

- [ ] **Step 4: Tests PASS → Commit** — `feat(graph): cached LLM community labels on refresh`

**Stage 1 gate:** run full `pytest`, then code-verify subagent, fix to PASS, commit any fixes.

---

## Stage 2 — Typed KG + Temporal + Trails

### Task 9: SPO triple + typed entity extraction

**Files:**
- Modify: `scripts/rag.py` (add `extract_knowledge`, keep `extract_concepts` as thin wrapper for backward compat)
- Test: `tests/test_extract_knowledge.py`

- [ ] **Step 1: Write failing tests** for the parser `parse_knowledge_response(raw_text)` (pure function, no LLM): valid JSON → `{"entities": [...], "triples": [...]}` with categories validated against `{"is-a","part-of","uses","contrasts","causes","related"}` (invalid category → `"related"`); malformed JSON → falls back to `{"entities": [], "triples": []}`; entity types validated against `{"concept","person","tech","org","event","other"}` (invalid → `"concept"`).

- [ ] **Step 2: Implement parser + extractor** in `rag.py`:
```python
VALID_CATEGORIES = {"is-a", "part-of", "uses", "contrasts", "causes", "related"}
VALID_ENTITY_TYPES = {"concept", "person", "tech", "org", "event", "other"}


def parse_knowledge_response(raw_text: str) -> dict:
    data = _safe_json_value(raw_text) or {}
    entities, triples = [], []
    for e in data.get("entities", []) if isinstance(data, dict) else []:
        if isinstance(e, dict) and isinstance(e.get("name"), str) and e["name"].strip():
            etype = e.get("type") if e.get("type") in VALID_ENTITY_TYPES else "concept"
            entities.append({"name": e["name"].strip(), "type": etype})
    for t in data.get("triples", []) if isinstance(data, dict) else []:
        if not isinstance(t, dict):
            continue
        s, p, o = (t.get(k, "") for k in ("subject", "predicate", "object"))
        if all(isinstance(x, str) and x.strip() for x in (s, p, o)):
            cat = t.get("category") if t.get("category") in VALID_CATEGORIES else "related"
            triples.append({"subject": s.strip(), "predicate": p.strip(),
                            "object": o.strip(), "category": cat})
    return {"entities": entities[:15], "triples": triples[:20]}


def extract_knowledge(text: str, title: str = "", model_name: str | None = None) -> dict:
    excerpt = text[:3000]
    prompt = f"""Extract a small knowledge graph from this document.

Return ONLY JSON: {{"entities": [{{"name": "...", "type": "concept|person|tech|org|event|other"}}],
"triples": [{{"subject": "...", "predicate": "short verb phrase", "object": "...",
"category": "is-a|part-of|uses|contrasts|causes|related"}}]}}

Rules: 5-12 entities, 5-15 triples. Subjects/objects must be entity names. Keep names short and canonical.

Document title: {title}
Text:
{excerpt}"""
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model_name or RAG_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=900,
        response_format={"type": "json_object"},
    )
    return parse_knowledge_response(resp.choices[0].message.content or "{}")
```
`extract_concepts` becomes: call `extract_knowledge`, return entity names (keeps every existing caller working).

- [ ] **Step 3: Wire into ingestion + KB** — in `add_document_to_kb`, accept optional `knowledge: dict`; merge into `kb.setdefault("entities", {})` / `kb.setdefault("triples", [])` with `doc_id` and `created_at` stamps. In `server.py` upload/add-url/add-text handlers, call `extract_knowledge` where `extract_concepts` is called today and pass both. In `refresh_missing_concepts`, also backfill docs with no triples.

- [ ] **Step 4: Tests PASS → Commit** — `feat(rag): SPO triple and typed entity extraction`

### Task 10: Entity merge + typed edges in graph

**Files:**
- Modify: `scripts/graph.py` (`build_graph`)
- Test: `tests/test_graph_typed.py`

- [ ] **Step 1: Write failing tests** — KB with `entities`/`triples` produces `entity::` nodes (with `entity_type`), `triple` edges between subject/object entity nodes (with `category`, `label`=predicate, `created_at`), doc→entity mention edges; entity names canonicalized through the same `canonicalize_concepts` embedding pass (extend it to cover entity names: synonymous entities merge to one node).

- [ ] **Step 2: Implement** — extend `canonicalize_concepts` to also index `kb["entities"]` keys; in `build_graph`, after concepts: add entity nodes (`entity::<canonical>`), doc—entity edges (kind="concept", weight 1.0) from `entity["doc_ids"]`, and triple edges (kind="triple", category, label=predicate) between canonical entity nodes. Skip triples whose endpoints didn't survive validation.

- [ ] **Step 3: Frontend styles** — add to the Cytoscape stylesheet: entity node shapes by `entity_type` (diamond=person, hexagon=tech, ellipse=concept...), triple edge colors by `category` (`is-a` #4e79a7, `part-of` #59a14f, `uses` #76b7b2, `contrasts` #e15759, `causes` #f28e2b, `related` #6b6b75), legend entries, and a "relations" layer toggle button (same pattern as concepts toggle).

- [ ] **Step 4: Tests PASS → Commit** — `feat(graph): typed entity nodes and relation edges`

### Task 11: Temporal slider

**Files:**
- Modify: `frontend/index.html`, `frontend/app.js`, `frontend/styles.css`

- [ ] **Step 1:** All nodes/edges already carry `created_at` (Tasks 3/9/10 stamp it). Add to the toolbar:
```html
<label class="graph-slider-label">as of <span id="graph-date-value">now</span>
    <input type="range" id="graph-date" min="0" max="100" value="100">
</label>
```

- [ ] **Step 2:** In `app.js`: compute `[minDate, maxDate]` from payload `created_at` values; slider position → cutoff date; `cyElements()` additionally filters `created_at <= cutoff` (empty `created_at` always shown); label shows the cutoff date (`YYYY-MM-DD`); slider at max shows "now". Re-call `buildCy` on input (debounced 150ms).

- [ ] **Step 3: Manual check** (docs added on different dates appear/disappear as slider moves) **→ Commit** — `feat(frontend): temporal slider replays knowledge growth`

### Task 12: Session trails

**Files:**
- Modify: `scripts/graph.py` (`build_graph`)
- Test: `tests/test_graph_trails.py`

- [ ] **Step 1: Write failing test** — KB whose `qa_history` has an entry with sources covering docs d1+d3 yields edge (d1,d3) `kind="trail"` with `weight` = co-citation count; entries with a single source produce no edge.

- [ ] **Step 2: Implement** in `build_graph`: after similarity edges, iterate `kb.get("qa_history", [])`, collect unique `doc_id`s per entry from `sources`, for each pair increment a trail counter; add edges `kind="trail", weight=count, created_at=entry timestamp` only where no similarity/triple edge already exists.

- [ ] **Step 3: Frontend** — trail edge style (dotted, #4e79a7) + "trails" toggle.

- [ ] **Step 4: Tests PASS → Commit** — `feat(graph): session trail edges from Q&A co-citations`

**Stage 2 gate:** full pytest + code-verify subagent + live ingest test (upload 1 text note → entities/triples appear in `/api/graph`).

---

## Stage 3 — Graph-enhanced RAG

### Task 13: Personalized PageRank retrieval fusion

**Files:**
- Modify: `scripts/graph.py` (add `ppr_doc_scores`), `scripts/rag.py` (`answer_question`)
- Test: `tests/test_graph_ppr.py`

- [ ] **Step 1: Write failing tests** for `ppr_doc_scores(kb, query_embedding_fn, query, top_m=4)`:
```python
def test_ppr_boosts_docs_linked_via_shared_entities(sample_kb, fake_embed):
    canonicalize_concepts(sample_kb, embed_fn=fake_embed, threshold=0.9)
    scores = ppr_doc_scores(sample_kb, fake_embed, "agile")
    assert scores  # non-empty
    assert scores["d2"] > scores["d3"]  # d2 cites Agile; d3 only RAG


def test_empty_graph_returns_empty(fake_embed):
    kb = {"documents": {}, "qa_history": []}
    assert ppr_doc_scores(kb, fake_embed, "anything") == {}
```

- [ ] **Step 2: Implement** in `graph.py`:
```python
def ppr_doc_scores(kb: dict, embed_fn, query: str, top_m: int = 4) -> dict[str, float]:
    """Personalized PageRank from the concept/entity nodes nearest to the query."""
    G = build_graph(kb)
    label_nodes = [(n, d["label"]) for n, d in G.nodes(data=True) if d.get("kind") in ("concept", "entity")]
    if not label_nodes or len(G) == 0:
        return {}
    import numpy as np
    vecs = np.asarray(embed_fn([lbl for _, lbl in label_nodes]))
    qv = np.asarray(embed_fn([query]))[0]
    sims = vecs @ qv
    seeds = {label_nodes[i][0]: float(sims[i]) for i in np.argsort(-sims)[:top_m] if sims[i] > 0.5}
    if not seeds:
        return {}
    pr = nx.pagerank(G, alpha=0.85, personalization=seeds, weight="weight")
    doc_scores = {n: s for n, s in pr.items() if G.nodes[n].get("kind") == "document"}
    if not doc_scores:
        return {}
    mx = max(doc_scores.values())
    return {n: s / mx for n, s in doc_scores.items()}
```

- [ ] **Step 3: Fuse in `answer_question`** (rag.py): add keyword param `embed_fn=None` to `answer_question`; after `retrieval = retrieve(...)`, when both `kb` and `embed_fn` are provided compute `graph_scores = ppr_doc_scores(kb, embed_fn, query)`; re-score each chunk `r["score"] = 0.75 * r["score"] + 0.25 * graph_scores.get(r["doc_id"], 0.0)`, re-sort before slicing `LLM_CONTEXT_CHUNKS`. Wrap in try/except — graph failure must never break Q&A.

- [ ] **Step 3b: Concrete server plumbing** — `scripts/server.py:412-413` currently calls `answer_question(req.message, history, embed_model, kb)` where `embed_model` is a raw SentenceTransformer (not a `list[str] -> ndarray` callable). Define one shared wrapper near the top of server.py (reuse it in `/api/graph` from Task 5 to avoid drift):
```python
def _embed_labels(labels: list[str]):
    return embed_model.encode(labels, normalize_embeddings=True, show_progress_bar=False)
```
and change the chat call to `answer_question(req.message, history, embed_model, kb, embed_fn=_embed_labels)`.

- [ ] **Step 4: Tests PASS → live check** (`POST /api/chat` answer still sane, sources re-ordered when graph agrees) **→ Commit** — `feat(rag): fuse personalized pagerank into retrieval scoring`

### Task 14: Lazy global answers (community summaries)

**Files:**
- Modify: `scripts/rag.py`
- Test: `tests/test_global_question.py`

- [ ] **Step 1: Write failing tests** for `is_global_question(q)`: True for "내 문서 전체 주제는 뭐야?", "what are the main themes overall?", "모든 문서를 요약해줘"; False for "What is agile?", "RAG가 뭐야?".

- [ ] **Step 2: Implement** `is_global_question` (regex on 전체/전반/모든 문서/주제들/overall/themes/all my documents/across documents + question shorter than 200 chars) and `answer_global_question(query, kb, payload)`: group docs by community (from `build_graph_payload`), build a context of `community label + member doc titles + summaries` (truncate to ~6000 chars), single LLM call answering the query from that map. In `answer_question`, route to it when `is_global_question(query)` and `kb` has ≥3 docs; responses keep the same `{answer, sources, confidence, related_docs, connections}` shape (`sources` = one entry per community's top doc).

- [ ] **Step 3: Tests PASS → Commit** — `feat(rag): lazy community-summary path for global questions`

### Task 15: Docs + final verification

- [ ] **Step 1:** Update `README.md`: Knowledge Map feature description (communities, gaps, typed relations, temporal slider, trails), `/api/graph` row in the endpoint table, pipeline line mentions graph fusion.
- [ ] **Step 2:** Full `pytest` run; full manual pass (ingest → map → chat → global question).
- [ ] **Step 3:** Dispatch code-verify subagent over the whole diff; fix to PASS.
- [ ] **Step 4:** Final commit — `docs: document knowledge graph features`

---

## Self-Review Notes
- Spec coverage: 정규화(T2), NetworkX+커뮤니티/중심성/갭(T3-4), /api/graph(T5), Cytoscape+가독성 인터랙션(T6-7, the user's screenshot complaint), 커뮤니티 라벨(T8), SPO+타입드(T9-10), temporal(T11), 트레일(T12), PPR 융합(T13), 글로벌 질문(T14), 문서화(T15). ✓
- Type consistency: `canonicalize_concepts(kb, embed_fn, threshold)` / `build_graph(kb)` / `analyze_graph(G)` / `find_structural_gaps(G, max_density, min_size)` / `build_graph_payload(kb)` / `ppr_doc_scores(kb, embed_fn, query, top_m)` used identically across tasks. ✓
- Legacy removal: old canvas code paths enumerated in Task 7 Step 2 with line anchors. ✓
