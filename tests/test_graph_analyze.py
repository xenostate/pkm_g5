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
