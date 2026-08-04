"""Behavioural parity against the published `python-igraph` implementation."""

import math

import igraph as ig
import numpy as np
import pytest

import mojoigraph as mig


def same_partition(left, right):
    return {frozenset(group) for group in left} == {frozenset(group) for group in right}


@pytest.fixture
def directed_graphs():
    rng = np.random.default_rng(7)
    result = []
    for n, p in [(1, 0.0), (7, 0.22), (18, 0.13)]:
        edges = [(i, j) for i in range(n) for j in range(n) if i != j and rng.random() < p]
        result.append((n, edges))
    return result


def test_constructor_and_mutation_match_igraph():
    graph = mig.Graph(2, [(0, 1)], directed=True)
    graph.add_vertices(2)
    graph.add_edges([(1, 2), (2, 3)])
    assert graph.vcount() == 4
    assert graph.ecount() == 3
    assert graph.is_directed()
    assert graph.get_edgelist() == [(0, 1), (1, 2), (2, 3)]
    assert mig.Graph.Ring(4).get_edgelist() == [(0, 1), (1, 2), (2, 3), (0, 3)]
    assert mig.Graph.Full(3).ecount() == 3
    assert mig.Graph.Full(3, loops=True).ecount() == 6


def test_distances_and_shortest_paths_parity(directed_graphs):
    for n, edges in directed_graphs:
        theirs = ig.Graph(n, edges, directed=True)
        ours = mig.Graph(n, edges, directed=True)
        for mode in ("out", "in", "all"):
            assert np.allclose(theirs.distances(mode=mode), ours.distances(mode=mode), equal_nan=True)
        assert ours.distances(source=0, target=[0]) == theirs.distances(source=0, target=[0])
        assert ours.shortest_paths(source=0) == theirs.distances(source=0)


def test_bfs_and_vertex_paths_parity():
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
    theirs = ig.Graph(6, edges, directed=True)
    ours = mig.Graph(6, edges, directed=True)
    assert ours.bfs(0) == theirs.bfs(0)
    with pytest.warns(RuntimeWarning):
        expected = theirs.get_shortest_paths(0, to=[3, 4, 5])
    assert ours.get_shortest_paths(0, to=[3, 4, 5]) == expected


def test_components_and_connectivity_parity(directed_graphs):
    for n, edges in directed_graphs:
        theirs = ig.Graph(n, edges, directed=True)
        ours = mig.Graph(n, edges, directed=True)
        for mode in ("weak", "strong"):
            assert same_partition(ours.components(mode), theirs.components(mode))
            assert ours.is_connected(mode) == theirs.is_connected(mode)
    undirected = mig.Graph(5, [(0, 1), (2, 3)])
    assert undirected.components().sizes() == [2, 2, 1]


def test_pagerank_parity_with_dangling_vertices(directed_graphs):
    for n, edges in directed_graphs:
        theirs = ig.Graph(n, edges, directed=True)
        ours = mig.Graph(n, edges, directed=True)
        assert np.allclose(ours.pagerank(), theirs.pagerank(), rtol=1e-11, atol=1e-12)
        assert np.allclose(ours.pagerank(vertices=[0]), theirs.pagerank(vertices=[0]), rtol=1e-11)


def test_brandes_betweenness_matches_directed_and_undirected():
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (1, 4)]
    for directed in (False, True):
        theirs = ig.Graph(6, edges, directed=directed)
        ours = mig.Graph(6, edges, directed=directed)
        assert np.allclose(ours.betweenness(), theirs.betweenness(), atol=1e-12)
        assert np.allclose(ours.betweenness(vertices=[1, 3]), theirs.betweenness(vertices=[1, 3]))


def test_brandes_simd_tail_matches_igraph():
    edges = [(v, (v + 1) % 37) for v in range(37)] + [(v, (v + 5) % 37) for v in range(0, 37, 3)]
    theirs = ig.Graph(37, edges, directed=True)
    ours = mig.Graph(37, edges, directed=True)
    assert np.allclose(ours.betweenness(), theirs.betweenness(), atol=1e-12)


def test_brandes_multiedges_do_not_overrun_predecessor_storage():
    # Parallel edges are legal in igraph.  They exercise a predecessor count
    # larger than the vertex count, which must remain safe across the C ABI.
    edges = [(0, 1)] * 20 + [(1, 2)] * 20
    theirs = ig.Graph(3, edges, directed=True)
    ours = mig.Graph(3, edges, directed=True)
    assert np.allclose(ours.betweenness(), theirs.betweenness(), atol=1e-12)


def test_invalid_vertex_ids_are_rejected_instead_of_using_numpy_negative_indexes():
    graph = mig.Graph.Ring(4)
    with pytest.raises(ValueError):
        graph.distances(target=-1)
    with pytest.raises(ValueError):
        graph.degree(vertices=-1)


def test_vertex_counts_and_ids_do_not_silently_narrow_floats():
    with pytest.raises(TypeError):
        mig.Graph(1.5)
    with pytest.raises(TypeError):
        mig.Graph(2, [(0.5, 1)])
    with pytest.raises(TypeError):
        mig.Graph(2).add_vertices(1.5)
    with pytest.raises(ValueError):
        mig.Graph(-1, [(0, 0)])
    with pytest.raises(TypeError):
        mig.Graph(2, unsupported_option=True)


def test_empty_graph_centrality_avoids_zero_length_ffi_buffers():
    assert mig.Graph().betweenness() == []


def test_closeness_parity_and_nan_for_isolates():
    edges = [(0, 1), (1, 2), (2, 3)]
    theirs = ig.Graph(5, edges, directed=True)
    ours = mig.Graph(5, edges, directed=True)
    for mode in ("out", "in", "all"):
        assert np.allclose(ours.closeness(mode=mode), theirs.closeness(mode=mode), equal_nan=True)
    assert math.isnan(ours.closeness(vertices=4)[0])


def test_degree_and_density_parity():
    edges = [(0, 1), (1, 1), (2, 1)]
    theirs = ig.Graph(3, edges, directed=True)
    ours = mig.Graph(3, edges, directed=True)
    for mode in ("in", "out", "all"):
        assert ours.degree(mode=mode) == theirs.degree(mode=mode)
        assert ours.degree(1, mode=mode) == theirs.degree(1, mode=mode)
    assert ours.degree(mode="all", loops=False) == theirs.degree(mode="all", loops=False)
    assert ours.density() == pytest.approx(theirs.density())
    undirected_theirs = ig.Graph(3, [(0, 0), (0, 1)])
    undirected_ours = mig.Graph(3, [(0, 0), (0, 1)])
    assert undirected_ours.degree() == undirected_theirs.degree()
    assert undirected_ours.degree(loops=False) == undirected_theirs.degree(loops=False)


def test_weighted_requests_are_explicitly_not_covered():
    graph = mig.Graph.Ring(5)
    with pytest.raises(NotImplementedError):
        graph.distances(weights=[1] * 5)
    with pytest.raises(NotImplementedError):
        graph.pagerank(weights=[1] * 5)
    with pytest.raises(NotImplementedError):
        graph.betweenness(cutoff=2)
