# mojo-igraph

A focused, standalone Mojo port of the compute-heavy graph-algorithms core of
[python-igraph](https://python.igraph.org/). It provides a Python `Graph` API
with the same method names and signatures for the covered subset, while moving
the traversal and centrality loops into a single compiled Mojo shared library.

```python
import mojoigraph as ig

graph = ig.Graph(5, [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)], directed=True)
print(graph.distances(source=0))
# [[0.0, 1.0, 1.0, 2.0, 3.0]]
print(graph.betweenness())
# [0.0, 1.0, 1.0, 3.0, 0.0]
```

## Covered subset

| `igraph.Graph` API | coverage |
| --- | --- |
| construction and inspection | `Graph(n, edges, directed)`, `Full`, `Ring`, `vcount`, `ecount`, `get_edgelist`, `add_vertices`, `add_edges` |
| paths and traversal | `distances`, deprecated-compatible `shortest_paths`, `bfs`, `get_shortest_paths(output="vpath")` |
| connectivity | `components` / `connected_components`, `is_connected`, including weak and strong directed components |
| centrality | unweighted `pagerank`, unweighted Brandes `betweenness`, unweighted `closeness` |
| basic measures | `degree`, `density` |

`components()` returns a lightweight `VertexClustering` compatible result with
`membership`, iteration, indexing, `size`, and `sizes`.

This is deliberately a graph-algorithms subset, not a replacement for all of
igraph. Vertex/edge attributes, graph I/O and drawing, community detection,
isomorphism, flow and matching algorithms, weighted paths/centrality, and
`get_shortest_paths(output="epath")` are not yet covered. Requests for a
weighted covered algorithm fail explicitly rather than silently producing an
unweighted result. Brandes currently supports its full-graph form only (not
`cutoff`, `sources`, or `targets`).

## Install and run

`python-igraph` is a test/benchmark dependency; the public implementation is
the `mojoigraph` package to avoid shadowing an installed upstream `igraph`.

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

The example at the top can be run unchanged as:

```bash
pixi run python -c 'import mojoigraph as ig; print(ig.Graph(3, [(0, 1), (1, 2)]).distances(source=0))'
```

## Correctness

The test suite uses the real `python-igraph` package from conda-forge as its
oracle. It asserts numerical or behavioural parity for directed and undirected
graphs, disconnected vertices, all distance modes, BFS order/level boundaries,
strong and weak component partitions, dangling PageRank, Brandes betweenness,
closeness, degree, and density.

## Performance

Measured by `pixi run bench` on Linux 6.8.0-136-generic, x86_64, glibc 2.39.
Times are the best of three runs and include the Python result conversion but
exclude graph construction and library load.

| case | mojo-igraph | python-igraph | result |
| --- | ---: | ---: | --- |
| BFS distances, 50k vertices / 400k edges | 7.51 ms | 15.44 ms | 2.06x faster |
| PageRank, 50k vertices / 400k edges | 54.45 ms | 506.60 ms | 9.30x faster |
| weak components, 100k vertices / 300k edges | 6.11 ms | 24.93 ms | 4.08x faster |
| Brandes betweenness, 400 vertices / 3.2k edges | 4.76 ms | 17.08 ms | 3.59x faster |

These are machine-specific measurements, not promises. Brandes partitions
independent sources across four CPU workers at 256 vertices and above; smaller
graphs remain serial to avoid thread-dispatch overhead. BFS, components, and
PageRank are serial. Different native builds or workloads may move the
comparison either way.

There is no GPU path: BFS, connected components, and Brandes are irregular
CSR traversals with low arithmetic intensity, where host-device transfers and
divergence lose to the CPU. PageRank is already well beyond the benchmark
target on CPU, so adding a GPU implementation would not improve this library's
measured workload.

## How it works

`Graph` stores an edge list for the Python-facing API and materializes a
contiguous CSR adjacency representation for each required direction. NumPy owns
the offsets, neighbour lists, and algorithm scratch buffers. ctypes passes their
addresses as 64-bit integers to `src/capi.mojo`; the Mojo C ABI rebuilds typed
`UnsafePointer`s and allocates nothing. SIMD fills initialize traversal scratch,
and SIMD loads/stores reduce Brandes worker results with scalar remainder loops.
Component groups are materialized only when iteration or indexing needs them.
The one compilation unit exports BFS, component labelling/Kosaraju SCC,
PageRank, and Brandes kernels, so serial calls cross the language boundary once
per operation rather than once per vertex or edge.

## License

MIT
