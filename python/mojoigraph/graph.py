"""Covered `igraph.Graph` algorithms backed by Mojo CSR kernels."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ._lib import addr, lib

_BRANDES_PARALLEL_THRESHOLD = 256
_BRANDES_WORKERS = 4
_BRANDES_POOL = ThreadPoolExecutor(max_workers=_BRANDES_WORKERS)


def _vertices(vertices, n: int) -> list[int]:
    if vertices is None:
        return list(range(n))
    raw = [vertices] if isinstance(vertices, (int, np.integer)) else vertices
    result = [operator.index(v) for v in raw]
    if any(v < 0 or v >= n for v in result):
        raise ValueError("invalid vertex ID")
    return result


class VertexClustering:
    """Small compatible result object for `Graph.components()`."""

    def __init__(self, graph: Graph, membership: Iterable[int], n: int):
        self.graph = graph
        self.membership = membership.tolist() if isinstance(membership, np.ndarray) else list(membership)
        self._n = n
        self._groups: list[list[int]] | None = None

    def _materialize_groups(self) -> list[list[int]]:
        if self._groups is None:
            groups = [[] for _ in range(self._n)]
            for vertex, label in enumerate(self.membership):
                groups[label].append(vertex)
            self._groups = groups
        return self._groups

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        return iter(self._materialize_groups())

    def __getitem__(self, index):
        return self._materialize_groups()[index]

    def sizes(self) -> list[int]:
        return [len(group) for group in self._materialize_groups()]

    def size(self, index: int) -> int:
        return len(self._materialize_groups()[index])


class Graph:
    """An `igraph.Graph`-shaped graph for the covered algorithm subset."""

    def __init__(self, n: int = 0, edges=None, directed: bool = False):
        directed = bool(directed)
        n = operator.index(n)
        if n < 0:
            raise ValueError("vertex count must be non-negative")
        edges = [] if edges is None else [(operator.index(a), operator.index(b)) for a, b in edges]
        if not directed:
            edges = [(min(a, b), max(a, b)) for a, b in edges]
        inferred = max((max(a, b) for a, b in edges), default=-1) + 1
        self._n = max(n, inferred)
        if any(a < 0 or b < 0 or a >= self._n or b >= self._n for a, b in edges):
            raise ValueError("edge endpoint is outside the vertex range")
        self._edges = edges
        self._directed = directed
        self._csr_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def Full(cls, n: int, directed: bool = False, loops: bool = False):
        edges = ((i, j) for i in range(n) for j in range(n)
                 if (directed and (loops or i != j)) or
                 (not directed and (i < j or (loops and i == j))))
        return cls(n, edges, directed=directed)

    @classmethod
    def Ring(cls, n: int, directed: bool = False, mutual: bool = False, circular: bool = True):
        edges = [(v, v + 1) for v in range(max(0, n - 1))]
        if circular and n > 1:
            edges.append((n - 1, 0))
        if mutual and directed:
            edges += [(b, a) for a, b in edges]
        return cls(n, edges, directed=directed)

    def vcount(self) -> int:
        return self._n

    def ecount(self) -> int:
        return len(self._edges)

    def is_directed(self) -> bool:
        return self._directed

    def get_edgelist(self) -> list[tuple[int, int]]:
        return list(self._edges)

    def add_vertices(self, n: int):
        n = operator.index(n)
        if n < 0:
            raise ValueError("vertex count must be non-negative")
        self._n += n
        self._csr_cache.clear()

    def add_edges(self, edges):
        new = [(operator.index(a), operator.index(b)) for a, b in edges]
        if not self._directed:
            new = [(min(a, b), max(a, b)) for a, b in new]
        if any(a < 0 or b < 0 or a >= self._n or b >= self._n for a, b in new):
            raise ValueError("edge endpoint is outside the vertex range")
        self._edges.extend(new)
        self._csr_cache.clear()

    def _csr(self, mode: str = "out") -> tuple[np.ndarray, np.ndarray]:
        mode = str(mode).lower()
        if mode not in {"out", "in", "all"}:
            raise ValueError("mode must be 'out', 'in', or 'all'")
        if not self._directed:
            mode = "all"
        if mode in self._csr_cache:
            return self._csr_cache[mode]
        rows = [[] for _ in range(self._n)]
        for a, b in self._edges:
            if not self._directed:
                rows[a].append(b)
                if a != b:
                    rows[b].append(a)
            elif mode == "out":
                rows[a].append(b)
            elif mode == "in":
                rows[b].append(a)
            else:
                rows[a].append(b)
                if a != b:
                    rows[b].append(a)
        offsets = np.empty(self._n + 1, dtype=np.int64)
        offsets[0] = 0
        for v, row in enumerate(rows):
            offsets[v + 1] = offsets[v] + len(row)
        neighbors = np.fromiter((w for row in rows for w in row), dtype=np.int64,
                                count=int(offsets[-1]))
        self._csr_cache[mode] = (offsets, neighbors)
        return offsets, neighbors

    def _bfs(self, source: int, mode: str):
        source = int(source)
        if source < 0 or source >= self._n:
            raise ValueError("invalid vertex ID")
        offsets, neighbors = self._csr(mode)
        depth = np.empty(self._n, dtype=np.int64)
        parent = np.empty(self._n, dtype=np.int64)
        queue = np.empty(self._n, dtype=np.int64)
        reached = lib().mig_bfs(addr(offsets), addr(neighbors), self._n, source,
                                addr(depth), addr(parent), addr(queue))
        return depth, parent, queue[:reached].copy()

    def distances(self, source=None, target=None, weights=None, mode: str = "out", algorithm: str = "auto"):
        self._check_unweighted(weights)
        sources = _vertices(source, self._n)
        targets = _vertices(target, self._n)
        matrix = np.empty((len(sources), len(targets)), dtype=np.float64)
        for row, vertex in enumerate(sources):
            depth, _, _ = self._bfs(vertex, mode)
            values = depth[targets].astype(np.float64)
            values[values < 0] = math.inf
            matrix[row] = values
        return matrix.tolist()

    def shortest_paths(self, *args, **kwargs):
        return self.distances(*args, **kwargs)

    def bfs(self, vid, mode: str = "out"):
        depth, parent, order = self._bfs(vid, mode)
        rank = [0]
        for position in range(1, len(order)):
            if depth[order[position]] != depth[order[position - 1]]:
                rank.append(position)
        rank.append(len(order))
        return order.tolist(), rank, parent.tolist()

    def get_shortest_paths(self, v, to=None, weights=None, mode: str = "out", output: str = "vpath", algorithm: str = "auto"):
        self._check_unweighted(weights)
        if output != "vpath":
            raise NotImplementedError("only output='vpath' is covered")
        targets = _vertices(to, self._n)
        _, parent, _ = self._bfs(int(v), mode)
        paths = []
        for target in targets:
            if parent[target] == -2:
                paths.append([])
                continue
            path = [target]
            while parent[path[-1]] != -1:
                path.append(int(parent[path[-1]]))
            paths.append(path[::-1])
        return paths

    def components(self, mode: str = "strong") -> VertexClustering:
        mode = str(mode).lower()
        membership = np.empty(self._n, dtype=np.int64)
        if self._n == 0:
            return VertexClustering(self, [], 0)
        if mode == "weak" or not self._directed:
            offsets, neighbors = self._csr("all")
            queue = np.empty(self._n, dtype=np.int64)
            count = lib().mig_components(addr(offsets), addr(neighbors), self._n,
                                         addr(membership), addr(queue))
        elif mode == "strong":
            offsets, neighbors = self._csr("out")
            reverse_offsets, reverse_neighbors = self._csr("in")
            stack = np.empty(self._n, dtype=np.int64)
            cursor = np.empty(self._n, dtype=np.int64)
            order = np.empty(self._n, dtype=np.int64)
            count = lib().mig_strong_components(addr(offsets), addr(neighbors), addr(reverse_offsets),
                                                addr(reverse_neighbors), self._n, addr(membership),
                                                addr(stack), addr(cursor), addr(order))
        else:
            raise ValueError("mode must be 'weak' or 'strong'")
        return VertexClustering(self, membership, count)

    connected_components = components

    def is_connected(self, mode: str = "strong") -> bool:
        return len(self.components(mode=mode)) <= 1

    def pagerank(self, vertices=None, directed: bool = True, damping: float = 0.85, weights=None,
                 arpack_options=None, implementation: str = "prpack"):
        self._check_unweighted(weights)
        if self._n == 0:
            return []
        mode = "out" if directed else "all"
        offsets, neighbors = self._csr(mode)
        in_offsets, in_neighbors = self._csr("in" if directed else "all")
        degree = np.diff(offsets).astype(np.int64, copy=False)
        score = np.empty(self._n, dtype=np.float64)
        next_score = np.empty(self._n, dtype=np.float64)
        lib().mig_pagerank(addr(in_offsets), addr(in_neighbors), addr(degree), self._n,
                           float(damping), 10_000, 1e-13, addr(score), addr(next_score))
        return score[_vertices(vertices, self._n)].tolist()

    def betweenness(self, vertices=None, directed: bool = True, cutoff=None, weights=None,
                    sources=None, targets=None):
        self._check_unweighted(weights)
        if cutoff is not None or sources is not None or targets is not None:
            raise NotImplementedError("cutoff, sources, and targets are not covered")
        if self._n == 0:
            return []
        mode = "out" if directed else "all"
        offsets, neighbors = self._csr(mode)
        workers = _BRANDES_WORKERS if self._n >= _BRANDES_PARALLEL_THRESHOLD else 1
        shape = (workers, self._n)
        result = np.empty(shape, dtype=np.float64)
        queue = np.empty(shape, dtype=np.int64)
        stack = np.empty(shape, dtype=np.int64)
        depth = np.empty(shape, dtype=np.int64)
        sigma = np.empty(shape, dtype=np.float64)
        delta = np.empty(shape, dtype=np.float64)
        reverse_offsets, reverse_neighbors = self._csr("in" if self._directed and directed else "all")
        kernel = lib().mig_betweenness

        def run_worker(worker: int):
            kernel(addr(offsets), addr(neighbors), addr(reverse_offsets), addr(reverse_neighbors), self._n,
                   worker, workers, addr(result[worker]), addr(queue[worker]), addr(stack[worker]),
                   addr(depth[worker]), addr(sigma[worker]), addr(delta[worker]))

        if workers == 1:
            run_worker(0)
        else:
            for future in [_BRANDES_POOL.submit(run_worker, worker) for worker in range(workers)]:
                future.result()
        lib().mig_reduce_betweenness(addr(result), self._n, workers,
                                     int(not (self._directed and directed)))
        return result[0, _vertices(vertices, self._n)].tolist()

    def closeness(self, vertices=None, mode: str = "all", cutoff=None, weights=None, normalized: bool = True):
        self._check_unweighted(weights)
        if cutoff is not None:
            raise NotImplementedError("cutoff is not covered")
        result = []
        for vertex in _vertices(vertices, self._n):
            depth, _, _ = self._bfs(vertex, mode)
            reachable = depth[depth > 0]
            if not len(reachable):
                result.append(math.nan)
            else:
                result.append((len(reachable) if normalized else 1.0) / float(reachable.sum()))
        return result

    def degree(self, vertices=None, mode: str = "all", loops: bool = True):
        selected = _vertices(vertices, self._n)
        mode = str(mode).lower()
        if not self._directed:
            offsets, _ = self._csr("all")
            values = np.diff(offsets)
        elif mode == "out":
            values = np.diff(self._csr("out")[0])
        elif mode == "in":
            values = np.diff(self._csr("in")[0])
        else:
            values = np.diff(self._csr("out")[0]) + np.diff(self._csr("in")[0])
        for a, b in self._edges:
            if a != b:
                continue
            if not self._directed:
                values[a] += 1 if loops else -1
            elif not loops:
                values[a] -= 2 if mode == "all" else 1
        answer = values[selected].astype(int).tolist()
        return answer[0] if isinstance(vertices, (int, np.integer)) else answer

    def density(self, loops: bool = False, weights=None) -> float:
        self._check_unweighted(weights)
        possible = self._n * (self._n - 1) if self._directed else self._n * (self._n - 1) // 2
        if loops:
            possible += self._n
        return 0.0 if possible == 0 else len(self._edges) / possible

    @staticmethod
    def _check_unweighted(weights):
        if weights is not None:
            raise NotImplementedError("weighted algorithms are not covered")
