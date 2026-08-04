"""Run through `pixi run bench`; the task serializes machine-wide benchmarks."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import igraph as ig
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import mojoigraph as mig  # noqa: E402


def graph(n: int, edges_per_vertex: int, seed: int):
    rng = np.random.default_rng(seed)
    edges = [(int(rng.integers(n)), int(rng.integers(n))) for _ in range(n * edges_per_vertex)]
    return ig.Graph(n, edges, directed=True), mig.Graph(n, edges, directed=True)


def best(fn, repeat: int = 3) -> float:
    result = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        result = min(result, time.perf_counter() - start)
    return result


def run_case(name, make):
    theirs, ours = make()
    ours()  # build/load and page faults are not an algorithm measurement
    ours_time, theirs_time = best(ours), best(theirs)
    ratio = theirs_time / ours_time
    verdict = "faster" if ratio > 1 else "slower"
    print(f"| {name} | {ours_time * 1e3:.2f} ms | {theirs_time * 1e3:.2f} ms | {ratio:.2f}x {verdict} |")


def main():
    print(f"Machine: {platform.platform()} ({platform.processor() or 'unknown CPU'})")
    print("| case | mojo-igraph | python-igraph | result |")
    print("| --- | ---: | ---: | --- |")
    run_case("BFS distances, 50k vertices / 400k edges", lambda: (
        (lambda pair=graph(50_000, 8, 1): pair[0].distances(source=0)),
        (lambda pair=graph(50_000, 8, 1): pair[1].distances(source=0)),
    ))
    run_case("PageRank, 50k vertices / 400k edges", lambda: (
        (lambda pair=graph(50_000, 8, 2): pair[0].pagerank()),
        (lambda pair=graph(50_000, 8, 2): pair[1].pagerank()),
    ))
    run_case("weak components, 100k vertices / 300k edges", lambda: (
        (lambda pair=graph(100_000, 3, 3): pair[0].components(mode="weak")),
        (lambda pair=graph(100_000, 3, 3): pair[1].components(mode="weak")),
    ))
    run_case("Brandes betweenness, 400 vertices / 3.2k edges", lambda: (
        (lambda pair=graph(400, 8, 4): pair[0].betweenness()),
        (lambda pair=graph(400, 8, 4): pair[1].betweenness()),
    ))


if __name__ == "__main__":
    main()
