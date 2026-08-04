"""CSR graph kernels exported through a deliberately small C ABI."""

from std.sys import simd_width_of

comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def zero_f64(values: FPtr, n: Int):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    var zeros = SIMD[DType.float64, W](0.0)
    while i + W <= n:
        values.store(i, zeros)
        i += W
    while i < n:
        values[i] = 0.0
        i += 1


def fill_i64(values: IPtr, n: Int, value: Int64):
    comptime W = simd_width_of[DType.int64]()
    var i = 0
    var fill = SIMD[DType.int64, W](value)
    while i + W <= n:
        values.store(i, fill)
        i += W
    while i < n:
        values[i] = value
        i += 1


def bfs_kernel(offsets: IPtr, neighbors: IPtr, n: Int, source: Int, depth: IPtr, parent: IPtr, queue: IPtr) -> Int:
    for v in range(n):
        depth[v] = -1
        parent[v] = -2
    depth[source] = 0
    parent[source] = -1
    queue[0] = Int64(source)
    var head = 0
    var tail = 1
    while head < tail:
        var v = Int(queue[head])
        head += 1
        for edge in range(Int(offsets[v]), Int(offsets[v + 1])):
            var w = Int(neighbors[edge])
            if depth[w] < 0:
                depth[w] = depth[v] + 1
                parent[w] = Int64(v)
                queue[tail] = Int64(w)
                tail += 1
    return tail


def components_kernel(offsets: IPtr, neighbors: IPtr, n: Int, membership: IPtr, queue: IPtr) -> Int:
    for v in range(n):
        membership[v] = -1
    var count = 0
    for seed in range(n):
        if membership[seed] >= 0:
            continue
        membership[seed] = Int64(count)
        queue[0] = Int64(seed)
        var head = 0
        var tail = 1
        while head < tail:
            var v = Int(queue[head])
            head += 1
            for edge in range(Int(offsets[v]), Int(offsets[v + 1])):
                var w = Int(neighbors[edge])
                if membership[w] < 0:
                    membership[w] = Int64(count)
                    queue[tail] = Int64(w)
                    tail += 1
        count += 1
    return count


def strong_components_kernel(offsets: IPtr, neighbors: IPtr, reverse_offsets: IPtr, reverse_neighbors: IPtr, n: Int, membership: IPtr, stack: IPtr, cursor: IPtr, order: IPtr) -> Int:
    for v in range(n):
        membership[v] = 0
    var finished = 0
    for seed in range(n):
        if membership[seed] != 0:
            continue
        membership[seed] = 1
        var top = 0
        stack[0] = Int64(seed)
        cursor[0] = offsets[seed]
        while top >= 0:
            var v = Int(stack[top])
            var edge = Int(cursor[top])
            if edge < Int(offsets[v + 1]):
                cursor[top] = Int64(edge + 1)
                var w = Int(neighbors[edge])
                if membership[w] == 0:
                    membership[w] = 1
                    top += 1
                    stack[top] = Int64(w)
                    cursor[top] = offsets[w]
            else:
                order[finished] = Int64(v)
                finished += 1
                top -= 1
    for v in range(n):
        membership[v] = -1
    var count = 0
    var pos = n - 1
    while pos >= 0:
        var seed = Int(order[pos])
        pos -= 1
        if membership[seed] >= 0:
            continue
        membership[seed] = Int64(count)
        stack[0] = Int64(seed)
        var head = 0
        var tail = 1
        while head < tail:
            var v = Int(stack[head])
            head += 1
            for edge in range(Int(reverse_offsets[v]), Int(reverse_offsets[v + 1])):
                var w = Int(reverse_neighbors[edge])
                if membership[w] < 0:
                    membership[w] = Int64(count)
                    stack[tail] = Int64(w)
                    tail += 1
        count += 1
    return count


def pagerank_kernel(in_offsets: IPtr, in_neighbors: IPtr, out_degree: IPtr, n: Int, damping: Float64, iterations: Int, tolerance: Float64, score: FPtr, next_score: FPtr) -> Int:
    var initial = 1.0 / Float64(n)
    for v in range(n):
        score[v] = initial
    for step in range(iterations):
        var dangling = 0.0
        for v in range(n):
            if out_degree[v] == 0:
                dangling += score[v]
        var change = 0.0
        for v in range(n):
            var incoming = 0.0
            for edge in range(Int(in_offsets[v]), Int(in_offsets[v + 1])):
                var u = Int(in_neighbors[edge])
                incoming += score[u] / Float64(out_degree[u])
            var value = (1.0 - damping) / Float64(n) + damping * (incoming + dangling / Float64(n))
            next_score[v] = value
            var diff = value - score[v]
            change += diff if diff >= 0.0 else -diff
        for v in range(n):
            score[v] = next_score[v]
        if change <= tolerance:
            return step + 1
    return iterations


def betweenness_kernel(offsets: IPtr, neighbors: IPtr, reverse_offsets: IPtr, reverse_neighbors: IPtr, n: Int, undirected: Int, result: FPtr, queue: IPtr, stack: IPtr, depth: IPtr, sigma: FPtr, delta: FPtr):
    zero_f64(result, n)
    for source in range(n):
        fill_i64(depth, n, -1)
        zero_f64(sigma, n)
        zero_f64(delta, n)
        depth[source] = 0
        sigma[source] = 1.0
        queue[0] = Int64(source)
        var head = 0
        var tail = 1
        var stack_size = 0
        while head < tail:
            var v = Int(queue[head])
            head += 1
            stack[stack_size] = Int64(v)
            stack_size += 1
            for edge in range(Int(offsets[v]), Int(offsets[v + 1])):
                var w = Int(neighbors[edge])
                if depth[w] < 0:
                    depth[w] = depth[v] + 1
                    queue[tail] = Int64(w)
                    tail += 1
                if depth[w] == depth[v] + 1:
                    sigma[w] += sigma[v]
        var pos = stack_size - 1
        while pos >= 0:
            var v = Int(stack[pos])
            pos -= 1
            for edge in range(Int(reverse_offsets[v]), Int(reverse_offsets[v + 1])):
                var w = Int(reverse_neighbors[edge])
                if depth[w] == depth[v] - 1:
                    delta[w] += (sigma[w] / sigma[v]) * (1.0 + delta[v])
            if v != source:
                result[v] += delta[v]
    if undirected != 0:
        for v in range(n):
            result[v] *= 0.5


@export("mig_bfs")
def mig_bfs(offsets: Int, neighbors: Int, n: Int, source: Int, depth: Int, parent: Int, queue: Int) abi("C") -> Int:
    return bfs_kernel(ip(offsets), ip(neighbors), n, source, ip(depth), ip(parent), ip(queue))


@export("mig_components")
def mig_components(offsets: Int, neighbors: Int, n: Int, membership: Int, queue: Int) abi("C") -> Int:
    return components_kernel(ip(offsets), ip(neighbors), n, ip(membership), ip(queue))


@export("mig_strong_components")
def mig_strong_components(offsets: Int, neighbors: Int, reverse_offsets: Int, reverse_neighbors: Int, n: Int, membership: Int, stack: Int, cursor: Int, order: Int) abi("C") -> Int:
    return strong_components_kernel(ip(offsets), ip(neighbors), ip(reverse_offsets), ip(reverse_neighbors), n, ip(membership), ip(stack), ip(cursor), ip(order))


@export("mig_pagerank")
def mig_pagerank(in_offsets: Int, in_neighbors: Int, out_degree: Int, n: Int, damping: Float64, iterations: Int, tolerance: Float64, score: Int, next_score: Int) abi("C") -> Int:
    return pagerank_kernel(ip(in_offsets), ip(in_neighbors), ip(out_degree), n, damping, iterations, tolerance, fp(score), fp(next_score))


@export("mig_betweenness")
def mig_betweenness(offsets: Int, neighbors: Int, reverse_offsets: Int, reverse_neighbors: Int, n: Int, undirected: Int, result: Int, queue: Int, stack: Int, depth: Int, sigma: Int, delta: Int) abi("C"):
    betweenness_kernel(ip(offsets), ip(neighbors), ip(reverse_offsets), ip(reverse_neighbors), n, undirected, fp(result), ip(queue), ip(stack), ip(depth), fp(sigma), fp(delta))
