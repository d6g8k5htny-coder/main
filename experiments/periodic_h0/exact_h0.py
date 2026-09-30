"""Exact ordinary H0 for integer-valued periodic vertex cubical filtrations.

All endpoints stay in input integer units. See EXACT_H0.md for the graph,
elder, tie, independent connectivity-verification and bin-transfer proofs.
This module does not evaluate Fourier fields or certify their sampling law.
"""
from array import array
from collections import deque
from fractions import Fraction as Q


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(values, n):
    _require(type(n) is int and n >= 3, 'Integer grid side at least three required')
    _require(type(values) in (list, tuple) and len(values) == n*n,
             'Flat square list or tuple required')
    _require(all(type(value) is int for value in values), 'Strict integer samples required')
    _require(n*n < 2**31, 'Grid exceeds signed array-index capacity')


def _result(result):
    keys = {'intervals', 'birth_vertices', 'essential', 'essential_vertices', 'zero_count'}
    _require(type(result) is dict and set(result) == keys, 'Unexpected barcode schema')
    bars, vertices = result['intervals'], result['birth_vertices']
    _require(type(bars) is list and type(vertices) is list and len(bars) == len(vertices),
             'Invalid finite-bar arrays')
    _require(all(type(pair) is list and len(pair) == 2
                 and all(type(value) is int for value in pair) and pair[0] > pair[1]
                 for pair in bars), 'Finite bars require strict integer birth greater than death')
    _require(all(type(v) is int and v >= 0 for v in vertices)
             and len(set(vertices)) == len(vertices), 'Invalid or duplicate birth vertices')
    for key in ('essential', 'essential_vertices'):
        _require(type(result[key]) is list and len(result[key]) == 1
                 and type(result[key][0]) is int, 'Exactly one essential class required')
    _require(result['essential_vertices'][0] >= 0
             and result['essential_vertices'][0] not in vertices, 'Invalid essential vertex')
    _require(type(result['zero_count']) is int and result['zero_count'] >= 0,
             'Nonnegative integer zero count required')
    records = [(b, d, v) for (b, d), v in zip(bars, vertices)]
    _require(records == sorted(records), 'Finite bars must be canonically ordered')


def compute(values, n):
    """Return exact positive finite pairs and the one essential birth.

    ``values[x*n+y]`` is an integer sample. Axis edges wrap in both directions.
    Larger birth survives a merge; equal births prefer the smaller vertex id.
    Endpoint units are unchanged. Zero pairs are counted but not diagram points.
    """
    _grid(values, n)
    count = len(values)
    parent = array('i', [-1])*count
    sizes = array('I', [0])*count
    eldest = array('I', [0])*count
    # Python's stable sort retains increasing vertex ids within equal values.
    order = sorted(range(count), key=values.__getitem__, reverse=True)
    records, zero = [], 0

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for vertex in order:
        parent[vertex] = vertex
        sizes[vertex] = 1
        eldest[vertex] = vertex
        x, y = divmod(vertex, n)
        neighbors = (((x-1) % n)*n+y, ((x+1) % n)*n+y,
                     x*n+(y-1) % n, x*n+(y+1) % n)
        for neighbor in neighbors:
            if parent[neighbor] < 0:
                continue
            u, v = root(vertex), root(neighbor)
            if u == v:
                continue
            survivor, dying = eldest[u], eldest[v]
            if (values[survivor] < values[dying]
                    or (values[survivor] == values[dying] and survivor > dying)):
                survivor, dying = dying, survivor
            birth, death = values[dying], values[vertex]
            if birth > death:
                records.append((birth, death, dying))
            else:
                zero += 1
            # Balance the data structure independently of the mathematical elder.
            if sizes[u] < sizes[v]:
                u, v = v, u
            parent[v] = u
            sizes[u] += sizes[v]
            eldest[u] = survivor
    records.sort()
    essential = eldest[root(0)]
    return {'intervals': [[b, d] for b, d, _ in records],
            'birth_vertices': [v for _, _, v in records],
            'essential': [values[essential]], 'essential_vertices': [essential],
            'zero_count': zero}


def verify_by_connectivity(values, n, result):
    """Independently check every positive birth and exact elder death by BFS.

    No union-find or sweep replay is used. Every regional maximum plateau is
    found; each nonessential representative must reach an older vertex at d,
    and fail to do so at d+1. Integer levels make these two tests sufficient.
    Worst-case cost is O(N times the number of regional maxima), O(N) memory.
    """
    _grid(values, n)
    _result(result)
    count = len(values)
    _require(all(v < count for v in result['birth_vertices']+result['essential_vertices']),
             'Barcode vertex outside grid')
    _require(len(result['intervals'])+result['zero_count']+1 == count,
             'Finite, zero and essential counts do not sum to the vertex count')

    # Branch-based seam handling is deliberately separate from compute's
    # modular neighbor expression; the tiny test oracle builds undirected edges.
    def neighbors(vertex):
        row, column = divmod(vertex, n)
        return (vertex-n if row else vertex+n*(n-1),
                vertex+n if row+1 < n else vertex-n*(n-1),
                vertex-1 if column else vertex+n-1,
                vertex+1 if column+1 < n else vertex-n+1)

    seen = bytearray(count)
    maxima = []
    for vertex, level in enumerate(values):
        if seen[vertex]:
            continue
        seen[vertex] = 1
        queue = deque([vertex])
        has_higher_neighbor = False
        while queue:
            u = queue.popleft()
            for v in neighbors(u):
                if values[v] > level:
                    has_higher_neighbor = True
                elif values[v] == level and not seen[v]:
                    seen[v] = 1
                    queue.append(v)
        # The outer scan encounters the smallest plateau id first.
        if not has_higher_neighbor:
            maxima.append(vertex)
    essential = max(maxima, key=lambda v: (values[v], -v))
    _require(result['essential'] == [values[essential]]
             and result['essential_vertices'] == [essential], 'Incorrect essential elder')
    _require(set(result['birth_vertices']) == set(maxima)-{essential},
             'Finite births do not exhaust the nonessential regional maximum plateaus')

    visits = array('I', [0])*count
    generation = 0
    for (birth, death), vertex in zip(result['intervals'], result['birth_vertices']):
        _require(birth == values[vertex], 'Birth endpoint differs from its maximum plateau')
        for threshold, expected in ((death+1, False), (death, True)):
            generation += 1
            visits[vertex] = generation
            queue = deque([vertex])
            reached_older = False
            while queue:
                u = queue.popleft()
                if values[u] > birth or (values[u] == birth and u < vertex):
                    reached_older = True
                    break
                for v in neighbors(u):
                    if values[v] >= threshold and visits[v] != generation:
                        visits[v] = generation
                        queue.append(v)
            _require(reached_older == expected, 'Incorrect exact elder death level')
    return True


def bin_transfer(barcode, scale, epsilon, edges):
    """Apply the exact lifetime-bin sandwich, conditional on a diagram matching.

    The caller supplies the separate proof of an epsilon-matching to its target.
    No upper count is asserted for a bin starting at or below 2*epsilon.
    Input endpoints are integers divided by ``scale``; all bins are half open.
    """
    _result(barcode)
    _require(type(scale) is int and scale > 0, 'Positive strict integer scale required')
    _require(type(epsilon) is Q and epsilon >= 0, 'Nonnegative exact rational epsilon required')
    _require(type(edges) in (list, tuple) and len(edges) >= 2
             and all(type(edge) is Q and edge > 0 for edge in edges)
             and all(a < b for a, b in zip(edges, edges[1:])),
             'Strictly increasing positive exact rational bin edges required')
    lengths = [Q(b-d, scale) for b, d in barcode['intervals']]
    delta = 2*epsilon

    def count(lower, upper):
        return sum(lower <= length < upper for length in lengths)

    return [{'lower': a, 'upper': b, 'sample_count': count(a, b),
             'target_lower_count': count(a+delta, b-delta),
             'target_upper_count': count(a-delta, b+delta) if a > delta else None,
             'clean_upper': a > delta, 'nonempty_contraction': b-a > 2*delta}
            for a, b in zip(edges, edges[1:])]
