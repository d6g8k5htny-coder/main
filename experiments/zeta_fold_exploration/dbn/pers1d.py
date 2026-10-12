"""Superlevel-set H0 persistence of a function sampled on a path (elder rule).

Endpoint convention: this routine KEEPS bars born at the first or last sample of the path
(it is used on sampled windows in check_collision_persistence.py and
stats/check_Z_persistence.py). stats/analyze.py:barcode DROPS bars born at the path
endpoints. The two conventions therefore differ by one or two bars per window.
"""


def superlevel_h0(vals):
    """Return list of (birth_index, death_index, birth_value, death_value) for finite bars,
    plus the essential class index. Elder rule: at a merge the component with the lower
    birth value dies."""
    n = len(vals)
    order = sorted(range(n), key=lambda i: -vals[i])
    parent = [-1] * n  # -1 = not yet added
    birth = {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    bars = []
    for v in order:
        parent[v] = v
        birth[v] = v
        roots = []
        for nb in (v - 1, v + 1):
            if 0 <= nb < n and parent[nb] != -1:
                roots.append(find(nb))
        roots = list(dict.fromkeys(roots))
        if not roots:
            continue
        if len(roots) == 1:
            parent[v] = roots[0]
            continue
        a, b = roots
        ba, bb = birth[a], birth[b]
        # elder survives
        if vals[ba] >= vals[bb]:
            old, young = a, b
        else:
            old, young = b, a
        bars.append((birth[young], v, vals[birth[young]], vals[v]))
        parent[young] = old
        parent[v] = old
    root = find(order[0])
    return bars, birth[root]
