"""Exact finite companion to the P15 composition proof, not a theorem prover.

Only small supplied finite interfaces are exhausted. Analytic source primitives
and arbitrary-size existence theorems are not proved by this module.
"""
from fractions import Fraction as F
from functools import lru_cache
from itertools import combinations, product
from math import ceil, comb, prod

MAX_VERTICES = 10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def family(edges):
    return frozenset(frozenset(edge) for edge in edges)


def subsets(vertices):
    vertices = tuple(sorted(vertices))
    require(len(vertices) <= MAX_VERTICES, "finite exhaustion cap exceeded")
    return tuple(frozenset(s) for n in range(len(vertices) + 1)
                 for s in combinations(vertices, n))


def antichain(edges):
    edges = family(edges)
    return frozenset(e for e in edges if not any(f < e for f in edges))


def good(vertices, witnesses):
    return not any(e <= vertices for e in witnesses)


def price(generator, prices):
    return prod((prices[v] for v in generator), start=F(1))


def cover_cost(generators, prices):
    require(all(z >= 0 for z in prices.values()), "negative price")
    return sum((price(g, prices) for g in family(generators)), F(0))


def probability(vertices, probabilities, predicate):
    vertices = frozenset(vertices)
    require(set(probabilities) == vertices, "probability coordinates differ")
    require(all(0 <= p <= 1 for p in probabilities.values()), "invalid probability")
    return sum((prod((probabilities[v] if v in s else 1 - probabilities[v]
                      for v in vertices), start=F(1))
                for s in subsets(vertices) if predicate(s)), F(0))


def colorable(vertices, witnesses, palette):
    require(isinstance(palette, int) and palette >= 1, "positive palette required")
    vertices = frozenset(vertices)
    require(len(vertices) <= MAX_VERTICES, "finite exhaustion cap exceeded")
    if not vertices:
        return True  # Partition into zero parts is allowed by "at most K".
    relevant = family(e for e in witnesses if e <= vertices)
    if any(len(e) <= 1 for e in relevant):
        return False
    if not relevant or palette >= len(vertices):
        return True
    order = sorted(vertices, key=lambda v: (-sum(v in e for e in relevant), v))
    bins = [set() for _ in range(min(palette, len(vertices)))]

    def search(index, used):
        if index == len(order):
            return True
        vertex = order[index]
        for color in range(min(used + 1, len(bins))):
            bins[color].add(vertex)
            allowed = not any(e <= bins[color] for e in relevant)
            if allowed and search(index + 1, max(used, color + 1)):
                bins[color].remove(vertex)
                return True
            bins[color].remove(vertex)
        return False

    return search(0, 0)


def obstruction_cover(vertices, witnesses, palette, generators):
    """Return exact obstruction count; reject one missed set immediately."""
    vertices = frozenset(vertices)
    generators = family(generators)
    require(all(g <= vertices for g in generators), "cover outside support")
    count = 0
    for s in subsets(vertices):
        if not colorable(s, witnesses, palette):
            count += 1
            require(any(g <= s for g in generators), "uncovered obstruction")
    return count


def validate_blocks(blocks, local, macro):
    require(blocks and set(blocks) == set(local), "nonempty block index interface")
    require(all(blocks[i] for i in blocks), "empty block must be deleted")
    all_vertices = frozenset().union(*blocks.values())
    require(sum(map(len, blocks.values())) == len(all_vertices), "overlapping blocks")
    require(len(all_vertices) <= MAX_VERTICES, "finite exhaustion cap exceeded")
    require(all(e <= blocks[i] for i in blocks for e in local[i]), "nonlocal witness")
    require(all(e <= set(blocks) for e in macro), "macro edge outside indices")
    return all_vertices


def complete_witnesses(blocks, local, macro):
    validate_blocks(blocks, local, macro)
    edges = [e for i in blocks for e in local[i]]
    for edge in macro:
        edges.extend(frozenset(choices) for choices in
                     product(*(sorted(blocks[i]) for i in sorted(edge))))
    return antichain(edges)


def require_complete_actual(blocks, local, macro, actual):
    require(antichain(actual) == complete_witnesses(blocks, local, macro),
            "actual family is not the complete-transversal family")


def occupancy_cover(vertices, prices):
    require(vertices, "occupancy block must be nonempty")
    require(all(prices[v] >= 0 for v in vertices), "negative price")
    return (family(({v} for v in vertices))
            if sum((prices[v] for v in vertices), F(0)) <= 1
            else family([set()]))


def lift(macro_generators, occupancy):
    descriptions = []
    for g in sorted(family(macro_generators), key=lambda e: (len(e), sorted(e))):
        for selected in product(*(occupancy[i] for i in sorted(g))):
            descriptions.append(frozenset().union(*selected))
    return family(descriptions), tuple(descriptions)


def check_macro_certificate(indices, macro, colors, generators):
    require(set(colors) == set(indices), "macro coloring coordinates differ")
    require(all(isinstance(c, int) and c >= 0 for c in colors.values()), "invalid macro color")
    require(all(g <= set(indices) for g in generators), "macro cover outside indices")
    for e in macro:
        if len({colors[i] for i in e}) <= 1:
            require(any(g <= e for g in generators), "uncovered monochromatic macro edge")


def verify_composition(blocks, local, macro, local_palette, local_covers,
                       colors, macro_generators, prices):
    vertices = validate_blocks(blocks, local, macro)
    require(set(prices) == set(vertices), "price coordinates differ")
    require(set(local_covers) == set(blocks), "local cover indices differ")
    local_counts = {i: obstruction_cover(blocks[i], local[i], local_palette,
                                        local_covers[i]) for i in blocks}
    check_macro_certificate(blocks, macro, colors, macro_generators)
    occupancy = {i: occupancy_cover(blocks[i], prices) for i in blocks}
    occupancy_prices = {i: cover_cost(occupancy[i], prices) for i in blocks}
    lifted, descriptions = lift(macro_generators, occupancy)
    description_cost = sum((price(g, prices) for g in descriptions), F(0))
    require(description_cost == cover_cost(macro_generators, occupancy_prices),
            "lift description product identity failed")
    require(cover_cost(lifted, prices) <= description_cost, "deduplication raised price")
    for s in subsets(vertices):
        occupied = frozenset(i for i in blocks if s & blocks[i])
        if any(g <= occupied for g in macro_generators):
            require(any(g <= s for g in lifted), "setwise occupancy lift failed")
    palette = local_palette * len(set(colors.values()))
    total_cover = lifted | family(g for i in blocks for g in local_covers[i])
    global_witnesses = complete_witnesses(blocks, local, macro)
    count = obstruction_cover(vertices, global_witnesses, palette, total_cover)
    return {"vertices": len(vertices), "subsets_exhausted": 2 ** len(vertices),
            "local_palette": local_palette, "macro_colors": len(set(colors.values())),
            "product_palette": palette, "global_obstructions": count,
            "local_obstructions": local_counts, "lifted_generators": len(lifted),
            "indexed_description_cost": description_cost,
            "lifted_cost": cover_cost(lifted, prices),
            "total_cover_cost": cover_cost(total_cover, prices)}


def singleton_restriction(blocks, local, macro):
    actual = complete_witnesses(blocks, local, macro)
    require(frozenset() not in actual, "globally empty family has no singleton reduction")
    removed = frozenset().union(*(e for e in actual if len(e) == 1))
    reduced_blocks = {i: b - removed for i, b in blocks.items() if b - removed}
    reduced_local = {i: family(e for e in local[i] if not e & removed)
                     for i in reduced_blocks}
    reduced_macro = family(e for e in macro if e <= set(reduced_blocks))
    expected = antichain(e for e in actual if not e & removed)
    if reduced_blocks:
        require(complete_witnesses(reduced_blocks, reduced_local, reduced_macro) == expected,
                "singleton restriction changed the actual family")
    else:
        require(not expected, "empty remainder is not universally good")
    return removed, reduced_blocks, reduced_local, reduced_macro


def verify_scalar_sandwich(vertices, witnesses, weights, kappa):
    require(kappa >= 1 and set(weights) == set(vertices), "invalid scalar interface")
    require(all(0 <= a < 1 for a in weights.values()), "scalar singleton endpoint")
    for s in subsets(vertices):
        weight = sum((weights[v] for v in s), F(0))
        require(weight >= 1 or good(s, witnesses), "inner scalar implication failed")
        require(not good(s, witnesses) or weight < kappa, "strict outer scalar implication failed")


@lru_cache(None)
def primitive_palette(rank, multiplier=F(2)):
    """Evaluate the pinned P11-D recurrence; only ranks <=8 are run here."""
    multiplier = F(multiplier)
    require(2 <= rank <= 8 and multiplier >= 1, "bounded recurrence domain")
    if rank == 2:
        return 192 * ceil(multiplier ** 2)
    d = rank * (2 * rank - 1)
    a = F(d - 1, d)
    lam = d - 1
    u = {0: F(0)}
    for j in range(1, rank):
        u[j] = lam * (1 + sum((comb(j, i) * u[j - i] for i in range(1, j)), F(0)))
    t = 1 + sum((comb(rank, i) * u[rank - i] for i in range(1, rank)), F(0))
    k_c = 64 * ceil(multiplier ** 2)
    k_r = ceil(48 * t * (multiplier * d) ** rank)
    return k_c + 3 * primitive_palette(rank - 1, multiplier / a) * k_r
