"""Copied exact helpers consumed by the Reactor A finite control only."""

from array import array
from fractions import Fraction


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_periodic_h0(values, side):
    """Return positive finite superlevel H0 bars and the essential birth.

    This is the consumed union-find core of ``periodic_h0/exact_h0.py``.
    Endpoints remain in strict integer input units and axis edges wrap.
    """
    _require(type(side) is int and side >= 3, "integer grid side >= 3 required")
    _require(type(values) in (list, tuple) and len(values) == side * side,
             "flat square list or tuple required")
    _require(all(type(value) is int for value in values),
             "strict integer samples required")
    count = len(values)
    parent = array("i", [-1]) * count
    sizes = array("I", [0]) * count
    eldest = array("I", [0]) * count
    order = sorted(range(count), key=values.__getitem__, reverse=True)
    records = []

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for vertex in order:
        parent[vertex] = vertex
        sizes[vertex] = 1
        eldest[vertex] = vertex
        row, column = divmod(vertex, side)
        neighbors = (
            ((row - 1) % side) * side + column,
            ((row + 1) % side) * side + column,
            row * side + (column - 1) % side,
            row * side + (column + 1) % side,
        )
        for neighbor in neighbors:
            if parent[neighbor] < 0:
                continue
            left, right = root(vertex), root(neighbor)
            if left == right:
                continue
            survivor, dying = eldest[left], eldest[right]
            if (values[survivor] < values[dying] or
                    (values[survivor] == values[dying] and survivor > dying)):
                survivor, dying = dying, survivor
            birth, death = values[dying], values[vertex]
            if birth > death:
                records.append((birth, death, dying))
            if sizes[left] < sizes[right]:
                left, right = right, left
            parent[right] = left
            sizes[left] += sizes[right]
            eldest[left] = survivor
    records.sort()
    essential_vertex = eldest[root(0)]
    return {
        "intervals": [[birth, death] for birth, death, _ in records],
        "essential": [values[essential_vertex]],
    }


def _integer_root_exact(value, order):
    _require(type(value) is int and value >= 0, "nonnegative integer radicand required")
    _require(type(order) is int and order > 0, "positive integer root order required")
    low, high = 0, value + 1
    while low + 1 < high:
        middle = (low + high) // 2
        if middle ** order <= value:
            low = middle
        else:
            high = middle
    if low ** order != value:
        raise ValueError("rational root is not exact; exact control refused")
    return low


def rational_power_exact(value, exponent):
    """Return ``value**exponent`` only when the rational root is exact."""
    _require(type(value) in (int, Fraction) and value >= 0,
             "nonnegative exact rational base required")
    _require(type(exponent) is Fraction and exponent > 0,
             "positive Fraction exponent required")
    value = Fraction(value)
    root = Fraction(
        _integer_root_exact(value.numerator, exponent.denominator),
        _integer_root_exact(value.denominator, exponent.denominator),
    )
    return root ** exponent.numerator
