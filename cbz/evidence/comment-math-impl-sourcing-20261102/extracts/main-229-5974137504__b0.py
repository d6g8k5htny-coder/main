#!/usr/bin/env python3
"""C96 cross-provider review: a discrete control of the lemma's combinatorial core (standard library only).

Target: C96-DESIGNATED-MAXIMIN-TO-H0-20261003-v1, main issue 229, comment 5965141133
(frozen proof: final 7,657 bytes, SHA-256 7198ff63...7a7a).
Reviewer: Anthropic Claude, session_01NMeKEismAyeqgdB4sy2NJU, for Dylan Roy (delegated AI review).
Written without the author's or the first reviewer's code.

On a connected graph with distinct vertex values (here torus grids, 4- and 8-neighbour), two independent
computations are compared for every local maximum m:
  (a) superlevel H0 persistence by union-find with the elder rule: the vertex at which m's class dies;
  (b) the maximin (widest-path) level d(m) = max over paths from m to a vertex x with f(x) > f(m) of the
      minimum of f along the path, and -infinity if there is no such x.
Checks: d(m) = f(death vertex of m) for every non-global maximum; the global maximum alone has
d = -infinity and an essential class; and, for random designated pairs (m, s) with f(s) < f(m),
d(m) = f(s) iff m's class dies at s (C96 (D1), equivalence 1 <=> 2, graph version).
This is the discrete analogue of the lemma's merge argument. It does not prove the smooth statement,
[P] section 8's almost-sure Morse property, or C95's weighted bounds.

Usage: python3 c96_discrete.py [MUTANT]   (exit 0 pass, 1 fail, 2 unknown label)
Mutants (each must exit 1):
  ELDER_REVERSED   at a merge the class with the higher birth dies (anti-elder rule)
  GE_ENDPOINT      the maximin admits endpoints with f(x) >= f(m), so x = m itself qualifies
  FIRST_MERGE      a maximum's death is recorded at its first contact with any other component
"""
import heapq
import random
import sys

MUTANTS = ("ELDER_REVERSED", "GE_ENDPOINT", "FIRST_MERGE")
MUT = None
if len(sys.argv) > 1:
    MUT = sys.argv[1]
    if len(sys.argv) > 2 or MUT not in MUTANTS:
        print("unknown mutant label: %s" % " ".join(sys.argv[1:]))
        sys.exit(2)

RNG = random.Random(96_5965141133)
STATS = {}


def tally(key, ok):
    a, b = STATS.get(key, (0, 0))
    STATS[key] = (a + (1 if ok else 0), b + 1)


def field(n):
    """Integer-valued periodic 'smooth' field (sum of triangle waves) with a random tie-breaker."""
    terms = []
    for _ in range(RNG.randint(3, 6)):
        p, q = RNG.randint(-3, 3), RNG.randint(-3, 3)
        if p == 0 and q == 0:
            p = 1
        terms.append((p, q, RNG.randrange(n), RNG.randint(1, 9)))
    perm = list(range(n * n))
    RNG.shuffle(perm)
    vals = []
    for i in range(n):
        for j in range(n):
            s = 0
            for (p, q, ph, a) in terms:
                m = (p * i + q * j + ph) % n
                s += a * min(m, n - m)
            vals.append(s * (n * n) + perm[i * n + j])   # distinct; order of the smooth part kept
    return vals


def neighbours(n, eight):
    nb = []
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if eight:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for i in range(n):
        for j in range(n):
            nb.append([((i + di) % n) * n + (j + dj) % n for (di, dj) in steps])
    return nb


def persistence(vals, nb):
    """Superlevel H0 by union-find; returns {birth vertex: death vertex or None (essential)}."""
    order = sorted(range(len(vals)), key=lambda v: -vals[v])
    parent, birth, added = {}, {}, set()
    death = {}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    for v in order:
        roots = {find(u) for u in nb[v] if u in added}
        added.add(v)
        parent[v] = v
        if not roots:
            birth[v] = v
            death[v] = None
            continue
        roots = sorted(roots, key=lambda r_: -vals[birth[r_]])   # eldest first
        keep = roots[0]
        if MUT == "ELDER_REVERSED":
            keep = roots[-1]
        parent[v] = keep
        for r_ in roots:
            if r_ == keep:
                continue
            if death[birth[r_]] is None:
                death[birth[r_]] = v                  # the younger class dies at v (elder rule)
            parent[r_] = keep
            if MUT == "FIRST_MERGE" and death[birth[keep]] is None:
                death[birth[keep]] = v                # wrong: the surviving elder is marked dead too
    return death


def maximin(vals, nb, m):
    """Widest path from m to any vertex x with f(x) > f(m); None for -infinity."""
    fm = vals[m]
    best = {m: fm}
    heap = [(-fm, m)]
    while heap:
        nb_val, v = heapq.heappop(heap)
        b = -nb_val
        if b < best.get(v, -1):
            continue
        if (vals[v] >= fm) if MUT == "GE_ENDPOINT" else (vals[v] > fm):
            return b                                  # first higher vertex popped: the widest value
        for u in nb[v]:
            nbv = min(b, vals[u])
            if nbv > best.get(u, -1):
                best[u] = nbv
                heapq.heappush(heap, (-nbv, u))
    return None


for n in (12, 16, 24, 32):
    for eight in (False, True):
        for trial in range(25 if n <= 16 else 12):
            vals = field(n)
            g = neighbours(n, eight)
            tally("C0 values are distinct", len(set(vals)) == len(vals))
            death = persistence(vals, g)
            gmax = max(range(len(vals)), key=lambda v: vals[v])
            maxima = [v for v in range(len(vals)) if all(vals[u] < vals[v] for u in g[v])]
            tally("C1 births are exactly the local maxima", sorted(death) == sorted(maxima))
            for m in maxima:
                d = maximin(vals, g, m)
                if m == gmax:
                    tally("C2 global maximum: d = -infinity and an essential class", d is None and death[m] is None)
                    continue
                s = death[m]
                tally("C3 non-global maximum: finite class, d(m) = f(death vertex)", s is not None and d == vals[s])
                if s is not None:
                    tally("C4 the death vertex is lower than m and higher-born component exists", vals[s] < vals[m])
            for _ in range(40):
                m = RNG.choice(maxima)
                lower = [v for v in range(len(vals)) if vals[v] < vals[m]]
                if not lower:
                    continue
                s = RNG.choice(lower) if RNG.random() < 0.5 or death[m] is None else death[m]
                d = maximin(vals, g, m)
                tally("C5 designated pair: d(m) = f(s) iff m dies at s", (d is not None and d == vals[s]) == (death[m] == s))

ok = all(a == b for a, b in STATS.values())
for key in sorted(STATS):
    a, b = STATS[key]
    print("%-72s %6d/%-6d %s" % (key, a, b, "PASS" if a == b else "FAIL"))
print("mutant: %s" % (MUT if MUT else "none"))
print("total checks: %d; failures: %d" % (sum(b for _, b in STATS.values()), sum(b - a for a, b in STATS.values())))
sys.exit(0 if ok else 1)
