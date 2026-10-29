"""Exact population counts for two stationary NON-GAUSSIAN finite-grid fields.

The domain is G=(Z/4Z)^2, not a continuous torus. Before translation, put
27 at (0,0), V1,V2,V3 at (0,2),(2,0),(2,2), and zero elsewhere. World iid
uses three independent ideal continuous Uniform(0,1) variables. World
dependent uses (U,V,(U+V) mod 1) for independent ideal continuous uniforms.
Both worlds then use an independent uniform translation in the 16-element G.

Analytic field-law proof (not numerical quadrature): Haar measure on the
unit torus is preserved by (u,v)->(u,u+v mod 1), and also with u,v exchanged.
Thus every distinct pair of peak variables is independent uniform in both
worlds. Conditional on the translation, any two field coordinates select
constants, one repeated peak, or two distinct peaks. Their complete joint
laws therefore agree, including coincident coordinates. Averaging over the
same translation preserves equality and makes each full field G-stationary.
In particular their means and covariance agree; the common mean is 57/32.
The variables are not jointly independent in the dependent world.

Topology: use the exact_h0 vertex-cubical superlevel filtration, with four
periodic axis neighbors, every cell entering when all its vertices enter,
over any one fixed coefficient field. Above zero the four positive sites
are isolated. At zero the entire graph connects. Almost surely the positive
finite intervals are (0,Vi], lifetimes Vi; birth27 is the one essential
class. Zero pairs and the essential class are excluded. Equal positive
heights are harmless with the backend's simultaneous-level/elder convention.

For N_t=# {positive finite lifetimes <=t}, split the dependent parameter
square by u=t,v=t,u+v=t,1,1+t. Its positive-area convex cells have constant
count labels. Rational interior representatives are integer-rescaled and
checked by actual global exact_h0.compute plus verify_by_connectivity at
ALL 16 translations. Shoelace areas integrate these constant count labels
exactly. Iid threshold boxes similarly have product-volume weights. All
square/cut boundaries are null for the ideal input law; the inclusive
selector is explicit. Representatives are deterministic count witnesses,
NOT a finite distribution reproducing the continuous field law, moments
or covariance. The Haar proof above establishes the full pair-law identity.

This standard exposed finite example supplies no smooth/continuous-field
extension, Gaussian admission, asymptotic theorem, blind or novel discovery,
or scientific acceptance. It consumes no theorem conclusion or #437 code.
"""
import itertools
import json
import math
import sys
from fractions import Fraction as Q
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'periodic_h0'))
import exact_h0


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _threshold(t):
    _require(type(t) is Q and 0 <= t <= 1, 'Exact Fraction threshold in [0,1] required')


def build_field(peaks, *, shift=(0, 0)):
    """Return (strict integer grid, scale) for one rational field witness.

    This deterministic constructor does not sample the ideal population.
    Boundary peaks zero/one are supported; a zero peak gives no positive bar.
    The shift is addition in G, with values[x*4+y] as the backend layout.
    """
    _require(type(peaks) in (list, tuple) and len(peaks) == 3
             and all(type(p) is Q and 0 <= p <= 1 for p in peaks),
             'Three exact Fraction peaks in [0,1] required')
    _require(type(shift) is tuple and len(shift) == 2
             and all(type(s) is int and 0 <= s < 4 for s in shift),
             'A pair of strict integer shifts in range(4) required')
    scale = math.lcm(*(p.denominator for p in peaks))
    values = [0]*16
    for (x, y), height in zip(((0, 0), (0, 2), (2, 0), (2, 2)), (Q(27), *peaks)):
        values[((x+shift[0]) % 4)*4+(y+shift[1]) % 4] = int(height*scale)
    return values, scale


def dependent_peaks(u, v):
    """Evaluate the specified copula at a rational point of [0,1)^2."""
    _require(all(type(p) is Q and 0 <= p < 1 for p in (u, v)),
             'Exact Fraction torus coordinates in [0,1) required')
    return u, v, (u+v) % 1


def polygon_area(polygon):
    """Unsigned exact shoelace area, including zero-area/empty polygons."""
    _require(type(polygon) in (list, tuple)
             and all(type(p) in (list, tuple) and len(p) == 2
                     and all(type(x) is Q for x in p) for p in polygon),
             'A sequence of exact Fraction coordinate pairs required')
    if len(polygon) < 3:
        return Q(0)
    return abs(sum(p[0]*q[1]-p[1]*q[0]
                   for p, q in zip(polygon, (*polygon[1:], polygon[0]))))/2


def _clip(polygon, a, b, c):
    """Intersect a convex polygon with a*x+b*y<=c, using rational arithmetic."""
    result = []
    previous = polygon[-1]
    pv = a*previous[0]+b*previous[1]-c
    for current in polygon:
        cv = a*current[0]+b*current[1]-c
        if (pv <= 0) != (cv <= 0):
            ratio = pv/(pv-cv)
            result.append(tuple(x+ratio*(y-x) for x, y in zip(previous, current)))
        if cv <= 0:
            result.append(current)
        previous, pv = current, cv
    return tuple(result)


def dependent_cells(t):
    """Return positive-area exact cells; only their count labels are integrated."""
    _threshold(t)
    cells = [((Q(0), Q(0)), (Q(1), Q(0)), (Q(1), Q(1)), (Q(0), Q(1)))]
    for a, b, c in ((Q(1), Q(0), t), (Q(0), Q(1), t),
                    (Q(1), Q(1), t), (Q(1), Q(1), Q(1)), (Q(1), Q(1), 1+t)):
        cells = [part for cell in cells
                 for part in (_clip(cell, a, b, c), _clip(cell, -a, -b, -c))
                 if polygon_area(part) > 0]
    return tuple(cells)


def _witness_count(peaks, t, shift):
    values, scale = build_field(peaks, shift=shift)
    barcode = exact_h0.compute(values, 4)
    exact_h0.verify_by_connectivity(values, 4, barcode)
    return sum(Q(b-d, scale) <= t for b, d in barcode['intervals'])


def population_law(t, *, world='dependent'):
    """Return exact probabilities for N_t=0,1,2,3 in the fixed ideal model.

    Each witness executes the actual global H0 backend and its independent
    connectivity checker. Area/volume weights integrate only the constant
    count event, not arbitrary functions of the representative fields.
    """
    _threshold(t)
    _require(type(world) is str and world in ('dependent', 'iid'),
             'World must be dependent or iid')
    if world == 'dependent':
        weighted_peaks = []
        for cell in dependent_cells(t):
            u = sum(p[0] for p in cell)/len(cell)
            v = sum(p[1] for p in cell)/len(cell)
            weighted_peaks.append((dependent_peaks(u, v), polygon_area(cell)))
    else:
        intervals = [(a, b) for a, b in ((Q(0), t), (t, Q(1))) if a < b]
        weighted_peaks = [(tuple((a+b)/2 for a, b in box),
                           math.prod(b-a for a, b in box))
                          for box in itertools.product(intervals, repeat=3)]
    law = [Q(0)]*4
    for peaks, weight in weighted_peaks:
        for shift in itertools.product(range(4), repeat=2):
            count = _witness_count(peaks, t, shift)
            _require(0 <= count <= 3, 'Unexpected positive finite-bar count')
            law[count] += weight/16
    _require(sum(law) == 1 and all(p >= 0 for p in law), 'Invalid population partition')
    return tuple(law)


def report():
    dependent = population_law(Q(1, 4), world='dependent')
    iid = population_law(Q(1, 4), world='iid')
    return {'model': 'stationary discrete 4x4 periodic vertex-cubical fields',
            'dependent_quarter': list(map(str, dependent)),
            'iid_quarter': list(map(str, iid)),
            'mean_count_quarter': str(sum(k*p for k, p in enumerate(dependent))),
            'expected_unordered_pairs_quarter': str(sum(math.comb(k, 2)*p
                                                       for k, p in enumerate(dependent))),
            'second_factorial_moment_quarter': str(sum(k*(k-1)*p
                                                      for k, p in enumerate(dependent))),
            'translation_group_size': 16,
            'field_pair_law_identity': 'analytic Haar shear and conditional translation',
            'representatives_are_field_law_quadrature': False,
            'gaussian_or_continuum_admission': False,
            'blind_or_novel_discovery': False,
            'scientific_effect': 'NONE'}


if __name__ == '__main__':
    print(json.dumps(report(), sort_keys=True, indent=2))
