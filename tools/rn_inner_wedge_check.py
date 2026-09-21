#!/usr/bin/env python3
"""Replay one finite-width RN annular wedge, conditional on the imported H3 floor.

This is a scoped author candidate. The full annulus, all-radii and all-angle
problems and event identification remain open; no scientific status changes.
"""
import argparse
from fractions import Fraction as F
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.cover.ledger import Box, Cell, Ledger, check_exact_partition
from research.interval import Interval as I, pi
from research.rn.certificate import canonical_bytes
from research.rn.density_majorant import density_moment_majorant
from research.rn.n6_inputs import checked_n6
from research.rn.side24_wedge import OUTER_BOX, RADII, TURNS, certify_wedge, containment
from tools.rn_side24_check import load_report

CANDIDATE = 'research/rn/candidates/inner_wedge_20260920_v1.json'
SCHEMA = 'RN_SIDE24_INNER_AXIS_WEDGE_V1'
DOMAIN = Box(*RADII, *TURNS)
DETERMINANT_LOWER = F(8, 10**25)
MAHALANOBIS_LOWER = F(103)
INTEGRAND_UPPER = F(1, 100000)
AREA_PI_COEFFICIENT = F(21, 5120000)
INTEGRAL_UPPER = F(33, 256000000000)
BITS = 256
PREC = 110
SCOPE = dict(authority='NONE', organizational_independence_credit=0,
             scientific_status_changed=False, original_prize_closed=False,
             wedge_covered_under_imported_h3=True, full_annulus_covered=False,
             all_radii_certified=False, all_pin_orientations_certified=False,
             event_identification_established=False, h3_floor_reproved_here=False,
             mathematical_acceptance='SCOPED_AUTHOR_CANDIDATE; retained review predicates apply')


def encode(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, I):
        return [str(value.lo), str(value.hi)]
    if isinstance(value, Box):
        return value.as_json()
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def assemble_cover(bounds, majorant):
    """Check exact auxiliary partition, then integrate the polar wedge once."""
    records = bounds['pieces']
    if bounds['geometry'] != containment():
        raise ValueError('declared wedge geometry differs from the proved containment')
    if (type(bounds['piece_count']) is not int or bounds['piece_count'] != 12 or
            len(records) != 12 or type(bounds['pending_pieces']) is not int or bounds['pending_pieces'] != 0):
        raise ValueError('complete twelve-piece auxiliary cover required')
    check_exact_partition(OUTER_BOX, [record['box'] for record in records], what='RN wedge auxiliary cover')
    if (any(record['determinant_lower'] < DETERMINANT_LOWER or record['q_lower'] < MAHALANOBIS_LOWER
            for record in records) or bounds['conditioning_energy_upper'] >= 8):
        raise ValueError('uniform determinant, energy or Mahalanobis premise failed')
    if (majorant['determinant_lower'] != DETERMINANT_LOWER or
            majorant['mahalanobis_lower'] != MAHALANOBIS_LOWER or majorant['jacobian_upper'] != 1 or
            majorant['integrand_upper'] >= INTEGRAND_UPPER or majorant['integrand_upper'] <= 0):
        raise ValueError('density-weighted moment bound does not meet the declared cap')
    p = pi(PREC).round_out(BITS)
    if not 3 < p.lo <= p.hi < F(22, 7):
        raise ValueError('rational pi bounds not certified')
    exact_factor = DOMAIN.dv() * (DOMAIN.u1**2 - DOMAIN.u0**2)
    if exact_factor != AREA_PI_COEFFICIENT or bounds['geometry']['area_pi_coefficient'] != exact_factor:
        raise ValueError('polar area factor mismatch')
    area = exact_factor * p
    value_range = I(0, majorant['integrand_upper'])
    contribution = area * value_range
    ledger = Ledger('RN5 inner positive-x wedge', DOMAIN, 'polar(radius, signed turn)',
                    integrand='SIDE24 six-pin density-weighted determinant majorant',
                    note='Exactly this wedge only. The auxiliary Cartesian rectangle is not the integration region.')
    ledger.add(Cell('wedge', DOMAIN, 0))
    ledger.accept('wedge', area, value_range, contribution)
    total = ledger.total().certified_enclosure()
    if total.lo != 0 or total.hi >= INTEGRAL_UPPER or INTEGRAL_UPPER != INTEGRAND_UPPER*exact_factor*F(22, 7):
        raise ValueError('exact wedge area-weighted budget failed')
    return ledger.receipt()


def build_report():
    n6, identities = checked_n6()
    caps = n6.moment_cap_certificate()['caps']
    if caps[2] > 4 or caps[1]**2 > 4:
        raise ValueError('normalized torus Hessian variance cap unavailable')
    bounds = certify_wedge(n6, pieces=12, order=6, bits=BITS)
    majorant = density_moment_majorant(DETERMINANT_LOWER, MAHALANOBIS_LOWER, bits=BITS)
    cover = assemble_cover(bounds, majorant)
    # Read fresh eligibility/identity again after numerical work; a cache is not custody.
    _, after = checked_n6()
    if identities != after:
        raise ValueError('input identities changed during wedge verification')
    return encode(dict(schema=SCHEMA, scope=SCOPE, source_identities=identities,
        fixed_r='1/20', fixed_b='6/5', fixed_pin_axis='x', full_height_window=['57599/48000', '6/5'],
        domain=DOMAIN, geometry='rho in [1/10,11/100], signed turns in [-1/1024,1/1024]',
        auxiliary=bounds, majorant=majorant, cover=cover,
        simple_bounds=dict(determinant_lower=DETERMINANT_LOWER, mahalanobis_lower=MAHALANOBIS_LOWER,
            integrand_strict_upper=INTEGRAND_UPPER, exact_area_pi_coefficient=AREA_PI_COEFFICIENT,
            integral_strict_upper=INTEGRAL_UPPER),
        accounting='Twelve strips partition one containing Cartesian rectangle. Its uniform cap bounds the contained wedge. Only the wedge polar area is multiplied, exactly once.',
        remaining='The rest of 1/10<=|y|<=5, the remote budget, all-small-r and all-pin-angle bounds, actual weighted-Palm/event interfaces and required independent review remain unresolved.'))


def check_report(report):
    if type(report) is not dict or report.get('schema') != SCHEMA or report.get('scope') != SCOPE:
        raise ValueError('candidate schema or scientific scope mismatch')
    expected = build_report()
    if canonical_bytes(report) != canonical_bytes(expected):
        raise ValueError('candidate differs from exact wedge replay')
    return expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--output', type=Path)
    group.add_argument('--check', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output:
            result = build_report()
            with args.output.open('xb') as stream:
                stream.write(canonical_bytes(result) + b'\n')
        else:
            result = check_report(load_report(args.check or ROOT / CANDIDATE))
        print('RN inner wedge: 12 auxiliary strips, zero pending; exact area 21*pi/5120000; integral <33/256000000000; fixed-r/axis/imported H3 only')
        return 0
    except Exception as error:
        print('RN inner wedge: REJECTED: ' + str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
