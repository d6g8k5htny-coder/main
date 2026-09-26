#!/usr/bin/env python3
"""Replay a complete local RN rectangle cover, conditional on the pinned H3 floor.

The small declared rectangle is not the entire near annulus. No scientific
promotion, far-budget assembly, all-small-r theorem or independence follows.

SIDE24 has no new source-of-truth carrier after 2026-08-06. Absent objects
stay ABSENT. Quarantine is not a source of truth. A green run is engineering
hygiene: inventable_attempt_accepted stays false, and OBL-H5-JETMOD stays OPEN.
"""
import argparse
from fractions import Fraction as F
import hashlib
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.cover import Box, DriverConfig, run
from research.cover.audit import checked_total
from research.rn.certificate import canonical_bytes
from research.rn.spatial_cover import RNRectangleRegion, RNSide24Integrand
from tools import rn_side24_check as point_check
from tools.rn_side24_density_check import checked_imports, serialize

SCHEMA = 'RN_SIDE24_LOCAL_SPATIAL_COVER_V1'
CANDIDATE = 'research/rn/candidates/side24_spatial_20260920_v1.json'
RECTANGLE = Box(F(1999, 2000), F(2001, 2000), F(1999, 2000), F(2001, 2000))
LOCAL_BUDGET = F(3, 500000000000)
TARGET_MEMBER = 'round5/intake/rn.txt'
TARGET_IDENTITY = (12956, '0c9446b7cb49e6e1e45e43f84bb72ce680f022882e225fdff61c8e6edede1373')
SCOPE = {
    'authority': 'NONE', 'independence_credit': 0,
    'scientific_status_changed': False, 'original_prize_closed': False,
    'local_rectangle_covered': True, 'near_annulus_covered': False,
    'near_restoration_target_established': False, 'far_budget_replayed': False,
    'all_small_r_certified': False, 'h3_floor_reproved_here': False,
    'h3_floor_status': 'EXPLICIT_IMPORTED_HYPOTHESIS',
    'statement': 'Uniform full-mark RN integrand and integral bound on the declared local rectangle, fixed r=1/20, conditional on the pinned H3 floor.',
    'review': 'AUTHOR CANDIDATE; technical and required independent mathematical acceptance remain separate.',
}


def checked_target(root):
    with zipfile.ZipFile(root / point_check.ARCHIVE) as archive:
        found = [item for item in archive.infolist() if item.filename == TARGET_MEMBER]
        if len(found) != 1 or found[0].file_size != TARGET_IDENTITY[0]:
            raise ValueError('near-target source member identity mismatch')
        raw = archive.read(TARGET_MEMBER)
    if hashlib.sha256(raw).hexdigest() != TARGET_IDENTITY[1]:
        raise ValueError('near-target source digest mismatch')
    return {point_check.ARCHIVE + '::' + TARGET_MEMBER: point_check.identity(raw)}


def cell_data(proof):
    # Complete certificates include each FamilyLaw; avoid duplicate dataclasses.
    return serialize({key: value.as_json() if key == 'box' else value
                      for key, value in proof.items() if key != 'laws'})


def build_report(root=None):
    root = ROOT if root is None else Path(root)
    sources = {**point_check.checked_sources(root), **checked_imports(root),
               **checked_target(root)}
    region, integrand = RNRectangleRegion(RECTANGLE), RNSide24Integrand(bits=192)
    ledger = run(region, integrand, DriverConfig(upper_budget=LOCAL_BUDGET,
                 max_depth=1, max_cells=5, prec=88, sig_bits=192))
    total = checked_total(ledger).certified_enclosure()
    receipt = ledger.receipt()
    if receipt['pending_count'] or total.lo < 0 or total.hi > LOCAL_BUDGET:
        raise ValueError('complete nonnegative local upper-budget cover required')
    attempts = [{**item, 'proof': cell_data(item['proof'])} if 'proof' in item
                else item for item in integrand.attempts]
    if len(attempts) != 5 or attempts[0]['status'] != 'INCONCLUSIVE':
        raise ValueError('expected retained parent refusal and four child attempts')
    if any(item['status'] != 'BOUNDED' for item in attempts[1:]):
        raise ValueError('all four exact child cells must be bounded')
    return {
        'schema': SCHEMA, 'scope': dict(SCOPE), 'round_bits': 192,
        'domain': RECTANGLE.as_json(), 'domain_area': str(RECTANGLE.param_area()),
        'sources': sources, 'attempts': attempts, 'cover': receipt,
        'composition': {'integral_range': serialize(total),
            'declared_local_upper_budget': str(LOCAL_BUDGET),
            'simple_rational_upper': str(LOCAL_BUDGET),
            'area_multiplicity': 'Each leaf area is multiplied exactly once by its uniform full-mark typed-integrand upper.'},
        'unresolved_near_target': {
            'domain': '1/10 <= |y| <= 5', 'fixed_r': '1/20',
            'coefficient': '44201/2500', 'target': '44201/20000000',
            'comparison_required': 'strict integral < target after a complete corrected near-annulus cover',
            'source_lines': 'round5/intake/rn.txt:249-255',
            'status': 'SOURCE_RESTORATION_TARGET_ONLY; historical certification held; not proved by this local cover',
            'search_limitation': 'N0 L2 transport is too wide near the inner edge; cancellation-preserving source N6 jets or another proved majorant are needed for a practical full cover.',
        },
    }


def check_report(report, root=None):
    if (type(report) is not dict or report.get('schema') != SCHEMA or
            report.get('scope') != SCOPE or report.get('domain') != RECTANGLE.as_json() or
            type(report.get('round_bits')) is not int or report['round_bits'] != 192):
        raise ValueError('candidate scope, rectangle, precision or schema mismatch')
    expected = build_report(root)
    if canonical_bytes(report) != canonical_bytes(expected):
        raise ValueError('candidate differs from reconstructed spatial cover and witnesses')
    return expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check', type=Path)
    group.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output:
            result = build_report()
            with args.output.open('xb') as stream:
                stream.write(canonical_bytes(result) + b'\n')
        else:
            result = check_report(point_check.load_report(args.check or ROOT / CANDIDATE))
        print('RN SIDE24 spatial: four full-mark cell witnesses; complete local rectangle integral <= '
              + result['composition']['simple_rational_upper']
              + '; H3 imported; full near annulus OPEN; authority=NONE')
        return 0
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as error:
        print('RN SIDE24 spatial: REJECTED: ' + str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
