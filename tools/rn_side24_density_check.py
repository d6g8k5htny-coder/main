#!/usr/bin/env python3
"""Replay the fixed SIDE24 point integrand with an explicitly imported H3 floor.

Exact interval arithmetic supplies the density/window numerator. The H3 floor
is a pinned hypothesis, not reproved here. No spatial or canonical closure.
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

from research.interval import Interval
from research.rn.certificate import canonical_bytes
from tools import rn_side24_check as point_check

SCHEMA = 'RN_SIDE24_POINT_DENSITY_V1'
CANDIDATE = 'research/rn/candidates/side24_density_20260920_v1.json'
H3_MEMBER = 'intake/rn_source/K3_SIDE24_LB/UPPER2D/H3_closure/H3_RUNG_FLOOR.md'
H3_IDENTITY = (7003, '6347275d86c56842b719b36180e535bc1793d6995ddb68464db2960820440dfa')
PIN_TRANSFORM = 'engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/H2_foundations/pin_transform.py'
PIN_IDENTITY = (17623, 'c6988ac7fcfa32dcc03c1374e13fb4c26af1c12c25cb824a315e6dce8991fd41')
H3_FLOOR = F('0.0077592917375327855')
SCOPE = {
    'authority': 'NONE', 'independence_credit': 0,
    'scientific_status_changed': False, 'original_prize_closed': False,
    'spatial_cover_certified': False, 'all_small_r_certified': False,
    'density_window_factor_included': True, 'h3_floor_reproved': False,
    'h3_floor_status': 'EXPLICIT_IMPORTED_HYPOTHESIS',
    'statement': 'Pointwise RN integrand upper at y=(1,1), r=1/20, SIDE24, conditional on the pinned H3 floor.',
    'review': 'AUTHOR CANDIDATE; mathematical acceptance and required independence remain separate.',
}


def serialize(value):
    if isinstance(value, Interval):
        return [str(value.lo), str(value.hi)]
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    if value is None or type(value) in (str, int, bool):
        return value
    raise ValueError('unsupported report value: ' + type(value).__name__)


def checked_imports(root):
    """Additional source custody; reading code bytes does not execute them."""
    with zipfile.ZipFile(root / point_check.ARCHIVE) as archive:
        matches = [item for item in archive.infolist() if item.filename == H3_MEMBER]
        if len(matches) != 1 or matches[0].file_size != H3_IDENTITY[0]:
            raise ValueError('H3 source member identity mismatch')
        raw = archive.read(H3_MEMBER)
    if hashlib.sha256(raw).hexdigest() != H3_IDENTITY[1]:
        raise ValueError('H3 source digest mismatch')
    with (root / PIN_TRANSFORM).open('rb') as stream:
        pin = stream.read(PIN_IDENTITY[0] + 1)
    if len(pin) != PIN_IDENTITY[0] or hashlib.sha256(pin).hexdigest() != PIN_IDENTITY[1]:
        raise ValueError('pin-transform source identity mismatch')
    return {
        point_check.ARCHIVE + '::' + H3_MEMBER: point_check.identity(raw),
        PIN_TRANSFORM: point_check.identity(pin),
    }


def build_report(root=None):
    root = ROOT if root is None else Path(root)
    # Reconstruct every moment certificate from the actual pinned field inputs.
    point = point_check.build_report(root)
    sources = {**point['sources'], **checked_imports(root)}
    from research.rn.side24_density import density_window
    density = density_window(point=(F(1), F(1)), bits=192)
    holder = F(point['conditional_holder']['upper'])
    mass = density['density_mass_upper']
    if not isinstance(mass, F) or mass < 0 or holder < 0:
        raise ValueError('nonnegative exact factors required')
    numerator = holder * mass
    normalized = numerator / H3_FLOOR
    scale = 10**12
    scaled = normalized * scale
    simple = F(-(-scaled.numerator // scaled.denominator), scale)
    pivots = [*density['six_pin_pivots'], *density['jet_pivots'],
              *density['gradient_pivots']]
    if not pivots or any(p.lo <= 0 for p in pivots):
        raise ValueError('positive pivot admission required')
    return {
        'schema': SCHEMA, 'point': ['1', '1'], 'round_bits': 192,
        'scope': dict(SCOPE), 'sources': sources,
        'moment_reconstruction': {
            'canonical_report_sha256': hashlib.sha256(canonical_bytes(point)).hexdigest(),
            'holder_upper': str(holder), 'powers': {'M': 4, 'S': 4, 'y': 2},
        },
        'density_window': serialize(density),
        'composition': {
            'numerator_upper': str(numerator), 'imported_h3_floor': str(H3_FLOOR),
            'normalized_upper': str(normalized), 'simple_rational_upper': str(simple),
            'formula': 'Holder_upper * p_grad_zero_upper * window_mass_upper / imported_H3_floor',
            'height_window_conditioning': 'six fixed pins AND gradient(y)=0',
        },
        'admission_margins': {
            'minimum_pivot_lower': str(min(p.lo for p in pivots)),
            'height_variance_lower': str(density['height_variance'].lo),
            'window_cap_below_one': str(F(1) - density['window_mass_upper']),
            'interpretation': 'Exact admission margins only; no sampled counterexample search or spatial robustness claim.',
        },
    }


def check_report(report, root=None):
    if (type(report) is not dict or report.get('schema') != SCHEMA or
            report.get('scope') != SCOPE or report.get('point') != ['1', '1'] or
            type(report.get('round_bits')) is not int or report['round_bits'] != 192):
        raise ValueError('candidate scope, point, precision or schema mismatch')
    expected = build_report(root)
    if canonical_bytes(report) != canonical_bytes(expected):
        raise ValueError('candidate differs from reconstructed density/window integrand')
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
        print('RN SIDE24 density: exact six-pin density/window replay; point integrand <= ' +
              result['composition']['simple_rational_upper'] +
              '; H3 floor imported; no spatial cover; authority=NONE')
        return 0
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as error:
        print('RN SIDE24 density: REJECTED: ' + str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
