#!/usr/bin/env python3
"""Replay the fixed SIDE24 point-law candidate and its three moment witnesses.

This author-side candidate bounds a conditional determinant factor at one
spatial point. It supplies no spatial cover, density/window factor, H3
normalizer, canonical promotion or organizational independence.

SIDE24 has no new source-of-truth carrier after 2026-08-06. Absent objects
stay ABSENT. Quarantine is not a source of truth. A green run is engineering
hygiene: inventable_attempt_accepted stays false, and OBL-H5-JETMOD stays OPEN.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.interval import Interval, sqrt
from research.rn.certificate import canonical_bytes, produce, verify_bytes
from tools.rn_certificate import context_data

DIRECTORY = 'drive/mirrors/2026-09-17_RN5_MOMENT_REPAIR_AND_REVIEW_ERRATUM/'
SOURCE = DIRECTORY + 'RN5_NEAR_MOMENT_REPAIR.md'
ARCHIVE = DIRECTORY + 'RN5_REPAIR_AND_ERRATUM_BUNDLE.zip'
SOURCES = {
    SOURCE: (13725, 'ac89f60b8206bfe011e6c2bc653e7acb39bc83a55c2a2c8fd1e17fd70c6c0383'),
    ARCHIVE: (140170, '28c1c385406452a706f11aea1646a87585b002841005d57febdf32222c368c6e'),
}
MEMBERS = {
    'closure_round2/rn_field.py': (16701, 'd9167ae821684f716fdbae11cd39eb13818422f50b6566f677d62c292e4a93c8'),
    'round5/near_moments.py': (8673, '03c6e35c426eee8ed91139b93f0ada4e96c7293e5afb4176677850596fbfca6e'),
    'round5/verify_round5.py': (5074, '614daf280f5037757d261281b73ea66edb350da4619066ee56ea5a4216b7737e'),
}
CANDIDATE = 'research/rn/candidates/side24_point_20260920_v1.json'
SCHEMA = 'RN_SIDE24_POINT_MOMENTS_V1'
MAX_REPORT_BYTES = 4 * 1024 * 1024
SCOPE = {
    'authority': 'NONE', 'independence_credit': 0,
    'scientific_status_changed': False, 'original_prize_closed': False,
    'field_certified': False,
    'spatial_cover_certified': False, 'all_small_r_certified': False,
    'density_window_factor_included': False, 'h3_normalizer_included': False,
    'statement': 'Uniform mark bound for the conditional determinant factor at y=(1,1), r=1/20, SIDE24.',
    'review': 'AUTHOR CANDIDATE; source construction and algebra require technical review.',
}


def identity(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def checked_sources(root):
    """Read exact pinned sources as data; never import or execute ZIP members."""
    out = {}
    for path, (size, digest) in SOURCES.items():
        with (root / path).open('rb') as stream:
            data = stream.read(size + 1)
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('source identity mismatch: ' + path)
        out[path] = identity(data)
    with zipfile.ZipFile(root / ARCHIVE) as archive:
        for path, (size, digest) in MEMBERS.items():
            if len([item for item in archive.infolist() if item.filename == path]) != 1:
                raise ValueError('nonunique source member: ' + path)
            if archive.getinfo(path).file_size != size:
                raise ValueError('source member size mismatch: ' + path)
            data = archive.read(path)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError('source member digest mismatch: ' + path)
            out[ARCHIVE + '::' + path] = identity(data)
    return out


def endpoints(value):
    return [str(value.lo), str(value.hi)]


def build_report(root=None):
    root = ROOT if root is None else Path(root)
    sources = checked_sources(root)
    from research.rn.side24 import point_laws
    derived = point_laws(point=(F(1), F(1)), bits=192)
    certificates = {}
    caps = {}
    for name, degree in (('M', 4), ('S', 4), ('y', 2)):
        law = derived['laws'][name]
        certificate = produce(law, degree=degree, depth=2)
        result = verify_bytes(canonical_bytes(certificate),
                              source_bytes=(root / SOURCE).read_bytes(),
                              expected_context=context_data(law))
        if not result['certificate_valid'] or not result['requested_checks_passed']:
            raise ValueError('moment witness replay failed: ' + name + ': ' + result['reason'])
        certificates[name] = certificate
        # Use the certificate's declared upper, whose value is checked by replay.
        caps[name] = F(certificate['proof']['upper'])
    fourth_power = caps['M'] * caps['S'] * caps['y'] ** 2
    holder = sqrt(sqrt(Interval.exact(fourth_power), 70), 70).round_out(192)
    if holder.lo < 0 or holder.hi ** 4 < fourth_power:
        raise ValueError('Holder root enclosure failed')
    scaled = holder.hi * 10**9
    display_upper = F(-(-scaled.numerator // scaled.denominator), 10**9)
    return {
        'schema': SCHEMA, 'point': ['1', '1'], 'round_bits': 192,
        'sources': sources, 'scope': dict(SCOPE),
        'field_specification': {
            'normalization': derived['source_binding']['normalization'],
            'conditioning_order': list(derived['conditioning_order']),
            'target_order': list(derived['target_order']),
            'conditioning_intercept': list(map(str, derived['conditioning_intercept'])),
            'conditioning_slope': list(map(str, derived['conditioning_slope'])),
            'centered_mark_domain': ['-1/96000', '1/96000'],
            'raw_covariance': [[endpoints(x) for x in row] for row in derived['raw_covariance']],
            'joint_psd_basis': 'Positive Fourier weights of the normalized periodized Gaussian; derivative Gram representation.',
        },
        'conditioning_pivots': list(map(endpoints, derived['pivot_intervals'])),
        'marginal_pivots': {key: list(map(endpoints, values))
                            for key, values in derived['marginal_pivot_intervals'].items()},
        'moment_certificates': certificates,
        'conditional_holder': {
            'powers': {'M': 4, 'S': 4, 'y': 2},
            'moment_caps': {key: str(value) for key, value in caps.items()},
            'fourth_power_upper': str(fourth_power),
            'root_enclosure': endpoints(holder), 'upper': str(holder.hi),
            'simple_rational_upper': str(display_upper),
            'event': 'Any intersection of the three Hessian type indicators; indicators are dropped.',
            'dependence': 'Holder requires no independence between the three Hessian blocks.',
        },
    }


def load_report(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_REPORT_BYTES + 1)
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError('report byte limit exceeded')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('floating/nonfinite JSON numbers are unsupported')
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_float=invalid,
                          parse_constant=invalid)
    except RecursionError:
        raise ValueError('JSON nesting limit exceeded') from None


def check_report(report, root=None):
    if (type(report) is not dict or report.get('schema') != SCHEMA or
            report.get('scope') != SCOPE or report.get('point') != ['1', '1'] or
            type(report.get('round_bits')) is not int or report['round_bits'] != 192):
        raise ValueError('candidate scope, point, precision or schema mismatch')
    expected = build_report(root)
    if canonical_bytes(report) != canonical_bytes(expected):
        raise ValueError('candidate differs from reconstructed SIDE24 point law or moment witnesses')
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
            result = check_report(load_report(args.check or ROOT / CANDIDATE))
        print('RN SIDE24: 9 positive conditioning pivots; M4/S4/y2 witnesses replayed; '
              'one exact spatial point and complete mark interval; authority=NONE')
        return 0
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as error:
        print('RN SIDE24: REJECTED: ' + str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
