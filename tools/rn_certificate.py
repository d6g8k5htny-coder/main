#!/usr/bin/env python3
"""Produce/replay bounded RN algebra certificates; no scientific status moves."""
import argparse
from dataclasses import asdict
from fractions import Fraction as F
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.operations.rn_applicability import Context, RN5_ID, RN5_SHA
from engine.operations.rn_family_applicability import FamilyLaw
from research.interval import Interval as I
from research.rn.certificate import (MAX_BYTES, MAX_OPERATIONS, ResourceLimit,
                                     canonical_bytes, load_bytes, produce, verify_bytes)

SOURCE = ROOT/'drive/mirrors/2026-09-17_RN5_MOMENT_REPAIR_AND_REVIEW_ERRATUM/RN5_NEAR_MOMENT_REPAIR.md'
CANDIDATES = ROOT/'research/rn/candidates'
PILOTS = ('variance2', 'variance4', 'affine4')


def pilot(name):
    """Explicit synthetic fixtures, not data imported from an identified field."""
    if name not in PILOTS:
        raise ValueError('unknown certificate pilot')
    c = Context(RN5_ID, RN5_SHA, 2, ('xx', 'yy', 'xy'),
                'exact synthetic Hessian units', 'certificate-pilot-'+name,
                (F(-1), F(1)))
    if name.startswith('variance'):
        law = FamilyLaw(c, (0, 0, 0), (0, 0, 0),
                        ((1, 0, 0), (0, 1, 0), (0, 0, I(0, 1))),
                        c.conditioned_law, c.conditioned_law)
        return law, int(name[-1])
    law = FamilyLaw(c, (I(-1, F(-3, 4)), I(-1), I(0)),
                    (I(F(3, 4), 1), I(-1), I(0)),
                    ((I(1, 2), I(F(1, 8), F(1, 4)), 0),
                     (I(F(1, 8), F(1, 4)), I(1, 2), 0), (0, 0, I(F(1, 4), F(1, 2)))),
                    c.conditioned_law, c.conditioned_law)
    return law, 4


def context_data(law):
    # Use the same public representation as the certificate context.
    c = asdict(law.context)
    c['order'] = list(c['order'])
    c['domain'] = list(map(str, c['domain']))
    return c


def read_bounded(path, limit=MAX_BYTES):
    with Path(path).open('rb') as stream:
        return stream.read(limit+1)


def check_candidates():
    expected_names = {f'{name}_certificate_v1.json' for name in PILOTS}
    actual_names = {p.name for p in CANDIDATES.glob('*certificate*.json')}
    if actual_names != expected_names:
        print('RN certificates: FAIL candidate set differs from explicit pilot manifest')
        return 1
    source = read_bounded(SOURCE)
    total = 0
    for name in PILOTS:
        law, degree = pilot(name)
        path = CANDIDATES/f'{name}_certificate_v1.json'
        data = read_bounded(path)
        result = verify_bytes(data, source_bytes=source, expected_context=context_data(law))
        if not result['certificate_valid'] or not result['requested_checks_passed']:
            print(f"RN certificates: FAIL {name}: {result['outcome']}: {result['reason']}")
            return 1
        # The committed fixtures are pinned by complete expected mathematical
        # inputs, not only their context labels or the certificates' own claims.
        expected = produce(law, degree=degree, depth=2)
        if canonical_bytes(load_bytes(data)) != canonical_bytes(expected):
            print(f'RN certificates: FAIL {name}: explicit pilot input/output mismatch')
            return 1
        total += result['operations']
    print(f'RN certificates: {len(PILOTS)} exact replays; source/context/pilot inputs match; operations={total}; authority=NONE')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check-candidates')
    create = commands.add_parser('produce')
    create.add_argument('pilot', choices=PILOTS)
    create.add_argument('--output', type=Path, required=True)
    check = commands.add_parser('verify')
    check.add_argument('certificate', type=Path)
    check.add_argument('--source', type=Path)
    check.add_argument('--expected-context', type=Path)
    check.add_argument('--max-operations', type=int, default=MAX_OPERATIONS)
    args = parser.parse_args(argv)
    try:
        if args.command == 'check-candidates':
            return check_candidates()
        if args.command == 'produce':
            law, degree = pilot(args.pilot)
            certificate = produce(law, degree=degree)
            # Refuse to overwrite an existing artifact; issue a successor.
            with args.output.open('xb') as stream:
                stream.write(canonical_bytes(certificate)+b'\n')
            print(f'RN certificate written: {args.output}; authority=NONE')
            return 0
        result = verify_bytes(read_bounded(args.certificate),
                              source_bytes=read_bounded(args.source) if args.source else None,
                              expected_context=load_bytes(read_bounded(args.expected_context)) if args.expected_context else None,
                              max_operations=args.max_operations)
        print(canonical_bytes(result).decode('ascii'))
        return 0 if result['certificate_valid'] and result['requested_checks_passed'] else 2 if result['outcome'] == 'INCONCLUSIVE' else 1
    except ResourceLimit as error:
        print(f'RN certificates: INCONCLUSIVE: {error}')
        return 2
    except (OSError, ValueError, TypeError) as error:
        print(f'RN certificates: REJECTED: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
