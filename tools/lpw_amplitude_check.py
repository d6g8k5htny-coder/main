#!/usr/bin/env python3
"""Check the source-bound LPW amplitude candidate; no scientific status moves."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.lpw.amplitude import SOURCE, COMPANION, verify_bytes

MIRROR = ROOT/'drive/mirrors/02_RESEARCH_CARRY_FORWARD_CANON/LPW — RAW ARCHIVE INTAKE R05'
CANDIDATE = ROOT/'research/lpw/candidates/amplitude_certificate_v1.json'
SOURCE_PATH = MIRROR/'03_RAYLEIGH_REPAIR_AND_INTERVAL_CERTIFICATE.md'
COMPANION_PATH = MIRROR/'checks/interval_repair.py'


def source_payloads():
    return {SOURCE['id']: SOURCE_PATH.read_bytes(), COMPANION['id']: COMPANION_PATH.read_bytes()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--certificate', type=Path, default=CANDIDATE)
    args = parser.parse_args(argv)
    try:
        with args.certificate.open('rb') as stream:
            data = stream.read(131073)
        result = verify_bytes(data, source_payloads())
    except OSError as error:
        print(f'LPW amplitude: FAIL: {error}')
        return 1
    if not result['certificate_valid']:
        print('LPW amplitude: FAIL: '+result['reason'])
        return 1
    print('LPW amplitude: exact parent moments rho^4=8, L1^4=12+32/pi; norm/phase/radius checks and 2 source hashes match; analytic imports retained; authority=NONE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
