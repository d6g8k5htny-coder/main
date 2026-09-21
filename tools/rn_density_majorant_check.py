"""Conditional majorant CLI: conditional exact arithmetic, never a spatial certification."""
import argparse
from fractions import Fraction
import re

from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.rn.density_majorant import density_moment_majorant


def rational(text):
    if len(text) > 2500 or not re.fullmatch(r'-?(0|[1-9][0-9]*)(/[1-9][0-9]*)?', text):
        raise argparse.ArgumentTypeError('bounded integer or rational p/q required')
    value = Fraction(text)
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 4096:
        raise argparse.ArgumentTypeError('rational exceeds resource limit')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', required=True)
    parser.add_argument('--determinant-lower', required=True, type=rational)
    parser.add_argument('--mahalanobis-lower', required=True, type=rational)
    parser.add_argument('--jacobian-upper', type=rational, default=1)
    parser.add_argument('--bits', type=int, default=192)
    parser.add_argument('--ceiling', type=rational,
                        help='require the proved majorant upper endpoint <= this ceiling')
    args = parser.parse_args()
    try:
        result = density_moment_majorant(args.determinant_lower, args.mahalanobis_lower,
            jacobian_upper=args.jacobian_upper, bits=args.bits, repo_root=args.repo_root)
        if args.ceiling is not None and result['integrand_upper'] > args.ceiling:
            raise ValueError('claimed ceiling is below the proved sufficient upper bound')
    except (ValueError, TypeError, ArithmeticError, OSError) as error:
        print('REJECTED:', error)
        return 1
    print('CONDITIONAL_ARITHMETIC_PASS source_custody=PASS spatial_premises=UNCHECKED '
          'field_certified=false authority=NONE integrand_upper=' + str(result['integrand_upper']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
