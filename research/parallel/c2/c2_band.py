#!/usr/bin/env python3
"""Source-bound ONE-jet SIDE24 interval candidate. No scientific promotion.

Uses exact rational endpoints and the repository's certified interval API.
The r-variable extension is newly derived here; RN5's source has fixed r=1/20.
No frozen source code is imported or executed. Independent review: pending.
"""
from __future__ import annotations
import argparse
import ast
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.interval import Interval as I, exp
from research.rn.side24 import image_tail, kernel_derivative

BAND = (F(7071, 200000), F(1, 20))
SOURCE_SHA = {
    'sources/H5_PROMOTE.md': '7258c84a7720704e577f341252aa809888735d9f8ace1674429e004da6c6172a',
    'sources/rn_field.py': 'd9167ae821684f716fdbae11cd39eb13818422f50b6566f677d62c292e4a93c8',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rational(value):
    if type(value) not in (int, F):
        raise TypeError('exact int/Fraction required; float/bool refused')
    return F(value)


def checked_band(lo, hi):
    lo, hi = rational(lo), rational(hi)
    require(0 < lo <= hi <= F(1, 20), 'band must satisfy 0 < lo <= hi <= 1/20')
    return lo, hi


def source_binding():
    for dependency in json.loads((HERE/'DEPENDENCIES.json').read_text())['repository_python_imports']:
        data = (REPO/dependency['path']).read_bytes()
        require(len(data) == dependency['bytes'] and hashlib.sha256(data).hexdigest() == dependency['sha256'],
                'repository dependency identity mismatch: '+dependency['path'])
    for path, digest in SOURCE_SHA.items():
        require(hashlib.sha256((HERE/path).read_bytes()).hexdigest() == digest,
                'source identity mismatch: '+path)
    text = (HERE/'sources/H5_PROMOTE.md').read_text()
    require('c₂(r) = (Var f(M) − Cov(f(M),f(S)))/r²' in text,
            'c2 definition missing')
    require('0.035355' in text and '0.05' in text, 'published rung literals missing')
    syntax = ast.parse((HERE/'sources/rn_field.py').read_text())
    assignments = {node.targets[0].id: node.value for node in syntax.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    require(ast.literal_eval(assignments['JETS']) == [(0, 0), (1, 0), (0, 1)],
            'six-pin derivative indices changed')
    # Exact AST equality, not eval/exec of source. Covariance signs are separately
    # bound to source bytes and reconstructed in DERIVATION.md and geometry().
    expected = ast.parse('[(p,a) for p in (-R/2,R/2) for a in JETS] + '
                         '[(p,a) for p in (-R/2,R/2) for a in HJETS]', mode='eval').body
    require(ast.dump(assignments['RAW']) == ast.dump(expected), 'pin order changed')
    return dict(SOURCE_SHA)


def geometry(lo, hi):
    lo, hi = checked_band(lo, hi)
    jets = ((0, 0), (1, 0), (0, 1))
    coordinates = [(label, coefficient, alpha) for label, coefficient in
                   (('M', F(-1, 2)), ('S', F(1, 2))) for alpha in jets]
    pairs = []
    for i, (_, c, alpha) in enumerate(coordinates):
        for j, (_, d, beta) in enumerate(coordinates):
            displacement = (c-d)*I(lo, hi)
            pairs.append(dict(i=i, j=j, x_r_coefficient=str(c-d),
                              x_box=iv(displacement), y_box=iv(I.exact(0)),
                              derivative_orders=[alpha[k]+beta[k] for k in (0, 1)],
                              second_argument_sign=(-1)**sum(beta)))
    return {'coordinates': [{'label': label, 'x_r_coefficient': str(c),
                              'y': '0', 'derivative': list(alpha)}
                             for label, c, alpha in coordinates], 'covariance_pairs': pairs}


def central_series(r, terms=96):
    """g(r)=(1-exp(-r²/2))/r², without subtractive cancellation.

    g=1/2 sum_{n>=0} (-1)^n x^n/(n+1)!, x=r²/2<=1/800.
    Alternating, decreasing terms: consecutive partial sums bracket g.
    r=0 gives the exact continuous extension 1/2, not the original quotient.
    """
    r = rational(r)
    require(0 <= r <= F(1, 20), 'central-series radius outside 0..1/20')
    require(type(terms) is int and 2 <= terms <= 512, 'terms outside 2..512')
    x = r*r/2
    term = F(1, 2)
    partial = F(0)
    for n in range(terms):
        partial += term
        term *= -x/F(n+2)
    other = partial+term
    return I(min(partial, other), max(partial, other))


@lru_cache(maxsize=1)
def denominator():
    """Full positive normalizer. High precision resolves the omitted-image control.

    Precision concerns tightness only. All endpoints remain exact rationals.
    The first omitted pair gives a strict positive lower witness. The existing
    geometric image_tail encloses ALL j>=2, not a second finite truncation.
    """
    finite = I.exact(1)+2*exp(I.exact(-288), 610)
    omitted_first = 2*exp(I.exact(-1152), 400)
    omitted_hi = image_tail(0, F(0), prec=400)
    omitted = I(omitted_first.lo, omitted_hi)
    result = finite+omitted
    require(result.lo > 0 and omitted.lo > 0, 'positive denominator tail missing')
    return result, finite, omitted


def image_curvature(radius):
    """Uniform bound for sum_{j != 0} phi''(s+24j), 0<=s<=radius.

    Since |s+24j|>=24-radius>1, every term is positive. The first pair
    is interval-evaluated on the whole s-box. All |j|>=2 use image_tail(2).
    Integrating the factor (1-t) multiplies this bound by exactly 1/2.
    """
    radius = rational(radius)
    require(0 <= radius <= F(1, 20), 'image-curvature radius outside 0..1/20')
    s = I(0, radius)
    first_pair = I.exact(0)
    for j in (-1, 1):
        x = s+24*j
        require((x*x-1).lo > 0, 'image curvature positivity not proved')
        first_pair += (x*x-1)*exp(-x*x/2, 220)
    tail = image_tail(2, radius, prec=220)
    require(tail > 0, 'omitted image curvature tail missing')
    return first_pair+I(0, tail), tail


def enclose(lo=BAND[0], hi=BAND[1], *, omit_images=False, flip_image_sign=False):
    """Whole-band c2 enclosure; mutation switches are TEST-ONLY invalid variants."""
    lo, hi = checked_band(lo, hi)
    left, right = central_series(lo), central_series(hi)
    central = I(right.lo, left.hi)  # g decreases, by its integral representation.
    curvature, tail = image_curvature(hi)
    correction = I(curvature.lo/2, curvature.hi/2)
    z, _, _ = denominator()
    if omit_images:
        numerator = central  # INVALID, used only to exercise a containment failure.
    elif flip_image_sign:
        numerator = central+correction  # INVALID.
    else:
        numerator = central-correction
    result = (numerator/z).round_out(2048)
    require(result.lo > 0, 'c2 lower endpoint not positive')
    return result, {'central': central, 'image_integral': correction,
                    'second_derivative_tail': tail, 'normalizer': z}


def direct_point(r):
    """Independent-shaped point interval via the full periodic kernel difference."""
    r = rational(r)
    return (I.exact(1)-kernel_derivative(0, r, prec=400, bits=1024))/(r*r)


def iv(value):
    return {'lo': str(value.lo), 'hi': str(value.hi)}


def decimal_outward(value, places=15):
    scale = 10**places
    low = value.lo.numerator*scale//value.lo.denominator
    high = -((-value.hi.numerator*scale)//value.hi.denominator)
    def fmt(n):
        sign = '-' if n < 0 else ''
        n = abs(n)
        return sign+str(n//scale)+'.'+str(n%scale).zfill(places)
    return [fmt(low), fmt(high)]


def build_report():
    sources = source_binding()
    result, terms = enclose()
    z, finite_z, ztail = denominator()
    dependencies = ['research/rn/side24.py', 'research/bands/hermite_gaussian.py',
                    'research/interval/core.py', 'research/interval/transcendental.py']
    return {
        'schema_version': 1, 'target': 'PARALLEL-H5-C2-BAND-20260920-v1',
        'claim_id': '4fa1fee4-fcb4-4a53-b337-4c88ab80302c',
        'technical_state': 'AUTHOR_SIDE_CANDIDATE_REPLAYED',
        'jet': {'raw': 'Var f(M)-Cov(f(M),f(S))', 'p_J': 2,
                'normalized': 'c2(r)=(1-k(r))/r^2', 'jets_bound': 1, 'jets_required': 24},
        'band': {'lo': str(BAND[0]), 'hi': str(BAND[1]),
                 'role': 'candidate chosen adjacent published literal rungs; not an authoritative full band partition'},
        'source_sha256': sources,
        'dependencies_sha256': {p: hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in dependencies},
        'geometry': geometry(*BAND),
        'enclosure': iv(result), 'enclosure_outward_decimal': decimal_outward(result),
        'width': str(result.width()), 'width_upper': str(F(156186, 10**9)),
        'decomposition': {k: iv(v) if isinstance(v, I) else str(v) for k, v in terms.items()},
        'denominator_finite': iv(finite_z), 'denominator_omitted': iv(ztail),
        'arithmetic': 'exact Fraction + certified interval exp; no floating-point exploration',
        'mathematical_argument': 'DERIVATION.md; author reconstruction, independent review pending',
        'scope': 'normalized SIDE24 field, fixed axis, one scalar jet, every r in the exact named band',
        'scientific_promotions': 0, 'independence_credit': 0, 'obligation_closed': False,
        'modulus_comparison': 'INSUFFICIENT_DATA: authoritative whole-band modulus not supplied',
        'does_not_establish': [
            'The full 24-jet definitions, powers, or certified G12-to-cover map',
            'An authoritative all-r band partition or a modulus acceptance result',
            'H5 excluded certificate restoration, H3 replay, RN spatial cover, or theorem closure',
            'Independent acceptance, organizational independence, or public publication'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path, default=HERE/'candidate.json',
                        help='recompute and compare an exact stored candidate')
    args = parser.parse_args()
    report = build_report()
    require(F(report['width']) < F(report['width_upper']), 'width target not met')
    if args.verify:
        from tools.rn_side24_check import load_report
        from research.rn.certificate import canonical_bytes
        require(canonical_bytes(load_report(args.verify)) == canonical_bytes(report),
                'stored candidate differs from exact replay')
    if args.output:
        with args.output.open('xb') as stream:
            stream.write((json.dumps(report, indent=2, ensure_ascii=False)+'\n').encode())
    print('PASS: one c2 band '+str(report['enclosure_outward_decimal'])+
          '; width < '+report['width_upper']+'; 0 promotions; independent review pending')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, TypeError, ArithmeticError, OSError) as error:
        print('FAIL: '+str(error), file=sys.stderr)
        raise SystemExit(1)
