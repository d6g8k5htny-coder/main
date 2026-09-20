#!/usr/bin/env python3
"""Exact author-side fixed-r H3 floor reconstruction, not a status promotion.

Reads existing repository/source bytes. Writes only its explicit report path.
No historical source is executed. Standard library and repo interval code only.
"""
from fractions import Fraction as F
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
from functools import lru_cache
import argparse
import hashlib
import json
import re
import sys

HERE = Path(__file__).resolve().parent
early = argparse.ArgumentParser(add_help=False)
early.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
REPO = early.parse_known_args()[0].repo.resolve()
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True
from research.interval import Interval as I, sqrt, Phi, normal_pdf
from research.rn import side24
from research.rn.conditioning import condition_gaussian, interval_ldlt
from tools.drive_coverage import Store, DEFAULT

BITS = 256
PREC = 104
R = F(1, 20)
HEIGHT = F(6, 5)
IMPORTED = F('0.0077592917375327855')
ZERO = (0,) * 6
BASE = 'K3_SIDE24_LB/UPPER2D/H3_closure/'
SCRIPT = BASE + 'h3_rung_floor.py'
SCRIPT_SHA = '094d382451ff37e8d3123e7203c743f359cb0af1c2c12cf336adf74fc0b25d8f'
ARCHIVE_SHA = 'a2136bc033f349382f9896896347da7a6dabde3334103276ad04db9205aa2b5b'
H3_MEMBER = 'intake/rn_source/' + BASE + 'H3_RUNG_FLOOR.md'
H3_SHA = '6347275d86c56842b719b36180e535bc1793d6995ddb68464db2960820440dfa'
LOW = tuple(map(F, (-7, -8, -6, -8))) + (F(-12, 5), F(-8))
HIGH = (F(18, 25), F(2), F(6), F(8), F(12, 5), F(8))
CHECKS = []


def require(condition, message):
    if not condition:
        raise ValueError(message)
    CHECKS.append(message)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def rnd(x):
    return x.round_out(BITS)


def enc(x):
    if isinstance(x, I):
        return {'lo': str(x.lo), 'hi': str(x.hi)}
    if isinstance(x, F):
        return str(x)
    if isinstance(x, dict):
        return {str(k): enc(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [enc(v) for v in x]
    return x


def custody():
    exclusions = json.loads((REPO/'quarantine/EXCLUSIONS.json').read_text())['exclusions']
    require(not any(e.get('member_path') in (SCRIPT, H3_MEMBER, BASE+'H3_RUNG_FLOOR.md')
                    or e.get('payload_sha256') in (SCRIPT_SHA, H3_SHA) for e in exclusions),
            'target H3 members are not logically excluded')
    coverage = [json.loads(l) for l in (REPO/DEFAULT/'coverage.jsonl').read_text().splitlines()]
    row = next(x for x in coverage if x['id'] == '1vSI-evINWskhXVyiZ0slt-rLT74sPpXH')
    require(row['sha256'] == ARCHIVE_SHA and row['status'] == 'EXACT_SOURCE_BYTES',
            'active archive source identity and eligible custody')
    archive = b''.join(Store(REPO, DEFAULT).object_chunks(ARCHIVE_SHA))
    with ZipFile(BytesIO(archive)) as z:
        require(z.namelist().count(SCRIPT) == 1, 'historical script unique member')
        source = z.read(SCRIPT)
    require(len(source) == 16960 and sha(source) == SCRIPT_SHA, 'historical script exact bytes')
    rn5 = side24.source_binding()
    with ZipFile(side24.ARCHIVE) as z:
        body = z.read(H3_MEMBER)
    require(len(body) == 7003 and sha(body) == H3_SHA, 'H3 proof member exact bytes')
    floors = re.findall(r'Z_\{0\.05\} ∈ \[\s*([\d.e+-]+)\s*,', body.decode())
    require(len(floors) == 1 and F(floors[0]) == IMPORTED, 'exact imported decimal parsed from unique source rung display')
    return {'historical_archive': {'drive_id': row['id'], 'sha256': ARCHIVE_SHA, 'bytes': len(archive)},
            'historical_script': {'member': SCRIPT, 'sha256': sha(source), 'bytes': len(source), 'executed': False},
            'proof': {'member': H3_MEMBER, 'sha256': sha(body), 'bytes': len(body)}, 'rn5': rn5}


def dot(xs, ys):
    out = I.exact(0)
    for x, y in zip(xs, ys):
        out = rnd(out + rnd(x*y))
    return out


def functionals():
    m, s = ((-R/2, F(0)), (R/2, F(0)))
    raw = tuple((p, jet) for p in (m, s) for jet in ((0, 0), (1, 0), (0, 1)))
    transform = ((1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0),
                 (0, 0, 1, 0, 0, 0), (0, -1/R, 0, 0, 1/R, 0),
                 (0, 0, -1/R, 0, 0, 1/R), (-1/R**3, -1/(2*R**2), 0, 1/R**3, -1/(2*R**2), 0))
    pins = [[(*raw[j], F(c)) for j, c in enumerate(row) if c] for row in transform]
    soft = [[(p, jet, scale)] for jet, scale in (((0, 2), F(1)), ((2, 0), 1/R), ((1, 1), 1/R)) for p in (m, s)]
    # The transformation is invertible: recover raw fS,fxS,fyS from V and first three pins.
    raw_values = (HEIGHT, F(0), F(0), HEIGHT-R**3/6, F(0), F(0))
    values = tuple(sum(F(c)*x for c, x in zip(row, raw_values)) for row in transform)
    require(values == (HEIGHT, F(0), F(0), F(0), F(0), F(-1, 6)), 'six-pin transformed values match exact RN law')
    recovered = (values[0], values[1], values[2],
                 R**3*values[5]+values[0]+R*(R*values[3]+2*values[1])/2,
                 R*values[3]+values[1], R*values[4]+values[2])
    require(recovered == raw_values, 'six-pin invertible transformation exact roundtrip')
    return pins, soft, values, transform


@lru_cache(maxsize=256)
def cov_term(p, alpha, q, beta):
    return side24.covariance(p, alpha, q, beta, prec=PREC, bits=BITS)


def lcov(a, b):
    result = I.exact(0)
    for p, alpha, scale_a in a:
        for q, beta, scale_b in b:
            result = rnd(result + rnd(cov_term(p, alpha, q, beta)*scale_a*scale_b))
    return result


def gram(items):
    out = [[None]*len(items) for _ in items]
    for i, a in enumerate(items):
        for j in range(i+1):
            out[i][j] = out[j][i] = lcov(a, items[j])
    return out


def law(pin_last=F(-1, 6)):
    pins, soft, values, transform = functionals()
    all_gram = gram(pins+soft)
    a = [r[:6] for r in all_gram[:6]]
    c = [r[:6] for r in all_gram[6:]]
    b = [r[6:] for r in all_gram[6:]]
    result = condition_gaussian(a, c, b, (*values[:5], pin_last), (0,)*6,
        conditioning_order=tuple('V'+str(i) for i in range(6)),
        target_order=('qM', 'qS', 'aM', 'aS', 'bM', 'bS'),
        covariance_law='SIDE24:normalized:r=1/20:six-pins', mean_law='SIDE24:normalized:r=1/20:six-pins',
        conditioning_law='SIDE24:normalized:r=1/20:six-pins', mark_domain=(0, 0), round_bits=BITS)
    factor = interval_ldlt(result.covariance, round_bits=BITS)
    t = [[rnd(factor.lower[i][j]*sqrt(factor.pivots[j], PREC)) if j <= i else I.exact(0)
          for j in range(6)] for i in range(6)]
    require(all(d.lo > 0 for d in (*result.pivots, *factor.pivots)), 'six pin and six soft LDL pivots strictly positive')
    return {'mu': result.intercept, 'covariance': result.covariance, 'T': t,
            'pin_pivots': result.pivots, 'soft_pivots': factor.pivots, 'pin_transform': transform}


def safe_box(law, high=HIGH):
    mu, t = law['mu'], law['T']
    ranges = [rnd(mu[i]+dot(t[i], [I(a, b) for a, b in zip(LOW, high)])) for i in range(6)]
    qm, qs, am, ass, bm, bs = ranges
    dm = rnd(am*qm-R*bm**2)
    ds = rnd(ass*qs-R*bs**2)
    require(qm.hi < 0 and qs.hi < 0 and am.hi < 0 and ass.lo > 0, 'box has strict Hessian diagonal signs')
    require(dm.lo > 0 and ds.hi < 0, 'box is type-safe: D_M>0 and D_S<0 everywhere')
    return {'soft_ranges': ranges, 'DM': dm, 'DS': ds, 'low': LOW, 'high': high}


@lru_cache(maxsize=64)
def truncated(a, b):
    require(type(a) is F and type(b) is F and a < b, 'exact ordered truncated-moment endpoints')
    pa, pb = rnd(normal_pdf(I.exact(a), PREC)), rnd(normal_pdf(I.exact(b), PREC))
    values = [rnd(Phi(I.exact(b), PREC)-Phi(I.exact(a), PREC)), rnd(pa-pb)]
    for n in range(2, 5):
        values.append(rnd(a**(n-1)*pa-b**(n-1)*pb+(n-1)*values[n-2]))
    require(values[0].lo >= 0 and values[0].hi <= 1, 'truncated probability lies in [0,1]')
    return tuple(values)


def affine(mu, row):
    p = {ZERO: mu}
    for i, c in enumerate(row):
        if c.lo != 0 or c.hi != 0:
            e = [0]*6; e[i] = 1; p[tuple(e)] = c
    return p


def add(p, q, scale=F(1)):
    out = dict(p)
    for e, c in q.items():
        out[e] = rnd(out.get(e, I.exact(0))+scale*c)
    return out


def multiply(p, q):
    out = {}
    for e, x in p.items():
        for f, y in q.items():
            g = tuple(a+b for a, b in zip(e, f))
            out[g] = rnd(out.get(g, I.exact(0))+rnd(x*y))
    require(len(out) <= 210 and max(map(sum, out)) <= 4, 'polynomial resource and degree bounds')
    return out


def det_polynomial(law):
    qm, qs, am, ass, bm, bs = [affine(m, t) for m, t in zip(law['mu'], law['T'])]
    dm = add(multiply(am, qm), multiply(bm, bm), -R)
    ds = add(multiply(ass, qs), multiply(bs, bs), -R)
    return {e: -c for e, c in multiply(dm, ds).items()}


def integral(poly, high):
    moments = [truncated(a, b) for a, b in zip(LOW, high)]
    result = I.exact(0)
    for e, c in sorted(poly.items()):
        term = c
        for i, degree in enumerate(e):
            term = rnd(term*moments[i][degree])
        result = rnd(result+term)
    return rnd(result*R**2)


def fail_control(name, action):
    try:
        action()
    except (ValueError, TypeError, ZeroDivisionError):
        return {'name': name, 'rejected': True}
    raise ValueError('negative control accepted: '+name)


def scale_identity(scale):
    """Independent exact formal-polynomial determinant scaling, no intervals."""
    def monomial(coefficient, *indices):
        e = [0]*6
        for i in indices:
            e[i] += 1
        return {tuple(e): F(coefficient)}
    def plus(p, q):
        out = dict(p)
        for e, c in q.items():
            out[e] = out.get(e, F(0))+c
        return {e:c for e,c in out.items() if c}
    def times(p, q):
        out = {}
        for e, c in p.items():
            for f, d in q.items():
                g = tuple(x+y for x,y in zip(e,f))
                out[g] = out.get(g,F(0))+c*d
        return {e:c for e,c in out.items() if c}
    dm = plus(monomial(1, 0, 2), monomial(-R, 4, 4))
    ds = plus(monomial(1, 1, 3), monomial(-R, 5, 5))
    raw_m = plus(times(monomial(R, 2), monomial(1, 0)),
                 {e:-c for e,c in times(monomial(R,4),monomial(R,4)).items()})
    raw_s = plus(times(monomial(R, 3), monomial(1, 1)),
                 {e:-c for e,c in times(monomial(R,5),monomial(R,5)).items()})
    raw_product = times(raw_m, raw_s)
    scaled = {e:scale*c for e,c in times(dm,ds).items()}
    require(raw_product == scaled, 'exact raw Hessian determinant product equals r² D_M D_S coefficient by coefficient')
    return {'raw_product': raw_product, 'scaled_product': scaled, 'scale': scale}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mutation', choices=['wrong_window', 'unsafe_box', 'wrong_sign', 'oversized_floor', 'wrong_scale'])
    args = parser.parse_args()
    output = args.output.resolve()
    require(REPO not in output.parents and output != REPO, 'output outside read-only repository')
    require(not output.exists() and output.parent.is_dir(), 'fresh output file in existing output directory required')
    source = custody()
    files = sorted({str(Path(m.__file__).resolve().relative_to(REPO)) for m in tuple(sys.modules.values())
                    if getattr(m, '__file__', None) and REPO in Path(m.__file__).resolve().parents
                    and Path(m.__file__).suffix == '.py'})
    before = {p: sha((REPO/p).read_bytes()) for p in files}
    identity = scale_identity(F(1) if args.mutation == 'wrong_scale' else R**2)
    model = law(F(-1, 12) if args.mutation == 'wrong_window' else F(-1, 6))
    original_box = safe_box(model, (F(4, 5), *HIGH[1:]) if args.mutation == 'unsafe_box' else HIGH)
    poly = det_polynomial(model)
    if args.mutation == 'wrong_sign':
        poly = {e: -c for e, c in poly.items()}
    original = integral(poly, HIGH)
    # A numbered successor box, not a silent alteration of the frozen source.
    successor_high = (F(721, 1000), *HIGH[1:])
    successor_box = safe_box(model, successor_high)
    successor = integral(poly, successor_high)
    target = F(1, 100) if args.mutation == 'oversized_floor' else IMPORTED
    require(successor.lo > target, 'successor box proves imported exact floor with strict rational slack')
    require(successor.lo > original.hi, 'successor box integral strictly improves original box integral')
    require(original.lo > IMPORTED, 'original frozen source box itself proves exact imported floor')
    new_floor = F('0.007759315')
    require(successor.lo > new_floor, 'new separately labeled conservative successor floor proved')
    controls = [fail_control('wrong six-pin height drop', lambda: safe_box(law(F(-1, 12)))),
                fail_control('unsafe enlarged U1 box', lambda: safe_box(model, (F(4, 5), *HIGH[1:]))),
                fail_control('oversized floor 1/100', lambda: require(successor.lo > F(1, 100), 'oversized floor')),
                fail_control('determinant product wrong sign', lambda: require((-successor).lo > IMPORTED, 'wrong sign')),
                fail_control('omitted r² determinant-product scale', lambda: scale_identity(F(1)))]
    # Analytic recurrence identities on symmetric box; these catch sign/index faults.
    symmetric = truncated(F(-2), F(2))
    require(F(0) in symmetric[1] and F(0) in symmetric[3], 'odd moments vanish on symmetric interval')
    density2 = normal_pdf(I.exact(2), PREC)
    require(symmetric[2].intersect(symmetric[0]-4*density2) is not None, 'second moment integration by parts identity')
    require(symmetric[4].intersect(3*symmetric[2]-16*density2) is not None, 'fourth moment integration by parts identity')
    after = {p: sha((REPO/p).read_bytes()) for p in files}
    require(before == after, 'all imported computational source files unchanged during run')
    payload = {'schema': 'H3_FIXED_R_FLOOR_RECONSTRUCTION_V1', 'claim_id': '15219ecb-5b2e-4f44-9d58-d349cd244866',
      'source': source, 'repository_root': str(REPO), 'computational_sources': before, 'checker_sha256': sha(Path(__file__).read_bytes()),
      'precision': {'binary_round_bits': BITS, 'transcendental_decimal_precision': PREC},
      'r': R, 'height': HEIGHT, 'determinant_scale_identity': identity, 'model': model, 'original_box': original_box,
      'original_box_Z_lower_integral': original, 'original_box_proves_imported_floor': original.lo >= IMPORTED,
      'original_box_integral_below_imported_floor': original.hi < IMPORTED,
      'successor_box': successor_box, 'successor_box_Z_lower_integral': successor,
      'imported_floor': IMPORTED, 'strict_slack_above_imported_floor': successor.lo-IMPORTED,
      'original_strict_slack_above_imported_floor': original.lo-IMPORTED,
      'new_conservative_floor': new_floor,
      'new_floor_proved': True,
      'quartic_monomials': len(poly), 'polynomial_coefficients': poly,
      'checks': CHECKS, 'negative_controls': controls, 'technical_disposition': 'AUTHOR_SIDE_EXPOSED_RECONSTRUCTION_PASS',
      'fixed_r_floor_reproved_by_this_candidate': True, 'historical_upper_bound_replayed': False,
      'scientific_status_changed': False, 'external_independence_credit': 0, 'peer_review': False,
      'all_small_r_certified': False, 'rn_uniform_lemma_closed': False, 'original_prize_closed': False,
      'does_not_establish': 'No spatial integral, all-r extension, canonical promotion, historical upper-endpoint replay, peer review or independence.'}
    payload = enc(payload)
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    envelope = {'payload': payload, 'payload_sha256': sha(canonical)}
    with output.open('x') as f:
        f.write(json.dumps(envelope, sort_keys=True, indent=2, ensure_ascii=False)+'\n')
    print('H3 fixed-r reconstruction PASS; original_floor_comparison='+str(payload['original_box_proves_imported_floor'])+
          '; successor_proves_imported=True; checks='+str(len(CHECKS))+'; negative_controls='+str(len(controls)))


if __name__ == '__main__':
    main()
