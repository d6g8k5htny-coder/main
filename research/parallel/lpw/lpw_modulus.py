#!/usr/bin/env python3
"""Author-side exact R05 tails/profile replay and quadratic covariance modulus.

No K, density, headline, scientific status, or independence is established.
Only pinned source bytes are read. Source programs are never executed.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
DEFAULT_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(DEFAULT_REPO))
from research.interval import Interval as I, exp, pi, sqrt, sin, cos

PREFIX = 'drive/mirrors/02_RESEARCH_CARRY_FORWARD_CANON/LPW — RAW ARCHIVE INTAKE R05/'
SOURCES = [
    ('03_RAYLEIGH_REPAIR_AND_INTERVAL_CERTIFICATE.md', '1ngaO6hzeIXdCYWMwKl8JYPPZTeryGFTx', 9628, '840c75a7825c67b8d99a536394beb18976475fe9bd0e474ed5f80c375004ec81'),
    ('checks/interval_repair.py', '1cdE15_2dd0VLHJnDGcFQ5RDQJhTkXfr-', 6843, 'b37150f0eb79ff5d11b9e8b60f80afd94c677f8150aa701573ce24b337f24069'),
    ('raw_reports/lpw_constant.py', '1fRWIm7IsufB2xkrCxaExFp2TFdcd4oaG', 25380, 'e258322cfbb71dbb8665c2d0c0517399dca5ca5913e9ddf70eb7247ba74d3035'),
]
POWERS = ((0,0),(1,0),(0,1),(2,0),(1,1),(3,0),(0,2),(2,1),(1,2),(0,3))
PREC = 24
BITS = 160
R = F(1,100000)
N = 70


def require(value, reason):
    if not value:
        raise ValueError(reason)


def rr(x):
    return x.round_out(BITS)


def isum(values):
    result = I(0)
    for value in values:
        result = rr(result + value)
    return result


def cap(x):
    return {'lo':str(x.lo), 'hi':str(x.hi)}


def source_identity(repo):
    result = []
    for path, drive_id, size, digest in SOURCES:
        raw = (repo / PREFIX / path).read_bytes()
        require(len(raw) == size and hashlib.sha256(raw).hexdigest() == digest,
                'source identity mismatch: ' + path)
        result.append({'path':PREFIX+path, 'drive_id':drive_id, 'bytes':size,
                       'sha256':digest, 'extraction':'complete raw file bytes'})
    for relative in ('research/interval/__init__.py', 'research/interval/core.py',
                     'research/interval/transcendental.py'):
        raw = (DEFAULT_REPO / relative).read_bytes()
        result.append({'path':relative, 'bytes':len(raw),
                       'sha256':hashlib.sha256(raw).hexdigest(),
                       'extraction':'complete local dependency bytes'})
    return result


def moment_tail(h, a, power, first=N+1):
    require(type(power) is int and 0 <= power <= 12, 'moment order outside 0..12')
    require(type(first) is int and first > 0, 'tail start must be positive')
    ratio = rr(exp(-a*(2*first+1), PREC) * I(F(first+1,first))**power)
    require(ratio.hi < F(1,2), 'moment tail ratio not below 1/2')
    initial = rr(2 * exp(-a*first*first, PREC) * (h*first)**power)
    bound = rr(initial/(1-ratio))
    require(bound.lo > 0, 'tail must be strictly positive')
    return bound, ratio, initial


def shell_tail(h, a, power, first=N+1):
    require(power in (3,4), 'shell power must be 3 or 4')
    root2 = sqrt(I(2), PREC)
    # Algebraically cancelled ratio, avoiding tiny-number division.
    ratio = rr(I(F(first+1,first))*exp(-a*F(2*first+1,2), PREC)
               * ((1+h*root2*(first+1))/(1+h*root2*first))**power)
    require(ratio.hi < 1, 'shell ratio not below one')
    initial = rr(8*first*exp(-a*F(first*first,2), PREC)*(1+h*root2*first)**power)
    return rr(initial/(1-ratio)), ratio, initial


def normalize_moment(numerator, numerator_tail, ztr, zall):
    require(ztr.lo > 0 and zall.lo >= ztr.lo, 'invalid normalization envelope')
    require(numerator_tail.lo >= 0, 'negative numerator tail')
    return rr(I((numerator/zall).lo, ((numerator+numerator_tail)/ztr).hi))


def lattice():
    h = pi(PREC)/12
    a = rr(h*h/2)
    weights = [exp(-a*n*n, PREC) for n in range(N+1)]
    ztr = isum([weights[0], 2*isum(weights[1:])])
    tail_records = []
    tails = []
    for p in range(13):
        t, ratio, first = moment_tail(h,a,p)
        tails.append(t)
        tail_records.append({'power':p, 'tail_upper':str(t.hi),
                             'ratio':cap(ratio), 'first_two_sided_term':cap(first)})
    zall = I(ztr.lo, (ztr+tails[0]).hi)
    moments = [I(1)]
    for p in range(2,13,2):
        numerator = 2*isum(rr(weights[n]*(h*n)**p) for n in range(1,N+1))
        moments.append(normalize_moment(numerator,tails[p],ztr,zall))
    shell_records = []
    for p in (3,4):
        t, ratio, first = shell_tail(h,a,p)
        shell_records.append({'power':p, 'unnormalized_tail_upper':str(t.hi),
                              'normalized_tail_upper':str((t/ztr).hi),
                              'ratio':cap(ratio), 'first_shell_majorant':cap(first)})
    return h,a,moments,{'N':N,'first_omitted':N+1,'h':cap(h),'a':cap(a),
                       'Z1_truncated':cap(ztr),'Z1_full':cap(zall),
                       'even_moments':[cap(m) for m in moments],
                       'one_dimensional_tails':tail_records,
                       'two_dimensional_shell_tails':shell_records}


def poly_product(a,b):
    out = {}
    for u,x in a.items():
        for v,y in b.items():
            out[u+v] = out.get(u+v,F(0)) + x*y
    return out


def poly_sum(a,b):
    out = dict(a)
    for p,c in b.items():
        out[p] = out.get(p,F(0)) + c
    return out


def profile_bounds(kind):
    if kind == 'linear_replay':
        pb = [{0:F(v)} for v in (1,2,1,F(1,2),1,F(1,4),1,F(1,2),F(1,2),F(1,6))]
        pb[0] = {0:F(1),1:R/4}
        db = [{0:F(v)} for v in (F(1,2),F(5,4),1,F(1,4),F(1,2),F(5,4),0,0,0,0)]
        db[0] = {0:F(1,2),1:R/4}
        return pb,db,1,F(1,2)
    require(kind == 'quadratic', 'unknown profile method')
    pb = [{0:F(v)} for v in (1,2,1,F(1,2),1,F(1,6),1,F(1,2),F(1,2),F(1,6))]
    pb[0] = {0:F(1),2:R*R/8}
    # |g_i'(theta)| <= d_i |theta|. Integration in r gives r^2/8.
    db = [{0:F(v)} for v in (1,1,1,F(1,6),F(1,3),F(1,30),0,0,0,0)]
    return pb,db,2,F(1,8)


def modulus(moments, kind):
    pb,db,offset,factor = profile_bounds(kind)
    @lru_cache(None)
    def absolute_moment(order):
        require(0 <= order <= 12, 'insufficient certified moments')
        if order % 2 == 0:
            return moments[order//2].hi
        return sqrt(I(moments[(order-1)//2].hi*moments[(order+1)//2].hi),PREC).hi
    matrix = []
    for i,(xi,yi) in enumerate(POWERS):
        row = []
        for j,(xj,yj) in enumerate(POWERS):
            if (xi+yi-xj-yj)%2 or (yi+yj)%2:
                row.append(F(0))
                continue
            polynomial = poly_sum(poly_product(db[i],pb[j]),poly_product(pb[i],db[j]))
            value = factor*absolute_moment(yi+yj)*sum(
                c*absolute_moment(xi+xj+offset+p) for p,c in polynomial.items())
            require(value >= 0, 'negative entry majorant')
            row.append(rr(I(value)).hi)
        matrix.append(row)
    norm = sqrt(isum(I(x*x) for row in matrix for x in row), PREC)
    return matrix,norm


def make_report(repo=DEFAULT_REPO, linear_claim=F('19.072'), quadratic_claim=F('1.804')):
    identities = source_identity(repo)
    h,a,moments,data = lattice()
    old,linear = modulus(moments,'linear_replay')
    new,quadratic = modulus(moments,'quadratic')
    require(linear.hi < linear_claim, 'linear modulus claim rejected')
    require(quadratic.hi < quadratic_claim, 'quadratic modulus claim rejected')
    old_loss = linear.hi*R
    new_loss = quadratic.hi*R*R
    require(new_loss < old_loss, 'quadratic loss does not improve old bound')
    endpoint = F(31,250)
    require(endpoint-new_loss > 0, 'conditional floor must be positive')
    return {
        'schema':'lpw-tail-profile-modulus-v1',
        'status':'AUTHOR_SIDE_EXACT_CANDIDATE',
        'sources':identities,
        'parameters':{'R':str(R),'endpoint_floor_import':str(endpoint),
                      'precision_hint':PREC,'outward_significant_bits':BITS},
        'lattice':data,
        'profile_powers':[list(x) for x in POWERS],
        'linear_replay':{'entry_majorants':[[str(x) for x in row] for row in old],
                         'frobenius_coefficient':cap(linear),'ceiling':str(linear_claim),
                         'max_covariance_loss':str(old_loss)},
        'quadratic':{'entry_majorants':[[str(x) for x in row] for row in new],
                     'frobenius_coefficient':cap(quadratic),'ceiling':str(quadratic_claim),
                     'derivative_slope_bounds':[str(x[0]) for x in profile_bounds('quadratic')[1]],
                     'max_covariance_loss':str(new_loss),
                     'conditional_uniform_eigenfloor':str(endpoint-new_loss),
                     'simple_conditional_uniform_eigenfloor':str(endpoint-quadratic_claim*R*R),
                     'computed_loss_budget_ratio':str(old_loss/new_loss)},
        'authority':{'scientific_status_changed':False,'original_prize_closed':False,
                     'organizational_independence_credit':0,
                     'reviewer_role':'source-exposed author/coauthor internal verification'},
        'limitations':[
            'Ordinary written analytic proof; no proof assistant formalization.',
            'The ten source profiles and derivative coordinates identify the model conditionally; the six-pin transform was not rederived.',
            'The exact endpoint eigenfloor 31/250 is imported, not proved.',
            'No finite two-dimensional Fourier amplitude sum, K<=9432, conditional density/norm, LPW topology, or headline is certified here.',
            'No scientific status, external review gate, 2D/3D composition, or original prize closure follows.'
        ]
    }


def strict_json(path):
    with path.open('rb') as stream:
        raw = stream.read(2_000_001)
    require(len(raw) <= 2_000_000, 'certificate too large')
    def pairs(items):
        d = {}
        for k,v in items:
            require(k not in d, 'duplicate certificate key')
            d[k] = v
        return d
    def bad_number(x):
        raise ValueError('noninteger JSON numeric literal')
    return json.loads(raw,object_pairs_hook=pairs,parse_float=bad_number,parse_constant=bad_number)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=DEFAULT_REPO)
    parser.add_argument('--certificate',type=Path,default=Path(__file__).with_name('candidate.json'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--claim-linear',default='19.072')
    parser.add_argument('--claim-quadratic',default='1.804')
    args=parser.parse_args()
    try:
        report=make_report(args.repo,F(args.claim_linear),F(args.claim_quadratic))
        if args.certificate:
            require(json.dumps(strict_json(args.certificate),sort_keys=True,ensure_ascii=False)
                    ==json.dumps(report,sort_keys=True,ensure_ascii=False),
                    'certificate reconstruction mismatch')
        raw=(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+'\n').encode()
        if args.output:
            with args.output.open('xb') as output:
                output.write(raw)
        print('PASS LPW full tails/profile modulus; payload_sha256='+hashlib.sha256(raw).hexdigest())
    except (ValueError, ArithmeticError, OSError) as exc:
        print('FAIL '+str(exc),file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    sys.exit(main())
