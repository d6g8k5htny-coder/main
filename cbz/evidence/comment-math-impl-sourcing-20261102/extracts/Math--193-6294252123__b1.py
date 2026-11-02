"""R83 source-only frozen proposal. Do not execute without exact-version review.

This file computes no fields, samples, barcodes, coefficient integral, scientific
tolerance, or study decision. The current author is forbidden to execute it.
"""
import hashlib
import json
import os
from fractions import Fraction as F

BITS = 192
SCALE = 1 << BITS
MAX_INTERVAL_CONSTRUCTIONS = 2000000
INTERVAL_CONSTRUCTIONS = 0


class Hold(Exception):
    pass


def tick():
    global INTERVAL_CONSTRUCTIONS
    INTERVAL_CONSTRUCTIONS += 1
    if INTERVAL_CONSTRUCTIONS > MAX_INTERVAL_CONSTRUCTIONS:
        raise Hold('HOLD_INTERVAL_CONSTRUCTION_CAP')


def down(x):
    return F((x.numerator * SCALE) // x.denominator, SCALE)


def up(x):
    return -down(-x)


class I:
    def __init__(self, lo, hi=None):
        tick()
        lo, hi = F(lo), F(lo if hi is None else hi)
        if lo > hi:
            raise Hold('HOLD_REVERSED_INTERVAL')
        self.lo, self.hi = down(lo), up(hi)

    def __add__(self, y):
        y = box(y)
        return I(self.lo + y.lo, self.hi + y.hi)

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, y):
        return self + -box(y)

    def __rsub__(self, y):
        return box(y) + -self

    def __mul__(self, y):
        y = box(y)
        p = [self.lo*y.lo, self.lo*y.hi, self.hi*y.lo, self.hi*y.hi]
        return I(min(p), max(p))

    __rmul__ = __mul__

    def __truediv__(self, y):
        y = box(y)
        if y.lo <= 0 <= y.hi:
            raise Hold('HOLD_DIVISOR_CONTAINS_ZERO')
        return self * I(1/y.hi, 1/y.lo)

    def __rtruediv__(self, y):
        return box(y) / self

    def __pow__(self, n):
        if type(n) is not int or n < 0 or n > 32:
            raise Hold('HOLD_POWER_DOMAIN')
        ans = I(1)
        for unused in range(n):
            ans = ans * self
        return ans

    def pack(self):
        return {'lo': [str(self.lo.numerator), str(self.lo.denominator)],
                'hi': [str(self.hi.numerator), str(self.hi.denominator)]}


def box(x):
    return x if isinstance(x, I) else I(x)


def imin(*xs):
    xs = [box(x) for x in xs]
    return I(min(x.lo for x in xs), min(x.hi for x in xs))


def imax(*xs):
    xs = [box(x) for x in xs]
    return I(max(x.lo for x in xs), max(x.hi for x in xs))


def require_positive(x, name):
    if box(x).lo <= 0:
        raise Hold('HOLD_POSITIVITY_' + name)
    return x


def sqrt_rat(x):
    if x < 0:
        raise Hold('HOLD_SQRT_DOMAIN')
    # Exact integer isqrt, written locally to keep primitive semantics explicit.
    n = (x.numerator * SCALE * SCALE) // x.denominator
    if n == 0:
        k = 0
    else:
        k = 1 << ((n.bit_length() + 1) // 2)
        for unused in range(1024):
            z = (k + n // k) // 2
            if z >= k:
                break
            k = z
        else:
            raise Hold('HOLD_SQRT_ITERATION_CAP')
    if not (k*k <= n < (k+1)*(k+1)):
        raise Hold('HOLD_SQRT_INTEGER_POSTCONDITION')
    lo = F(k, SCALE)
    hi = lo if lo*lo == x else F(k+1, SCALE)
    return lo, hi


def sqrt_i(x):
    x = box(x)
    return I(sqrt_rat(x.lo)[0], sqrt_rat(x.hi)[1])


def arctan_recip(n):
    # Alternating series: consecutive partial sums bracket arctan(1/n).
    total = F(0)
    for j in range(256):
        term = F((-1)**j, (2*j+1) * n**(2*j+1))
        total += term
        nxt = F((-1)**(j+1), (2*j+3) * n**(2*j+3))
        if abs(nxt) <= F(1, 1 << (BITS+16)):
            return I(min(total, total+nxt), max(total, total+nxt))
    raise Hold('HOLD_ARCTAN_CAP')


def pi_i():
    return 16*arctan_recip(5) - 4*arctan_recip(239)


def exp_neg_rat(x):
    # x >= 0. Range reduction is exact; Taylor brackets are rational.
    if x < 0:
        raise Hold('HOLD_EXP_DOMAIN')
    y, shifts = x, 0
    while y > F(1, 8):
        shifts += 1
        if shifts > 32:
            raise Hold('HOLD_EXP_RANGE_CAP')
        y /= 2
    total, term = F(1), F(1)
    for n in range(1, 256):
        term *= -y/F(n)
        total += term
        nxt = term * -y/F(n+1)
        if abs(nxt) <= F(1, 1 << (BITS+16)):
            ans = I(min(total, total+nxt), max(total, total+nxt))
            break
    else:
        raise Hold('HOLD_EXP_SERIES_CAP')
    for unused in range(shifts):
        ans = ans * ans
    if ans.lo < 0 or ans.hi > 1:
        # Enclosure intersects [0,1], justified for the explicit nonnegative x.
        ans = I(max(F(0), ans.lo), min(F(1), ans.hi))
    return ans


def exp_neg(x):
    x = box(x)
    if x.lo < 0:
        raise Hold('HOLD_EXP_INTERVAL_DOMAIN')
    return I(exp_neg_rat(x.hi).lo, exp_neg_rat(x.lo).hi)


def choose(n, k):
    a = 1
    for j in range(1, k+1):
        a = a * (n-j+1) // j
    return a


def add_omitted_tail(finite,tail):
    if tail.lo < 0:
        raise Hold('HOLD_NEGATIVE_OMITTED_TAIL')
    return I(finite.lo,finite.hi+tail.hi)


def normalized_moments(sums):
    require_positive(sums[0],'normalizer')
    return [I(1)] + [s/sums[0] for s in sums[1:]]


def radial_moments(mu):
    return [sum((choose(k,j)*mu[j]*mu[k-j] for j in range(k+1)),I(0))
            for k in range(len(mu))]


def full_moments(pi, cutoff=64):
    if cutoff != 64:
        raise Hold('HOLD_UNREVIEWED_SPECTRAL_CUTOFF')
    h, alpha = pi/12, (pi/12)**2/2
    if alpha.lo * cutoff * cutoff < 7:
        raise Hold('HOLD_MOMENT_MONOTONICITY')
    weights = [exp_neg(alpha*(m*m)) for m in range(cutoff+1)]
    sums = []
    tails = []
    # METHOD M14-M15, uniform alpha_lo and h_hi tail majorant.
    em = exp_neg_rat(alpha.lo * cutoff * cutoff)
    tail_integral = em / I(2*alpha.lo*cutoff)
    for k in range(8):
        if k:
            tail_integral = (I(cutoff**(2*k-1))*em/(2*alpha.lo)
                             + I(2*k-1)*tail_integral/(2*alpha.lo))
        finite = I(1 if k == 0 else 0)
        for m in range(1, cutoff+1):
            finite += 2 * (h*m)**(2*k) * weights[m]
        tail = 2 * I(h.hi)**(2*k) * tail_integral
        sums.append(add_omitted_tail(finite,tail))
        tails.append(tail)
    mu = normalized_moments(sums)
    if not all(tail.hi > 0 for tail in tails):
        raise Hold('HOLD_MISSING_POSITIVE_GAUSSIAN_TAIL')
    # M_(2k) in d2: binomial expansion of (Q1^2+Q2^2)^k.
    radial = radial_moments(mu)
    return h, alpha, sums, tails, mu, radial, weights


def embedding(h, cutoff=32):
    if cutoff != 32:
        raise Hold('HOLD_UNREVIEWED_EMBEDDING_CUTOFF')
    partial = I(0)
    for m in range(-cutoff, cutoff+1):
        for n in range(-cutoff, cutoff+1):
            partial += 1 / (1 + h**2 * (m*m+n*n))**2
    tail = 4 / (I(h.lo)**4 * cutoff**2)
    return I(partial.lo, partial.hi+tail.hi), partial, tail


def constants():
    p = pi_i()
    h, alpha, sums, tails, mu, radial, w = full_moments(p)
    a,b,c = mu[1:4]
    # Direct interval subtraction is retained; an independent positive witness
    # is intersected only as a lower bound, never silently substituted as value.
    beta = b-a*a
    nu = c-b*b/a
    beta_witness = 2*h**4*w[1]/sums[0]**2
    nu_witness = 72*h**6*w[1]*w[2]/(sums[0]*(w[1]+4*w[2]))
    beta = I(max(beta.lo,beta_witness.lo),beta.hi)
    nu = I(max(nu.lo,nu_witness.lo),nu.hi)
    require_positive(a,'a'); require_positive(beta,'beta'); require_positive(nu,'nu3')
    bstar, vstar = imin(beta,2*a*a)/2, nu/4
    M2,M4,M6,M8,M10 = radial[1:6]
    LA = sqrt_i(M2*M6)/4
    LB = sqrt_i(M4*M8)/12
    Lg = M6/8 + sqrt_i(M2*M10)/40
    Lw = sqrt_i(M6*M10)/20
    LV = Lw + 4*sqrt_i(M2*M6)*Lg/a + 2*M2*M6*LA/(a*a)
    r0 = imin(F(1,4),sqrt_i(a/(2*LA)),sqrt_i(bstar/(2*LB)),sqrt_i(vstar/(2*LV)))
    vf = vstar/2
    pstar = 4/(a*bstar) * sqrt_i(2/vstar) / ((2*p)**2*sqrt_i(2*p))
    JA = imax(c+a*b,(c+7*a*b)/2)
    JB = b+a*a
    Dstar = 9*JA*JB*pstar
    dstar, cstar = 144/vf, 72/M6
    theta = sum((choose(7,k)*radial[k] for k in range(8)),I(0))
    e2, epartial, etail = embedding(h)
    e = sqrt_i(e2)
    d2 = sqrt_i(M6/64+M8/576+M10/1600)
    U = M2+M4+M6
    KSigma = 2*sqrt_i(U)*d2
    tstar = 1+2*sqrt_i(M2*M6)/a
    Istar = imax(2/bstar,tstar*tstar*imax(2/a,1/vf))
    breg = e*sqrt_i(theta*U)*Istar
    dreg = e*sqrt_i(theta)*(d2*Istar+sqrt_i(U)*Istar**2*KSigma)
    L = imax(e*(1+sqrt_i(I(3))*breg),breg)
    K = imax(dreg,sqrt_i(I(3))*e*dreg+F(23,120)*e*breg)
    # All far statements use the exact R79 r0, not its outward lower endpoint
    # as a newly declared source separation. Monotonicity propagates its box.
    gram = (r0/12)**6 / 2787840000
    pfar = 1 / ((2*p)**2*sqrt_i(2*p)*gram**2*sqrt_i(gram))
    cfar = 576*pfar*3*e**4*theta**2*(1+1/gram)**2
    data = dict(pi=p,h=h,alpha=alpha,z=sums[0],a=a,b=b,c6=c,
                beta=beta,nu3=nu,beta_witness=beta_witness,nu3_witness=nu_witness,
                bstar=bstar,vstar=vstar,LA=LA,LB=LB,Lg=Lg,Lw=Lw,LV=LV,
                r0=r0,vf=vf,pstar=pstar,JA=JA,JB=JB,Dstar=Dstar,
                dstar=dstar,cstar=cstar,Theta=theta,e_emb_squared=e2,e_emb=e,
                embedding_partial=epartial,embedding_tail=etail,d2=d2,U=U,
                KSigma=KSigma,tstar=tstar,Istar=Istar,breg=breg,dreg=dreg,
                L=L,K=K,mu_far=gram,p_far=pfar,C_far_new=cfar)
    for name,value in data.items():
        require_positive(value,name)
    # Precision acceptance concerns primitive moments only. Composite interval
    # widths are recorded, never declared adequate for a scientific tolerance.
    if any(x.hi-x.lo > F(1,1<<96) for x in mu[1:]):
        raise Hold('HOLD_MOMENT_WIDTH')
    return {'constants': {name:value.pack() for name,value in data.items()},
            'one_dimensional_moments': [x.pack() for x in mu],
            'radial_even_moments': [x.pack() for x in radial],
            'spectral_sums': [x.pack() for x in sums],
            'spectral_tail_bounds': [x.pack() for x in tails],
            'scientific_effect':'NONE','scientific_acceptance':False,
            'E':None,'Delta':None,'J':None,'ell_star':None,
            'window_status':'NOT_EVALUATED','source_law':'FULL_GAUSSIAN_DC_INCLUDED_D2_L24'}


def controls(spec):
    # Analytically specified controls; current author has NOT executed them.
    if not (I(F(1,3)).lo <= F(1,3) <= I(F(1,3)).hi):
        raise Hold('CONTROL_OUTWARD_THIRDS')
    negprod = I(-2,3)*I(-4,5)
    if not (negprod.lo <= -12 and negprod.hi >= 15):
        raise Hold('CONTROL_MULTIPLY_ALL_ENDPOINTS')
    root = sqrt_i(I(4))
    if root.lo != 2 or root.hi != 2:
        raise Hold('CONTROL_EXACT_SQRT')
    root2 = sqrt_i(I(2))
    if not (root2.lo*root2.lo <= 2 <= root2.hi*root2.hi):
        raise Hold('CONTROL_SQRT_BRACKET')
    pp = pi_i()
    if not (pp.lo > 3 and pp.hi < F(22,7)):
        raise Hold('CONTROL_MACHIN_RANGE')
    ee = exp_neg(I(1))
    if not (ee.lo > F(1,3) and ee.hi < F(1,2)):
        raise Hold('CONTROL_EXP_NEG_ONE')
    # Independent exact finite algebraic fixture; NOT the Gaussian source:
    # equal weights on {0,-1,+1} give S0=3, S2=S4=S6=2. DC, the common
    # denominator, d2 cross terms and axial cubic rank loss are all observable.
    finite_mu = normalized_moments([I(3),I(2),I(2),I(2)])
    if not all(x.lo <= F(2,3) <= x.hi for x in finite_mu[1:]):
        raise Hold('CONTROL_DC_COMMON_NORMALIZER')
    finite_radial = radial_moments(finite_mu)
    if not (finite_radial[1].lo <= F(4,3) <= finite_radial[1].hi
            and finite_radial[2].lo <= F(20,9) <= finite_radial[2].hi):
        raise Hold('CONTROL_D2_RADIAL_CROSS_TERMS')
    rank_loss = finite_mu[3]-finite_mu[2]*finite_mu[2]/finite_mu[1]
    if not (rank_loss.lo <= 0 <= rank_loss.hi):
        raise Hold('CONTROL_FINITE_SUPPORT_RANK_LOSS')
    # A genuine positive omitted denominator tail [0,1] changes 2/3 to a
    # range including1/2; dropping it would fail this independent endpoint.
    z_with_tail = add_omitted_tail(I(3),I(0,1))
    augmented_mu = normalized_moments([z_with_tail,I(2)])
    if not (z_with_tail.lo == 3 and z_with_tail.hi == 4
            and augmented_mu[1].lo <= F(1,2) <= augmented_mu[1].hi
            and augmented_mu[1].lo <= F(2,3) <= augmented_mu[1].hi):
        raise Hold('CONTROL_POSITIVE_DENOMINATOR_TAIL')
    expected = [('zero_divisor',lambda: I(1)/I(-1,1)),
                ('negative_sqrt',lambda: sqrt_i(I(-1))),
                ('reversed_interval',lambda: I(2,1)),
                ('negative_exp_argument',lambda: exp_neg(I(-1))),
                ('unreviewed_spectral_cutoff',lambda: full_moments(pp,1)),
                ('unreviewed_embedding_cutoff',lambda: embedding(pp,1))]
    for name,fn in expected:
        try:
            fn()
        except Hold:
            continue
        raise Hold('CONTROL_EXPECTED_REJECTION_' + name)
    mutations = [('include_dc',False),('full_spectrum',False),
                 ('physical_derivatives',False),('L',[12,1]),
                 ('precision_bits',96),('scientific_tolerance','1/20'),
                 ('d',True),('unreviewed_extra_key',1)]
    for key,value in mutations:
        changed = dict(spec)
        changed[key] = value
        try:
            validate_spec(changed)
        except Hold:
            continue
        raise Hold('CONTROL_EXPECTED_INPUT_REJECTION_' + key)
    try:
        no_duplicate_keys([('d',2),('d',2)])
    except Hold:
        pass
    else:
        raise Hold('CONTROL_EXPECTED_DUPLICATE_REJECTION')
    return {'six_positive_controls':'PASS','six_expected_rejections':'PASS',
            'four_arithmetic_source_controls':'PASS',
            'eight_semantic_input_mutations':'REJECTED','duplicate_key':'REJECTED',
            'scientific_effect':'NONE','scientific_acceptance':False}


def no_duplicate_keys(pairs):
    ans = {}
    for key,value in pairs:
        if key in ans:
            raise Hold('HOLD_DUPLICATE_JSON_KEY')
        ans[key] = value
    return ans


def load_and_validate_spec(path):
    if os.path.getsize(path) > 16384:
        raise Hold('HOLD_INPUT_SIZE')
    with open(path,'r',encoding='utf-8') as stream:
        spec = json.load(stream,object_pairs_hook=no_duplicate_keys,
                         parse_float=lambda unused: (_ for _ in ()).throw(Hold('HOLD_FLOAT_INPUT')))
    validate_spec(spec)
    return spec


def validate_spec(spec):
    expected = {'schema':'R83_CONSTANTS_V1','d':2,'L':[24,1],
                'include_dc':True,'full_spectrum':True,'physical_derivatives':True,
                'height':'integrated','bar':'ordinary_finite_positive_elder_H0',
                'precision_bits':192,'spectral_cutoff':64,'embedding_cutoff':32,
                'moment_width_bits':96,'max_interval_constructions':2000000,
                'scientific_tolerance':None,'Delta':None,'J':None}
    # Strict type-sensitive canonical representation rejects bool-as-int, extra
    # keys and unauthorized parameter changes rather than Python's loose ==.
    if json.dumps(spec,sort_keys=True,separators=(',',':')) != json.dumps(expected,sort_keys=True,separators=(',',':')):
        raise Hold('HOLD_EXACT_INPUT_CONTRACT_MISMATCH')
    return spec


def run(mode,spec_path):
    if mode not in ('controls','constants'):
        raise Hold('HOLD_MODE_NOT_ALLOWED')
    spec = load_and_validate_spec(spec_path)
    if mode == 'controls':
        result = controls(spec)
    else:
        result = constants()
    result.update({'schema':'R83_RESULT_V1','mode':mode,'interval_construction_count':INTERVAL_CONSTRUCTIONS,
                   'input_sha256':hashlib.sha256(open(spec_path,'rb').read()).hexdigest(),
                   'contract_input':spec})
    return result
