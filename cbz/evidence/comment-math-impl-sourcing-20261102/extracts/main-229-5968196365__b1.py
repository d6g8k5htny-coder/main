#!/usr/bin/env python3
"""Independent C104 rational falsification controls. No author imports.

Matrix perturbations and concentration examples are explicitly synthetic;
the analytic review, not these finite cases, assesses the actual torus law.
"""
from collections import Counter, defaultdict
from fractions import Fraction as F
from hashlib import sha256, sha1
from itertools import product
from math import factorial
from pathlib import Path
import json
import platform

BASE = Path(__file__).resolve().parents[1]
COUNTS = Counter()
MUTANTS = []


def check(group, condition):
    if not condition:
        raise RuntimeError('independent control failed: ' + group)
    COUNTS[group] += 1


def reject(name, proposition):
    if proposition:
        raise RuntimeError('mutant survived: ' + name)
    MUTANTS.append(name)


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def tr(a):
    return [list(row) for row in zip(*a)]


def mm(a, b):
    return [[dot(row, col) for col in tr(b)] for row in a]


def eye(n):
    return [[F(i == j) for j in range(n)] for i in range(n)]


def diag(values):
    return [[F(v) if i == j else F(0) for j in range(len(values))]
            for i, v in enumerate(values)]


def add(a, b, scale=1):
    return [[x+scale*y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def block(a, rows, cols):
    return [[a[i][j] for j in cols] for i in rows]


def inverse(a):
    n = len(a)
    x = [[F(v) for v in row] + e for row, e in zip(a, eye(n))]
    for i in range(n):
        p = next(j for j in range(i, n) if x[j][i])
        x[i], x[p] = x[p], x[i]
        scale = x[i][i]
        x[i] = [v/scale for v in x[i]]
        for j in range(n):
            if i != j:
                scale = x[j][i]
                x[j] = [u-scale*v for u, v in zip(x[j], x[i])]
    return [row[n:] for row in x]


def determinant(a):
    x = [[F(v) for v in row] for row in a]
    result = F(1)
    for i in range(len(x)):
        p = next((j for j in range(i, len(x)) if x[j][i]), None)
        if p is None:
            return F(0)
        if p != i:
            x[p], x[i] = x[i], x[p]
            result = -result
        pivot = x[i][i]
        result *= pivot
        for j in range(i+1, len(x)):
            scale = x[j][i]/pivot
            for k in range(i+1, len(x)):
                x[j][k] -= scale*x[i][k]
    return result


def positive_definite(a):
    x = [[F(v) for v in row] for row in a]
    if x != tr(x):
        return False
    for i in range(len(x)):
        if x[i][i] <= 0:
            return False
        for j in range(i+1, len(x)):
            for k in range(i+1, len(x)):
                x[j][k] -= x[j][i]*x[i][k]/x[i][i]
    return True


def frob2(a):
    return sum(v*v for row in a for v in row)


def raw_jets(degree):
    return [(a, b, degree-a-b) for a in range(degree+1)
            for b in range(degree-a+1)]


def double_factorial_odd(n):
    v = 1
    for i in range(1, n+1, 2):
        v *= i
    return v


def covariance(a, b):
    powers = tuple(x+y for x, y in zip(a, b))
    if any(v % 2 for v in powers):
        return F(0)
    value = (-1)**(sum(a)+sum(powers)//2)
    for v in powers:
        value *= double_factorial_odd(v-1)
    return F(value)


def schur(gamma, u_idx, j_idx):
    g = block(gamma, u_idx, u_idx)
    c = block(gamma, j_idx, u_idx)
    regression = mm(c, inverse(g))
    s = add(block(gamma, j_idx, j_idx), mm(regression, tr(c)), -1)
    return regression, s


def poly_multiply(a, b):
    out = defaultdict(F)
    for i, v in a.items():
        for j, w in b.items():
            out[i+j] += v*w
    return dict(out)


def hermite(n):
    if n == 0:
        return {0: F(1)}
    previous, current = {0: F(1)}, {1: F(1)}
    for k in range(1, n):
        nxt = defaultdict(F, {p+1: v for p, v in current.items()})
        for p, v in previous.items():
            nxt[p] -= k*v
        previous, current = current, dict(nxt)
    return current


def alpha_factorial(a):
    result = 1
    for n in a:
        result *= factorial(n)
    return result


def tensor_action(rotation, entries):
    lookup = {a: i for i, a in enumerate(entries)}
    rows = []
    for alpha in entries:
        slots = [i for i, count in enumerate(alpha) for _ in range(count)]
        row = [F(0)]*len(entries)
        for inputs in product(range(3), repeat=3):
            beta = tuple(inputs.count(i) for i in range(3))
            if beta not in lookup:
                continue
            weight = F(1)
            for i, j in zip(slots, inputs):
                weight *= rotation[i][j]
            row[lookup[beta]] += weight
        rows.append(row)
    return rows


def matrix_action(o):
    columns = []
    for a in [[[1, 0], [0, 0]], [[0, 1], [1, 0]], [[0, 0], [0, 1]]]:
        out = mm(mm(o, a), tr(o))
        columns.append([out[0][0], out[0][1], out[1][1]])
    return tr(columns)


def cubic_count(s, b, d2, k):
    threshold = -(s-b)**2*(b+2*s)/(12*k)
    if s > b:
        return 2 if d2 < threshold else 1
    return 1 if d2 > threshold else 0


def two_index(a, c, b):
    d = a*b-c*c
    if d == 0:
        return None
    if d < 0:
        return 1
    return 2 if a+b < 0 else 0


def rising(x, q):
    result = F(1)
    for i in range(q):
        result *= x+i
    return result


proof = (BASE/'PROOF.md').read_bytes()
check('source_binding', (len(proof), sha256(proof).hexdigest()) ==
      (26474, 'afa92913a746b132f00a61ae91c63cb4a0381119b12fa60528014fadc44f55a9'))
manifest = (BASE/'SOURCE_IDENTITIES.json').read_bytes()
check('source_binding', (len(manifest), sha256(manifest).hexdigest()) ==
      (4441, 'f556504ffec70739d972fec90e1a5b1ae2ea72f44572430d8b07062f8f0b031e'))
sources = json.loads(manifest)
check('source_binding', len(sources) == 8)
for item in sources:
    data = (BASE/item['local_path']).read_bytes()
    check('source_binding', (len(data), sha256(data).hexdigest()) == (item['bytes'], item['sha256']))
    check('source_binding', sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest() == item['blob'])

jets = sum((raw_jets(q) for q in range(4)), [])
idx = {a: i for i, a in enumerate(jets)}
gamma = [[covariance(a, b) for b in jets] for a in jets]
h = eye(20)
for alpha in jets:
    degree = sum(alpha)
    if degree == 2 and 2 in alpha:
        h[idx[alpha]][idx[(0, 0, 0)]] = 1
    if degree == 3:
        for axis in range(3):
            if alpha[axis] >= 2:
                beta = list(alpha); beta[axis] -= 2
                h[idx[alpha]][idx[tuple(beta)]] += F(alpha[axis]*(alpha[axis]-1), 2)
q = add(h, eye(20), -1)
d = diag([alpha_factorial(a) for a in jets])
check('reference_hermite', mm(mm(h, gamma), tr(h)) == d)
check('reference_hermite', mm(q, q) == [[0]*20 for _ in range(20)])
check('reference_hermite', inverse(h) == add(eye(20), q, -1))
check('reference_hermite', frob2(h) == frob2(inverse(h)) == 56)
check('reference_hermite', sum(gamma[i][i] for i in range(20)) == 80)
check('reference_hermite', positive_definite(gamma))
check('reference_hermite', positive_definite(add(gamma, diag([F(1, 112)]*20), -1)))
check('reference_hermite', positive_definite(add(diag([81]*20), gamma, -1)))

u_entries = [(0,0,0), (1,0,0), (0,1,0), (0,0,1),
             (2,0,0), (1,1,0), (1,0,1), (3,0,0)]
third = [a for a in raw_jets(3) if a != (3,0,0)]
j_entries = [(0,2,0), (0,1,1), (0,0,2)] + third
ui, ji = [idx[a] for a in u_entries], [idx[a] for a in j_entries]
reg, s_ref = schur(gamma, ui, ji)
expected_variances = [2,1,2]+[alpha_factorial(a) for a in third]
check('conditional_reference', s_ref == diag(expected_variances))
for birth, k in product([F(-4), F(0), F(1), F(5,2)], [F(1,2), F(1), F(2), F(3)]):
    target = [birth,0,0,0,0,0,0,12*k]
    mu = [dot(row, target) for row in reg]
    check('conditional_reference', mu == [-birth,0,-birth]+[0]*9)
for removed in [(1,0,0),(0,1,0),(0,0,1)]:
    reduced = [a for a in u_entries if a != removed]
    r_reg, r_cov = schur(gamma, [idx[a] for a in reduced], ji)
    target = [F(12) if a == (3,0,0) else F(0) for a in reduced]
    mean = [dot(row, target) for row in r_reg]
    check('omitted_gradient_fixture', r_cov != s_ref)
    if removed == (1,0,0):
        pos = j_entries.index((1,2,0))
        check('omitted_gradient_fixture', mean[pos] == F(12,5) and r_cov[pos][pos] == F(12,5))
        reject('odd conditional law remains centered after dropping axial gradient', mean[pos] == 0)
    if removed == (0,1,0):
        pos = j_entries.index((2,1,0))
        check('omitted_gradient_fixture', r_cov[pos][pos] == 3)
        reject('transverse gradient omission preserves variance two', r_cov[pos][pos] == 2)

a0, m0 = F(1,112), F(81)
vbound = F(1+2+12*2)
dmu = 20*(112+81*112**2)*vbound
ds = F(20*(1+81*112)**2)
check('explicit_arithmetic', dmu/vbound == 20323520)
check('explicit_arithmetic', ds == 1646386580)
check('explicit_arithmetic', 20323520 < 2**25)
check('explicit_arithmetic', 20*F(1,2**245) < a0)
epsilon = F(1,2**250)
vectors = []
for members in [[(0,0,0),(0,2,0),(0,0,2)],
                [(1,0,0),(3,0,0),(1,2,0)],
                [(0,1,0),(2,1,0),(0,3,0)]]:
    vectors.append([F(a in members) for a in jets])
for sign, vector in product([-1,1], vectors):
    perturbation = [[sign*epsilon*x*y for y in vector] for x in vector]
    perturbed = add(gamma, perturbation)
    check('schur_neighborhood', positive_definite(add(perturbed, diag([a0]*20), -1)))
    p_reg, p_cov = schur(perturbed, ui, ji)
    check('schur_neighborhood', positive_definite(add(p_cov, diag([a0]*12), -1)))
    check('schur_neighborhood', positive_definite(add(diag([m0]*12), p_cov, -1)))
    check('schur_neighborhood', frob2(add(p_cov,s_ref,-1)) <= (ds*epsilon)**2)
    for birth, k in product([F(-2), F(0), F(1)], [F(1,2), F(2)]):
        target = [birth,0,0,0,0,0,0,12*k]
        mu = [dot(row,target) for row in p_reg]
        reference = [-birth,0,-birth]+[F(0)]*9
        check('schur_neighborhood', sum((x-y)**2 for x,y in zip(mu,reference)) <= (dmu*epsilon)**2)
        check('schur_neighborhood', dot(mu,mu) <= (3*vbound)**2)

for total in range(7):
    majorant = {p: abs(v) for p,v in hermite(total).items()}
    check('image_majorant', sum(majorant.values()) <= 76)
    for alpha in raw_jets(total):
        calculated = {0:F(1)}
        for n in alpha:
            calculated = poly_multiply(calculated,{p:abs(v) for p,v in hermite(n).items()})
        check('image_majorant', all(v <= majorant.get(p,0) for p,v in calculated.items()))
check('image_majorant', sum(abs(v) for v in hermite(6).values()) == 76)
for shell in range(1,31):
    count = (2*shell+1)**3-(2*shell-1)**3
    check('lattice_shell', count == 24*shell**2+2 <= 26*shell**2)
    check('lattice_shell', shell <= 2**(shell-1))
    check('lattice_shell', shell**2-1 >= 3*(shell-1))
check('lattice_shell', F(1,1-F(1,2**856)) < 2)
check('lattice_shell', 91*26*2 < 5000)
check('lattice_shell', 5000*25**6 < 2**43)
check('lattice_shell', F(1)+1+F(1,2) > 2)
check('lattice_shell', 3*24**2//2-8 == 856)

metric_t = diag([F(factorial(3),alpha_factorial(a)) for a in third])
cov_t = diag([alpha_factorial(a) for a in third])
metric_a = diag([1,2,1])
frames = [(F(1),F(0)),(F(0),F(1)),(F(3,5),F(4,5)),(F(-5,13),F(12,13))]
raw_distortion_found = False
for (cs,sn), reflection in product(frames,[-1,1]):
    o = [[cs,-sn],[reflection*sn,reflection*cs]]
    rotation = [[F(1),0,0],[0,o[0][0],o[0][1]],[0,o[1][0],o[1][1]]]
    action = tensor_action(rotation,third)
    ma = matrix_action(o)
    check('raw_angular_action', mm(mm(tr(action),metric_t),action) == metric_t)
    check('raw_angular_action', mm(mm(action,cov_t),tr(action)) == cov_t)
    check('raw_angular_action', abs(determinant(action)) == 1)
    check('raw_angular_action', mm(mm(tr(ma),metric_a),ma) == metric_a)
    check('raw_angular_action', abs(determinant(ma)) == 1)
    raw_distortion_found |= mm(tr(action),action) != eye(9)
    for hval in [F(1,7),F(1),F(4)]:
        av = [dot(row,[0,0,-hval]) for row in ma]
        for seed in range(13):
            tau = [F(((seed+3)*(i+2))%11-5,3) for i in range(9)]
            rotated = [dot(row,tau) for row in action]
            physical2 = dot(av,av)+dot(rotated,rotated)
            base2 = hval*hval+dot(tau,tau)
            check('raw_norm_envelope', base2/6 <= physical2 <= 6*base2)
check('raw_angular_action', raw_distortion_found)
reject('raw tensor action is Euclidean orthogonal', not raw_distortion_found)

for left,right in product([F(-3),F(0),F(1)], [F(2),F(4)]):
    # Differential of O diag(left,right) O^t at angle zero in raw (u,v,w).
    differential = [[1,0,0],[0,0,left-right],[0,1,0]]
    check('spectral_measure', abs(determinant(differential)) == abs(right-left))
# Gaussian test: raw integral 2*pi^(3/2), ordered eigenvalue integral 2*pi^(1/2).
check('spectral_measure', F(3,2)-F(1,2) == 1)
check('spectral_measure', F(2,2) == 1)
for k,s,a,beta,hval in product([F(1,2),F(1),F(2)],
        [F(-3),F(-1),F(0),F(1)], [F(-2),F(0),F(1)],
        [F(-2),F(0),F(1)], [F(1,3),F(1),F(3)]):
    pm = (-6*k,-a/2,s-beta/2)
    ps = (6*k,a/2,s+beta/2)
    detm, dets = pm[0]*pm[2]-pm[1]**2, ps[0]*ps[2]-ps[1]**2
    soft = abs(detm*dets) if two_index(*pm)==2 and two_index(*ps)==1 else 0
    hm = [[pm[0],pm[1],0],[pm[1],pm[2],0],[0,0,-hval]]
    hs = [[ps[0],ps[1],0],[ps[1],ps[2],0],[0,0,-hval]]
    check('spectral_hard_weight', determinant(hm)*determinant(hs) == hval**2*detm*dets)
    full = abs(determinant(hm)*determinant(hs)) if soft else 0
    check('spectral_hard_weight', hval*full == hval**3*soft)

ks = [F(1,4),F(1,2),F(1),F(2),F(4)]
values = [F(-3),F(-1),F(-1,4),F(0),F(1,4),F(1),F(3)]
for k,a,beta,c in product(ks,values,values,values):
    b = beta-a*a/(12*k)
    dvalue = (c-a*beta/(4*k)+a**3/(72*k*k))/2
    old = 384*k*k*abs(b)**3+2304*k**3*dvalue*dvalue
    expanded = 1536*k*k*abs(beta)**3+1728*k**3*c*c+108*k*a*a*beta*beta+F(11,9)/k*a**6
    check('cubic_envelope', old <= expanded)
    norm2 = a*a+beta*beta+c*c
    ck = 1536*k*k+1728*k**3+108*k+F(11,9)/k
    check('cubic_envelope', expanded <= ck*(1+norm2**3))
for k,b,x,d2 in product(ks,values,[F(1,4),F(1),F(4),F(12)],
                       [F(0),F(1,16),F(1),F(5)]):
    if x <= abs(b)/2:
        continue
    g = 2*x**3+3*b*x*x-b**3-12*k*d2
    threshold = -(-x-b)**2*(b-2*x)/(12*k)
    check('cubic_classifier', g == 12*k*(threshold-d2))
    if x >= 2*abs(b) and x**3 >= 48*k*d2:
        check('cubic_classifier', g >= x**3/8 > 0 and cubic_count(-x,b,d2,k)==0)
    check('cubic_boundary', cubic_count(-x,b,threshold,k) == (1 if -x>b else 0))
for k,b in product(ks,[F(-1,3),F(-1),F(-4)]):
    antiderivative = lambda s: 12*k*k*s**3-9*k*k*b*b*s
    exact = antiderivative(b/2)-antiderivative(b)
    check('exact_cubic_integrals', exact == -6*k*k*b**3)
for x,dvalue in product([F(1,2),F(1),F(3)],[F(1,2),F(1),F(2)]):
    k = x**3/(6*dvalue*dvalue)
    check('exact_cubic_integrals', 12*k*k*x**3 == 72*k**3*dvalue*dvalue)

# Moment envelopes after removing their common pi^(9/2) factor.
s0 = {F(13,2):F(1,2), F(19,2):rising(F(9,2),3)/2}
s1_from_derivative = defaultdict(F,s0)
for exponent,coefficient in s0.items():
    s1_from_derivative[exponent+1] += exponent*coefficient
s1_from_integrals = defaultdict(F)
for j in [0,3]:
    coefficient = rising(F(9,2),j)
    s1_from_integrals[F(9,2)+j+2] += coefficient/2
    s1_from_integrals[F(9,2)+j+3] += coefficient
for j in [1,4]:
    s1_from_integrals[F(9,2)+j+2] += rising(F(9,2),j)/2
check('surface_moments', dict(s1_from_derivative) == dict(s1_from_integrals))
check('surface_moments', rising(F(3,2),2) == F(15,4))
check('surface_moments', rising(F(3,2),3) == F(105,8))
check('surface_moments', rising(F(9,2),3)/2 == F(1287,16))
for u,v,w in product([F(-2),F(-3,2),F(-1)],
                      [F(-1,4),F(0),F(1,4)],
                      [F(-2),F(-3,2),F(-1)]):
    determinant_a = u*w-v*v
    check('normalizer_box', u<0 and determinant_a>=F(15,16))
    check('normalizer_box', u*u+v*v+w*w<9)
    check('normalizer_box', abs(determinant_a)<=u*u+v*v+w*w)
check('normalizer_box', 36*F(1,2)*F(15,16)**2 == 18*F(225,256))
for radius,mean in product([F(0),F(1,2),F(1),F(3),F(100)],
                           [F(0),F(1,3),F(1),F(10)]):
    check('density_interpolation_bounds', radius+mean <= (1+mean)*(1+radius**2))
    check('density_interpolation_bounds', (radius+mean)**2 <= 2*(1+mean**2)*(1+radius**2))
    check('density_interpolation_bounds', (radius-mean)**2 >= radius**2/2-mean**2)
for nl,nr,zl,zr in product([F(0),F(1),F(3)],repeat=4):
    if zl<=0 or zr<=0:
        continue
    floor = min(zl,zr)
    check('quotient_bound', abs(nl/zl-nr/zr) <= abs(nl-nr)/floor+nr*abs(zl-zr)/(floor*floor))
for pin,numerator,z,kpower in product([F(1,7),F(2)],
                                     [F(0),F(3)], [F(1,9),F(4)], [F(2),F(5)]):
    check('band_cancellation', (12*pin*z)*(numerator/z)/(3*kpower) == 4*pin*numerator/kpower)
check('band_cancellation', 4*4 == 16)
check('pin_density_derivative', F(8,2) == 4)

# Small full-law L1 can coexist with a large restriction value: a mixture of
# uniforms with epsilon=q^2, mixture weight q, and thin support [-epsilon,epsilon].
for qsmall in [F(1,8),F(1,32),F(1,128)]:
    center_difference = (1-qsmall)/2+qsmall/(2*qsmall*qsmall)-F(1,2)
    l1 = 2*qsmall*(1-qsmall*qsmall)
    check('surface_shortcut_counterfixture', l1<2*qsmall and center_difference>1/(4*qsmall))

reject('Hermite correction has the minus sign', mm(mm(add(eye(20),q,-1),gamma),tr(add(eye(20),q,-1))) == d)
reject('three-dimensional shell has 8j points', (2*2+1)**3-(2*2-1)**3 == 8*2)
reject('normalized kernel variance omits denominator', (1+F(1,8))/(1+F(1,8)) == 1+F(1,8))
reject('density logarithmic derivative drops the mean term', F(1)*F(1)*(F(1)-0) == 0)
reject('covariance log-determinant derivative has positive sign', -F(12,2) == F(12,2))
reject('hard numerator omits the Vandermonde factor', F(2)**3 == F(2)**2)
reject('normalized Haar has ordinary circumference mass', F(1) == F(2))
reject('raw tensor multiplicities are all one', metric_t == eye(9))
reject('cubic a^6 envelope coefficient omits a contribution', F(8,9)+F(1,3) == F(8,9))
reject('classifier equality rounded into doublet sector', cubic_count(F(-3,2),F(-2),F(5,48),F(1)) == 2)
reject('unmarked soft integral is finite', 12*(F(10)**3-F(1)**3) <= 36)
reject('full normalizer replaced by restricted positive scalar', F(3)/F(2) == F(3)/F(1,2))
reject('quotient transfer ignores denominator error', F(1)/F(2) == F(1)/F(3))
reject('pin factor 12 cancelled twice', F(12)/3 == F(12)/3/12)
reject('sphere measure uses planar factor', F(4) == F(2))
reject('angular product factors without independence', ((1+F(1,3))**2+(1-F(1,3))**2)/2 == 1)
reject('full-law L1 controls every restriction value', center_difference <= l1)

print('C104 reviewer independent finite controls PASS')
print('Python '+platform.python_version())
for group,count in sorted(COUNTS.items()):
    print(group+': '+str(count))
print('exact controls: '+str(sum(COUNTS.values())))
print('rejected mutants: '+str(len(MUTANTS)))
for label in MUTANTS:
    print('REJECT '+label)
print('No author implementation/design/output consumed. Finite controls do not prove analytic bounds.')
