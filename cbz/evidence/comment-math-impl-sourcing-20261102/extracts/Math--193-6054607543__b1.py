"""Exact rational diagnostics for the conventional enlarged P3 proof, not proof.

No inference, numerical Gaussian sampler, source modification or paid API call.
Rank diagnostics omit positive mode rescalings, which preserve exact rank.
"""
from fractions import Fraction as F
from itertools import combinations
from math import comb
from pathlib import Path
import hashlib
import json

CHECKS = []


def check(label, condition):
    if not condition:
        raise AssertionError(label)
    CHECKS.append(label)


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def transpose(a):
    return list(map(list, zip(*a)))


def mv(a, v):
    return [dot(row, v) for row in a]


def mm(a, b):
    return [[dot(row, col) for col in transpose(b)] for row in a]


def identity(n):
    return [[F(i == j) for j in range(n)] for i in range(n)]


def rank(a):
    a = [list(row) for row in a]
    cursor = 0
    for col in range(len(a[0])):
        pivot = next((i for i in range(cursor, len(a)) if a[i][col]), None)
        if pivot is None:
            continue
        a[cursor], a[pivot] = a[pivot], a[cursor]
        divisor = a[cursor][col]
        a[cursor] = [v/divisor for v in a[cursor]]
        for i in range(len(a)):
            if i != cursor:
                coefficient = a[i][col]
                a[i] = [x-coefficient*y for x, y in zip(a[i], a[cursor])]
        cursor += 1
        if cursor == len(a):
            break
    return cursor


def unpack(z, n):
    A, lam = z[0], z[1:n+1]
    B, D, C = z[n+1], z[n+2:2*n+1], z[2*n+1]
    pairs = list((i, j) for i in range(1, n) for j in range(i, n))
    H = dict(zip(pairs, z[2*n+2:]))
    q = [[F(0) for _ in range(n)] for _ in range(n)]
    q[0][0] = B
    for j in range(1, n):
        q[0][j] = q[j][0] = D[j-1]/2
    for (i, j), value in H.items():
        q[i][j] = q[j][i] = value if i == j else value/2
    return A, lam, B, D, C, H, q


def symbols(k, z, t, c=F(2, 3)):
    n = len(t)
    A, lam, _, _, C, _, q = unpack(z, n)
    xi = [c*v for v in k]
    u = [row[0] for row in t]
    physical_q = mm(mm(t, q), transpose(t))
    return A-dot(xi, mv(physical_q, xi)), dot(xi, mv(t, lam))-C*dot(xi, u)**3


def frequencies(n, doubled=True, paired=True):
    e = identity(n)
    result = list(e)
    if doubled:
        result += [[2*v for v in row] for row in e]
    if paired:
        result += [[e[i][k]+e[j][k] for k in range(n)] for i, j in combinations(range(n), 2)]
    return result


def coefficient_matrix(n, t, doubled=True, paired=True):
    N = 2+n+n*(n+1)//2
    basis = identity(N)
    rows = [[z[0] for z in basis]]
    for k in frequencies(n, doubled, paired):
        values = [symbols(k, z, t) for z in basis]
        rows.extend([[value[channel] for value in values] for channel in (0, 1)])
    return rows


def derivative(p, x):
    return F(0) if p == 0 else p*x**(p-1)


def peano(p, x, r):
    if p < 3:
        return F(0)
    degree = p-3
    integral = sum((F(comb(degree, i))*(x-r/2)**(degree-i)*r**i
                    *(F(1, i+2)-F(1, i+3)) for i in range(degree+1)), F(0))
    return 6*p*(p-1)*(p-2)*integral


for n in (2, 3, 4):
    N = 2+n+n*(n+1)//2
    # P's native axial4/transversepair order needs one common permutation.
    permutation = [0, 1] + list(range(4, 2*n+2, 2)) + [2] + list(range(5, 2*n+2, 2)) + [3]
    native_labels = ['value', 'axial_gradient', 'axial_second', 'axial_third']
    native_labels += [label for j in range(1, n) for label in (f'gradient{j}', f'mixed{j}')]
    grouped_labels = ['value', 'axial_gradient'] + [f'gradient{j}' for j in range(1, n)]
    grouped_labels += ['axial_second'] + [f'mixed{j}' for j in range(1, n)] + ['axial_third']
    check(f'n{n}_shared_row_permutation', [native_labels[i] for i in permutation] == grouped_labels)
    target_native = [F(7)-F(2)*F(1, 3)**3/2, -F(2)*F(1, 3)**2, F(0), F(24)] + [F(0)]*(2*n-2)
    target_grouped = [target_native[0], target_native[1]] + [F(0)]*(n-1) + [F(0)]*n + [F(24)]
    check(f'n{n}_prescribed_target_same_permutation', [target_native[i] for i in permutation] == target_grouped)
    # f(x,y)=y1: native row4=1, grouped row2=1, for every radius.
    native_linear_field = [F(0)]*(2*n+2)
    native_linear_field[4] = F(1)
    contact_linear_field = [F(0)]*(2*n+2)
    contact_linear_field[2] = F(1)
    check(f'n{n}_wrong_order_nonconvergence_detected', native_linear_field != contact_linear_field
          and [native_linear_field[i] for i in permutation] == contact_linear_field)
    ident = identity(n)
    reflected = [row[::-1] for row in ident]
    rotated = identity(n)
    rotated[0][0], rotated[0][1] = F(3, 5), F(-4, 5)
    rotated[1][0], rotated[1][1] = F(4, 5), F(3, 5)
    shear = identity(n)
    shear[0][1] = F(1)
    for label, t in [('identity', ident), ('reflection', reflected), ('rational_rotation', rotated), ('shear', shear)]:
        matrix = coefficient_matrix(n, t)
        check(f'n{n}_{label}_full_rank', rank(matrix) == N)
        check(f'n{n}_{label}_duplicate_observation_singular', rank([row+[row[-1]] for row in matrix]) == N < N+1)
        z = [F(i+1, i+2) for i in range(N)]
        _, _, B, D, _, H, q = unpack(z, n)
        xi = [F(2*i+1, 3) for i in range(n)]
        eta = mv(transpose(t), xi)
        raw = B*eta[0]**2 + sum((D[j-1]*eta[0]*eta[j] for j in range(1, n)), F(0))
        raw += sum((value*eta[i]*eta[j] for (i, j), value in H.items()), F(0))
        check(f'n{n}_{label}_offdiagonal_half_packing', dot(xi, mv(mm(mm(t, q), transpose(t)), xi)) == raw)
    check(f'n{n}_no_doubled_modes_cubic_alias', rank(coefficient_matrix(n, ident, doubled=False)) == N-1)
    check(f'n{n}_no_pair_modes_quadratic_missing', rank(coefficient_matrix(n, ident, paired=False)) == N-n*(n-1)//2)
    f = frequencies(n)
    check(f'n{n}_witness_set_distinct_positive_half', len({tuple(k) for k in f}) == len(f)
          and all(next(v for v in k if v) > 0 for k in f))

co, si, R, S = F(3, 5), F(4, 5), F(7, 3), F(-11, 4)
paired_square = (R*co-S*si)**2 + (R*si+S*co)**2
check('paired_phase_removal', paired_square == R**2+S**2)
check('extra_factor_two_rejected', 2*paired_square != R**2+S**2)
check('peano_nonnegative_kernel_unit_mass', 6*(F(1, 2)-F(1, 3)) == 1)
for x, r in ((F(0), F(1, 7)), (F(3, 5), F(2, 3)), (F(-4, 3), F(5, 7))):
    a, b = x-r/2, x+r/2
    for p in range(9):
        fa, fb = a**p, b**p
        da, db = derivative(p, a), derivative(p, b)
        u0, u1 = (fa+fb)/2, (fb-fa)/r
        u2 = (db-da)/r
        u3 = 6/r**2*(da+db-2*u1)
        check(f'peano_p{p}_x{x}_r{r}', u3 == peano(p, x, r))
        check(f'inverse_p{p}_x{x}_r{r}',
              u0-r*u1/2 == fa and u0+r*u1/2 == fb
              and u1-r*u2/2+r**2*u3/12 == da
              and u1+r*u2/2+r**2*u3/12 == db)

r = F(2, 7)
a, b = -r/2, r/2
phi = lambda s: s**3-3*r**2*s/4
grad = lambda s: 3*s**2-3*r**2/4
gap = phi(a)-phi(b)
u3 = 6/r**2*(grad(a)+grad(b)-2*(phi(b)-phi(a))/r)
check('stationary_cubic_physical_gap_factor12', grad(a) == grad(b) == 0 and u3 == 6 == 12*gap/r**3)
check('weakened_peano_factor3_rejected', u3/2 != 12*gap/r**3)
raw = [F(2), F(3), F(5)]
raw_sq = sum(v*v for v in raw)
frob_sq = raw[0]**2+2*raw[1]**2+raw[2]**2
check('raw_symmetric_frobenius_bounds', raw_sq <= frob_sq <= 2*raw_sq)

proof = Path(__file__).with_name('ACTUAL_ENLARGED_CONTACT_SMALL_RADIUS.md')
print(json.dumps({'status': 'PASS', 'exact_rational_checks': len(CHECKS),
                  'checks': CHECKS, 'proof_bytes': len(proof.read_bytes()),
                  'proof_sha256': hashlib.sha256(proof.read_bytes()).hexdigest(),
                  'scope': 'diagnostics only; no sampler, floor enclosure, kernel or scientific acceptance'}, indent=2))

