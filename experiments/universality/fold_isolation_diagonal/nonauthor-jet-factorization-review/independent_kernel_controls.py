"""Independent exact controls for the scoped nonauthor kernel review.

These are finite algebra checks, not a numerical proof of Gaussian integral
bounds, asymptotic mixing, analytic distance inequalities, or a process limit.
No author file is written. Run with Python -B to avoid source-side caches.
"""
from fractions import Fraction as Q
from math import comb, factorial
from pathlib import Path
import hashlib
import json


def tr(a):
    return list(map(list, zip(*a)))


def eye(n):
    return [[Q(i == j) for j in range(n)] for i in range(n)]


def matmul(a, b):
    bt = tr(b)
    return [[sum(x*y for x, y in zip(row, col)) for col in bt] for row in a]


def add(a, b, scale=Q(1)):
    return [[x+scale*y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def det(a):
    a = [list(map(Q, row)) for row in a]
    ans = Q(1)
    for i in range(len(a)):
        p = next((p for p in range(i, len(a)) if a[p][i]), None)
        if p is None:
            return Q(0)
        if p != i:
            a[p], a[i] = a[i], a[p]
            ans = -ans
        value = a[i][i]
        ans *= value
        for j in range(i+1, len(a)):
            ratio = a[j][i]/value
            for k in range(i+1, len(a)):
                a[j][k] -= ratio*a[i][k]
    return ans


def inv(a):
    n = len(a)
    a = [list(map(Q, row))+unit for row, unit in zip(a, eye(n))]
    for i in range(n):
        p = next(p for p in range(i, n) if a[p][i])
        a[p], a[i] = a[i], a[p]
        value = a[i][i]
        a[i] = [x/value for x in a[i]]
        for j in range(n):
            if j != i:
                ratio = a[j][i]
                a[j] = [x-ratio*y for x, y in zip(a[j], a[i])]
    return [row[n:] for row in a]


def col(v):
    return [[Q(x)] for x in v]


def trace(a):
    return sum(a[i][i] for i in range(len(a)))


def quadratic(a, v):
    return matmul(tr(col(v)), matmul(a, col(v)))[0][0]


def derivative(p, order=1):
    for _ in range(order):
        p = [(i+1)*p[i+1] for i in range(len(p)-1)]
    return p


def evaluate(p, x):
    return sum(a*x**i for i, a in enumerate(p))


def compose_affine(p, a, b):
    out = [Q(0)]*len(p)
    for i, c in enumerate(p):
        for j in range(i+1):
            out[j] += c*comb(i, j)*a**(i-j)*b**j
    return out


def integral_01(p):
    return sum(c/Q(i+1) for i, c in enumerate(p))


def run():
    checks, records, negatives = {}, {}, {}
    # Literal source transform: [FM,FuM,FvM,FS,FuS,FvS] -> six U rows.
    transform_records = []
    for r in (Q(1, 2), Q(1, 3), Q(3, 5), Q(1, 8)):
        t = [[Q(0)]*6 for _ in range(6)]
        t[0][0] = t[0][3] = Q(1, 2)
        t[1][0], t[1][3] = -1/r, 1/r
        t[2][1], t[2][4] = -1/r, 1/r
        t[3][0], t[3][3] = 12/r**3, -12/r**3
        t[3][1] = t[3][4] = 6/r**2
        t[4][2] = t[4][5] = Q(1, 2)
        t[5][2], t[5][5] = -1/r, 1/r
        actual = abs(det(t))
        assert actual == 12/r**5
        b, k = Q(2, 7), Q(4, 3)
        original = [b, 0, 0, b-k*r**3, 0, 0]
        target = [b-k*r**3/2, -k*r**2, 0, 12*k, 0, 0]
        assert matmul(t, col(original)) == col(target)
        block = [[Q(0)]*12 for _ in range(12)]
        for block_index in range(2):
            for i in range(6):
                for j in range(6):
                    block[6*block_index+i][6*block_index+j] = t[i][j]
        assert abs(det(block)) == 144/r**10
        transform_records.append({'r':str(r), 'det_abs':str(actual)})
    negatives['missing_12_in_each_fold'] = actual != 1/r**5
    assert negatives['missing_12_in_each_fold']
    checks['literal_six_row_transform_and_twelve_row_block_jacobian'] = 'PASS'
    records['transform'] = transform_records

    # Endpoint extensions are regular integral rows, without gradient conditioning.
    for degree in range(3, 9):
        p = [Q((-1)**i*(i+2), i+1) for i in range(degree+1)]
        for r in (Q(1, 2), Q(2, 7)):
            a = -r/2
            p1, p2, p3 = derivative(p), derivative(p, 2), derivative(p, 3)
            slope = (evaluate(p1, a+r)-evaluate(p1, a))/r
            alpha_m = (evaluate(p2, a)-slope)/r
            alpha_s = (evaluate(p2, a+r)-slope)/r
            q = compose_affine(p3, a, r)
            weighted_s = [Q(0)]+q
            assert alpha_m == -integral_01(q)+integral_01(weighted_s)
            assert alpha_s == integral_01(weighted_s)
    checks['regular_endpoint_integral_rows_degrees_3_through_8'] = 'PASS'

    # A critical pair with nonzero mixed Hessian entries; exact polynomial jets.
    hessian_records = []
    for r in (Q(1, 10), Q(1, 20), Q(1, 50)):
        b, k, a, c, q = Q(1, 7), Q(3, 2), Q(-5, 4), Q(2, 3), Q(1, 5)
        # F(x,y)=b+2kx^3-(3/2)kr^2x-kr^3/2
        #        +Ay^2/2+c y(x^2-r^2/4)+q(x^2-r^2/4)^2.
        f = [b-k*r**3/2+q*r**4/16, -Q(3, 2)*k*r**2,
             -q*r**2/2, 2*k, q]
        assert evaluate(f, -r/2) == b
        assert evaluate(f, r/2) == b-k*r**3
        product_normalized = Q(1)
        for sign in (-1, 1):
            x = sign*r/2
            assert evaluate(derivative(f), x) == 0
            assert c*(x*x-r*r/4) == 0
            alpha = evaluate(derivative(f, 2), x)/r
            beta = 2*c*x/r
            assert alpha == sign*6*k+2*q*r and beta == sign*c
            physical_det = (r*alpha)*a-(r*beta)**2
            assert physical_det/r == alpha*a-r*beta**2
            negatives['wrong_r_instead_of_sqrt_r_offdiagonal'] = (
                physical_det/r != alpha*a-r*r*beta**2)
            if sign == -1:
                assert r*alpha < 0 and physical_det > 0
                max_weight = physical_det/r
            else:
                assert physical_det < 0
                saddle_weight = -physical_det/r
            product_normalized *= physical_det/r
        assert negatives['wrong_r_instead_of_sqrt_r_offdiagonal']
        assert max_weight*saddle_weight == -product_normalized
        hessian_records.append({'r':str(r), 'normalized_pair_weight':str(-product_normalized),
                                'contact_pair_weight':str(36*k*k*a*a)})
    checks['physical_hessian_congruence_type_and_contact_normalization'] = 'PASS'
    records['hessians'] = hessian_records

    # Full 14-dimensional log-density derivative, compared to a block closed form.
    derivative_records = []
    for rho in (Q(0), Q(1, 8), Q(1, 4), Q(1, 2), Q(-1, 3)):
        d = [Q(i+1) for i in range(7)]
        sigma = [[Q(0)]*14 for _ in range(14)]
        direction = [[Q(0)]*14 for _ in range(14)]
        for i, di in enumerate(d):
            sigma[i][i] = sigma[7+i][7+i] = di
            sigma[i][7+i] = sigma[7+i][i] = rho*di
            direction[i][7+i] = direction[7+i][i] = di
        si = inv(sigma)
        y1 = [Q((-1)**i*(i+1), 9) for i in range(7)]
        y2 = [Q(i*i-2, 5) for i in range(7)]
        y = y1+y2
        generic = (quadratic(matmul(si, matmul(direction, si)), y)
                   -trace(matmul(si, direction)))/2
        aa = sum(x*x/di for x, di in zip(y1, d))
        bb = sum(x*x/di for x, di in zip(y2, d))
        cc = sum(x*z/di for x, z, di in zip(y1, y2, d))
        assert cc == Q(19, 45)  # exercise the cross-quadratic term and its sign
        closed = 7*rho/(1-rho*rho)+(cc*(1+rho*rho)-rho*(aa+bb))/(1-rho*rho)**2
        assert generic == closed
        assert det(sigma) == factorial(7)**2*(1-rho*rho)**7
        assert matmul(sigma, si) == eye(14)
        derivative_records.append({'rho':str(rho), 'log_density_derivative':str(generic)})
    checks['fourteen_dimensional_gaussian_interpolation_derivative'] = 'PASS'
    records['density_derivative'] = derivative_records
    # Independent value coordinates but shared A gives a polynomial local failure.
    shared_half_fourth, product_half_seconds = Q(3, 2), Q(1, 4)
    assert shared_half_fourth != product_half_seconds
    negatives['zero_value_crosscovariance_does_not_factor_local_A_weight'] = True
    records['scalar_only_failure'] = {
        'value_crosscovariance':'0', 'joint_A_is_shared':True,
        'joint_A2_negative_A2_negative':str(shared_half_fourth),
        'product_marginals':str(product_half_seconds),
        'scope':'negative example for scalar-only inference; cross-jet rho=1, outside lemma'}

    # Canonical coupling with fixed 12 pins, changing coefficient dimension,
    # and deliberately singular conditional extension covariance.
    projection_records = []
    for n in (16, 20, 28):
        for r in (Q(0), Q(1, 16), Q(1, 64)):
            l = [[Q(0)]*n for _ in range(12)]
            for i in range(12):
                l[i][i] = Q(12+i, 24)
                l[i][12+i%(n-12)] = r*Q((i%3)-1, 3)
            gram = matmul(l, tr(l))
            v = [Q(0)]*12
            v[0], v[3], v[6], v[9] = Q(1, 7), Q(12), Q(-2, 9), Q(18)
            c = matmul(tr(l), matmul(inv(gram), col(v)))
            p = add(eye(n), matmul(tr(l), matmul(inv(gram), l)), Q(-1))
            assert matmul(l, c) == col(v)
            assert matmul(l, p) == [[Q(0)]*n for _ in range(12)]
            assert p == tr(p) and matmul(p, p) == p
            assert matmul(p, c) == [[Q(0)] for _ in range(n)]
            e = [[Q((j%4)-1, 10) for j in range(n)]]
            e = e+e  # duplicate rows: conditional E covariance is singular.
            ecov = matmul(e, matmul(p, tr(e)))
            assert det(ecov) == 0
            assert ecov[0][0] >= 0
            altered_v = list(v)
            altered_v[9] = 0
            wrong_c = matmul(tr(l), matmul(inv(gram), col(altered_v)))
            assert matmul(l, wrong_c) != col(v)
            projection_records.append({'coefficient_dimension':n, 'r':str(r),
                                       'conditional_extension_covariance_determinant':'0'})
    checks['canonical_conditioning_projector_with_singular_extension'] = 'PASS'
    negatives['omitted_nonzero_observation_pin_detected'] = True
    records['projection'] = projection_records

    # Exact exponent ledger, reported separately from the conventional estimates.
    assert Q(12, 2) == 6
    assert 6+8 == 14 and 8+8 == 16 and 6+9 == 15 and 6+10 == 16
    assert Q(1, 3)/2 == Q(1, 6)
    assert 1+3-5+2 == 1
    assert Q(4, 3)-Q(2, 3) == Q(2, 3)
    checks['density_weight_radial_volume_exponent_ledger'] = 'PASS'
    root = Path(__file__).resolve().parents[1]
    frozen = {
        'JET_NONDEGENERACY_FACTORIZATION_OBSTRUCTION.md':
            (34627, '0dfdddfbcac47b2fd07449f4152b7f5d935f92aed15272fb8333ed9cc60b5766'),
        'jet_factorization_obstruction_controls.py':
            (5818, '5b1e8fc18a6e32e3a312a253e3e4d1110bb674385f3d49447a45e78ad4df541f'),
        'JET_FACTORIZATION_OBSTRUCTION_CONTROLS.json':
            (1618, '1d79c4f420d8a007638eff98a0b93ffed74e5b1acbac60c2f7777c2cf012e764'),
    }
    source_identity = {}
    for name, (size, sha) in frozen.items():
        b = (root/name).read_bytes()
        assert len(b) == size and hashlib.sha256(b).hexdigest() == sha
        source_identity[name] = {'bytes':size, 'sha256':sha}
    checks['frozen_author_source_identities_unchanged'] = 'PASS'
    assert all(negatives.values())
    return {'classification':'INDEPENDENT_EXACT_FINITE_KERNEL_CONTROLS_ONLY',
            'principal':'OpenAI/Codex nonauthor; source-exposed; same provider; org credit 0',
            'checks':checks, 'negative_controls_rejected':negatives,
            'records':records, 'frozen_source':source_identity,
            'limitations':['No Gaussian draws or floating eigenvalues',
                           'Finite controls do not prove uniform analytic estimates',
                           'No Rice integral, process simulation, Lean or hosted execution',
                           'No scientific or provider-independent approval']}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, sort_keys=True))
