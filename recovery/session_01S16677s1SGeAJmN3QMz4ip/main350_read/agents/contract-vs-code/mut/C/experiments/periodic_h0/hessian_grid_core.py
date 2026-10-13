"""Exact Hessian grid bounds with fourth-derivative interpolation covers.

The input dyadics define one finite polynomial. All arithmetic is integer or
rational. No FFT library, Gaussian law, barcode or infinite field is certified.
"""
from fractions import Fraction as Q
from math import isqrt
import finite_certificate as fc
import nodal_core as nc


def bound_record(record, side, grid):
    """Return JSON-safe bounds; a sample mesh h has spatial budget h²*coefficient.

    `grid` is the Hessian evaluation mesh, not necessarily the sample mesh.
    Sparse records are accepted; the consuming coefficient validator binds the
    complete source record. Bounds on transformed centers are before q² scaling
    only in `grid_center_abs_max` and `grid_hessian_center_norm`, with q=2π/side.
    """
    fc.require(type(side) is int and side > 0, 'Positive integer side required')
    # Validate the ORIGINAL object before zero derivative multipliers or the
    # zero derivative DC could hide invalid source coefficients/metadata.
    nc._input(record, grid)
    q_hi = 2*fc.pi_bounds()[1]/side
    q2, q4 = q_hi**2, q_hi**4
    h2 = Q(side**2, grid**2)

    # A = q_hi^4 sum rho_plus (|kx|+|ky|)^2 k k^T is PSD.
    # The absolute cross moment is also needed for the scalar f_xy cover.
    axx = axy = ayy = cross_absolute = Q(0)
    for x, y, real, imag in record['modes']:
        a, b = fc.rational(real), fc.rational(imag)
        weight = 2*fc.sqrt_bounds(a*a+b*b)[1]*(abs(x)+abs(y))**2
        axx += weight*x*x
        axy += weight*x*y
        ayy += weight*y*y
        cross_absolute += weight*abs(x*y)
    axx, axy, ayy, cross_absolute = (q4*v for v in (axx, axy, ayy, cross_absolute))
    a_norm = (axx+ayy+fc.sqrt_bounds((axx-ayy)**2+4*axy*axy)[1])/2

    components, centers_by_component, errors = {}, [], []
    for key, dx, dy, moment in (('xx', 2, 0, axx),
                                ('xy', 1, 1, cross_absolute),
                                ('yy', 0, 2, ayy)):
        transformed = {'seed': record['seed'], 'dc': ['0', '0'],
                       'modes': [[x, y, str(-x**dx*y**dy*fc.rational(a)),
                                  str(-x**dx*y**dy*fc.rational(b))]
                                 for x, y, a, b in record['modes']]}
        centers, stages, error = nc.evaluate(transformed, grid)
        reals = [z[0] for z in centers]
        maximum = Q(max(abs(v) for v in reals), nc.SCALE)
        nodal = q2*(maximum+error)
        cover = h2*moment/8
        components[key] = {
            'center_object_sha256': fc.digest([[str(a), str(b)] for a, b in centers]),
            'stages': stages, 'center_error_bound': str(error),
            'grid_center_abs_max': str(maximum), 'grid_abs_bound': str(nodal),
            'interpolation_cover': str(cover), 'global_bound': str(nodal+cover)}
        centers_by_component.append(reals)
        errors.append(error)

    # For a real symmetric 2x2 matrix C, ||C||op equals
    # (|Cxx+Cyy| + sqrt((Cxx-Cyy)^2+4Cxy^2))/2. The integer
    # ceiling square root encloses this for every grid node.
    largest = 0
    for a, b, c in zip(*centers_by_component):
        square = (a-c)**2+4*b*b
        root = isqrt(square)
        root += int(root*root != square)
        largest = max(largest, abs(a+c)+root)
    center_norm = Q(largest, 2*nc.SCALE)
    exx, exy, eyy = errors
    # Symmetric error matrix: ||E||op <= maximum absolute row sum.
    matrix_error = q2*max(exx+exy, eyy+exy)
    grid_norm = q2*center_norm+matrix_error
    operator_cover = h2*a_norm/8
    H = grid_norm+operator_cover
    Mxx, Mxy, Myy = (Q(components[k]['global_bound']) for k in ('xx', 'xy', 'yy'))
    coefficient = min(H/4, (Mxx+Myy)/8+Mxy/4)
    return {
        'grid': grid, 'side': side, 'fixed_point_scale': str(nc.SCALE),
        'components': components,
        'fourth_matrix': {'xx': str(axx), 'xy': str(axy), 'yy': str(ayy),
                          'operator_bound': str(a_norm)},
        'grid_hessian_center_norm': str(center_norm),
        'grid_hessian_error_bound': str(matrix_error),
        'grid_hessian_norm_bound': str(grid_norm),
        'operator_interpolation_cover': str(operator_cover),
        'Mxx': str(Mxx), 'Mxy': str(Mxy), 'Myy': str(Myy), 'H': str(H),
        'spatial_coefficient': str(coefficient)}
