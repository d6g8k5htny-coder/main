"""Exact fixed-point inverse DFT centers and proved component error majorants.

Input dyadics define a finite Fourier polynomial, not a Gaussian sampling law.
The returned error bounds each real/imaginary component of every exact node.
No floating-point operation, NumPy FFT, or persistence calculation is used.
"""
from fractions import Fraction as Q
from functools import lru_cache
from math import factorial
import finite_certificate as fc

SCALE = 1 << 96


def nearest_integer(q):
    """Round exact rational to nearest integer, breaking ties toward +infinity."""
    fc.require(type(q) is Q, 'Exact rational required for rounding')
    return (2*q.numerator+q.denominator)//(2*q.denominator)


def _grid_size(n):
    fc.require(type(n) is int and 2 <= n <= 1024 and n & (n-1) == 0,
               'Supported grid is a power of two from 2 through 1024')


@lru_cache(maxsize=1)
def twiddle_component_error():
    """Enclose the Taylor remainder plus angle error and fixed-point rounding."""
    lo, hi = fc.pi_bounds()
    # After quadrant/reflection reduction, theta=q*pi with 0<=q<=1/4.
    # Both degree31 Taylor polynomials have remainder <= |theta|^32/32!;
    # the zero odd coefficients make cos's displayed degree only 30.
    # Centering pi gives angle error <=(hi-lo)/8. Sin/cos are 1-Lipschitz.
    analytic_error = (hi/4)**32/factorial(32)+(hi-lo)/8
    fc.require(analytic_error <= Q(1, 2*SCALE), 'Twiddle analytic error exceeds budget')
    # Nearest integer rounding contributes <=1/(2*SCALE) per component.
    return Q(1, SCALE)


def twiddle_center(k, n):
    """Integer center for exp(+2*pi*i*k/n), each component error <=1/SCALE."""
    _grid_size(n)
    fc.require(type(k) is int, 'Integer frequency required')
    twiddle_component_error()
    k %= n
    quadrant = (4*k)//n
    q = Q(2*k, n)-Q(quadrant, 2)
    reflect = q > Q(1, 4)
    if reflect:
        q = Q(1, 2)-q
    lo, hi = fc.pi_bounds()
    x = q*(lo+hi)/2
    sine = sum((Q((-1)**j, factorial(2*j+1))*x**(2*j+1) for j in range(16)), Q(0))
    cosine = sum((Q((-1)**j, factorial(2*j))*x**(2*j) for j in range(16)), Q(0))
    c, s = nearest_integer(cosine*SCALE), nearest_integer(sine*SCALE)
    if reflect:
        c, s = s, c
    return ((c, s), (-s, c), (-c, -s), (s, -c))[quadrant]


@lru_cache(maxsize=10)
def _twiddles(n):
    return tuple(twiddle_center(k, n) for k in range(n//2))


def _scaled(value):
    q = fc.rational(value)*SCALE
    fc.require(q.denominator == 1, 'Coefficient is not exactly representable at fixed precision')
    return q.numerator


def _input(record, n):
    _grid_size(n)
    fc.require(type(record) is dict and set(record) == {'seed', 'dc', 'modes'}, 'Unexpected Fourier record schema')
    fc.require(type(record['seed']) is int and record['seed'] >= 0, 'Invalid seed label')
    dc = record['dc']
    fc.require(type(dc) is list and len(dc) == 2, 'Malformed DC coefficient')
    dc_real, dc_imag = _scaled(dc[0]), _scaled(dc[1])
    fc.require(dc_imag == 0, 'DC coefficient must be real')
    modes = record['modes']
    fc.require(type(modes) is list, 'Fourier modes must be a list')
    real, imag = [0]*(n*n), [0]*(n*n)
    real[0] = dc_real
    previous = None
    for mode in modes:
        fc.require(type(mode) is list and len(mode) == 4, 'Malformed Fourier mode')
        x, y, a, b = mode
        fc.require(type(x) is int and type(y) is int and (x > 0 or (x == 0 and y > 0)), 'Invalid half-plane frequency')
        fc.require(n > 2*max(abs(x), abs(y)), 'Grid aliases a retained frequency')
        fc.require(previous is None or (x, y) > previous, 'Duplicate or unordered Fourier mode')
        previous = x, y
        u, v = _scaled(a), _scaled(b)
        positive, negative = (x % n)*n+(y % n), ((-x) % n)*n+((-y) % n)
        real[positive], imag[positive] = u, v
        real[negative], imag[negative] = u, -v
    return real, imag


def evaluate(record, n):
    """Return (row-major integer complex centers, JSON-safe stages, rational E).

    Each exact finite-polynomial node differs from its returned center/SCALE
    by at most E in each component. Transform rows (axis1) before columns
    (axis0), with positive exponent and no normalization in either axis.
    Sparse records are permitted here; a parent coefficient-object validator
    enforces any claimed complete mode set or source identity separately.
    """
    real, imag = _input(record, n)
    twiddles = _twiddles(n)
    bits = n.bit_length()-1
    reverse = [int(f'{j:0{bits}b}'[::-1], 2) for j in range(n)]
    error, stages = Q(0), []
    for axis in ('rows', 'columns'):
        stride = 1 if axis == 'rows' else n
        for line in range(n):
            base = line*n if axis == 'rows' else line
            for j, reflected in enumerate(reverse):
                if j < reflected:
                    a, b = base+j*stride, base+reflected*stride
                    real[a], real[b] = real[b], real[a]
                    imag[a], imag[b] = imag[b], imag[a]
        for exponent in range(1, bits+1):
            length = 1 << exponent
            half = length//2
            maximum = max(abs(a)+abs(b) for a, b in zip(real, imag))
            for line in range(n):
                base = line*n if axis == 'rows' else line
                for start in range(0, n, length):
                    for j in range(half):
                        a = base+(start+j)*stride
                        b = a+half*stride
                        wr, wi = twiddles[j*n//length]
                        vr, vi = real[b], imag[b]
                        # Full complex numerator is formed before one rounding
                        # per component. All additions and products are integers.
                        tr = (2*(wr*vr-wi*vi)+SCALE)//(2*SCALE)
                        ti = (2*(wr*vi+wi*vr)+SCALE)//(2*SCALE)
                        ur, ui = real[a], imag[a]
                        real[a], imag[a] = ur+tr, ui+ti
                        real[b], imag[b] = ur-tr, ui-ti
            # |cos|+|sin| <= sqrt(2) <3/2. Each twiddle component
            # error <=1/SCALE; V's component l1 norm <=maximum/SCALE.
            error = Q(5, 2)*error+Q(maximum, SCALE*SCALE)+Q(1, 2*SCALE)
            stages.append({'axis': axis, 'length': length,
                           'input_l1_integer_max': maximum, 'error_bound': str(error)})
    return list(zip(real, imag)), stages, error
