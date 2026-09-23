#!/usr/bin/env python3
"""
C095 exact six-pin conditional covariance factorization.

For the exact normalized periodized Bargmann--Fock covariance

    K_L((x,y),(x',y')) = k_L(x-x') k_L(y-y'),

and six observations (f, f_x, f_y) at two points on y=0, parity splits the
pin Gram matrix into an even block E=(f,f_x at both points) and an odd block
O=(f_y at both points).

For target values X=(x,y), X'=(x',y'),

  C_6(X,X')
    = k(x-x') k(y-y')
      - k(y)k(y') e_r(x)^T E_r^{-1} e_r(x')
      - k'(y)k'(y') o_r(x)^T O_r^{-1} o_r(x').

The identity is exact for every r and every target pair. Differentiating it
gives every conditional derivative covariance needed by BR-MARK.

Outputs:
    C095_SIX_PIN_COVARIANCE_FACTORIZATION.json
"""

from __future__ import annotations

import json
from pathlib import Path
import mpmath as mp

BASE = Path(__file__).resolve().parent
mp.mp.dps = 90


def theta_derivative(t: mp.mpf, L: mp.mpf, order: int, shells: int = 5) -> mp.mpf:
    total = mp.mpf("0")
    for n in range(-shells, shells + 1):
        z = t + n * L
        if order == 0:
            he = 1
        elif order == 1:
            he = z
        elif order == 2:
            he = z**2 - 1
        elif order == 3:
            he = z**3 - 3*z
        elif order == 4:
            he = z**4 - 6*z**2 + 3
        elif order == 5:
            he = z**5 - 10*z**3 + 15*z
        elif order == 6:
            he = z**6 - 15*z**4 + 45*z**2 - 15
        else:
            raise ValueError("orders 0--6 supported")
        total += (-1)**order * he * mp.e**(-z*z/2)
    return total


def k(t: mp.mpf, L: mp.mpf, order: int = 0) -> mp.mpf:
    return theta_derivative(t, L, order) / theta_derivative(mp.mpf("0"), L, 0)


def cov(
    p: tuple[mp.mpf, mp.mpf],
    a: tuple[int, int],
    q: tuple[mp.mpf, mp.mpf],
    c: tuple[int, int],
    L: mp.mpf,
) -> mp.mpf:
    dx, dy = p[0]-q[0], p[1]-q[1]
    return (
        (-1)**(c[0]+c[1])
        * k(dx, L, a[0]+c[0])
        * k(dy, L, a[1]+c[1])
    )


def full_conditional_value_cov(
    X: tuple[mp.mpf, mp.mpf],
    Y: tuple[mp.mpf, mp.mpf],
    r: mp.mpf,
    L: mp.mpf,
) -> mp.mpf:
    pins = [(-r/2, mp.mpf("0")), (r/2, mp.mpf("0"))]
    funcs = []
    for p in pins:
        funcs.extend([(p,(0,0)),(p,(1,0)),(p,(0,1))])
    G = mp.matrix(6)
    cX = mp.matrix(1,6)
    cY = mp.matrix(6,1)
    for i,(p,a) in enumerate(funcs):
        cX[0,i] = cov(X,(0,0),p,a,L)
        cY[i,0] = cov(p,a,Y,(0,0),L)
        for j,(q,b) in enumerate(funcs):
            G[i,j] = cov(p,a,q,b,L)
    return cov(X,(0,0),Y,(0,0),L) - (cX*mp.inverse(G)*cY)[0]


def blocks(r: mp.mpf, L: mp.mpf):
    pins = [(-r/2,mp.mpf("0")),(r/2,mp.mpf("0"))]
    even = []
    odd = []
    for p in pins:
        even.extend([(p,(0,0)),(p,(1,0))])
        odd.append((p,(0,1)))
    E = mp.matrix(4)
    O = mp.matrix(2)
    for i,(p,a) in enumerate(even):
        for j,(q,b) in enumerate(even):
            E[i,j] = cov(p,a,q,b,L)
    for i,(p,a) in enumerate(odd):
        for j,(q,b) in enumerate(odd):
            O[i,j] = cov(p,a,q,b,L)
    return pins, even, odd, E, O


def e_vector(x: mp.mpf, r: mp.mpf, L: mp.mpf) -> mp.matrix:
    pins = [-r/2, r/2]
    values = []
    for p in pins:
        # Cov(f(x,0), f(p,0)); Cov(f(x,0), f_x(p,0)).
        values.extend([k(x-p,L,0), -k(x-p,L,1)])
    return mp.matrix(values)


def o_vector(x: mp.mpf, r: mp.mpf, L: mp.mpf) -> mp.matrix:
    return mp.matrix([k(x+r/2,L,0), k(x-r/2,L,0)])


def factored_conditional_value_cov(
    X: tuple[mp.mpf,mp.mpf],
    Y: tuple[mp.mpf,mp.mpf],
    r: mp.mpf,
    L: mp.mpf,
) -> mp.mpf:
    _,_,_,E,O = blocks(r,L)
    ex = e_vector(X[0],r,L)
    ey = e_vector(Y[0],r,L)
    ox = o_vector(X[0],r,L)
    oy = o_vector(Y[0],r,L)
    base = k(X[0]-Y[0],L,0)*k(X[1]-Y[1],L,0)
    even_term = k(X[1],L,0)*k(Y[1],L,0)*(ex.T*mp.inverse(E)*ey)[0]
    odd_term = k(X[1],L,1)*k(Y[1],L,1)*(ox.T*mp.inverse(O)*oy)[0]
    return base-even_term-odd_term


def main() -> None:
    L=mp.mpf("24")
    rungs=[mp.mpf("0.05"),mp.mpf("0.025"),mp.mpf("0.0125")]
    targets=[
        ((mp.mpf("-0.4"),mp.mpf("0.2")),(mp.mpf("0.7"),mp.mpf("0.8"))),
        ((mp.mpf("0.1"),mp.mpf("-0.5")),(mp.mpf("1.2"),mp.mpf("0.3"))),
        ((mp.mpf("1.7"),mp.mpf("0.9")),(mp.mpf("-0.8"),mp.mpf("-0.4"))),
    ]
    rows=[]
    worst=mp.mpf("0")
    block_cross_worst=mp.mpf("0")
    for r in rungs:
        pins,even,odd,E,O=blocks(r,L)
        # Explicit parity block check.
        for p,a in even:
            for q,b in odd:
                block_cross_worst=max(block_cross_worst,abs(cov(p,a,q,b,L)))
        for X,Y in targets:
            direct=full_conditional_value_cov(X,Y,r,L)
            factored=factored_conditional_value_cov(X,Y,r,L)
            err=abs(direct-factored)
            worst=max(worst,err)
            rows.append({
                "r":str(r),
                "X":[str(X[0]),str(X[1])],
                "Y":[str(Y[0]),str(Y[1])],
                "direct":mp.nstr(direct,50),
                "factored":mp.nstr(factored,50),
                "absolute_error":mp.nstr(err,25),
            })

    report={
        "cycle":"C095",
        "result_id":"SIX_PIN_CONDITIONAL_COVARIANCE_FACTORIZATION",
        "grade":"DERIVED-EXACT",
        "statement":(
            "C6(X,Y)=kx(x-x')ky(y-y')-ky(y)ky(y') e(x)^T E^-1 e(x')"
            "-ky'(y)ky'(y') o(x)^T O^-1 o(x')"
        ),
        "proof":[
            "The exact periodized covariance factorizes into one-dimensional x and y kernels.",
            "At y=0, parity makes the (f,fx) pin block orthogonal to the fy block.",
            "The six-pin Gram matrix is block diagonal after permutation.",
            "Target cross-covariances with the even block factor by k(y); with the odd block they factor by -k'(y).",
            "Substitution into the Schur complement gives the displayed identity.",
            "Differentiation under the exact analytic identity gives all derivative covariance formulas."
        ],
        "block_dimensions":{"even":4,"odd":2},
        "parity_cross_block_worst":mp.nstr(block_cross_worst,30),
        "numeric_checks_90dps":rows,
        "worst_absolute_error":mp.nstr(worst,30),
        "gate_effect":{
            "BR_MARK":(
                "Substantially reshaped. Six-pin value/derivative covariance "
                "reduces to two one-dimensional Schur kernels E_r and O_r. "
                "Uniform eigenvalue bounds, two-saddle Hessian typing, "
                "determinant Palm weights, and the final (A,Z) density bound remain."
            ),
            "COMMON_MODE":(
                "The algebraic proof is exact and exempt from agreement-based "
                "promotion. The full-6x6 numerical comparison is diagnostic."
            )
        }
    }
    path=BASE/"C095_SIX_PIN_COVARIANCE_FACTORIZATION.json"
    path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
