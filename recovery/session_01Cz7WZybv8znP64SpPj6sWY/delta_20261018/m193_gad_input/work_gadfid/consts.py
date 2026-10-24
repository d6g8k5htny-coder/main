import sys
from fractions import Fraction
try:
    import mpmath as mp
except Exception as e:
    mp=None; print("no mpmath",e)
if mp:
    mp.mp.dps=40
    # 144*12^(-7/3) == 12^(-1/3)
    print("144*12^-7/3 - 12^-1/3 =", 144*mp.power(12,-mp.mpf(7)/3)-mp.power(12,-mp.mpf(1)/3))
    # 2^(2/3)/(2*12^(1/3)) == 24^(-1/3)
    print("2^(2/3)/(2*12^(1/3)) - 24^(-1/3) =", mp.power(2,mp.mpf(2)/3)/(2*mp.cbrt(12))-mp.power(24,-mp.mpf(1)/3))
    # E|N(0,1)|^(4/3) by quadrature vs closed form
    q=mp.quad(lambda t: abs(t)**(mp.mpf(4)/3)*mp.exp(-t*t/2)/mp.sqrt(2*mp.pi),[-mp.inf,0,mp.inf])
    cf=mp.power(2,mp.mpf(2)/3)*mp.gamma(mp.mpf(7)/6)/mp.sqrt(mp.pi)
    print("E|Z|^(4/3) quad - closed =", q-cf)
    # E (Z_+)^(4/3) = half
    qp=mp.quad(lambda t: t**(mp.mpf(4)/3)*mp.exp(-t*t/2)/mp.sqrt(2*mp.pi),[0,mp.inf])
    print("E Z_+^(4/3) - half =", qp-cf/2)
    # (27) with T_+ half moment == (28) with |T| moment
    print("12^(-1/3)*1/2 - 1/(2*12^(1/3)) =", mp.power(12,-mp.mpf(1)/3)/2-1/(2*mp.cbrt(12)))
    # full: (28) prefactor * p_D p_He * E|N(0,V)|^(4/3) => Gamma(7/6)/(24^(1/3) sqrt(pi)) (2pi)^-d a^-d/2 detB^-1/2 V^(2/3)
    for d in (2,3,4):
        a=mp.mpf('0.7'); detB=mp.mpf('1.3'); V=mp.mpf('2.1'); Fe=mp.mpf('0.37')
        lhs=1/(2*mp.cbrt(12))*(2*mp.pi)**(-mp.mpf(d)/2)*a**(-mp.mpf(d)/2)*(2*mp.pi)**(-mp.mpf(d)/2)*detB**(-mp.mpf(1)/2)*mp.power(V,mp.mpf(2)/3)*cf*Fe
        rhs=mp.gamma(mp.mpf(7)/6)/(mp.cbrt(24)*mp.sqrt(mp.pi)*(2*mp.pi)**d*a**(mp.mpf(d)/2))*Fe*mp.power(V,mp.mpf(2)/3)/mp.sqrt(detB)
        print("d",d,"(17)->(1) integrand diff", lhs-rhs)
