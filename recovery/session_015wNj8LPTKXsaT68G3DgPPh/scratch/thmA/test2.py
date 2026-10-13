from fractions import Fraction as Fr
import theorem_a as ta
from theorem_a import IV
law = ta.band_law(Fr(1, 1024), Fr(1, 512))
b = IV(0, Fr(1, 64)); k = IV(Fr(2) - Fr(1, 512), Fr(2))
mu_w = b * law.alpha['w'] + k * law.beta['w']
m_lam = -mu_w
xq = m_lam / law.sw
print('mu_w', float(mu_w.lo), float(mu_w.hi), 'sw', float(law.sw.lo), float(law.sw.hi))
print('xq', float(xq.lo), float(xq.hi))
P = ta.Phi(xq); ph = ta.phi_interval(xq)
print('Phi', float(P.lo), float(P.hi), 'phi', float(ph.lo), float(ph.hi))
El2 = (m_lam.sq() + law.sw2) * P + m_lam * law.sw * ph
print('El2', float(El2.lo), float(El2.hi))
print('36 klo^2 El2.lo', float(36 * k.lo ** 2 * El2.lo))
