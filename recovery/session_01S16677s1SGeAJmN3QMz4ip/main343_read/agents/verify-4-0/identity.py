from mpmath import mp, mpf, quad, sqrt
mp.dps = 50
# core integrand of direct_mass in theta variable, at saturation s=1 (root of 36 s^3 = (9/4)(3+s^2)^2)
f = lambda t: t * (9 - t**2) * (mpf(3) / (3 + t**2)) ** mpf('2.5')
core = quad(f, [0, 1]) / 15
print('(1/15) int_0^1 t(9-t^2)(3/(3+t^2))^(5/2) dt =', core)
print('minus 1/5 =', core - mpf(1) / 5)
# check s=1 solves the split-root equation at tau=9/4 exactly
print('36*1^3 - (9/4)*(3+1)^2 =', 36 - mpf(9) / 4 * 16)
