def build(N=6):
    J = {(i, j): sp.Symbol(f'J{i}{j}') for i in range(N + 1) for j in range(N + 1 - i)}
    F = sum(J[(i, j)] * u**i * v**j / (sp.factorial(i) * sp.factorial(j)) for (i, j) in J)
    Fu, Fv = sp.diff(F, u), sp.diff(F, v)

    def at(expr, uu, vv):
        return sp.expand(expr.subs({u: uu, v: vv}, simultaneous=True))

    # ---- S06 eq. (18),(20): exact contact rows, pins at u=-r/2 (minus) and u=+r/2 (plus), v=0
    Fm, Fp = at(F, -rr / 2, 0), at(F, rr / 2, 0)
    Fum, Fup = at(Fu, -rr / 2, 0), at(Fu, rr / 2, 0)
    Fvm, Fvp = at(Fv, -rr / 2, 0), at(Fv, rr / 2, 0)
    U0 = sp.expand((Fm + Fp) / 2)
    U1 = sp.expand(sp.cancel((Fp - Fm) / rr))
    U2 = sp.expand(sp.cancel((Fup - Fum) / rr))
    U3 = sp.expand(sp.cancel(6 / rr**2 * (Fum + Fup - 2 * U1)))
    V0 = sp.expand((Fvm + Fvp) / 2)
    V1 = sp.expand(sp.cancel((Fvp - Fvm) / rr))
    pins = [U0, U1, U2, U3, V0, V1]
    pins = [sp.expand(p.subs(rr, q * s)) for p in pins]   # r = q s
    U0, U1, U2, U3, V0, V1 = pins

    # ---- witness rows (SHARP-SECTOR-01 lines 41-47): x = s z, y = s(z + eta w)
    xu, xv = s * c, s * t
    yu, yv = s * (c + eta * a), s * (t + eta * b)
    Gu, Gv = at(Fu, xu, xv), at(Fv, xu, xv)
    Du = sp.expand(sp.cancel((at(Fu, yu, yv) - Gu) / eta))
    Dv = sp.expand(sp.cancel((at(Fv, yu, yv) - Gv) / eta))

    # ---- residuals, SHARP-SECTOR-01 (3), lines 50-62
    Ru = sp.expand(Gu - U1 - s * (c * U2 + t * V1) - s**2 * c**2 / 2 * U3)
    Rv = sp.expand(Gv - V0 - s * c * V1)
    Su = sp.expand(Du - s * (a * U2 + b * V1) - s**2 * a * c * U3)
    Sv = sp.expand(Dv - s * a * V1)
    W0 = sp.expand(sp.cancel(Rv / (s * t)))
    W1 = sp.expand(sp.cancel(Ru / s**2))
    W2 = sp.expand(sp.cancel(Su / s**2))
    W3 = sp.expand(sp.cancel((Sv - (b / t) * Rv) / s**2))
    # ---- RANK9-01 v2 line 9
    W1s = sp.expand(W1 + q**2 / 24 * U3)
    # ---- candidate
    d = b * c - a * t
    Kz = c**3 * J[(4, 0)] + 3 * c**2 * t * J[(3, 1)] + 3 * c * t**2 * J[(2, 2)] + t**3 * J[(1, 3)]
    Hb = sp.expand(2 * a * W1s + (-c + eta / (2 * a)) * W2 - t * W3 + (c * eta / 2 - eta**2 / (4 * a)) * U3)
    kap = 2 * b / t + eta * b**2 / t**2
    Hd = sp.expand(W2 - kap * W1s - eta * a**2 / 2 * U3)
