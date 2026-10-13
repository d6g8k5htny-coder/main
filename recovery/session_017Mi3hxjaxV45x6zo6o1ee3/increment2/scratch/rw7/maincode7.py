

# ---------------- Owen's T closed form of the Owen-type kinds, and the interval value of F_5 (rules OWEN_EXACT, INNER_LAYERS)
def K_owen(al, g, T):
    """K_{al,g}(T) = int_0^T e^{-al w^2} E_g(w) dw = (pi/sqrt(al g)) [arctan(a)/(2 pi) - T_Owen(sqrt(2 al) T, a)], a = sqrt(g/al) > 1,
    through the reflection T(h, a) = (Phi(h) + Phi(a h))/2 - Phi(h) Phi(a h) - T(a h, 1/a) (h >= 0) and arctan a = pi/2 - arctan(1/a);
    Owen's series of the Math-#197 toolkit with N = 1400 terms (independent of the cumulative Taylor layer)."""
    A = (Iv.frac(g) / Iv.frac(al)).sqrt()
    h = (2 * Iv.frac(al)).sqrt() * Iv.frac(T)
    ah = A * h
    Th = (Phi(h) + Phi(ah)) / 2 - Phi(h) * Phi(ah) - owen_T(ah, 1 / A, N=1400)
    return PI / (Iv.frac(al) * Iv.frac(g)).sqrt() * ((PI / 2 - atan_iv(1 / A, N=400)) / (2 * PI) - Th)


def F_iv(pieces, T):
    """Interval value of F_m(T) from its closed-form pieces (the Owen-type kinds by K_owen); exact rational T."""
    tot = Iv(ZERO)
    Ti = Iv.frac(T)
    for key, poly in pieces.items():
        poly = trim(poly)
        if not poly:
            continue
        pv = Iv.frac(sum(c * Fr(T) ** i for i, c in enumerate(poly)))
        if key[0] == "1":
            tot = tot + pv
        elif key[0] == "e":
            tot = tot + pv * (-(Iv.frac(key[1]) * Ti.sq())).exp()
        elif key[0] == "E":
            tot = tot + pv * (PI / Iv.frac(key[1])).sqrt() / 2 * erf_iv(Iv.frac(key[1]).sqrt() * Ti)
        elif key[0] == "eE":
            al, g = key[1], key[2]
            tot = tot + pv * (-(Iv.frac(al) * Ti.sq())).exp() * (PI / Iv.frac(g)).sqrt() / 2 * erf_iv(Iv.frac(g).sqrt() * Ti)
        elif key[0] == "K":
            tot = tot + pv * K_owen(key[1], key[2], T)
        elif key[0] == "eK":
            tot = tot + pv * (-(Iv.frac(key[1]) * Ti.sq())).exp() * K_owen(key[2], key[3], T)
        else:
            raise ValueError(key)
    return tot


# ---------------- main
S6, S21, S14 = Iv.frac(6).sqrt(), Iv.frac(21).sqrt(), Iv.frac(14).sqrt()
ATAN2 = PI / 2 - atan_iv(Iv.frac(Fr(1, 2)))
D3_CF = (50 * PI + 200 * ATAN2 - 228) / (9 * PI)                  # Math-#201
D4_CF = (Iv.frac(Fr(6695, 54)) - Iv.frac(Fr(405, 32)) * S6        # Math-#202
         - (Iv.frac(Fr(1375, 72)) * S21 + Iv.frac(Fr(6695, 27)) * atan_iv(S21 / 7) + Iv.frac(Fr(405, 16)) * S6 * atan_iv(S14 / 14)) / PI)
R8 = [Fr(3, 1024) * x for x in (11243520, 0, -522240, 0, 28800, 0, -288, 0, 1)]
R6 = [Fr(-256, 729) * x for x in (405, 0, 405, 0, 45, 0, 1)]
STATED_F4 = {("e", Fr(1, 16)): [Fr(1, 31104) * x for x in (0, -991830528, 0, 18489408, 0, -346856, 0, 729)],   # Math-#213, verbatim
             ("E", Fr(1, 16)): R8,
             ("eE", Fr(1, 48), Fr(1, 24)): R6, ("eE", Fr(1, 48), Fr(1, 6)): [2 * x for x in R6],
             ("e", Fr(3, 16)): [Fr(512, 243) * x for x in (0, -297, 0, -42, 0, -1)]}
R10 = (7189875000, 0, 554203125, 0, -14962500, 0, 176750, 0, -700, 0, 1)
R9 = (0, -349281450000, 0, -34608500625, 0, 39341075, 0, -521235, 0, 729)
STATED_F5 = {("e", Fr(1, 5)): [Fr(256, 10546875) * x for x in (-131412150000, 0, -13096721250, 0, 7502375, 0, -174960, 0, 243)],
             ("e", Fr(3, 40)): [Fr(-1, 337500000) * x for x in (-1076528332800000, 0, 241731011520000, 0, -113418584000, 0,
                                                                    4666529880, 0, 150240501)],
             ("eE", Fr(1, 30), Fr(1, 24)): [Fr(256, 158203125) * x for x in R9],
             ("eE", Fr(1, 30), Fr(1, 6)): [Fr(512, 158203125) * x for x in R9],
             ("eE", Fr(1, 80), Fr(1, 16)): [Fr(-3, 500000000) * x for x in (0, -211803379200000, 0, 4362224640000, 0, -26148412800, 0,
                                                                           290979840, 0, 9372409)],
             ("K", Fr(1, 80), Fr(1, 16)): [Fr(576, 9765625) * x for x in R10],
             ("K", Fr(1, 30), Fr(1, 24)): [Fr(768, 9765625) * x for x in R10],
             ("K", Fr(1, 30), Fr(1, 6)): [Fr(1536, 9765625) * x for x in R10]}
R12 = (17156269670400, 0, -361184624640, 0, 17244057600, 0, -174182400, 0, 777600, 0, -1440, 0, 1)
S12 = (2392508671875, 0, -66431531250, 0, 898734375, 0, -7822500, 0, 49125, 0, -210, 0, 1)
Q11 = (0, -4306515046875, 0, 23876996875, 0, -236297250, 0, 1429650, 0, -6075, 0, 27)
STATED_F6 = {("K", Fr(1, 48), Fr(1, 16)): [Fr(5, 39366) * x for x in R12],
             ("eK", Fr(1, 120), Fr(1, 80), Fr(1, 16)): [Fr(-55296, 244140625) * x for x in S12],
             ("eK", Fr(1, 120), Fr(1, 30), Fr(1, 24)): [Fr(-73728, 244140625) * x for x in S12],
             ("eK", Fr(1, 120), Fr(1, 30), Fr(1, 6)): [Fr(-147456, 244140625) * x for x in S12],
             ("e", Fr(1, 12)): [Fr(-1, 170859375000) * x for x in (696502101362688000000, 0, -19424726020382400000, 0,
                                                                     1543288909140360000, 0, 4490688495364200, 0, 15081527796345, 0,
                                                                     12354341056)],
             ("e", Fr(5, 24)): [Fr(-24576, 9765625) * x for x in (-1619841281250, 0, 8110196875, 0, -81472500, 0, 487800, 0, -2070, 0, 9)],
             ("eE", Fr(1, 24), Fr(1, 24)): [Fr(-8192, 48828125) * x for x in Q11],
             ("eE", Fr(1, 24), Fr(1, 6)): [Fr(-16384, 48828125) * x for x in Q11],
             ("eE", Fr(1, 48), Fr(1, 16)): [Fr(-1, 20503125000000) * x for x in (0, 30469360222513728000000, 0, -451869635195452800000, 0,
                                                                               26977353149594304000, 0, 54126362457032400, 0,
                                                                               261305873316375, 0, 123252092672)]}
STAGES_F6 = [(Fr(1, 24), ["('1',)"]),
             (Fr(1, 48), ["('1',)", "('E', Fraction(3, 16))", "('K', Fraction(1, 48), Fraction(1, 24))",
                          "('K', Fraction(1, 48), Fraction(1, 6))"]),
             (Fr(1, 80), ["('1',)", "('E', Fraction(1, 5))", "('E', Fraction(3, 40))"]),
             (Fr(1, 120), ["('1',)", "('E', Fraction(1, 12))", "('E', Fraction(5, 24))", "('K', Fraction(1, 24), Fraction(1, 24))",
                           "('K', Fraction(1, 24), Fraction(1, 6))", "('KK', Fraction(1, 120), Fraction(1, 30), Fraction(1, 24))",
                           "('KK', Fraction(1, 120), Fraction(1, 30), Fraction(1, 6))",
                           "('KK', Fraction(1, 120), Fraction(1, 80), Fraction(1, 16))"])]   # generated by the derivation and zero
MEHTA_F6 = {("K", Fr(1, 48), Fr(1, 16)): [Fr(276480)]}   # the only kind of the Vandermonde-alone layers that survives T -> infinity
OWEN_KINDS = ((Fr(1, 48), Fr(1, 16)), (Fr(1, 80), Fr(1, 16)), (Fr(1, 30), Fr(1, 24)), (Fr(1, 30), Fr(1, 6)))


def K_limit(al, g):
    """K_{al,g}(infinity) = arctan(a)/(2 sqrt(al g)), a = sqrt(g/al), through arctan a = pi/2 - arctan(1/a) for a > 1."""
    A = (Iv.frac(g) / Iv.frac(al)).sqrt()
    at = PI / 2 - atan_iv(1 / A, N=400) if g > al else atan_iv(A, N=400)
    return at / (2 * (Iv.frac(al) * Iv.frac(g)).sqrt())


def label(W):
    def f(x):
        return ("inf" if x > 0 else "-inf") if isinstance(x, float) else str(x)
    return "[%s,%s]" % (f(W[0]), f(W[1]))


def within(iv, fv, tol):
    f, t = Decimal(repr(fv)), Decimal(repr(tol))
    return CE.subtract(iv.lo, f) <= t and CE.subtract(f, iv.hi) <= t


def common_prefix(iv):
    lo, hi = str(iv.lo), str(iv.hi)
    i = 0
    while i < min(len(lo), len(hi)) and lo[i] == hi[i]:
        i += 1
    return lo[:i]


def rel_width(iv):
    return CC.divide(iv.width(), CC.multiply(Decimal(2), iv.lo))


def main():
    global MUT, C_EVEN_REF, C_ODD_REF, CPRIME_REF, DETV_REF, LAMBDA0
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant", choices=MUTANTS)
    MUT = ap.parse_args().mutant
    C_EVEN_REF, C_ODD_REF = ref_matrix(EVEN), ref_matrix(ODD)            # rebuilt: they depend on the mutant
    CPRIME_REF, DETV_REF = schur(C_EVEN_REF, IDX_X, IDX_V)
    LAMBDA0 = Fr(23, 100) if MUT == "d6-floor" else Fr(1, 5)
    Z_MEHTA[6] = 4423680 * PI * SQRTPI * (ONE if MUT == "mehta" else SQRT2)
    t0 = time.time()
    checks = {}
    full, fullK = (-INF, INF), (Fr(0), INF)
    band, bandK = (Fr(0), Fr(1)), (Fr(1, 2), Fr(2))

    # the reference law, exactly
    stated = [[Fr(0)] * 22 for _ in range(22)]
    stated[0][0] = Fr(2, 3)
    for i in DIAG_A:
        stated[0][i] = stated[i][0] = Fr(-2, 3)
        for j in DIAG_A:
            stated[i][j] = Fr(8, 3) if i == j else Fr(2, 3)
    for i in OFF_A:
        stated[i][i] = Fr(1)
    s2 = CPRIME_REF[0][0]
    mean = [CPRIME_REF[i][0] / s2 for i in range(1, 22)]
    condA = [[CPRIME_REF[i][j] - CPRIME_REF[i][0] * CPRIME_REF[0][j] / s2 for j in range(1, 22)] for i in range(1, 22)]
    goe = [[(Fr(2) if i + 1 in DIAG_A else Fr(1)) if i == j else Fr(0) for j in range(21)] for i in range(21)]
    Ginv, detG = inv_n([[C_ODD_REF[i][j] for j in range(7)] for i in range(7)])
    tau2_ref = C_ODD_REF[7][7] - sum(C_ODD_REF[7][i] * Ginv[i][j] * C_ODD_REF[7][j] for i in range(7) for j in range(7))
    checks["REFERENCE_EXACT"] = bool(CPRIME_REF == stated and mean == [-1 if i + 1 in DIAG_A else 0 for i in range(21)]
                                     and condA == goe and DETV_REF == 3 and detG == 1 and tau2_ref == 6
                                     and tilted_cprime(Fr(2, 3), Fr(1), Fr(0)) == CPRIME_REF)
    checks["LAMBDA_FLOOR"] = bool(LAMBDA0 == Fr(1, 5) and sylvester_pd(shifted(CPRIME_REF, LAMBDA0))
                                  and sylvester_pd(shifted(CPRIME_REF, Fr(2063, 10000)))
                                  and not sylvester_pd(shifted(CPRIME_REF, Fr(2064, 10000)))
                                  and not sylvester_pd(shifted(CPRIME_REF, Fr(23, 100))))

    # the exact inner layers: the generated cancellations, the stated closed form, m = 4 and m = 5 against Math-#213 and #222, and the
    # interval value of F_6 (Owen's T for the Owen-type kinds) against a direct float quintuple integral
    pieces6, gamma6, stages6 = F_pieces(6)
    nonzero = {k: trim(v) for k, v in pieces6.items() if trim(v)}
    cancelled = sorted("%s" % (k,) for k, v in pieces6.items() if not trim(v))
    layer_ctrl = {}
    for T in (3, 6, 10):
        fi = F_iv(pieces6, Fr(T))
        layer_ctrl[str(T)] = {"F6_interval": fi.pair(), "F6_direct_float_n16": "%.15g" % Fm_direct_float(6, float(T), 16)}
    checks["INNER_LAYERS"] = bool(nonzero == STATED_F6 and stages6 == STAGES_F6 and gamma6 == Fr(1, 72)
                                  and cancelled == STAGES_F6[-1][1]
                                  and {k: trim(v) for k, v in F_pieces(5)[0].items() if trim(v)} == STATED_F5
                                  and {k: trim(v) for k, v in F_pieces(4)[0].items() if trim(v)} == STATED_F4
                                  and all(within(Iv(Decimal(v["F6_interval"][0]), Decimal(v["F6_interval"][1])),
                                                 float(v["F6_direct_float_n16"]), 1e-10 * abs(float(v["F6_direct_float_n16"])))
                                          for v in layer_ctrl.values()))

    # Mehta's Z_6 through the same layers (the Vandermonde alone): Z_6 = sqrt(24 pi) lim_{T -> infinity} F_6^Vandermonde(T)
    mehta_pieces = {k: trim(v) for k, v in F_pieces(6, mehta=True)[0].items() if trim(v)}
    surviving = {k: v for k, v in mehta_pieces.items() if k[0] in ("1", "E", "K")}
    Zlim = Iv(ZERO)
    for k, v in surviving.items():
        lim = Iv(ONE) if k[0] == "1" else ((PI / Iv.frac(k[1])).sqrt() / 2 if k[0] == "E" else K_limit(k[1], k[2]))
        Zlim = Zlim + Iv.frac(v[0]) * lim
    Zlim = (24 * PI).sqrt() * Zlim
    checks["MEHTA_Z6"] = bool(surviving == MEHTA_F6 and all(len(v) == 1 for v in surviving.values())
                              and all(k[0] in ("e", "eE", "eK") for k in mehta_pieces if k not in surviving)
                              and Zlim.intersects(Z_MEHTA[6]) and Zlim.width() < Zlim.lo * Decimal("1e-55")
                              and Fr(276480) * Fr(8, 3) * 6 == 4423680)       # sqrt(24 pi) (pi/3) 8 sqrt3 = 6 sqrt2 (8/3) pi^(3/2)

    # the reference birth integral: the cumulative layer against Owen's T, D_6, coarse quadrature, m = 2, ..., 5, float control
    ref6 = Birth(6)
    owen = {}
    ok = True
    for kk in OWEN_KINDS:
        rows = {}
        for s0, v in [ref6.M.Kvals[kk][0], ref6.M.Kvals[kk][40]] + [(T_CUT, ref6.M.K_at_cut[kk])]:
            w = K_owen(kk[0], kk[1], s0)
            ok &= v.intersects(w) and v.width() < Decimal("1e-50") and w.width() < Decimal("1e-50")
            rows[str(s0)] = {"cumulative_layer": v.pair(), "owen_T": w.pair()}
        owen["K_%s_%s" % kk] = rows
    checks["OWEN_EXACT"] = bool(ok)
    I_full = ref6.I(full)
    checks["D6_CERTIFIED"] = bool(I_full.width() < Decimal("1e-50") and I_full.lo > 0 and ref6.M.Rsum < Decimal("1e-50")
                                  and ref6.M.tail < Decimal("1e-60"))
    coarse = ref6.pref * Model(g_terms(6), K=8, cell=Fr(2), T=Fr(96), rho=Fr(6)).integral()
    checks["QUAD_COARSE"] = bool(coarse.contains(I_full))
    ref2 = Birth(2)
    cross = {}
    ok = ref2.I(full).intersects(D2_EXACT) and ref2.I(full).width() < Decimal("1e-45")
    for B in ((Fr(0), Fr(1)), (Fr(3), Fr(7, 2)), (-INF, Fr(0))):
        x, y = ref2.I(B), I3_ref(*B)
        ok &= x.intersects(y) and x.width() < Decimal("1e-45")
        cross["m2 " + label(B)] = {"T_quadrature": x.pair(), "Math197_closed_form": y.pair()}
    ref3 = Birth(3)
    F4band = ref3.I(band) / D3_CF
    ok &= ref3.I(full).intersects(D3_CF) and str(F4band.lo).startswith(F4B_209) and str(F4band.hi).startswith(F4B_209)
    cross["m3 full"] = ref3.I(full).pair()
    cross["m3 F^(4)_[0,1]"] = F4band.pair()
    ref4 = Birth(4)
    F5band = ref4.I(band) / D4_CF
    ok &= ref4.I(full).intersects(D4_CF) and str(F5band.lo).startswith(F5B_213) and str(F5band.hi).startswith(F5B_213)
    cross["m4 full"] = ref4.I(full).pair()
    cross["m4 F^(5)_[0,1]"] = F5band.pair()
    ref5 = Birth(5)
    D5_full = ref5.I(full)
    F6band = ref5.I(band) / D5_full
    ok &= (D5_full.intersects(Iv(Decimal(D5_222[0]), Decimal(D5_222[1]))) and D5_full.intersects(Iv(Decimal(D5_204[0]), Decimal(D5_204[1])))
           and str(F6band.lo).startswith(F6B_222) and str(F6band.hi).startswith(F6B_222))
    cross["m5 full"] = D5_full.pair()
    cross["m5 F^(6)_[0,1]"] = F6band.pair()
    checks["CROSSCHECK_M2_M5"] = bool(ok)
    fctrl = {}
    ok = True
    for B in (full, (Fr(0), Fr(1)), (Fr(1), Fr(2))):
        fv, cv = Im_float(6, B), ref6.I(B)
        ok &= within(cv, fv, 1e-9 * fv)
        fctrl[label(B)] = {"float_outer_quadrature": "%.12g" % fv, "certified": cv.pair()}
    checks["FLOAT_CONTROL"] = bool(ok)

    # image bound and epsilon
    E10, E12, E24 = image_bound7(10), image_bound7(12), image_bound7(24)
    checks["IMAGE_BOUND"] = bool(Decimal("2.19e-8") < E10.lo and E10.hi < Decimal("2.20e-8")
                                 and Decimal("1.82e-109") < E24.lo and E24.hi < Decimal("1.83e-109")
                                 and (Iv.frac(4096) * (-Iv.frac(150)).exp()).hi < HALF)
    S0, S10, S12, S24 = Space7(0, ref6), Space7(E10, ref6), Space7(E12, ref6), Space7(E24, ref6)
    checks["EPSILON"] = bool(S0.eps < Decimal("1e-55") and S10.eps < Decimal("6e-6") and S12.eps < Decimal("5e-15")
                             and S24.eps < Decimal("1e-54"))

    # Lemma S against exact scalar perturbations and an exact non-scalar (tilted) covariance
    ok = True
    for eta in (Fr(1, 50), Fr(-1, 50)):
        for B in ((Fr(0), Fr(1)), (Fr(3), Fr(7, 2)), (Fr(-7, 2), Fr(-3))):
            ok &= sandwich(ref6, B, Iv.frac(abs(eta))).contains(scaled_exact(ref6, B, eta))
    checks["SANDWICH_EXACT_SCALING"] = bool(ok)
    tb = Birth(6, *TILT)
    eps_t = frob_eps(tilted_cprime(*TILT), CPRIME_REF)
    full_t = tb.I(full)
    tilt = {"v_mu_sigma2": [str(x) for x in TILT], "eps": str(eps_t), "full_window": full_t.pair(), "windows": {}}
    ok = full_t.intersects(I_full) and Decimal("0.133") < eps_t < Decimal("0.134")
    for B in ((Fr(0), Fr(1)), (Fr(1), Fr(2))):
        if not ok:                                                     # eps outside its certified range: no bracket is formed
            break
        ex, sw, rf = tb.I(B), sandwich(ref6, B, eps_t), ref6.I(B)
        ok &= sw.contains(ex) and not ex.intersects(rf)
        tilt["windows"][label(B)] = {"tilted_exact": ex.pair(), "sandwich": sw.pair(), "reference": rf.pair()}
    checks["TILTED_EXACT"] = bool(ok)

    # the coefficient: reference, every L >= 10, L = 24
    C7_REF = GAMMA76 * CBRT32 * I_full / (120 * SQRT3 * PI.sq().sq() * SQRTPI)   # c_{7,ref} = Gamma(7/6) (3/2)^(1/3) D_6/(120 sqrt3 pi^(9/2))
    C7_GEN = GAMMA76 * CBRT32 * (16 * PI.sq() * PI / 15) * I_full / (SQRT3 * SQRTPI * ipow(2 * PI, 7))   # |S^6| D_6/(sqrt3 sqrt(pi) (2 pi)^7)
    b_windows = [(Fr(0), Fr(1)), (Fr(-1), Fr(1)), (Fr(-2), Fr(2)), (Fr(0), INF), (-INF, Fr(0)), full]
    table = {}
    for B in b_windows:
        F7 = ref6.I(B) / I_full
        for K in (bandK, fullK):
            GK = G_ref(K)
            table["B=%s K=%s" % (label(B), label(K))] = {
                "F7_B": F7.pair(), "G_K": GK.pair(), "c7_ref": (C7_REF * F7 * GK).pair(),
                "c7_torus_L_ge_10": S10.coefficient(B, K).pair(), "c7_torus_L_24": S24.coefficient(B, K).pair()}
    F7_band = ref6.I(band) / I_full
    c_band_ref = C7_REF * F7_band * G_ref(bandK)
    c_band_0, c_band_10, c_band_24 = S0.coefficient(band, bandK), S10.coefficient(band, bandK), S24.coefficient(band, bandK)
    c_full_0, c_full_10, c_full_24 = S0.coefficient(full, fullK), S10.coefficient(full, fullK), S24.coefficient(full, fullK)
    checks["FACTORIZATION"] = bool(c_band_0.intersects(c_band_ref) and c_full_0.intersects(C7_REF))
    checks["C7_CONSISTENT"] = bool(C7_REF.intersects(C7_GEN) and c_full_24.intersects(C7_REF) and c_full_10.contains(C7_REF)
                                   and C7_REF.width() < Decimal("1e-50"))
    checks["NESTING"] = bool(c_band_10.contains(c_band_ref) and c_band_10.intersects(c_band_24)
                             and c_band_10.width() > c_band_24.width() * Decimal(10) ** 30
                             and all(Iv(Decimal(v["c7_torus_L_ge_10"][0]), Decimal(v["c7_torus_L_ge_10"][1])).contains(
                                 Iv(Decimal(v["c7_ref"][0]), Decimal(v["c7_ref"][1]))) for v in table.values()))
    rel10 = rel_width(c_band_10)
    worst = max(rel_width(Iv(Decimal(v["c7_torus_L_ge_10"][0]), Decimal(v["c7_torus_L_ge_10"][1]))) for v in table.values())
    checks["WIDTHS"] = bool(worst < Decimal("2e-4") and c_band_24.width() < c_band_24.lo * Decimal("1e-40"))
    checks["SANDWICH_ORDERED"] = bool(not SANDWICH_VIOLATIONS)
    checks["LIBRARY_EXACT"] = bool(library_exact())
    pinned = {"c_7_ref": C7_REF, "D_6": I_full, "F7|[0,1]": F7_band, "c7|[0,1]x[1/2,2] ref": c_band_ref, "pi": PI}
    checks["PINNED"] = bool(all(str(pinned[k].lo).startswith(p) and str(pinned[k].hi).startswith(p) for k, p in PINNED.items()))
    if len(checks) != 20:
        raise ValueError("rule count changed")
    passed = all(checks.values())

    out = {
        "object": "CL-C8-WINDOW-COEFFICIENT-D7-20261001-v1",
        "scientific_effect": "NONE",
        "certified": True,
        "mutant": MUT,
        "passed": passed,
        "checks": checks,
        "c8_band": {"B": "[0,1]", "K": "[1/2,2]",
                    "c7_reference": c_band_ref.pair(), "c7_reference_digits": common_prefix(c_band_ref),
                    "F7_B": F7_band.pair(), "c7_torus_every_L_ge_10_every_frame": c_band_10.pair(),
                    "c7_torus_L_24": c_band_24.pair(), "relative_half_width_L_ge_10": str(rel10)},
        "full_window": {"D_6": I_full.pair(), "D_6_digits": common_prefix(I_full), "c_7_ref": C7_REF.pair(),
                        "c_7_ref_digits": common_prefix(C7_REF), "c_7_ref_general_formula": C7_GEN.pair(),
                        "c7_torus_every_L_ge_10": c_full_10.pair(), "c7_torus_L_24": c_full_24.pair()},
        "windows": table,
        "worst_relative_half_width_L_ge_10": str(worst),
        "transfer": {"E7_10": E10.pair(), "E7_12": E12.pair(), "E7_24": E24.pair(),
                     "eps_L_ge_10": str(S10.eps), "eps_L_ge_12": str(S12.eps), "eps_L_24": str(S24.eps), "eps_E0": str(S0.eps),
                     "lambda_floor": str(LAMBDA0), "lambda_min_exact": "(10 - 2 sqrt22)/3",
                     "tau2_L_ge_10": S10.tau2.pair(), "pG_pV_L_ge_10": (S10.pG * S10.pV).pair()},
        "reference": {"C_prime": [[str(x) for x in r] for r in CPRIME_REF], "A_given_f_mean": [str(x) for x in mean],
                      "tau2": str(tau2_ref), "det_Cov_V": str(DETV_REF), "coarse_quadrature": coarse.pair(),
                      "F_6_pieces": {"%s" % (k,): [str(x) for x in v] for k, v in sorted(nonzero.items(), key=lambda kv: str(kv[0]))},
                      "layer_cancellations": [[str(c), z] for c, z in stages6],
                      "quadrature_remainder_sum": str(ref6.M.Rsum), "quadrature_tail": str(ref6.M.tail),
                      "Z_6_layers": Zlim.pair(), "Z_6_closed_form": Z_MEHTA[6].pair(),
                      "Z_6_surviving_kinds": {"%s" % (k,): [str(x) for x in v] for k, v in surviving.items()}},
        "owen_type_layer": owen,
        "layers_interval_vs_float": layer_ctrl,
        "crosschecks": cross,
        "float_control": fctrl,
        "tilted": tilt,
        "constants": {"pi": PI.pair(), "Gamma(7/6)": GAMMA76.pair(), "(3/2)^(1/3)": CBRT32.pair()},
        "exact_relations": {
            "window coefficient": "c^(7)_{B,K} = 144 int_{S^6} p_G(0) p_V(0) I_B(u) J_K(u) dsigma(u), I_B = E[1{f in B} det(A)^2 1{A<0} | V=0] ([LP] (11.3), section 15)",
            "reference": "(f, A) | V=0: f ~ N(0, 2/3), A = Q - f I, Q the 6x6 GOE; I^ref_B = (1/(sqrt3 Z_6)) int_0^inf e^{-T^2/72} F_6(T) [Phi((3/sqrt2)(b_+ - T/9)) - Phi((3/sqrt2)(b_- - T/9))] dT",
            "Z_6": "Mehta's integral int |Delta| e^{-|mu|^2/4} over R^6 = 4423680 sqrt2 pi^(3/2) = sqrt(24 pi) * 276480 * K_{1/48,1/16}(infinity), K(infinity) = arctan(sqrt3)/(2 sqrt(1/768)) = 8 sqrt3 pi/3",
            "Owen-type kinds": "K_{al,g}(T) = int_0^T e^{-al w^2} E_g(w) dw = (pi/sqrt(al g)) [arctan(a)/(2 pi) - T_Owen(sqrt(2 al) T, a)], a = sqrt(g/al); eK = e^{-T^2/120} K",
            "sandwich": "(1-eps) C'_ref <= C' <= (1+eps) C'_ref  =>  (1-eps)^17/(1+eps)^11 I^ref_{B/sqrt(1-eps)} <= I'_B <= (1+eps)^17/(1-eps)^11 I^ref_{B/sqrt(1+eps)}",
            "epsilon": "eps = ||C' - C'_ref||_F / lambda0, lambda0 = 1/5 <= lambda_min(C'_ref) = (10 - 2 sqrt22)/3",
            "image bound": "E^(7)_L = 1499596 (76 L^6 + 15) exp(-L^2/2)",
            "c_7_ref": "Gamma(7/6) (3/2)^(1/3) |S^6| D_6 / (sqrt3 sqrt(pi) (2 pi)^7) = Gamma(7/6) (3/2)^(1/3) D_6 / (120 sqrt3 pi^(9/2)), |S^6| = 16 pi^3/15",
        },
    }
    sys.stdout.write(json.dumps(out, indent=1) + "\n")
    sys.stderr.write("seconds %.1f\n" % (time.time() - t0))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
