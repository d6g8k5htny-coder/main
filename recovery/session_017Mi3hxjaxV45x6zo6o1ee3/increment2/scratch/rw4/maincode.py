

# ---------------- main
ATAN2 = PI / 2 - atan_iv(Iv.frac(Fr(1, 2)))
D3_CF = (50 * PI + 200 * ATAN2 - 228) / (9 * PI)                  # Math-#201: D_3 = (50 pi + 200 arctan 2 - 228)/(9 pi)
C4_REF = GAMMA76 * CBRT32 * D3_CF / (8 * SQRT3 * PI.sq() * SQRTPI)  # Math-#199: c_{4,ref} = Gamma(7/6) (3/2)^(1/3) D_3/(8 sqrt3 pi^(5/2))


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
    LAMBDA0 = Fr(1, 3) if MUT == "d3-floor" else Fr(3, 10)
    t0 = time.time()
    checks = {}
    full, fullK = (-INF, INF), (Fr(0), INF)
    band, bandK = (Fr(0), Fr(1)), (Fr(1, 2), Fr(2))

    # the reference law, exactly
    a, c, t, o = Fr(2, 3), Fr(-2, 3), Fr(8, 3), Fr(0)
    stated = [[a, c, o, o, c, o, c], [c, t, o, o, a, o, a], [o, o, 1, o, o, o, o], [o, o, o, 1, o, o, o],
              [c, a, o, o, t, o, a], [o, o, o, o, o, 1, o], [c, a, o, o, a, o, t]]
    s2 = CPRIME_REF[0][0]
    mean = [CPRIME_REF[i][0] / s2 for i in range(1, 7)]
    condA = [[CPRIME_REF[i][j] - CPRIME_REF[i][0] * CPRIME_REF[0][j] / s2 for j in range(1, 7)] for i in range(1, 7)]
    goe = [[Fr(2) if (i == j and i in (0, 3, 5)) else (Fr(1) if i == j else Fr(0)) for j in range(6)] for i in range(6)]
    Ginv, detG = inv_n([[C_ODD_REF[i][j] for j in range(4)] for i in range(4)])
    tau2_ref = C_ODD_REF[4][4] - sum(C_ODD_REF[4][i] * Ginv[i][j] * C_ODD_REF[4][j] for i in range(4) for j in range(4))
    checks["REFERENCE_EXACT"] = bool(CPRIME_REF == stated and mean == [-1, 0, 0, -1, 0, -1] and condA == goe and DETV_REF == 3
                                     and detG == 1 and tau2_ref == 6 and tilted_cprime(Fr(2, 3), Fr(1), Fr(0)) == CPRIME_REF)
    checks["LAMBDA_FLOOR"] = bool(LAMBDA0 == Fr(3, 10) and sylvester_pd(shifted(CPRIME_REF, LAMBDA0))
                                  and sylvester_pd(shifted(CPRIME_REF, Fr(305, 1000)))
                                  and not sylvester_pd(shifted(CPRIME_REF, Fr(306, 1000)))
                                  and not sylvester_pd(shifted(CPRIME_REF, Fr(1, 3))))

    # the exact inner layers against a direct float double integral
    pieces3 = F_pieces(3)[0]
    R6 = [Fr(4, 729) * x for x in (11340, 0, 2835, 0, -90, 0, 1)]                  # (4/729)(T^6 - 90T^4 + 2835T^2 + 11340)
    stated_F3 = {("E", Fr(1, 24)): R6, ("E", Fr(1, 6)): [2 * x for x in R6],         # E_{1/24} = sqrt(6pi) erf, E_{1/6} = sqrt(6pi)/2 erf
                 ("e", Fr(1, 24)): [Fr(8, 243) * x for x in (0, -8802, 0, -933, 0, 2)],
                 ("e", Fr(1, 6)): [Fr(8, 243) * x for x in (0, 3132, 0, -93, 0, 1)]}
    layer_ctrl = {str(T): ["%.12g" % F_float(pieces3, T), "%.12g" % F3_direct_float(float(T))] for T in (2, 4, 6)}
    checks["INNER_LAYERS"] = bool(all(abs(float(x) - float(y)) <= 1e-6 * abs(float(y)) for x, y in layer_ctrl.values())
                                  and {k: trim(v) for k, v in pieces3.items() if trim(v)} == stated_F3)

    # the reference birth integral: full window, coarse quadrature, m = 2, float control
    ref3 = Birth(3)
    I_full = ref3.I(full)
    checks["D3_EXACT"] = bool(I_full.intersects(D3_CF) and I_full.width() < Decimal("1e-45") and D3_CF.width() < Decimal("1e-50"))
    coarse = Model(g_terms(3), K=8, cell=Fr(2), T=Fr(60), rho=Fr(6)).integral() / (96 * PI)
    checks["QUAD_COARSE"] = bool(coarse.contains(D3_CF))
    ref2 = Birth(2)
    m2 = {}
    ok = ref2.I(full).intersects(D2_EXACT) and ref2.I(full).width() < Decimal("1e-45")
    for B in ((Fr(0), Fr(1)), (Fr(-1), Fr(1)), (Fr(3), Fr(7, 2)), (Fr(0), INF), (-INF, Fr(0))):
        x, y = ref2.I(B), I3_ref(*B)
        ok &= x.intersects(y) and x.width() < Decimal("1e-45")
        m2[label(B)] = {"T_quadrature": x.pair(), "Math197_closed_form": y.pair()}
    checks["M2_CROSSCHECK"] = bool(ok)
    fctrl = {}
    ok = True
    for B in ((Fr(0), Fr(1)), (Fr(1), Fr(2))):
        fv, cv = I3_float(B), ref3.I(B)
        ok &= within(cv, fv, 1e-8 * fv)
        fctrl[label(B)] = {"float_eigenvalue_quadrature": "%.12g" % fv, "certified": cv.pair()}
    checks["FLOAT_CONTROL"] = bool(ok)

    # image bound and epsilon
    E10, E12, E24 = image_bound4(10), image_bound4(12), image_bound4(24)
    checks["IMAGE_BOUND"] = bool(Decimal("1.50e-10") < E10.lo and E10.hi < Decimal("1.51e-10")
                                 and Decimal("1.24e-111") < E24.lo and E24.hi < Decimal("1.25e-111")
                                 and (Iv.frac(512) * (-Iv.frac(150)).exp()).hi < HALF)
    S0, S10, S12, S24 = Space4(0, ref3), Space4(E10, ref3), Space4(E12, ref3), Space4(E24, ref3)
    checks["EPSILON"] = bool(S0.eps < Decimal("1e-55") and S10.eps < Decimal("1e-8") and S12.eps < Decimal("1e-17")
                             and S24.eps < Decimal("1e-55"))

    # Lemma S against exact scalar perturbations and an exact non-scalar (tilted) covariance
    ok = True
    for eta in (Fr(1, 50), Fr(-1, 50)):
        for B in ((Fr(0), Fr(1)), (Fr(3), Fr(7, 2)), (Fr(-7, 2), Fr(-3))):
            ok &= sandwich(ref3, B, Iv.frac(abs(eta))).contains(scaled_exact(ref3, B, eta))
    checks["SANDWICH_EXACT_SCALING"] = bool(ok)
    tb = Birth(3, *TILT)
    eps_t = frob_eps(tilted_cprime(*TILT), CPRIME_REF)
    full_t = tb.I(full)
    tilt = {"v_mu_sigma2": [str(x) for x in TILT], "eps": str(eps_t), "full_window": full_t.pair(), "windows": {}}
    ok = full_t.intersects(D3_CF) and Decimal("0.07") < eps_t < Decimal("0.071")
    for B in ((Fr(0), Fr(1)), (Fr(1), Fr(2))):
        if not ok:                                                     # eps outside its certified range: no bracket is formed
            break
        ex, sw, rf = tb.I(B), sandwich(ref3, B, eps_t), ref3.I(B)
        ok &= sw.contains(ex) and not ex.intersects(rf)
        tilt["windows"][label(B)] = {"tilted_exact": ex.pair(), "sandwich": sw.pair(), "reference": rf.pair()}
    checks["TILTED_EXACT"] = bool(ok)

    # the coefficient: reference, every L >= 10, L = 24
    b_windows = [(Fr(0), Fr(1)), (Fr(-1), Fr(1)), (Fr(-2), Fr(2)), (Fr(0), INF), (-INF, Fr(0)), full]
    table = {}
    for B in b_windows:
        F4 = ref3.I(B) / D3_CF
        for K in (bandK, fullK):
            GK = G_ref(K)
            table["B=%s K=%s" % (label(B), label(K))] = {
                "F4_B": F4.pair(), "G_K": GK.pair(), "c4_ref": (C4_REF * F4 * GK).pair(),
                "c4_torus_L_ge_10": S10.coefficient(B, K).pair(), "c4_torus_L_24": S24.coefficient(B, K).pair()}
    F4_band = ref3.I(band) / D3_CF
    c_band_ref = C4_REF * F4_band * G_ref(bandK)
    c_band_0, c_band_10, c_band_24 = S0.coefficient(band, bandK), S10.coefficient(band, bandK), S24.coefficient(band, bandK)
    c_full_0, c_full_10, c_full_24 = S0.coefficient(full, fullK), S10.coefficient(full, fullK), S24.coefficient(full, fullK)
    checks["FACTORIZATION"] = bool(c_band_0.intersects(c_band_ref) and c_full_0.intersects(C4_REF))
    checks["C4_CONSISTENT"] = bool(str(C4_REF.lo).startswith(C4_REF_199) and str(C4_REF.hi).startswith(C4_REF_199)
                                   and c_full_24.intersects(C4_REF) and c_full_10.contains(C4_REF))
    checks["NESTING"] = bool(c_band_10.contains(c_band_ref) and c_band_10.intersects(c_band_24)
                             and c_band_10.width() > c_band_24.width() * Decimal(10) ** 30
                             and all(Iv(Decimal(v["c4_torus_L_ge_10"][0]), Decimal(v["c4_torus_L_ge_10"][1])).contains(
                                 Iv(Decimal(v["c4_ref"][0]), Decimal(v["c4_ref"][1]))) for v in table.values()))
    rel10 = rel_width(c_band_10)
    worst = max(rel_width(Iv(Decimal(v["c4_torus_L_ge_10"][0]), Decimal(v["c4_torus_L_ge_10"][1]))) for v in table.values())
    checks["WIDTHS"] = bool(worst < Decimal("1e-7") and c_band_24.width() < c_band_24.lo * Decimal("1e-40"))
    checks["SANDWICH_ORDERED"] = bool(not SANDWICH_VIOLATIONS)
    checks["LIBRARY_EXACT"] = bool(library_exact())
    pinned = {"c_4_ref": C4_REF, "D_3": I_full, "F4|[0,1]": F4_band, "c4|[0,1]x[1/2,2] ref": c_band_ref, "pi": PI}
    checks["PINNED"] = bool(all(str(pinned[k].lo).startswith(p) and str(pinned[k].hi).startswith(p) for k, p in PINNED.items()))
    if len(checks) != 18:
        raise ValueError("rule count changed")
    passed = all(checks.values())

    out = {
        "object": "CL-C8-WINDOW-COEFFICIENT-D4-20261001-v1",
        "scientific_effect": "NONE",
        "certified": True,
        "mutant": MUT,
        "passed": passed,
        "checks": checks,
        "c8_band": {"B": "[0,1]", "K": "[1/2,2]",
                    "c4_reference": c_band_ref.pair(), "c4_reference_digits": common_prefix(c_band_ref),
                    "F4_B": F4_band.pair(), "c4_torus_every_L_ge_10_every_frame": c_band_10.pair(),
                    "c4_torus_L_24": c_band_24.pair(), "relative_half_width_L_ge_10": str(rel10)},
        "full_window": {"c_4_ref": C4_REF.pair(), "c4_torus_every_L_ge_10": c_full_10.pair(), "c4_torus_L_24": c_full_24.pair(),
                        "Math199_c_4_ref_digits": C4_REF_199},
        "windows": table,
        "worst_relative_half_width_L_ge_10": str(worst),
        "transfer": {"E4_10": E10.pair(), "E4_12": E12.pair(), "E4_24": E24.pair(),
                     "eps_L_ge_10": str(S10.eps), "eps_L_ge_12": str(S12.eps), "eps_L_24": str(S24.eps), "eps_E0": str(S0.eps),
                     "lambda_floor": str(LAMBDA0), "lambda_min_exact": "(7 - sqrt37)/3",
                     "tau2_L_ge_10": S10.tau2.pair(), "pG_pV_L_ge_10": (S10.pG * S10.pV).pair()},
        "reference": {"C_prime": [[str(x) for x in r] for r in CPRIME_REF], "A_given_f_mean": [str(x) for x in mean],
                      "A_given_f_cov": [[str(x) for x in r] for r in condA], "tau2": str(tau2_ref), "det_Cov_V": str(DETV_REF),
                      "D_3_quadrature": I_full.pair(), "D_3_closed_form_Math201": D3_CF.pair(), "coarse_quadrature": coarse.pair(),
                      "F_3_pieces": {"%s" % (k,): [str(x) for x in trim(v)] for k, v in sorted(pieces3.items(), key=lambda kv: str(kv[0]))}},
        "layers_float_control": layer_ctrl,
        "m2_crosscheck": m2,
        "float_control": fctrl,
        "tilted": tilt,
        "constants": {"pi": PI.pair(), "Gamma(7/6)": GAMMA76.pair(), "(3/2)^(1/3)": CBRT32.pair(), "arctan 2": ATAN2.pair()},
        "exact_relations": {
            "window coefficient": "c^(4)_{B,K} = 144 int_{S^3} p_G(0) p_V(0) I_B(u) J_K(u) dsigma(u), I_B = E[1{f in B} det(A)^2 1{A<0} | V=0] ([LP] (11.3), section 15)",
            "reference": "(f, A) | V=0: f ~ N(0, 2/3), A = Q - f I, Q the 3x3 GOE; I^ref_B = (1/(96 pi)) int_0^inf e^{-T^2/24} F_3(T) [Phi(sqrt3(b_+ - T/6)) - Phi(sqrt3(b_- - T/6))] dT",
            "sandwich": "(1-eps) C'_ref <= C' <= (1+eps) C'_ref  =>  (1-eps)^(13/2)/(1+eps)^(7/2) I^ref_{B/sqrt(1-eps)} <= I'_B <= (1+eps)^(13/2)/(1-eps)^(7/2) I^ref_{B/sqrt(1+eps)}",
            "epsilon": "eps = ||C' - C'_ref||_F / lambda0, lambda0 = 3/10 <= lambda_min(C'_ref) = (7 - sqrt37)/3",
            "image bound": "E^(4)_L = 10240 (76 L^6 + 15) exp(-L^2/2)",
        },
    }
    sys.stdout.write(json.dumps(out, indent=1) + "\n")
    sys.stderr.write("seconds %.1f\n" % (time.time() - t0))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
