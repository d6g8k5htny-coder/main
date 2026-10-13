
# ---------------- main
def m3_closed(b):
    b = Iv.frac(b)
    return (b * b.sq() + b) * phi(b) + (b.sq().sq() + 2 * b.sq() + 7) * Phi(b) - 4 * SQRT2 * (-(b.sq()) / 4).exp() * Phi(b / SQRT2)


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


def main():
    global MUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant", choices=MUTANTS)
    MUT = ap.parse_args().mutant
    # module-level objects that depend on the mutant are rebuilt here
    global CPRIME_REF, DETV_REF, LAMBDA0, C_EVEN_REF, C_ODD_REF
    C_EVEN_REF, C_ODD_REF = ref_matrix(EVEN), ref_matrix(ODD)
    CPRIME_REF, DETV_REF = schur(C_EVEN_REF, IDX_X, IDX_V)
    LAMBDA0 = Fr(1) if MUT == "lambda-min" else Fr(1, 3)
    t0 = time.time()
    checks = {}
    full, fullK = (-INF, INF), (Fr(0), INF)
    band, bandK = (Fr(0), Fr(1)), (Fr(1, 2), Fr(2))

    # reference: exact conditional laws
    stated = [[Fr(2, 3), Fr(-2, 3), 0, Fr(-2, 3)], [Fr(-2, 3), Fr(8, 3), 0, Fr(2, 3)], [0, 0, 1, 0], [Fr(-2, 3), Fr(2, 3), 0, Fr(8, 3)]]
    s2 = CPRIME_REF[0][0]
    mean = [CPRIME_REF[i][0] / s2 for i in (1, 2, 3)]
    condA = [[CPRIME_REF[i][j] - CPRIME_REF[i][0] * CPRIME_REF[0][j] / s2 for j in (1, 2, 3)] for i in (1, 2, 3)]
    Gref = [[C_ODD_REF[i][j] for j in range(3)] for i in range(3)]
    Ginv, detG = inv3(Gref)
    tau2_ref = C_ODD_REF[3][3] - sum(C_ODD_REF[3][i] * Ginv[i][j] * C_ODD_REF[3][j] for i in range(3) for j in range(3))
    checks["REFERENCE_EXACT"] = bool(CPRIME_REF == stated and mean == [-1, 0, -1] and condA == [[2, 0, 0], [0, 1, 0], [0, 0, 2]]
                                     and DETV_REF == 3 and detG == 1 and tau2_ref == 6)
    checks["LAMBDA_FLOOR"] = bool(LAMBDA0 == Fr(1, 3) and lambda_floor_ok())
    I_full = I3_ref(*full)
    checks["D2_EXACT"] = bool(I_full.intersects(D2_EXACT) and I_full.width() < Decimal("1e-50"))
    checks["M3_FLOAT"] = bool(all(within(m3_closed(b), m3_float(float(b)), 1e-8) for b in (Fr(-2), Fr(-1), Fr(0), Fr(1, 2), Fr(1), Fr(2))))

    # image bound and epsilon
    E10, E12, E24 = image_bound3(10), image_bound3(12), image_bound3(24)
    checks["IMAGE_BOUND"] = bool(Decimal("2.0e-11") < E10.lo and E10.hi < Decimal("2.1e-11")
                                 and Decimal("1.7e-112") < E24.lo and E24.hi < Decimal("1.8e-112")
                                 and (Iv.frac(256) * (-Iv.frac(150)).exp()).hi < HALF)
    S0, S10, S12, S24 = Space3(0), Space3(E10), Space3(E12), Space3(E24)
    checks["EPSILON"] = bool(S0.eps < Decimal("1e-55") and S10.eps < Decimal("1e-9") and S24.eps < Decimal("1e-55")
                             and S12.eps < Decimal("1e-18"))

    # the sandwich against exact scaling perturbations C' = (1 + eta) C_ref' (tight eps = |eta|)
    scal_windows = [(Fr(0), Fr(1)), (Fr(3), Fr(7, 2)), (Fr(-7, 2), Fr(-3)), (Fr(-1), Fr(2)), (Fr(0), INF)]
    ok = True
    for eta in (Fr(1, 50), Fr(-1, 50), Fr(1, 200), Fr(-1, 200)):
        for B in scal_windows:
            ok &= sandwich(B, abs(eta)).contains(scaled_exact(B, eta))
    checks["SANDWICH_EXACT_SCALING"] = bool(ok)

    # float control: a non-scalar perturbation (the conditional mean of A12 tilts), float quadrature inside the sandwich
    Cp, eps_c = control_matrices()
    Cpf = [[float(x) for x in r] for r in Cp]
    Cref_f = [[float(x) for x in r] for r in CPRIME_REF]
    ctrl = {}
    ok = True
    for B in ((Fr(0), Fr(1)), (Fr(-1, 2), Fr(3, 2)), (Fr(1), Fr(2))):
        fr_, fp_ = I_float(Cref_f, B), I_float(Cpf, B)
        sw = sandwich(B, eps_c)
        ok &= within(I3_ref(*B), fr_, 1e-7 * fr_) and within(sw, fp_, 1e-7 * fp_)
        ctrl[label(B)] = {"float_reference": "%.10g" % fr_, "float_perturbed": "%.10g" % fp_, "sandwich": sw.pair()}
    checks["FLOAT_CONTROL"] = bool(ok and Decimal("0.005") < eps_c < Decimal("0.01"))

    # the coefficient: reference, every L >= 10, L = 24
    b_windows = [(Fr(0), Fr(1)), (Fr(-1), Fr(1)), (Fr(-2), Fr(2)), (Fr(0), INF), (-INF, Fr(0)), full]
    k_windows = [bandK, fullK]
    table = {}
    for B in b_windows:
        F3 = I3_ref(*B) / D2_EXACT
        for K in k_windows:
            GK = G_ref(K)
            table["B=%s K=%s" % (label(B), label(K))] = {
                "F3_B": F3.pair(), "G_K": GK.pair(), "c3_ref": (C3_REF * F3 * GK).pair(),
                "c3_pipeline_E0": S0.coefficient(B, K).pair(),
                "c3_torus_L_ge_10": S10.coefficient(B, K).pair(), "c3_torus_L_24": S24.coefficient(B, K).pair()}
    c_band_ref = C3_REF * (I3_ref(*band) / D2_EXACT) * G_ref(bandK)
    c_band_0, c_band_10, c_band_24 = S0.coefficient(band, bandK), S10.coefficient(band, bandK), S24.coefficient(band, bandK)
    c_full_0, c_full_10, c_full_24 = S0.coefficient(full, fullK), S10.coefficient(full, fullK), S24.coefficient(full, fullK)
    checks["FACTORIZATION"] = bool(c_band_0.intersects(c_band_ref) and c_full_0.intersects(C3_REF)
                                   and all(Iv(Decimal(v["c3_pipeline_E0"][0]), Decimal(v["c3_pipeline_E0"][1])).intersects(
                                       Iv(Decimal(v["c3_ref"][0]), Decimal(v["c3_ref"][1]))) for v in table.values()))
    side = Iv(Decimal(SIDE24_D3[0]), Decimal(SIDE24_D3[1]))
    checks["SIDE24_CONSISTENT"] = bool(c_full_24.intersects(side) and c_full_10.contains(C3_REF) and C3_REF.intersects(side))
    checks["NESTING"] = bool(c_band_10.contains(c_band_ref) and c_band_10.intersects(c_band_24)
                             and c_band_10.width() > c_band_24.width() * Decimal(10) ** 40)
    rel10 = CC.divide(c_band_10.width(), CC.multiply(Decimal(2), c_band_10.lo))
    checks["WIDTHS"] = bool(rel10 < Decimal("1e-8") and c_band_24.width() < Decimal("1e-50") and CC.divide(c_full_10.width(), c_full_10.lo) < Decimal("1e-8"))
    checks["LIBRARY_EXACT"] = bool(library_exact())
    pinned = {"c_3_ref": C3_REF, "F3|[0,1]": I3_ref(*band) / D2_EXACT, "c3|[0,1]x[1/2,2] ref": c_band_ref, "pi": PI}
    checks["PINNED"] = bool(all(str(pinned[k].lo).startswith(p) and str(pinned[k].hi).startswith(p) for k, p in PINNED.items()))
    if len(checks) != 14:
        raise ValueError("rule count changed")
    passed = all(checks.values())

    out = {
        "object": "CL-C8-WINDOW-COEFFICIENT-D3-TORUS-20260930-v1",
        "scientific_effect": "NONE",
        "certified": True,
        "mutant": MUT,
        "passed": passed,
        "checks": checks,
        "c8_band": {"B": "[0,1]", "K": "[1/2,2]",
                    "c3_reference": c_band_ref.pair(), "c3_reference_digits": common_prefix(c_band_ref),
                    "c3_torus_every_L_ge_10_every_frame": c_band_10.pair(),
                    "c3_torus_L_24": c_band_24.pair(),
                    "relative_half_width_L_ge_10": str(rel10)},
        "full_window": {"c_3_ref": C3_REF.pair(), "c3_torus_every_L_ge_10": c_full_10.pair(), "c3_torus_L_24": c_full_24.pair(),
                        "SIDE24_c_3_24": list(SIDE24_D3)},
        "windows": table,
        "transfer": {"E3_10": E10.pair(), "E3_12": E12.pair(), "E3_24": E24.pair(),
                     "eps_L_ge_10": str(S10.eps), "eps_L_ge_12": str(S12.eps), "eps_L_24": str(S24.eps), "eps_E0": str(S0.eps),
                     "lambda_floor": str(LAMBDA0), "lambda_min_exact": "2 - sqrt(8/3)",
                     "tau2_L_ge_10": S10.tau2.pair(), "pG_pV_L_ge_10": (S10.pG * S10.pV).pair()},
        "reference": {"C_prime": [[str(x) for x in r] for r in CPRIME_REF], "A_given_f_mean": [str(x) for x in mean],
                      "A_given_f_cov": [[str(x) for x in r] for r in condA], "tau2": str(tau2_ref), "det_Cov_V": str(DETV_REF),
                      "D_2": D2_EXACT.pair(), "I_full": I_full.pair()},
        "float_control": {"perturbation_of_even_block": {"%d,%d" % k: str(v) for k, v in CONTROL_DELTA.items()},
                          "eps": str(eps_c), "windows": ctrl},
        "constants": {"pi": PI.pair(), "Gamma(7/6)": GAMMA76.pair(), "(3/2)^(1/3)": CBRT32.pair()},
        "exact_relations": {
            "window coefficient": "c^(3)_{B,K} = 144 int_{S^2} p_G(0) p_V(0) I_B(u) J_K(u) dsigma(u), I_B = E[1{f in B} det(A)^2 1{A<0} | V=0] ([LP] (11.3), section 15)",
            "reference": "(f, A11, A12, A22) | V=0 has covariance C'_ref (above); A | f=b, V=0 = -b I + Q; I^ref_B = int_B phi_{2/3} m_{3,b} db (Math-#197 (4.1))",
            "sandwich": "(1-eps) C'_ref <= C' <= (1+eps) C'_ref  =>  (1-eps)^4/(1+eps)^2 I^ref_{B/sqrt(1-eps)} <= I'_B <= (1+eps)^4/(1-eps)^2 I^ref_{B/sqrt(1+eps)}",
            "epsilon": "eps = ||C' - C'_ref||_F / lambda0, lambda0 = 1/3 <= lambda_min(C'_ref) = 2 - sqrt(8/3)",
            "image bound": "E^(3)_L = 1404 (76 L^6 + 15) exp(-L^2/2)",
        },
    }
    sys.stdout.write(json.dumps(out, indent=1) + "\n")
    sys.stderr.write("seconds %.1f\n" % (time.time() - t0))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
