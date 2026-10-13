
# ---------------- m = 4 through the certified cumulative layer (regression against the exact closed form of Math-#202)
def plist_mul(u, v):
    out = [Fr(0)] * (len(u) + len(v) - 1)
    for i, a in enumerate(u):
        for j, b in enumerate(v):
            out[i + j] += a * b
    return out


def D4_certified_path(**kw):
    """D_4 with the v-layer done by the certified cumulative machinery instead of the exact incomplete Gaussian moments:
    Omega_a(T) = int_0^T e^{-v^2/24} sum_k e_{a,k}(v) J_k(v) dv = int_0^T [Q_{a,B}(v) e^{-v^2/24} + Q_{a,C}(v) e^{-v^2/6}] dv."""
    r = m4_reduction()
    JK = r["JK"]

    def spec(inner, g, erfc):
        cum, tail = {}, {}
        for a, ks in inner.items():
            qB, qC = [Fr(0)], [Fr(0)]
            for k, ek in ks.items():
                A, B, C = JK[k]
                if A:
                    raise ValueError("even p3-power at m = 4")
                pb = [B * c for c in ek]
                pc = plist_mul(ek, C) if C else [Fr(0)]
                qB = [x + y for x, y in itertools.zip_longest(qB, pb, fillvalue=Fr(0))]
                qC = [x + y for x, y in itertools.zip_longest(qC, pc, fillvalue=Fr(0))]
            cum[a] = {(Fr(1, 24), None): [Iv.frac(c) for c in qB], (Fr(1, 6), None): [Iv.frac(c) for c in qC]}
            bnd = Iv(ZERO)
            for gamma, Q in ((Fr(1, 24), qB), (Fr(1, 6), qC)):
                for n, c in enumerate(Q):
                    if c:
                        bnd = bnd + Iv.frac(abs(c)) * gauss_moment(n, gamma)
            tail[a] = Fr(bnd.hi)
        kappa_iv = Iv.frac(r["kappa"][0]) * root_frac(Fr(r["kappa"][1]), 2)
        return {"const": Iv(ONE), "prefactors": [Gauss(g)] + ([ErfLin(kappa_iv, -1)] if erfc else []), "cum": cum,
                "tail_poly": [tail.get(a, Fr(0)) for a in range(max(tail) + 1)], "gamma_tail": g + (surd_sq(r["kappa"]) if erfc else 0)}
    certE = enclose_layered([spec(r["innerE"], r["gE"], True)], **kw)
    certR = enclose_layered([spec(r["innerR"], r["gR"], False)], **kw)
    return iv_eval(r["pref"]) * (iv_eval(r["constE"]) * certE + certR)


# ---------------- float control (not part of the certificate): the same layer decomposition with trapezoid quadrature
def float_layered(red, Tmax=70.0, n=3500):
    def fcert(lam, g, erfc, kap):
        h = Tmax / n
        grid = [i * h for i in range(n + 1)]
        total = 0.0
        for a, (_, cert) in lam.items():
            for (gamma, mu), Q in cert.items():
                qc = {nn: fl_eval(e) for nn, e in Q.items()}
                smu = math.sqrt(mu) if mu is not None else None
                f = [sum(c * w ** nn for nn, c in qc.items()) * math.exp(-float(gamma) * w * w) * (math.erf(smu * w) if smu else 1.0) for w in grid]
                Om = [0.0]
                for i in range(1, n + 1):
                    Om.append(Om[-1] + 0.5 * h * (f[i - 1] + f[i]))
                out = [T ** a * math.exp(-g * T * T) * (math.erfc(kap * T) if erfc else 1.0) * Om[i] for i, T in enumerate(grid)]
                total += h * (sum(out) - 0.5 * (out[0] + out[-1]))
        return total
    lamE, lamR = layer_w(red["innerE"], red["JK"]), layer_w(red["innerR"], red["JK"])
    exE = fl_eval(outer_exact(lamE, red["gE"], True, red["kappa"]))
    exR = fl_eval(outer_exact(lamR, red["gR"], False, red["kappa"]))
    kap = float(red["kappa"][0]) * math.sqrt(red["kappa"][1])
    cE, cR = fcert(lamE, float(red["gE"]), True, kap), fcert(lamR, float(red["gR"]), False, kap)
    prefE, prefR = fl_eval(mul(red["pref"], red["constE"])), fl_eval(red["pref"])
    return prefE * (exE + cE) + prefR * (exR + cR)


# ---------------- pinned values and companions
SIDE24 = {2: ("0.07340691930603427103", "0.07340691930603427104"), 3: ("0.04177593184059834334", "0.04177593184059834335")}
PINNED = {"D_5": "PIN_D5", "c_6_ref": "PIN_C6", "pi": "3.14159265358979323846264338327950288419716939937510"}
CLOSED_FORM_D3 = {"1": Fr(50, 3), "1/pi": Fr(-76, 3), "arctan(1/2)/pi": Fr(-200, 9)}                                     # Math-#201
CLOSED_FORM_D4 = {"1": Fr(6695, 54), "sqrt6": Fr(-405, 32), "sqrt21/pi": Fr(-1375, 72), "arctan(sqrt21/7)/pi": Fr(-6695, 27),
                  "sqrt6 arctan(sqrt14/4)/pi": Fr(-405, 16), "sqrt6 arctan(sqrt14/7)/pi": Fr(405, 16)}                   # Math-#202
CERTIFIED_200 = {"D_4": ("14.2187634589773576013064656778740783173406003782976930345926", "14.2187634589773576013064656778740783173406042538663501338228"),
                 "c_5_ref": ("0.0132193193800849680760334875146413965210528883241234179643777", "0.0132193193800849680760334875146413965210528919272763929401087")}


def main():
    global MUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant", choices=MUTANTS)
    args = ap.parse_args()
    MUT = args.mutant
    t0 = time.time()
    checks = {}

    # exact regressions: D_1, D_2, D_3, D_4, Z_4 (Math-#201/#202 code paths)
    d1 = D1_exact()
    d2, _ = D2_exact()
    d3, _ = D3_exact()
    d4, info4 = D4_exact()
    z4 = Z4_exact()

    def target(spec, keys):
        out = {}
        for name, c in spec.items():
            out = add(out, scale(keys[name], c))
        return out
    a_half, a14_4, a14_7, a21_7 = (atan_elem(t) for t in ((Fr(1, 2), 1), (Fr(1, 4), 14), (Fr(1, 7), 14), (Fr(1, 7), 21)))
    keys3 = {"1": E(1), "1/pi": E(1, 1, -2), "arctan(1/2)/pi": mul_pi_half(a_half, -2)}
    keys4 = {"1": E(1), "sqrt6": E(1, 6), "sqrt21/pi": E(1, 21, -2), "arctan(sqrt21/7)/pi": mul_pi_half(a21_7, -2),
             "sqrt6 arctan(sqrt14/4)/pi": mul_pi_half(mul_surd(a14_4, (Fr(1), 6)), -2),
             "sqrt6 arctan(sqrt14/7)/pi": mul_pi_half(mul_surd(a14_7, (Fr(1), 6)), -2)}
    D3_target, D4_target = target(CLOSED_FORM_D3, keys3), target(CLOSED_FORM_D4, keys4)

    # m = 5
    red = m5_reduction()
    D5, info5 = D5_enclosure(red)
    D5_route2, _ = D5_enclosure(red, certify_e=True)
    coarse_cells = [(Fr(i), Fr(i + 1)) for i in range(0, 90)]
    if MUT == "remainder-dropped":
        D5 = D5_enclosure(red, K=10, cells=coarse_cells, remainder=False)[0]
        D5_coarse = D5
    else:
        D5_coarse = D5_enclosure(red, K=20, cells=coarse_cells)[0]
    if MUT == "tail-dropped":
        D5 = D5_enclosure(red, cells=CELLS[:60])[0]
    Z5, zinfo = Z5_enclosure(**({"K": 10, "cells": coarse_cells, "remainder": False} if MUT == "remainder-dropped" else
                                ({"cells": CELLS[:60]} if MUT == "tail-dropped" else {})))
    Z5_closed = iv_eval(mehta_Z(5))
    Z5_literal = 46080 * PI * SQRTPI
    D4_cert = D4_certified_path(**({"K": 10, "cells": coarse_cells, "remainder": False} if MUT == "remainder-dropped" else
                                   ({"cells": CELLS[:60]} if MUT == "tail-dropped" else {})))
    D4_iv, D3_iv = iv_eval(d4), iv_eval(d3)

    # coefficients c_{d,ref} = Gamma(7/6) (3/2)^(1/3) |S^{d-1}| D_{d-1} / (sqrt3 sqrt pi (2 pi)^d)
    def sphere_area(d):
        pi_pow = ipow(PI, d // 2) * (SQRTPI if d % 2 else Iv(ONE))
        return 2 * pi_pow / gamma_half_integer(d)

    def c_ref(d, D):
        return GAMMA76 * CBRT32 * sphere_area(d) * D / (SQRT3 * SQRTPI * ipow(2 * PI, d))
    c2, c3, c4, c5, c6 = c_ref(2, iv_eval(d1)), c_ref(3, iv_eval(d2)), c_ref(4, D3_iv), c_ref(5, D4_iv), c_ref(6, D5)
    c6_closed = GAMMA76 * CBRT32 * D5 / (64 * SQRT3 * PI * PI * PI * SQRTPI)

    # torus transfer for d = 6, L = 24 (NOTE section 4): image bound, covariance sandwich, density and cone-moment ratios
    shells = 728 if MUT != "image-shells" else 364                                   # (2j+1)^6 - (2j-1)^6 = 384 j^5 + 320 j^3 + 24 j <= 728 j^5
    E6 = Iv.frac(2 * shells * 216 * (76 * 24 ** 6 + 15)) * (-Iv.frac(288)).exp()   # |n|^2 <= 6 j^2, so |n|^6 <= 216 j^6
    n_jet = 6 + 1 + 21                                                              # dim of (G, t, svec H) in d = 6
    eps = Iv.frac(6 * n_jet) * E6                                                   # spectral norm <= n . 2E; C_ref >= I/3
    b_exp = 14 if MUT != "sandwich-exponent" else 12                                # a = 79/6, b = 27/2, both <= 14
    num = Iv(ZERO)
    for k in range(1, 15):
        num = num + Iv.frac(math.comb(14, k) - (-1) ** k * math.comb(b_exp, k)) * ipow(eps, k)
    delta = num / ipow(1 - eps, b_exp)
    ratio = Iv((1 / (1 + delta)).lo, (1 + delta).hi)
    c6_torus = c6 * ratio

    # ---- rules
    kq = sorted({k for inner in (red["innerE"], red["innerR"]) for bs in inner.values() for ks in bs.values() for k in ks})
    checks["STRUCTURE"] = bool(red["alpha"] == Fr(15, 32) and red["beta_T"] == Fr(3, 16) and red["kappa"] == (Fr(1, 40), 30)
                               and surd_sq(red["kappa"]) == Fr(3, 160) and red["gE"] == Fr(1, 80) and red["gR"] == Fr(1, 32)
                               and red["gE"] + surd_sq(red["kappa"]) == red["gR"] and kq == [1, 3, 5, 7, 9, 11] and red["kmax"] == 11
                               and info5["n_cert"] == (28, 26))
    checks["JK_EXACT"] = bool(jk_identities(red["JK"]))
    checks["LAYER_EXACT"] = bool(g_identities(24, (Fr(1, 24), Fr(1, 6), Fr(1, 48), Fr(1, 16), Fr(3, 16))))
    checks["D1_EXACT"] = bool(d1 == E(Fr(4, 3)))
    checks["D2_EXACT"] = bool(d2 == add(E(Fr(29, 6)), E(-1, 6)))
    checks["D3_EXACT"] = bool(d3 == D3_target)
    checks["D4_EXACT"] = bool(d4 == D4_target and z4 == E(1536, 1, 2))
    checks["D4_CERTIFIED_PATH"] = bool(D4_cert.contains(D4_iv) and D4_cert.width() < Decimal("1e-30"))
    checks["MEHTA_Z5"] = bool(Z5.contains(Z5_literal) and Z5.intersects(Z5_closed) and Z5.width() < Decimal("1e-30"))
    checks["ROUTE_CONSISTENT"] = bool(D5.intersects(D5_route2) and D5_route2.width() < Decimal("1e-30"))
    checks["TRUNCATION_NESTING"] = bool(D5_coarse.intersects(D5) and D5_coarse.width() >= D5.width())
    checks["WIDTHS"] = bool(D5.width() < Decimal("1e-30") and c6.width() < Decimal("1e-30"))
    fl5 = float_layered(red)
    checks["FLOAT_INSIDE"] = bool(abs(Decimal(repr(fl5)) - D5.lo) < Decimal("5e-3") * D5.lo)
    side = {d: Iv(Decimal(a), Decimal(b)) for d, (a, b) in SIDE24.items()}
    checks["SIDE24_CONSISTENT"] = bool(all(c.intersects(side[d] + Iv(-side[d].hi * Decimal("1e-106"), side[d].hi * Decimal("1e-106")))
                                           for d, c in ((2, c2), (3, c3)))
                                       and Iv(Decimal(CERTIFIED_200["D_4"][0]), Decimal(CERTIFIED_200["D_4"][1])).contains(D4_iv)
                                       and Iv(Decimal(CERTIFIED_200["c_5_ref"][0]), Decimal(CERTIFIED_200["c_5_ref"][1])).contains(c5))
    checks["CLOSED_FORM_D6"] = bool(c6.intersects(c6_closed))
    checks["MONOTONE"] = bool(D4_iv.hi < D5.lo and c5.lo > c6.hi and c6.lo > 0)
    checks["IMAGE_BOUND"] = bool(Decimal("3.8e-110") < E6.lo and E6.hi < Decimal("3.9e-110"))
    # proving direction: 27 eps <= delta <= 29 eps for every represented value; the enclosures of delta and 28 eps intersect
    checks["TRANSFER_BOUND"] = bool(delta.lo >= CC.multiply(Decimal(27), eps.hi) and delta.hi <= CF.multiply(Decimal(29), eps.lo)
                                    and delta.hi >= CF.multiply(Decimal(28), eps.lo) and delta.lo <= CC.multiply(Decimal(28), eps.hi)
                                    and c6_torus.contains(c6))
    checks["LIBRARY_EXACT"] = bool(library_exact())
    pinned = {"D_5": D5, "c_6_ref": c6, "pi": PI}
    checks["PINNED"] = bool(all(str(pinned[k].lo).startswith(p) and str(pinned[k].hi).startswith(p) for k, p in PINNED.items()))
    if len(checks) != 20:
        raise ValueError("rule count changed")
    passed = all(checks.values())

    def fr(x):
        return str(x)
    out = {
        "object": "CL-SIDE24-D6-COEFFICIENT-20260930-v1",
        "scientific_effect": "NONE",
        "certified": True,
        "mutant": MUT,
        "passed": passed,
        "checks": checks,
        "cone_moments": {"D_1": show(d1), "D_2": show(d2), "D_3": show(d3), "D_4": show(d4), "D_4_interval": D4_iv.pair(),
                         "D_4_certified_path": D4_cert.pair(), "D_5": D5.pair(), "D_5_route_certified_e": D5_route2.pair(),
                         "D_5_coarse": D5_coarse.pair()},
        "coefficients": {"c_2_ref": c2.pair(), "c_3_ref": c3.pair(), "c_4_ref": c4.pair(), "c_5_ref": c5.pair(),
                         "c_6_ref": c6.pair(), "c_6_ref_closed": c6_closed.pair(), "c_6_24": c6_torus.pair()},
        "transfer": {"E_6": E6.pair(), "epsilon": eps.pair(), "delta": delta.pair(), "n_jet": n_jet,
                     "exponents": {"a": "79/6 (<= 14 used)", "b": "27/2 (<= 14 used)"}},
        "constants": {"pi": PI.pair(), "Gamma(7/6)": GAMMA76.pair(), "(3/2)^(1/3)": CBRT32.pair(),
                      "Z_5": Z5_closed.pair(), "Z_5_layered": Z5.pair(), "46080 pi^(3/2)": Z5_literal.pair(), "|S^5|": sphere_area(6).pair()},
        "quadrature": {"K": K_ORDER, "cells": "width 1/2 on [0, 30], width 1 on [30, 90]", "rho": str(RHO)},
        "layers": {"m=5": {"alpha": fr(red["alpha"]), "beta": "%s T" % fr(red["beta_T"]), "kappa": "%s sqrt(%d)" % red["kappa"],
                           "kappa^2": fr(surd_sq(red["kappa"])), "lambda_erfc": fr(red["gE"]), "lambda_poly": fr(red["gR"]),
                           "prefactor": show(red["pref"]), "G_0_constant": show(red["constE"]),
                           "q_powers": kq, "T_powers_erfc": sorted(red["innerE"]), "T_powers_poly": sorted(red["innerR"]),
                           "certified_functions_(erfc, poly)": list(info5["n_cert"]),
                           "exact_part_erfc_branch": show(info5["exact_E"]), "exact_part_poly_branch": show(info5["exact_R"]),
                           "certified_part_erfc_branch": info5["cert_E"].pair(), "certified_part_poly_branch": info5["cert_R"].pair()},
                   "Z_5": {"kmax": zinfo["kmax"], "exact_part": show(zinfo["exact"]), "certified_part": zinfo["cert"].pair()},
                   "atoms": ["1"] + ["arctan(%s sqrt(%d))" % a for a in ATOMS[1:]]},
        "float_reference_10sig": {"D_5_layered_trapezoid": "%.10g" % fl5},
        "exact_relations": {
            "D_m": "sqrt(3/(m+3)) Z_m^-1 int_{mu<0} prod mu_i^2 |Delta(mu)| exp(-|mu|^2/4 + (sum mu)^2/(4(m+3))) dmu",
            "m=5 coordinates": "T = 4p1 + 3p2 + 2p3 + p4, w = 3p2 + 2p3 + p4, v = 2p3 + p4, q = p4; dp = dT dw dv dq/24; 0 < q < v < w < T",
            "exponents": "erfc branch T^2/80 + w^2/48 + v^2/24 + q^2/8 with erfc(sqrt30 T/40); polynomial branch T^2/32 + w^2/48 + v^2/24 + q^2/8",
            "layers": "q: J_k (odd k, no erf); v: incomplete Gaussian moments G_n(mu, w), mu in {1/24, 1/6}; w: exact for the polynomial and e^{-mu w^2} parts (G_n(1/48 + mu, T)), certified cumulative Omega_{a,mu}(T) = int_0^T Q(w) e^{-w^2/48} erf(sqrt(mu) w) dw for the erf parts; T: exact two-erf moments plus the certified Taylor quadrature",
            "Z_5": "5! (1/24) 2 sqrt(pi/5) int_{0<q<v<w<T} V e^{-T^2/80 - w^2/48 - v^2/24 - q^2/8} = 2^(15/2) (2 pi)^(5/2) prod_{j<=5} Gamma(1+j/2)/Gamma(3/2)^5 = 46080 pi^(3/2)",
            "c_d_ref": "Gamma(7/6) (3/2)^(1/3) |S^{d-1}| D_{d-1} / (sqrt3 sqrt pi (2 pi)^d); d = 6: |S^5| = pi^3, c_{6,ref} = Gamma(7/6) (3/2)^(1/3) D_5 / (64 sqrt3 pi^(7/2))",
            "transfer": "E_6 = 314496 (76 24^6 + 15) e^-288; eps = 168 E_6; c_{6,24}/c_{6,ref} in [1/(1+delta), 1+delta], delta = (1+eps)^14/(1-eps)^14 - 1",
        },
    }
    sys.stdout.write(json.dumps(out, indent=1) + "\n")
    sys.stderr.write("seconds %.1f\n" % (time.time() - t0))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
