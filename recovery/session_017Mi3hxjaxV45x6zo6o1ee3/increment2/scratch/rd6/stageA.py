
# ---------------- m = 5: nested coordinates, two exact layers, the certified cumulative layer
def organize_twvq(P):
    """Polynomial dict in (T, w, v, q) -> {a: {b: {k: [coefficients in v by power]}}}."""
    out = {}
    for (a, b, j, k), c in P.items():
        out.setdefault(a, {}).setdefault(b, {}).setdefault(k, {})[j] = c
    return {a: {b: {k: [d.get(j, Fr(0)) for j in range(max(d) + 1)] for k, d in ks.items()} for b, ks in bs.items()} for a, bs in out.items()}


def m5_reduction():
    """The m = 5 reduction: T = 4p_1 + 3p_2 + 2p_3 + p_4, w = 3p_2 + 2p_3 + p_4, v = 2p_3 + p_4, q = p_4 (Jacobian 1/24,
    region 0 < q < v < w < T); the decoupled exponents T^2/80 + w^2/48 + v^2/24 + q^2/8 (erfc branch) and
    T^2/32 + w^2/48 + v^2/24 + q^2/8 (polynomial branch) are checked symbolically."""
    m = 5
    alpha, beta, PE, PR, expE, rest = a_integrated_terms(m)
    T, w, v, q = (pvar(i, 4) for i in range(4))
    img = [pscale(padd(T, pscale(w, Fr(-1))), Fr(1, 4)), pscale(padd(w, pscale(v, Fr(-1))), Fr(1, 3)),
           pscale(padd(v, pscale(q, Fr(-1))), Fr(1, 2)), q]
    eE, eR, b = psubst(expE, img, 4), psubst(rest, img, 4), psubst(beta, img, 4)
    diag = {(2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2)}
    for e in (eE, eR):
        if set(e) != diag or e[(0, 2, 0, 0)] != Fr(1, 48) or e[(0, 0, 2, 0)] != Fr(1, 24) or e[(0, 0, 0, 2)] != Fr(1, 8):
            raise ValueError("m = 5 decoupling failed: " + repr(e))
    if set(b) != {(1, 0, 0, 0)}:
        raise ValueError("beta is not a multiple of T: " + repr(b))
    gE, gR = eE[(2, 0, 0, 0)], eR[(2, 0, 0, 0)]
    kappa = surd_mul((b[(1, 0, 0, 0)] / 2, 1), surd_sqrt(1 / alpha))
    constE = mul_pi_half(mul_surd(E(Fr(1, 2)), surd_sqrt(1 / alpha)), 1)
    if MUT == "erfc-branch":
        constE = scale(constE, Fr(1, 2))
    jac = Fr(1, 24) if MUT != "jacobian" else Fr(1, 6)
    pref = prefactor(5, jac)
    innerE, innerR = organize_twvq(psubst(PE, img, 4)), organize_twvq(psubst(PR, img, 4))
    kmax = max(k for inner in (innerE, innerR) for bs in inner.values() for ks in bs.values() for k in ks)
    return {"alpha": alpha, "beta_T": b[(1, 0, 0, 0)], "kappa": kappa, "gE": gE, "gR": gR, "constE": constE, "pref": pref,
            "innerE": innerE, "innerR": innerR, "kmax": kmax, "JK": jk_exact(kmax)}


def layer_q(ks, JK, mu_outer):
    """int_0^x e^{-mu_outer y^2} sum_k e_k(y) J_k(y) dy as a T-function in x, for {k: [coefficients in y]} (odd k only)."""
    L = {}
    for k, ek in ks.items():
        A, B, C = JK[k]
        if A:
            raise ValueError("an even innermost power occurred; J_k would carry an error function")
        for j, c in enumerate(ek):
            if c == 0:
                continue
            part = tf_scale(G(j, mu_outer), B) if B else {}
            for i, ci in enumerate(C):
                if ci:
                    part = tf_add(part, tf_scale(G(j + i, mu_outer + Fr(1, 8)), ci))
            L = tf_add(L, tf_scale(part, c))
    return L


def layer_w(inner, JK, certify_e=False):
    """For each T-power a: (exact T-function of Lambda_a(T) = int_0^T e^{-w^2/48} sum_b w^b L_{a,b}(w) dw,
    certified entries {(gamma, mu): {n: element}} for the parts int_0^T w^n e^{-gamma w^2} erf(sqrt(mu) w) dw that are
    not elementary (mu None: no erf factor; only used when certify_e is set))."""
    out = {}
    for a, bs in inner.items():
        exact, cert = {}, {}
        for b, ks in bs.items():
            L = layer_q(ks, JK, Fr(1, 24))
            for (kind, mu), poly in L.items():
                for n, coef in poly.items():
                    if kind == "c" or (kind == "e" and not certify_e):
                        g_here = Fr(1, 48) + (mu if kind == "e" else 0)
                        exact = tf_add(exact, {k2: {n2: mul(coef, e2) for n2, e2 in p2.items()} for k2, p2 in G(b + n, g_here).items()})
                    else:
                        key = (Fr(1, 48) + mu, None) if kind == "e" else (Fr(1, 48), mu)
                        d = cert.setdefault(key, {})
                        d[b + n] = add(d.get(b + n, {}), coef)
        out[a] = (exact, cert)
    return out


def outer_exact(lam, g, erfc, kappa):
    tot = {}
    for a, (exact, _) in lam.items():
        for (kind, mu), poly in exact.items():
            for n, coef in poly.items():
                tot = add(tot, mul(coef, outer_moment(n + a, g, kind, mu, erfc, kappa)))
    return tot


def poly_taylor_short_iv(Q, s0):
    """Taylor coefficients at s0 of sum_j Q[j] s^j with interval coefficients Q."""
    deg = len(Q) - 1
    out = []
    for n in range(deg + 1):
        acc = Iv(ZERO)
        for j in range(n, deg + 1):
            acc = acc + Q[j] * Iv.frac(math.comb(j, n) * Fr(s0) ** (j - n))
        out.append(acc)
    return out


def poly_disc_bound_iv(Q, s0, rho):
    r = Iv.frac(abs(Fr(s0)) + Fr(rho))
    acc, rp = Iv(ZERO), Iv(ONE)
    for c in Q:
        acc = acc + Iv(c.mag()) * rp
        rp = rp * r
    return acc


def certified_entries(lam):
    """Convert the exact-coefficient certified entries of layer_w into interval coefficient lists and tail constants:
    returns ({a: {(gamma, mu): [Iv by power]}}, {a: Fr upper bound of sum |Omega_{a,.}(T)| for all T})."""
    cum, tail = {}, {}
    for a, (_, cert) in lam.items():
        if not cert:
            continue
        cum[a] = {}
        bound = Iv(ZERO)
        for key, Q in cert.items():
            gamma, mu = key
            deg = max(Q)
            coeffs = [iv_eval(Q[n]) if n in Q else Iv(ZERO) for n in range(deg + 1)]
            cum[a][key] = coeffs
            for n, c in enumerate(coeffs):
                if c.lo != 0 or c.hi != 0:
                    bound = bound + Iv(c.mag()) * gauss_moment(n, gamma)                 # |erf| <= 1
        tail[a] = Fr(bound.hi)
    return cum, tail


def enclose_layered(branches, K=None, rho=None, cells=None, remainder=True):
    """int_0^Tmax sum_branches const . prod(prefactors)(T) . sum_a T^a sum_{(gamma,mu)} Omega_{a,gamma,mu}(T) dT + tail, with
    Omega_{a,gamma,mu}(T) = int_0^T Q_{a,gamma,mu}(w) e^{-gamma w^2} erf(sqrt(mu) w) dw (no erf factor when mu is None),
    each Omega carried as a cumulative Taylor factor exactly as in Math-#200 (value at the centre = accumulated certified
    integral; higher coefficients = those of the integrand divided by n; disc bound |Omega(s_0)| + rho M_g; Cauchy remainders
    at the centre with the geometric factor).  A branch is {const, prefactors, cum: {a: {(gamma, mu): [Q by power, Iv]}},
    tail_poly: [Fr by power of T], gamma_tail}: for T >= Tmax the branch is bounded by |const| tail_poly(T) e^{-gamma_tail T^2}."""
    K = K_ORDER if K is None else K
    rho = RHO if rho is None else rho
    cells = CELLS if cells is None else cells
    rho_pow = Iv.frac(Fr(rho) ** (K + 1))
    keys = sorted({key for br in branches for a in br["cum"] for key in br["cum"][a]})
    L = {(bi, a, key): Iv(ZERO) for bi, br in enumerate(branches) for a in br["cum"] for key in br["cum"][a]}
    total = Iv(ZERO)
    for lo, hi in cells:
        h = Fr(hi) - Fr(lo)
        half = h / 2
        s0 = Fr(lo) + half
        if half >= Fr(rho):
            raise ValueError("cell half-width must be below the Cauchy radius")
        geom = 1 / (1 - half / Fr(rho))
        full_even = {n: Iv.frac(2 * half ** (n + 1) / (n + 1)) for n in range(0, K + 1, 2)}
        left = [Iv.frac((-1) ** n * half ** (n + 1) / (n + 1)) for n in range(K + 1)]
        rem_half = Iv.frac(half ** (K + 2) / (K + 2) * geom)
        rem_full = rem_half * 2
        base, base_bound = {}, {}
        for gamma, mu in keys:
            gf = Gauss(gamma)
            coeffs, bound = gf.coefficients(s0, K), gf.disc_bound(s0, rho)
            if mu is not None:
                ef = ErfLin(root_frac(Fr(mu), 2), +1 if MUT != "erf-inside" else -1)
                coeffs, bound = t_mul(coeffs, ef.coefficients(s0, K)), bound * ef.disc_bound(s0, rho)
            base[(gamma, mu)], base_bound[(gamma, mu)] = coeffs, bound
        outer_val, M_outer = Iv(ZERO), Iv(ZERO)
        for bi, br in enumerate(branches):
            acc, M_acc = None, Iv(ZERO)
            for a, entries in br["cum"].items():
                cum_sum, M_sum = None, Iv(ZERO)
                for key, Q in entries.items():
                    g = t_mul_poly(poly_taylor_short_iv(Q, s0), base[key])
                    M_g = poly_disc_bound_iv(Q, s0, rho) * base_bound[key]
                    Ihalf, Ifull = Iv(ZERO), Iv(ZERO)
                    for n in range(K + 1):
                        Ihalf = Ihalf + g[n] * left[n]
                        if n % 2 == 0:
                            Ifull = Ifull + g[n] * full_even[n]
                    if remainder:
                        Rh = (M_g / rho_pow * rem_half).hi
                        Ihalf = Ihalf + Iv(-Rh, Rh)
                        Ifull = Ifull + Iv(-2 * Rh, 2 * Rh)
                    Lc = L[(bi, a, key)] + (Ihalf if MUT != "cumulative-half" else Iv(ZERO))
                    L[(bi, a, key)] = L[(bi, a, key)] + Ifull
                    cum = [Lc] + [g[n - 1] / n for n in range(1, K + 1)]
                    cum_sum = cum if cum_sum is None else t_add(cum_sum, cum)
                    M_sum = M_sum + Iv(Lc.mag()) + Iv.frac(rho) * M_g
                ta = poly_taylor_short([Fr(0)] * a + [Fr(1)], s0)
                term = t_mul_poly(ta, cum_sum)
                M_term = Poly([Fr(0)] * a + [Fr(1)]).disc_bound(s0, rho) * M_sum
                acc = term if acc is None else t_add(acc, term)
                M_acc = M_acc + M_term
            Mb = M_acc
            for f in br["prefactors"]:
                acc = t_mul(acc, f.coefficients(s0, K))
                Mb = Mb * f.disc_bound(s0, rho)
            for n in range(0, K + 1, 2):
                outer_val = outer_val + br["const"] * acc[n] * full_even[n]
            M_outer = M_outer + Iv(br["const"].mag()) * Mb
        if remainder:
            R = (M_outer / rho_pow * rem_full).hi
            outer_val = outer_val + Iv(-R, R)
        total = total + outer_val
    Tmax = Fr(cells[-1][1])
    tail = Iv(ZERO)
    if MUT != "tail-dropped":
        for br in branches:
            g = Iv.of(br["gamma_tail"])
            x = g * Iv.frac(Tmax * Tmax)
            for j, cj in enumerate(br["tail_poly"]):
                if cj == 0:
                    continue
                a = Fr(j + 1, 2)
                if a.denominator == 2:
                    xa1 = ipow(x, int(a - Fr(1, 2))) * x.sqrt() / x
                    gpow = ipow(g, int(a)) * g.sqrt()
                else:
                    xa1 = ipow(x, int(a) - 1)
                    gpow = ipow(g, int(a))
                up = xa1 * (-x).exp()
                if a > 1:
                    up = up / (1 - Iv.frac(a - 1) / x)
                tail = tail + Iv(br["const"].mag()) * Iv.frac(abs(cj)) * up / gpow / 2
    return total + Iv(-tail.hi, tail.hi)


def branch_spec(lam, g, erfc, kappa):
    cum, tail = certified_entries(lam)
    tail_poly = [tail.get(a, Fr(0)) for a in range(max(tail) + 1)] if tail else [Fr(0)]
    kappa_iv = Iv.frac(kappa[0]) * root_frac(Fr(kappa[1]), 2)
    prefactors = [Gauss(g)] + ([ErfLin(kappa_iv, -1)] if erfc else [])
    gamma_tail = g + (surd_sq(kappa) if erfc else 0)
    return {"const": Iv(ONE), "prefactors": prefactors, "cum": cum, "tail_poly": tail_poly, "gamma_tail": gamma_tail}


def D5_enclosure(red, certify_e=False, **kw):
    """D_5 = pref [c_E (exact_E + cert_E) + exact_R + cert_R]; returns (interval, details)."""
    lamE, lamR = layer_w(red["innerE"], red["JK"], certify_e), layer_w(red["innerR"], red["JK"], certify_e)
    exE, exR = outer_exact(lamE, red["gE"], True, red["kappa"]), outer_exact(lamR, red["gR"], False, red["kappa"])
    specE, specR = branch_spec(lamE, red["gE"], True, red["kappa"]), branch_spec(lamR, red["gR"], False, red["kappa"])
    certE, certR = enclose_layered([specE], **kw), enclose_layered([specR], **kw)
    pref, cE = iv_eval(red["pref"]), iv_eval(red["constE"])
    val = pref * (cE * (iv_eval(exE) + certE) + iv_eval(exR) + certR)
    return val, {"exact_E": exE, "exact_R": exR, "cert_E": certE, "cert_R": certR, "n_cert": (sum(len(c) for _, c in lamE.values()), sum(len(c) for _, c in lamR.values()))}


def Z5_enclosure(**kw):
    """Mehta's Z_5 = 5! (1/24) 2 sqrt(pi/5) int_{0<q<v<w<T} V e^{-T^2/80 - w^2/48 - v^2/24 - q^2/8}, V the Vandermonde of the
    gaps p_1 = (T-w)/4, p_2 = (w-v)/3, p_3 = (v-q)/2, p_4 = q; must enclose 46080 pi^(3/2)."""
    T, w, v, q = (pvar(i, 4) for i in range(4))
    p1 = pscale(padd(T, pscale(w, Fr(-1))), Fr(1, 4))
    p2 = pscale(padd(w, pscale(v, Fr(-1))), Fr(1, 3))
    p3 = pscale(padd(v, pscale(q, Fr(-1))), Fr(1, 2))
    p4 = q
    V = pconst(1, 4)
    for f in (p1, p2, p3, p4, padd(p1, p2), padd(p2, p3), padd(p3, p4), padd(padd(p1, p2), p3), padd(padd(p2, p3), p4), padd(padd(padd(p1, p2), p3), p4)):
        V = pmul(V, f)
    inner = organize_twvq(V)
    kmax = max(k for bs in inner.values() for ks in bs.values() for k in ks)
    JK = jk_exact(kmax)
    lam = layer_w(inner, JK)
    ex = outer_exact(lam, Fr(1, 80), False, None)
    spec = branch_spec(lam, Fr(1, 80), False, (Fr(0), 1))
    cert = enclose_layered([spec], **kw)
    jac = Fr(1, 24) if MUT != "jacobian" else Fr(1, 6)
    const = mul_pi_half(mul_surd(E(120 * jac * 2), surd_sqrt(Fr(1, 5))), 1)         # 5! . jac . 2 sqrt(pi/5)
    return iv_eval(const) * (iv_eval(ex) + cert), {"exact": ex, "cert": cert, "kmax": kmax}


if __name__ == "__main__":
    t0 = time.time()
    red = m5_reduction()
    print("reduction", round(time.time() - t0, 1), "s", flush=True)
    t1 = time.time()
    D5, info = D5_enclosure(red)
    print("D_5", D5.pair(), "width", D5.width(), round(time.time() - t1, 1), "s", "n_cert", info["n_cert"], flush=True)
    print("cert parts", info["cert_E"].pair(), info["cert_R"].pair())
    print("float check: exact_E", fl_eval(info["exact_E"]), "exact_R", fl_eval(info["exact_R"]))
    t2 = time.time()
    Z5, zinfo = Z5_enclosure()
    print("Z_5", Z5.pair(), "width", Z5.width(), "target 46080 pi^1.5 =", (46080 * PI * SQRTPI).pair(), round(time.time() - t2, 1), "s", "kmax", zinfo["kmax"], flush=True)
    print("total", round(time.time() - t0, 1), "s")
