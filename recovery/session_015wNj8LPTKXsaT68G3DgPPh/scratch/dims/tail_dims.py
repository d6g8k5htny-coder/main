

# ----------------------------------------------------------------------------- floating controls (not part of the certificate)
def _Phi_f(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _phi_f(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def r3r_float(b, n=4000, top=14.0):
    """Floating R_3(b) R(b) = phi_2(b) N_3(b) / m_(3,b), and R_3(b) = N_3(b) m_(2,b) / m_(3,b):
    N_3(b) = sqrt(pi/2) E[X_+^3], X ~ N(b, 2); m_(3,b) = E[((b + t)^2 - rho^2)^2 1{b + t > rho}], t ~ N(0,1), rho Rayleigh(1)."""
    s = math.sqrt(2)
    mu = b
    ex3 = (mu ** 3 + 3 * mu * s * s) * _Phi_f(mu / s) + s * (mu * mu + 2 * s * s) * _phi_f(mu / s)
    N3 = math.sqrt(math.pi / 2) * ex3
    m2 = (2 + b * b) * _Phi_f(b / s) + s * b * _phi_f(b / s)

    def inner(rho):                                         # int_rho^inf (u^2 - rho^2)^2 phi(u - b) du
        z = rho - b
        P0 = 1 - _Phi_f(z)
        ph = _phi_f(z)
        # moments of u = b + x over x > z: E[x^j 1{x > z}] for x ~ N(0,1)
        m = [P0, ph]
        for j in range(2, 5):
            m.append(z ** (j - 1) * ph + (j - 1) * m[j - 2])
        U = lambda p: sum(math.comb(p, j) * b ** (p - j) * m[j] for j in range(p + 1))
        return U(4) - 2 * rho * rho * U(2) + rho ** 4 * U(0)
    h = top / n
    acc = 0.0
    for i in range(n + 1):
        rho = i * h
        w = 1 if i in (0, n) else (4 if i % 2 else 2)
        acc += w * rho * math.exp(-rho * rho / 2) * inner(rho)
    m3 = acc * h / 3
    phi2 = math.exp(-b * b / 4) / (2 * math.sqrt(math.pi))
    return {'b': b, 'N3': N3, 'm2': m2, 'm3': m3, 'R3': N3 * m2 / m3, 'R': phi2 / m2, 'R3R': phi2 * N3 / m3}


def _ynorm(T111, T112, T122, T222, n=64):
    best, bt = 0.0, 0.0
    for i in range(n):
        th = math.pi * i / n
        c, s = math.cos(th), math.sin(th)
        y = abs(T111 * c ** 3 + 3 * T112 * c * c * s + 3 * T122 * c * s * s + T222 * s ** 3)
        if y > best:
            best, bt = y, th
    f = lambda th: abs(T111 * math.cos(th) ** 3 + 3 * T112 * math.cos(th) ** 2 * math.sin(th)
                       + 3 * T122 * math.cos(th) * math.sin(th) ** 2 + T222 * math.sin(th) ** 3)
    lo, hi = bt - math.pi / n, bt + math.pi / n
    g = (math.sqrt(5) - 1) / 2
    for _ in range(40):
        a, c = hi - g * (hi - lo), lo + g * (hi - lo)
        if f(a) > f(c):
            hi = c
        else:
            lo = a
    return max(best, f((lo + hi) / 2))


def mc_J(d, k, n, seed):
    """Monte Carlo of J^(d)(k) (floating): v ~ N(0, I/2), Omega GOE(1), Y with T111, T222 ~ N(0,6), T112, T122 ~ N(0,2);
    for d = 2 only the first components enter M."""
    rng = random.Random(seed)
    tot = tot2 = 0.0
    k = float(k)
    for _ in range(n):
        v1, v2 = rng.gauss(0, math.sqrt(0.5)), rng.gauss(0, math.sqrt(0.5))
        o11, o22, o12 = rng.gauss(0, math.sqrt(2)), rng.gauss(0, math.sqrt(2)), rng.gauss(0, 1)
        T111, T222 = rng.gauss(0, math.sqrt(6)), rng.gauss(0, math.sqrt(6))
        T112, T122 = rng.gauss(0, math.sqrt(2)), rng.gauss(0, math.sqrt(2))
        if d == 3:
            vn = math.hypot(v1, v2)
            tr, df = (o11 + o22) / 2, math.hypot((o11 - o22) / 2, o12)
            on = max(abs(tr + df), abs(tr - df))
            yn = _ynorm(T111, T112, T122, T222)
        else:
            vn, on, yn = abs(v1), abs(o11), abs(T111)
        M = max(12 * k, 2 * vn, on, yn)
        a, c = v1 * v1, 6 * k * o11 - v1 * v1
        m0, S = max(a, c), 8 * M * M
        P = lambda s: s ** 3 / 3 - (a + c) * s * s / 2 + a * c * s
        x = (P(S) - P(m0)) / (216 * k ** 3)
        tot += x
        tot2 += x * x
    m = tot / n
    return {'d': d, 'k': str(Fr(k).limit_denominator(64)), 'n': n, 'seed': seed, 'J': m, 'se': math.sqrt(max(tot2 / n - m * m, 0) / n),
            'J_over_2359296k3': m / (2359296 * k ** 3)}


# ----------------------------------------------------------------------------- certified record, check, main
def dec_down(x, places):
    x = Fr(x)
    n = (x.numerator * 10 ** places) // x.denominator
    s = str(abs(n)).rjust(places + 1, '0')
    return ('-' if n < 0 else '') + s[:-places] + '.' + s[-places:]


def dec_up(x, places):
    x = Fr(x)
    n = -((-x.numerator * 10 ** places) // x.denominator)
    s = str(abs(n)).rjust(places + 1, '0')
    return ('-' if n < 0 else '') + s[:-places] + '.' + s[-places:]


def iv_str(x, places=10):
    return [dec_down(x.lo, places), dec_up(x.hi, places)]


def certified():
    check_contact_law()
    lhs, rhs = m3_0_polar()
    require(abs(lhs.mid() - rhs.mid()) < Fr(1, 10 ** 40), 'm_(3,0) closed form')
    r3 = R3_0()
    out = {'contact_law_d3': 'verified exactly (rational covariances)', 'm3_0': iv_str(rhs, 20), 'R3_0': iv_str(r3, 20),
           'R_0': iv_str(R_0, 20), 'J': {}, 'J3_from_G': {}, 'cG3_0': {}, 'G_containment': {}}
    for d in (2, 3):
        out['J'][str(d)] = {}
        for k in K_GRID:
            br, parts = J_bracket(d, k)
            out['J'][str(d)][str(k)] = {'bracket': iv_str(br, 6), 'B_P': dec_up(parts['B_P'], 6),
                                        'T_hi': dec_up(parts['T_hi'], 6), 'T_lo': dec_up(parts['T_lo'], 12)}
            if d == 2:
                g_lo, g_hi = (Fr(x) for x in JG[k])
                ok = br.lo <= g_lo and g_hi <= br.hi
                require(ok or MUT is not None, 'planar bracket does not contain [G] at k = %s' % k)
                out['G_containment'][str(k)] = ok
    for k in K_GRID:
        j3 = J3_from_G(k)
        out['J3_from_G'][str(k)] = iv_str(j3, 10)
        out['cG3_0'][str(k)] = iv_str(R3_0() * R_0 * j3, 6)
    return out


def write_results(res):
    with open(os.path.join(HERE, 'RESULTS.json'), 'w') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
        fh.write('\n')


def load_results():
    with open(os.path.join(HERE, 'RESULTS.json')) as fh:
        return json.load(fh)


MC_PLAN = [(d, k, 200000, seed) for d in (2, 3) for k in (Fr(1, 6), Fr(1, 2), Fr(3, 4)) for seed in (1, 2)]


def main():
    global MUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--mutant', choices=MUTANTS)
    ap.add_argument('--controls', action='store_true', help='recompute the floating controls (slow)')
    ap.add_argument('--procs', type=int, default=4)
    args = ap.parse_args()
    MUT = args.mutant
    if args.check:
        ref = load_results()
        cur = certified()
        require(cur == ref['certified'], 'certified values differ from RESULTS.json')
        require(ref['mutant'] is None, 'RESULTS.json produced under a mutant')
        # floating controls recorded: d = 3 Monte Carlo inside the certified enclosures where they are informative
        for row in ref['controls']['mc']:
            k = Fr(row['k'])
            if row['d'] == 3 and k in (Fr(3, 4),):
                lo, hi = (float(Fr(x)) for x in ref['certified']['J3_from_G'][str(k)])
                require(lo - 5 * row['se'] <= row['J'] <= hi + 5 * row['se'], 'MC control outside the enclosure')
        print(json.dumps({'check': 'ok', 'cG3_0': cur['cG3_0'], 'mutant': MUT}))
        return
    res = {'schema': 1, 'object': 'CL-C8-CAP-FAILURE-COEFFICIENT-DIMS-20261001-v1', 'scientific_effect': 'NONE',
           'certified': certified(), 'mutant': MUT}
    if args.controls:
        import multiprocessing
        with multiprocessing.Pool(args.procs) as pool:
            mc = pool.starmap(mc_J, MC_PLAN)
        res['controls'] = {'r3r_float': [r3r_float(b) for b in (0.0, 0.25, 0.5, 0.75, 1.0)], 'mc': mc}
    else:
        res['controls'] = load_results()['controls']
    write_results(res)
    print(json.dumps(res['certified']['cG3_0']))


if __name__ == '__main__':
    main()
