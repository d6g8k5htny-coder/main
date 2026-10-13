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



# ---------------- smoke test
import math as _m
def factor(poly):
    poly = trim(poly)
    nums = [c for c in poly if c]
    if not nums:
        return None
    num = 0
    for c in nums:
        num = _m.gcd(num, c.numerator)
    den = 1
    for c in nums:
        den = den * c.denominator // _m.gcd(den, c.denominator)
    s = Fr(num, den)
    if nums[-1] < 0:
        s = -s
    return s, [int(c / s) for c in poly]


def main():
    t0 = time.time()
    p6, g6, st = F_pieces(6)
    print("gamma", g6, "%.1fs" % (time.time() - t0), flush=True)
    for c, z in st:
        print("  stage", c, z)
    for k, v in sorted(p6.items(), key=lambda kv: str(kv[0])):
        f = factor(v)
        print(" ", k, "ZERO" if f is None else ("deg %d scale %s %s" % (len(f[1]) - 1, f[0], f[1])), flush=True)
    pm, _, stm = F_pieces(6, mehta=True)
    print("mehta kinds", {str(k): trim(v) for k, v in pm.items() if trim(v)}, flush=True)
    for T in (3, 6, 10):
        t1 = time.time()
        fi = F_iv(p6, Fr(T))
        print("T", T, "F_iv", fi.pair(), "%.1fs" % (time.time() - t1), flush=True)
        for n in (10, 12):
            t1 = time.time()
            print("   direct n=%d %.15g  %.1fs" % (n, Fm_direct_float(6, float(T), n), time.time() - t1), flush=True)
    print("m=5 direct T=3 n=12: %.15g (Math-#222 n=20: 5.71783857161613e-08)" % Fm_direct_float(5, 3.0, 12), flush=True)
    t1 = time.time()
    b6 = Birth(6)
    print("model built %.1fs" % (time.time() - t1), "Rsum", b6.M.Rsum, "tail", b6.M.tail, flush=True)
    full = (-INF, INF)
    I = b6.I(full)
    print("D6", I.pair(), "width", I.width(), flush=True)
    band = b6.I((Fr(0), Fr(1)))
    print("band", band.pair(), "F7B", (band / I).pair(), flush=True)
    print("float full", Im_float(6, (-80.0, 80.0)), "band", Im_float(6, (0.0, 1.0)), flush=True)
    C7 = GAMMA76 * CBRT32 * I / (120 * SQRT3 * PI.sq().sq() * SQRTPI)
    print("c7ref", C7.pair(), "total %.1fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
