def K_owen(al, g, T):
    """K_{al,g}(T) = (pi/sqrt(al g)) [arctan(a)/(2 pi) - T_Owen(sqrt(2 al) T, a)], a = sqrt(g/al) > 1, through the reflection
    T(h, a) = (Phi(h) + Phi(a h))/2 - Phi(h) Phi(a h) - T(a h, 1/a) (h >= 0)."""
    A = (Iv.frac(g) / Iv.frac(al)).sqrt()
    h = (2 * Iv.frac(al)).sqrt() * Iv.frac(T)
    ah = A * h
    Th = (Phi(h) + Phi(ah)) / 2 - Phi(h) * Phi(ah) - owen_T(ah, 1 / A, N=1400)
    return PI / (Iv.frac(al) * Iv.frac(g)).sqrt() * (atan_iv(A, N=400) / (2 * PI) - Th) if False else PI / (Iv.frac(al) * Iv.frac(g)).sqrt() * ((PI / 2 - atan_iv(1 / A, N=400)) / (2 * PI) - Th)


def F_iv(pieces, T):
    tot = Iv(ZERO)
    Ti = Iv.frac(T)
    for key, poly in pieces.items():
        poly = trim(poly)
        if not poly:
            continue
        pv = Iv.frac(sum(c * Fr(T) ** i for i, c in enumerate(poly)))
        if key[0] == "e":
            tot = tot + pv * (-(Iv.frac(key[1]) * Ti.sq())).exp()
        elif key[0] == "eE":
            al, g = key[1], key[2]
            tot = tot + pv * (-(Iv.frac(al) * Ti.sq())).exp() * (PI / Iv.frac(g)).sqrt() / 2 * erf_iv(Iv.frac(g).sqrt() * Ti)
        elif key[0] == "K":
            tot = tot + pv * K_owen(key[1], key[2], T)
        else:
            raise ValueError(key)
    return tot


def main():
    p5 = F_pieces(5)[0]
    for T in (Fr(3), Fr(6), Fr(10)):
        t0 = time.time()
        v = F_iv(p5, T)
        print("T", T, "F_iv", v.pair(), "%.1fs" % (time.time() - t0), flush=True)
        for n in (12, 16, 20, 24):
            print("   direct n=%d %.15g" % (n, F5_direct_float(float(T), n)), flush=True)
    for al, g in ((Fr(1, 80), Fr(1, 16)), (Fr(1, 30), Fr(1, 24)), (Fr(1, 30), Fr(1, 6))):
        print((al, g), "owen K(1/4)", K_owen(al, g, Fr(1, 4)).pair(), "K(81/4)", K_owen(al, g, Fr(81, 4)).pair())


if __name__ == "__main__":
    main()
