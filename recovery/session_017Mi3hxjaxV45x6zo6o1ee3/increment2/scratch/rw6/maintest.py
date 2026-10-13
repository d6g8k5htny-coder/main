# ---------------- smoke test
def main():
    t0 = time.time()
    p5, g5, st = F_pieces(5)
    print("stages", st, "gamma", g5, "kinds", sorted(str(k) for k, v in p5.items() if trim(v)), "%.1fs" % (time.time() - t0), flush=True)
    for m in (2, 3, 4):
        F_pieces(m)
    print("m=2..4 built %.1fs" % (time.time() - t0), flush=True)
    b5 = Birth(5)
    print("model built %.1fs" % (time.time() - t0), flush=True)
    full = (-INF, INF)
    I = b5.I(full)
    D5 = Iv(Decimal(D5_204[0]), Decimal(D5_204[1]))
    print("I_full", I.pair(), "width", I.width(), "intersects #204:", I.intersects(D5), "%.1fs" % (time.time() - t0), flush=True)
    band = b5.I((Fr(0), Fr(1)))
    print("band", band.pair(), "F6B", (band / I).pair(), "%.1fs" % (time.time() - t0), flush=True)
    M = b5.M
    for kk, vals in M.Kvals.items():
        al, g = kk
        for s0, v in vals[:2] + vals[40:41]:
            print(kk, s0, v.pair(), "float", K_float(float(al), float(g), float(s0)))
    print("K at cut", {str(k): v.pair() for k, v in M.K_at_cut.items()}, "tail", M.tail, "Rsum", M.Rsum)
    print("float control band", I5_float((Fr(0), Fr(1))), "full", I5_float((-60.0, 60.0)), "%.1fs" % (time.time() - t0))
    for T in (3.0, 6.0):
        print("F5 T=%g pieces %.12g direct %.12g" % (T, F_float(p5, T), F5_direct_float(T)), "%.1fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
