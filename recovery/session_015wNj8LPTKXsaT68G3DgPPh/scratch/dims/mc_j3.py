import math, random, sys
def ynorm(T111, T112, T122, T222, n=64):
    best = 0.0; bt = 0.0
    for i in range(n):
        th = math.pi * i / n
        c, s = math.cos(th), math.sin(th)
        y = abs(T111*c**3 + 3*T112*c*c*s + 3*T122*c*s*s + T222*s**3)
        if y > best: best, bt = y, th
    # refine by golden-section around bt
    lo, hi = bt - math.pi/n, bt + math.pi/n
    f = lambda th: -abs(T111*math.cos(th)**3 + 3*T112*math.cos(th)**2*math.sin(th) + 3*T122*math.cos(th)*math.sin(th)**2 + T222*math.sin(th)**3)
    g = (math.sqrt(5)-1)/2
    for _ in range(40):
        a = hi - g*(hi-lo); b = lo + g*(hi-lo)
        if f(a) < f(b): hi = b
        else: lo = a
    return max(best, -f((lo+hi)/2))
def F(a, c, M):
    m0 = max(a, c); S = 8*M*M
    if m0 >= S: return 0.0
    P = lambda s: s**3/3 - (a+c)*s*s/2 + a*c*s
    return P(S) - P(m0)
def run(k, n, seed, d3=True):
    rng = random.Random(seed); tot = 0.0; tot2 = 0.0
    for _ in range(n):
        v1, v2 = rng.gauss(0, math.sqrt(0.5)), rng.gauss(0, math.sqrt(0.5))
        o11, o22, o12 = rng.gauss(0, math.sqrt(2)), rng.gauss(0, math.sqrt(2)), rng.gauss(0, 1)
        T111, T222 = rng.gauss(0, math.sqrt(6)), rng.gauss(0, math.sqrt(6))
        T112, T122 = rng.gauss(0, math.sqrt(2)), rng.gauss(0, math.sqrt(2))
        if d3:
            vn = math.hypot(v1, v2)
            tr, df = (o11 + o22)/2, math.hypot((o11 - o22)/2, o12)
            on = max(abs(tr + df), abs(tr - df))
            yn = ynorm(T111, T112, T122, T222)
        else:
            vn, on, yn = abs(v1), abs(o11), abs(T111)
        M = max(12*k, 2*vn, on, yn)
        x = F(v1*v1, 6*k*o11 - v1*v1, M) / (216*k**3)
        tot += x; tot2 += x*x
    m = tot/n; se = math.sqrt(max(tot2/n - m*m, 0)/n)
    return m, se
k = eval(sys.argv[1]); n = int(sys.argv[2]); seed = int(sys.argv[3]); d3 = sys.argv[4] == '3'
m, se = run(k, n, seed, d3)
print(sys.argv[1], 'd=%s' % sys.argv[4], 'J = %.6g +- %.2g' % (m, se), 'J/(2359296 k^3) = %.6f +- %.6f' % (m/(2359296*k**3), se/(2359296*k**3)))
