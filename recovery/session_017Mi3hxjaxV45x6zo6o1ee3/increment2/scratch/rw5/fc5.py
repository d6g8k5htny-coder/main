import math, time
def gl(n):
    xs, ws = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1): p0, p1 = p1, ((2*k-1)*x*p1 - (k-1)*p0)/k
            dp = n*(x*p1 - p0)/(x*x - 1); dx = p1/dp; x -= dx
            if abs(dx) < 1e-15: break
        xs.append(x); ws.append(2/((1-x*x)*dp*dp))
    return xs, ws
def M4(b, n, R=16.0):
    xs, ws = gl(n); pts = [(R*(x+1)/2, w*R/2) for x, w in zip(xs, ws)]; tot = 0.0
    for a, wa in pts:
        l4 = b - a
        for p1, w1 in pts:
            l3 = l4 - p1
            for p2, w2 in pts:
                l2 = l3 - p2; e3 = l4*l4 + l3*l3 + l2*l2
                f3 = wa*w1*w2*(a*(a+p1)*(a+p1+p2))**2*p1*p2*(p1+p2)
                for p3, w3 in pts:
                    l1 = l2 - p3
                    tot += f3*w3*(a+p1+p2+p3)**2*p3*(p2+p3)*(p1+p2+p3)*math.exp(-(e3+l1*l1)/4)
    return 24/(1536*math.pi)*tot
for n in (14, 20, 26):
    t = time.time(); print(n, M4(0.5, n), M4(0.5, n, R=12.0), time.time()-t)
def I4(B, nb, n, R):
    xs, ws = gl(nb); lo, hi = B; tot = 0.0
    for x, w in zip(xs, ws):
        b = lo + (hi-lo)*(x+1)/2
        tot += w*math.sqrt(3/(4*math.pi))*math.exp(-3*b*b/4)*M4(b, n, R)
    return tot*(hi-lo)/2
t = time.time()
print(I4((0, 1), 12, 28, 14.0), "cert 0.621607797393378202790634837860", time.time()-t)
t = time.time()
print(I4((1, 2), 12, 28, 14.0), "cert 5.528374506406810830945596115349", time.time()-t)
