import math, itertools, time, sys
def mehta_Z_float(m):
    return 2 ** (m / 2 + m * (m - 1) / 4) * (2 * math.pi) ** (m / 2) * math.prod(math.gamma(1 + j / 2) for j in range(1, m + 1)) / math.gamma(1.5) ** m
def float_orthant(m, T, n):
    coef = math.sqrt(3 / (m + 3)) / mehta_Z_float(m)
    h = T / n
    grid = [-(i + 0.5) * h for i in range(n)]
    total = 0.0
    for mu in itertools.product(grid, repeat=m):
        S = sum(mu)
        w = 1.0
        for x in mu:
            w *= x * x
        for i in range(m):
            for j in range(i + 1, m):
                w *= abs(mu[i] - mu[j])
        total += w * math.exp(-sum(x * x for x in mu) / 4 + S * S / (4 * (m + 3)))
    return coef * total * h ** m
t0 = time.time()
for m, T, n in ((4, 16.0, 14), (5, 18.0, 14), (5, 18.0, 18)):
    print("m", m, "n", n, "orthant", float_orthant(m, T, n), round(time.time() - t0, 1), "s", flush=True)
print("reference D_4 = 14.21876")
