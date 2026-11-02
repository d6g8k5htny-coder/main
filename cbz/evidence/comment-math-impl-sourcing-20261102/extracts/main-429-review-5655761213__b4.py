"""Exact check of eq. (4): sum_{n=-2N}^{2N} exp(2 pi i n j/N) = 1 for every 1<=j<N (incl. non-primitive),
using high-precision mpmath, and the resulting fill correlation 1/(4N+1)."""
import mpmath as mp
mp.mp.dps = 50
bad = []
for N in range(2, 41):
    for j in range(1, N):
        s = mp.fsum(mp.cos(2 * mp.pi * n * j / N) for n in range(-2 * N, 2 * N + 1))
        if abs(s - 1) > mp.mpf(10) ** -40:
            bad.append((N, j, s))
        RH = s * (4 * N + 1) / (4 * N + 1) ** 2
        assert abs(RH - mp.mpf(1) / (4 * N + 1)) < mp.mpf(10) ** -40
print({"cases": sum(N - 1 for N in range(2, 41)), "failures": bad})
