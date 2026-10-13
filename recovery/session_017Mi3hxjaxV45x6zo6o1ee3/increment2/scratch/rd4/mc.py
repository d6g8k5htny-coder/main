import random, math, sys
from math import sqrt, exp, pi
def sample(rng, m, c):
    Q = [[0.0]*m for _ in range(m)]
    for i in range(m):
        Q[i][i] = rng.gauss(0, sqrt(2))
        for j in range(i+1, m):
            Q[i][j] = Q[j][i] = rng.gauss(0, 1)
    z = rng.gauss(0, 1)
    return [[Q[i][j] + (c*z if i == j else 0.0) for j in range(m)] for i in range(m)]
def det3(A):
    return (A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1]) - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]) + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]))
def negdef3(A):   # Sylvester on -A
    B = [[-x for x in row] for row in A]
    return B[0][0] > 0 and (B[0][0]*B[1][1]-B[0][1]**2) > 0 and det3(B) > 0
rng = random.Random(2026); n = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
c = sqrt(2/3); s = 0.0; s2 = 0.0
for _ in range(n):
    A = sample(rng, 3, c)
    v = det3(A)**2 if negdef3(A) else 0.0
    s += v; s2 += v*v
mean = s/n; se = sqrt(max(s2/n - mean*mean, 0)/n)
print("D_3 MC:", mean, "+-", se, "n =", n)
