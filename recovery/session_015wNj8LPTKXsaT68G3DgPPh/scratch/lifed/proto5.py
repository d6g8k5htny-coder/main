from fractions import Fraction as Fr
import lifecore as LC

def exact_cov(M, F, G, key, r):
    ser, w, _ = M.pair_series(F, G, key)
    return sum(c * r ** p for p, c in ser.items())

def solve(A, B):
    n = len(A)
    Mx = [list(A[i]) + [B[c][i] for c in range(len(B))] for i in range(n)]
    for col in range(n):
        p = max(range(col, n), key=lambda i: abs(Mx[i][col]))
        Mx[col], Mx[p] = Mx[p], Mx[col]
        for i in range(col + 1, n):
            f = Mx[i][col] / Mx[col][col]
            for j in range(col, len(Mx[i])):
                Mx[i][j] -= f * Mx[col][j]
    X = []
    for c in range(len(B)):
        x = [None] * n
        for i in reversed(range(n)):
            s = Mx[i][n + c] - sum(Mx[i][j] * x[j] for j in range(i + 1, n))
            x[i] = s / Mx[i][i]
        X.append(x)
    det = 1
    for i in range(n): det *= Mx[i][i]
    return X, det

def law_at(M, r, names):
    pins, targets = M.functionals()
    P = [pins[n] for n in M.PIN_NAMES]
    T = [targets[n] for n in names]
    np_ = len(P)
    Spp = [[exact_cov(M, P[i], P[j], ('p', i, j), r) for j in range(np_)] for i in range(np_)]
    Stp = [[exact_cov(M, T[i], P[j], ('tp', M.TARGET_NAMES.index(names[i]), j), r) for j in range(np_)] for i in range(len(T))]
    Stt = [[exact_cov(M, T[i], T[j], ('tt', M.TARGET_NAMES.index(names[i]), M.TARGET_NAMES.index(names[j])), r) for j in range(len(T))] for i in range(len(T))]
    kvec = [-r ** 3 / 2, -r ** 2, Fr(12)] + [Fr(0)] * (np_ - 3)
    e1 = [Fr(1)] + [Fr(0)] * (np_ - 1)
    W, det = solve(Spp, Stp)
    xs, _ = solve(Spp, [e1, kvec])
    alpha = [W[i][0] for i in range(len(T))]
    beta = [sum(W[i][j] * kvec[j] for j in range(np_)) for i in range(len(T))]
    C = [[Stt[i][j] - sum(W[i][l] * Stp[j][l] for l in range(np_)) for j in range(len(T))] for i in range(len(T))]
    ebb = xs[0][0]; ebk = sum(kvec[j] * xs[0][j] for j in range(np_)); ekk = sum(kvec[j] * xs[1][j] for j in range(np_))
    return alpha, beta, C, det, (ebb, ebk, ekk)

for d, M, names in ((2, LC.E2, ['w', 'om', 'v', 'T', 'tau', 'nu']), (3, LC.E3, ['a11', 'a22', 'a12', 'o11', 'o22', 'o12', 'v1', 'v2', 'T', 'tau'])):
    print('d =', d)
    for r in (Fr(1, 64), Fr(1, 128), Fr(1, 256)):
        al, be, C, det, e = law_at(M, r, names)
        i0 = 0
        out = ['r=1/%d' % (1 / r)]
        out.append('det-12=%.4e (/r^2 %.3f)' % (float(det - 12), float((det - 12) / r ** 2)))
        out.append('ebb-3/2 /r^2=%.3f ebk/r^3=%.3f ekk-24 /r^2=%.3f' % (float((e[0] - Fr(3, 2)) / r ** 2), float(e[1] / r ** 3), float((e[2] - 24) / r ** 2)))
        print('  ', ' '.join(out))
        for i, n in enumerate(names):
            print('     %-4s alpha=%+.6e beta=%+.6e  (beta/r=%+.4f beta/r^2=%+.4f beta/r^3=%+.4f) var=%.8f' % (n, float(al[i]), float(be[i]), float(be[i] / r), float(be[i] / r ** 2), float(be[i] / r ** 3), float(C[i][i])))
        if d == 2:
            w = 0
            print('     alpha_w+1 /r^2 = %.4f  var_w-2 /r^2 = %.4f  cov(om,w)/r = %.4f cov(v,w)= %.2e' % (float((al[0] + 1) / r ** 2), float((C[0][0] - 2) / r ** 2), float(C[1][0] / r), float(C[2][0])))
        else:
            print('     alpha_a11+1 /r^2 = %.4f  var_a11-2 /r^2 = %.4f var_a12-1 /r^2 = %.4f cov(a11,a22)/r^2=%.4f' % (float((al[0] + 1) / r ** 2), float((C[0][0] - 2) / r ** 2), float((C[2][2] - 1) / r ** 2), float(C[0][1] / r ** 2)))
