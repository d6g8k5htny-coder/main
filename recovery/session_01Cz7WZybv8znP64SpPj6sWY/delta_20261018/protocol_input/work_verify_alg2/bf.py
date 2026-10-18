from fractions import Fraction as F
# P8: exhaustive brute-force over small rational diagrams on a grid with delta-matchings
delta = F(1, 10)
grid = [F(k, 20) for k in range(1, 61)]  # lifetimes 0.05..3.0
def cnt(xs, lo, hi):
    return sum(lo <= x < hi for x in xs) if lo < hi else 0
viol = 0; viol_eq = 0; checked = 0
import random
random.seed(1)
for trial in range(20000):
    # random F lifetimes; P obtained by perturbing each by <=2delta, plus spurious P bars <=2delta, F bars <=2delta may vanish
    nF = random.randint(0, 6)
    Fl = [random.choice(grid) for _ in range(nF)]
    Pl = []
    for x in Fl:
        if x <= 2*delta and random.random() < 0.5:
            continue  # matched to diagonal
        y = x + F(random.randint(-4, 4), 20)
        if y > 0:
            Pl.append(y)
        elif x <= 2*delta:
            pass
        else:
            Pl.append(x)  # keep valid
    for _ in range(random.randint(0, 2)):
        Pl.append(random.choice([g for g in grid if g <= 2*delta]))
    for (lo, hi) in [(F(1), F(2)), (F(3, 10), F(1)), (F(21, 100), F(1, 2)), (2*delta, F(1))]:
        lower = cnt(Fl, lo+2*delta, hi-2*delta); mid = cnt(Pl, lo, hi); upper = cnt(Fl, lo-2*delta, hi+2*delta)
        checked += 1
        ok = lower <= mid <= upper
        if lo > 2*delta and not ok:
            viol += 1
        if lo == 2*delta and not ok:
            viol_eq += 1
print(checked,viol,viol_eq)
