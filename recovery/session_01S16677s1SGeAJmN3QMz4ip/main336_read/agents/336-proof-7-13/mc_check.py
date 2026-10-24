"""Monte Carlo check of (2)/(23) lifetime law and the Section 10 iid-copy Laplace functional.
Run: venv/bin/python -I mc_check.py"""
import random, math
random.seed(20261013)
gamma, lam0, C = 4/3, 0.37, 2.9          # arbitrary fixed constants
beta, ell0 = gamma/2, C*lam0**2
def draw_lambda():                        # density gamma v^(gamma-1)/lam0^gamma on (0,lam0): inverse CDF
    return lam0 * random.random()**(1/gamma)
N = 400000
ells = [C*draw_lambda()**2 for _ in range(N)]
# exact CDF (2): (t/ell0)^beta
worst = 0.0
for t in [0.001*ell0, 0.01*ell0, 0.1*ell0, 0.5*ell0, 0.9*ell0]:
    emp = sum(1 for e in ells if e <= t)/N
    exact = (t/ell0)**beta
    worst = max(worst, abs(emp-exact))
    print(f"t/ell0={t/ell0:<6} empirical P(ell<=t)={emp:.5f} exact={exact:.5f} diff={emp-exact:+.5f}")
print("PASS (2) CDF within 3 sigma everywhere" if worst < 3*0.5/math.sqrt(N) + 1e-9 else "FAIL (2)")
# local density slope: log P(ell<=t) vs log t should have slope beta=gamma/2
t1, t2 = 0.01*ell0, 0.1*ell0
p1 = sum(1 for e in ells if e <= t1)/N; p2 = sum(1 for e in ells if e <= t2)/N
slope = (math.log(p2)-math.log(p1))/(math.log(t2)-math.log(t1))
print(f"empirical cumulative power {slope:.3f} vs beta={beta:.3f}  (density power beta-1={beta-1:.3f})")
print("PASS Sec 9 cumulative power" if abs(slope-beta) < 0.03 else "FAIL Sec 9 cumulative power")

# Section 10: n iid copies, t_n with n t_n^beta = Lambda; count points with lifetime mark a in (0,T], x in a half-torus
Lambda, T = 2.0, 1.5
for n in [50, 500, 5000]:
    tn = (Lambda/n)**(1/beta)
    assert T*tn < ell0
    reps = 4000
    counts = []
    for _ in range(reps):
        k = 0
        for _ in range(n):
            lam = draw_lambda()
            if C*lam**2 <= T*tn and random.random() < 0.5:   # x in half the torus (Haar uniform, independent)
                k += 1
        counts.append(k)
    mean = sum(counts)/reps
    # Poisson intensity of (4) on (0,T] x half-torus: Lambda * (T^beta) * 1/2 ... times ell0^-beta: Lambda beta ell0^-beta int_0^T a^(beta-1) da * 1/2
    mu = Lambda*ell0**(-beta)*T**beta*0.5
    # Laplace functional at f = s*1_{window}: E[exp(-s N)] vs exp(-mu(1-e^{-s}))
    s = 0.7
    lap = sum(math.exp(-s*k) for k in counts)/reps
    lap_poisson = math.exp(-mu*(1-math.exp(-s)))
    var = sum((k-mean)**2 for k in counts)/reps
    print(f"n={n:<5} E[N]={mean:.4f} (Poisson mu={mu:.4f}) Var={var:.4f}  Laplace(0.7)={lap:.4f} Poisson={lap_poisson:.4f}")
