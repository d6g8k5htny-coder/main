import sys, importlib
from fractions import Fraction as Q
V = sys.argv[1]
def load(variant):
    p = f'{V}/{variant}/experiments/universality/two_parameter_cusp_population'
    for m in ('population_certificate','finite_certificate','gaussian_tail','mean_adaptive_confidence'):
        sys.modules.pop(m, None)
    sys.path.insert(0, p)
    import population_certificate as pc
    sys.path.remove(p)
    return pc

pc = load('pristine')
print('A. pristine negative_controls():', pc.negative_controls()['status'])
r = pc.negative_controls()
print('   whole_rectangle_mass value =', r['whole_rectangle_mass'], ' gates:', {k: r['gates'][k] for k in ('whole_rectangle_mass','missing_sign_detected','conditioning_detected')})

# B. monkeypatch: halve (and separately 5x) cdf_mass for 0<tau<9/4 only
orig = pc.cdf_mass
for name, factor in (('half', Q(1,2)), ('x5', Q(5))):
    def patched(tau, _o=orig, _f=factor):
        lo, hi = _o(tau)
        return (lo*_f, hi*_f) if 0 < tau < Q(9,4) else (lo, hi)
    pc.cdf_mass = patched
    r = pc.negative_controls()
    print(f'B. monkeypatch cdf_mass x{factor} on (0,9/4): negative_controls() =', r['status'], ' all gates True:', all(r['gates'].values()))
    print('   patched cdf_mass(1/8) =', [float(x) for x in pc.cdf_mass(Q(1,8))], ' vs orig', [float(x) for x in orig(Q(1,8))])
pc.cdf_mass = orig

# C. what CAN make the three gates fail? Only the literals.
print('C. rectangle_mass_control((1/10,1/10)) =', pc.rectangle_mass_control((Q(1,10),Q(1,10))),
      '; rectangle_mass_control((1,1)) =', pc.rectangle_mass_control((Q(1),Q(1))),
      '; these calls take no computed input in negative_controls (lines 458-459)')

# D. pristine closed form just below saturation (reader amendment (a) feasibility)
for k in (20, 40, 60):
    tau = Q(9,4) - Q(1, 1<<k)
    try:
        lo, hi = pc.cdf_mass(tau)
    except ValueError as e:
        print(f'D. cdf_mass(9/4 - 2^-{k}) raised ValueError: {e}'); continue
    print(f'D. cdf_mass(9/4 - 2^-{k}) encloses 1/5? {lo <= Q(1,5) <= hi}; |lo-1/5|={float(abs(lo-Q(1,5))):.3e} |hi-1/5|={float(abs(hi-Q(1,5))):.3e} width={float(hi-lo):.3e}')

# E. sed-halved mutant
pc2 = load('halved')
r2 = pc2.negative_controls()
print('E. halved-mutant negative_controls():', r2['status'], ' all gates True:', all(r2['gates'].values()))
print('   halved cdf_mass(1/8) =', [float(x) for x in pc2.cdf_mass(Q(1,8))], ' pristine =', [float(x) for x in pc.cdf_mass(Q(1,8))])
print('   halved cdf_mass(9/4) =', pc2.cdf_mass(Q(9,4)), '(hard-coded branch, unchanged)')
