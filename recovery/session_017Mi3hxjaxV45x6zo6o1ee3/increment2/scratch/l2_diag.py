import sys, time, json; sys.argv = ['x']
import pjc
from pjc import D, jet_data, directions, q_derivatives, coefficient_d2, cos_sin, PI, fmt, stirling_gamma

def kum(L):
    q2, q4, q6, tail = q_derivatives(L)
    m2, m4, m6 = -q2, q4, -q6
    return {2: m2, 4: m4 - 3*m2*m2, 6: m6 - 15*m4*m2 + 30*m2**3}, (q2, q4, q6, tail)

def integrand(u, k):
    jd = jet_data(u, k)
    pG = 1/(2*PI*jd['det_G'].sqrt()); pV = 1/(2*PI*jd['det_V'].sqrt())
    return pG*pV*jd['tau2']**(D(2)/3)*jd['S_AgV'][0][0]/2

pref = stirling_gamma(D(7)/6)/(D(24)**(D(1)/3)*PI.sqrt())
q2r, q4r, q6r, _ = q_derivatives(None)
kref = {2: -q2r, 4: q4r-3*q2r*q2r, 6: -q6r-15*q4r*(-q2r)+30*(-q2r)**3}
cref = coefficient_d2(kref, 4)
out = {'c2_reference': fmt(cref, 20), 'cases': {}}
for name, L in (('4', D(4)), ('3', D(3)), ('2', D(2))):
    t0 = time.time()
    k, (q2, q4, q6, tail) = kum(L)
    case = {'q2': fmt(q2, 20), 'q4': fmt(q4, 20), 'q6': fmt(q6, 20), 'tail': fmt(tail, 3),
            'k4_over_k2sq': fmt(k[4]/(k[2]*k[2]), 8), 'k6_over_k2cu': fmt(k[6]/k[2]**3, 8), 'dims': {}}
    for d in (2, 3, 4):
        data = [jet_data(u, k) for u in directions(d)]
        dev = D(0)
        for key in ('det_G', 'det_V', 'tau2', 'det_AgV', 'tr_AgV'):
            vals = [x[key] for x in data]; dev = max(dev, max(abs(v/vals[0]-1) for v in vals))
        taus = [x['tau2'] for x in data]
        case['dims'][str(d)] = {'min_pivot': fmt(min(x['min_pivot'] for x in data), 8),
                                'max_rel_aniso': fmt(dev, 6),
                                'tau2_min': fmt(min(taus), 12), 'tau2_max': fmt(max(taus), 12)}
    cs = {N: coefficient_d2(k, N) for N in (256, 512, 1024)}
    c_axis = pref*2*PI*integrand([D(1), D(0)], k)
    s = D(1)/D(2).sqrt()
    c_diag = pref*2*PI*integrand([s, s], k)
    kiso = {2: k[2], 4: D(0), 6: D(0)}        # Gaussian surrogate with the same k2 (mutant M1)
    c_gauss = coefficient_d2(kiso, 16)
    case['c2L'] = {('N%d' % N): fmt(v, 20) for N, v in cs.items()}
    case['c2L_N512_minus_N1024'] = fmt(abs(cs[512]-cs[1024]), 3)
    case['ratio_full_to_ref'] = fmt(cs[1024]/cref, 12)
    case['axis_frozen_substitution_ratio_to_full'] = fmt(c_axis/cs[1024], 12)
    case['diagonal_frozen_substitution_ratio_to_full'] = fmt(c_diag/cs[1024], 12)
    case['gaussian_surrogate_ratio_to_full'] = fmt(c_gauss/cs[1024], 12)
    case['seconds'] = round(time.time()-t0, 1)
    out['cases'][name] = case
    print(name, json.dumps(case), flush=True)
json.dump(out, open('l2_diag.json', 'w'), indent=1, sort_keys=True)
