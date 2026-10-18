import mpmath as mp
mp.mp.dps = 80
for L in (1, 2, 3):
    h = 2*mp.pi/L; al = h*h/2; ks = range(-40, 41)
    z = mp.fsum(mp.e**(-al*k*k) for k in ks)
    mom = lambda n: mp.fsum((h*k)**n*mp.e**(-al*k*k) for k in ks)/z
    a, b, c = mom(2), mom(4), mom(6)
    w1, w2 = mp.e**(-al), mp.e**(-4*al)
    m13 = 72*h**6*w1*w2/(z*(w1 + 4*w2))
    print(f'L={L}: c={mp.nstr(c,8)} c-b^2/a={mp.nstr(c-b*b/a,10)} M13 floor={mp.nstr(m13,10)} ratio={mp.nstr((c-b*b/a)/m13,8)}')
