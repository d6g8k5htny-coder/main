import mpmath as mp
mp.mp.dps = 30
phi8 = mp.e**(-32)/mp.sqrt(2*mp.pi)
for K in (24, 32, 64):
    N = (2*K+1)**2
    pclip = min(1, 2*N*phi8/8)                       # GAUSSIAN_COUPLING.md line 45
    t = 8
    eta = 4*mp.e**(-t**2/2)*((K+1)/(1-mp.e**(-t)) + mp.e**(-t)/(1-mp.e**(-t))**2)   # GAUSSIAN_TAIL.md line 90
    beta = pclip + eta
    print(K, "p_clip=", mp.nstr(pclip, 6), "eta_K,8=", mp.nstr(eta, 6), "beta=", mp.nstr(beta, 6), "sqrt(beta)=", mp.nstr(mp.sqrt(beta), 6))
