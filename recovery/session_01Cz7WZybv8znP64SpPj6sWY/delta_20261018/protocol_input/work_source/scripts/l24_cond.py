# Conditioning of beta, nu3 at L=24 and small L, original spectral law; mpmath high precision.
import mpmath as mp
mp.mp.dps = 400
def moments(L):
    h = 2*mp.pi/L; al = h**2/2
    M = int(mp.sqrt(2000/al))+5
    S = [mp.mpf(0)]*4
    for m in range(-M, M+1):
        w = mp.e**(-al*m*m)
        for k in range(4):
            S[k] += (h*m)**(2*k)*w
    z = S[0]
    return z, S[1]/z, S[2]/z, S[3]/z, h, al
for L in [mp.mpf(1)/2, 1, 3, 24]:
    z,a,b,c,h,al = moments(L)
    beta = b-a*a; nu3 = c-b*b/a; k4 = b-3*a*a; k6 = c-15*a*b+30*a**3
    amp_beta = (b+2*a*a)/abs(beta); amp_nu3 = (c+3*b*b/a)/abs(nu3)
    w1 = mp.e**(-al); w2 = mp.e**(-4*al)
    wit_nu3 = 72*h**6*w1*w2/(z*(w1+4*w2)); wit_beta = 2*h**4*w1/z**2; wit_a = 2*h*h*w1/z
    print(f"L={mp.nstr(L,4)} a={mp.nstr(a,12)} b={mp.nstr(b,12)} c={mp.nstr(c,12)} beta={mp.nstr(beta,8)} nu3={mp.nstr(nu3,8)} kappa4={mp.nstr(k4,5)} kappa6={mp.nstr(k6,5)}")
    print(f"   log10 amp beta={mp.nstr(mp.log10(amp_beta),4)} log10 amp nu3={mp.nstr(mp.log10(amp_nu3),4)}  witnesses a>={mp.nstr(wit_a,5)} beta>={mp.nstr(wit_beta,5)} nu3>={mp.nstr(wit_nu3,5)}")
