"""METHOD V2 witness floors M12/M13 and M6/M9/M11 floors evaluated at L=24, versus true values."""
import mpmath as mp
mp.mp.dps = 50
L = mp.mpf(24); h = 2*mp.pi/L; al = h*h/2
w1, w2 = mp.e**(-al), mp.e**(-4*al)
z = 1 + 2*mp.fsum(mp.e**(-al*m*m) for m in range(1, 400))
a_lo = 2*h**2*w1/z
beta_lo = 2*h**4*w1/z**2
nu3_lo = 72*h**6*w1*w2/(z*(w1+4*w2))
a, beta, nu3 = 1, 2, 6
print("a floor M12  %.4g (true ~1, ratio %.3g)" % (a_lo, a/a_lo))
print("beta floor M12 %.4g (true ~2, ratio %.3g)" % (beta_lo, beta/beta_lo))
print("nu3 floor M13 %.4g (true ~6, ratio %.3g)" % (nu3_lo, nu3/nu3_lo))
lamH_w = min(beta_lo, 2*a_lo**2)
print("lambda_H from witnesses %.4g vs true min(beta,2a^2)=2 ; b0=gamma0=%.4g vs true 1" % (lamH_w, lamH_w/2))
print("v0=nu3/d^2 from witness %.4g vs true V(e)=6 for all e" % (nu3_lo/4))
