# (17) barrier: F_lambda <= b - 3 delta/4 outside {|z| < r1/2}; new critical values > b - delta/4. Also Section 2: {h > b-R^2} = D_R.
import numpy as np, instance as I
lam = I.lam0
ph = np.linspace(0, 2*np.pi, 2001); rr = np.concatenate([np.linspace(I.r1/2, I.r1, 1501), np.linspace(I.r1, I.R, 1501)])
RR, PH = np.meshgrid(rr, ph, indexing='ij'); S, T = RR*np.cos(PH), RR*np.sin(PH)
Fv = I.Flam(S, T, lam)
print(f"max F_lambda on chart annulus r1/2<=|z|<=R: {Fv.max():.17g}; bound b-3delta/4 = {I.b-0.75*I.delta:.17g}; slack = {I.b-0.75*I.delta - Fv.max():.3e}")
print(f"min loss (b-F0) on |z|>=r1/2 vs delta: {(I.b - I.F0(S,T)).min():.6e} >= {I.delta:.6e} : {(I.b - I.F0(S,T)).min() >= I.delta}")
for zz, nm in ((I.zm, 'M-'), (I.z0, 'S'), (I.zp, 'M+')):
    val = I.b + lam**2*I.Phi(zz)
    print(f"  {nm}: F={val:.17g} > b-delta/4={I.b-I.delta/4:.17g}: {val > I.b - I.delta/4} ; |lam^2 Phi| = {abs(lam**2*I.Phi(zz)):.3e} < delta/4={I.delta/4:.3e}")
# Section 2: superlevel {h > b - R^2} equals the chart ball D_R (torus grid, d=2)
y = np.linspace(-I.L/2, I.L/2, 4001)[:-1]
Y1, Y2 = np.meshgrid(y, y, indexing='ij')
h = I.w[0]*np.cos(I.om*Y1) + I.w[1]*np.cos(I.om*Y2)
inside = h > I.b - I.R**2
z1 = np.sqrt(2*I.w[0])*np.sin(I.om*Y1/2); z2 = np.sqrt(2*I.w[1])*np.sin(I.om*Y2/2)
chart = (np.abs(Y1) < I.L/4) & (np.abs(Y2) < I.L/4)
ball = chart & (z1**2 + z2**2 < I.R**2)
print("{h > b-R^2} == D_R on grid:", bool(np.array_equal(inside, ball)), " #points:", int(inside.sum()))
print("max |y_i| on {h>b-R^2}:", float(np.abs(Y1[inside]).max()), float(np.abs(Y2[inside]).max()), " < L/4 =", I.L/4)
print("max h outside D_R:", float(h[~ball].max()), " <= b-R^2 =", I.b - I.R**2)
# old critical values: all <= b - Delta
import itertools
vals = sorted([sum(sg*wi for sg, wi in zip(sgn, I.w)) for sgn in itertools.product([1,-1], repeat=2)], reverse=True)
print("old critical values:", vals, " second <= b-Delta:", vals[1] <= I.b - I.Delta, " b-R^2=", I.b-I.R**2, "> b-Delta:", I.b-I.R**2 > I.b-I.Delta)
