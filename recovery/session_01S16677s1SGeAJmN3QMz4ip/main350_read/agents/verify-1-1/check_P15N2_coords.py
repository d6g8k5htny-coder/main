"""P15-N2 verifier check: annular exclusion (PROOF.md (14), line 186; use at lines 200-201)
under every coordinate reading of B_1, D_1 and g_*, for the concrete member of
ANALYTIC_GUARDS.md Sec. 1 (d=2, L=1, R=1/8, r1=1/32, A=2^-16, u0=3A^2, v0=2A^3).
Pure python3.12, no third-party packages."""
import math, sys
assert sys.version_info[:2] == (3, 12), sys.version

d = 2; L = 1.0; R = 1/8; r1 = 1/32; A = 2.0**-16; u0 = 3*A*A; v0 = 2*A**3
w1, w2, b = 1/3, 1/9, 4/9
astar = r1/4; delta = r1**4/64

def q(t):   return math.exp(-1/t) if t > 1e-9 else 0.0
def qp(t):  return math.exp(-1/t)/(t*t) if t > 1e-9 else 0.0
def psi(t):
    D = q(t)+q(1-t); return q(1-t)/D
def psip(t):
    D = q(t)+q(1-t); return -(qp(1-t)*q(t)+q(1-t)*qp(t))/(D*D)

# (G4) cutoff and its chart gradient
def chi(s, z):      return psi((s*s+z*z-r1*r1/4)/(3*r1*r1/4))
def grad_chi(s, z):
    k = psip((s*s+z*z-r1*r1/4)/(3*r1*r1/4))*8/(3*r1*r1)
    return (k*s, k*z)

# (G7) surgery profile
wid = R*R/2 - r1*r1
def c(r):   return 1-(1-r/4)*psi((r-r1*r1)/wid)
def cp(r):  return psi((r-r1*r1)/wid)/4-(1-r/4)*psip((r-r1*r1)/wid)/wid

def grad_F0(s, z):
    r = s*s+z*z
    return (-2*s*(c(r)+s*s*cp(r)), -2*z*(1+s*s*cp(r)))

def grad_P1(s, z):   # chi*s^2/2
    gx, gz = grad_chi(s, z); ch = chi(s, z)
    return (s*s/2*gx+ch*s, s*s/2*gz)
def grad_P2(s, z):   # chi*s
    gx, gz = grad_chi(s, z); ch = chi(s, z)
    return (s*gx+ch, s*gz)

# physical (torus y) coordinates: s=sqrt(2w1) sin(pi y1), z=sqrt(2w2) sin(pi y2); L=1
a1, a2 = math.sqrt(2*w1), math.sqrt(2*w2)
def jac(s, z):
    y1 = math.asin(s/a1)/math.pi; y2 = math.asin(z/a2)/math.pi
    return (a1*math.pi*math.cos(math.pi*y1), a2*math.pi*math.cos(math.pi*y2))
def nz(v):     return math.hypot(v[0], v[1])
def ny(v, J):  return math.hypot(J[0]*v[0], J[1]*v[1])

NR, NT = 800, 720
gz = gy = float('inf'); B1z = D1z = B1y = D1y = 0.0; Jmin = float('inf'); Jmax = 0.0
minFz = {}; minFy = {}
corners = [(su*u0, sv*v0) for su in (-1, 1) for sv in (-1, 1)]
for (u, v) in corners: minFz[(u, v)] = minFy[(u, v)] = float('inf')
for i in range(NR+1):
    rho = r1*i/NR                       # whole support disc 0..r1 for B,D sup norms
    for j in range(NT):
        th = 2*math.pi*j/NT
        s, z = rho*math.cos(th), rho*math.sin(th)
        J = jac(s, z); Jmin = min(Jmin, *J); Jmax = max(Jmax, *J)
        g1, g2 = grad_P1(s, z), grad_P2(s, z)
        B1z = max(B1z, nz(g1)); D1z = max(D1z, nz(g2))
        B1y = max(B1y, ny(g1, J)); D1y = max(D1y, ny(g2, J))
        if rho >= r1/2:                 # closed annulus r1/2..r1 for g_*
            g0 = grad_F0(s, z)
            gz = min(gz, nz(g0)); gy = min(gy, ny(g0, J))
            for (u, v) in corners:
                gf = (g0[0]+u*g1[0]+v*g2[0], g0[1]+u*g1[1]+v*g2[1])
                minFz[(u, v)] = min(minFz[(u, v)], nz(gf))
                minFy[(u, v)] = min(minFy[(u, v)], ny(gf, J))

print(f"grid: {NR+1} radii x {NT} angles over |(s,z)|<=r1; annulus r1/2..r1 for g_*")
print(f"Jacobian diag entries dz_i/dy_i over support: min={Jmin:.4f} max={Jmax:.4f}  (>1 => physical norms >= chart norms)")
print(f"chart   : g*_z={gz:.6e}  B1_z={B1z:.4f} (G6 bound 33*r1={33*r1:.4f})  D1_z={D1z:.4f} (G6 bound 65)")
print(f"physical: g*_y={gy:.6e}  B1_y={B1y:.4f}  D1_y={D1y:.4f}")
print(f"(G8) g*_z >= r1^3/16 = {r1**3/16:.6e} : {gz >= r1**3/16}")
readings = {
  "chart B,D / chart g*   (intended)": (B1z*u0+D1z*v0, gz/2),
  "phys  B,D / phys  g*   (consistent alt)": (B1y*u0+D1y*v0, gy/2),
  "phys  B,D / chart g*   (mixed 1)": (B1y*u0+D1y*v0, gz/2),
  "chart B,D / phys  g*   (mixed 2)": (B1z*u0+D1z*v0, gy/2),
}
for k, (lhs, rhs) in readings.items():
    print(f"{k:42s}: B1*u0+D1*v0={lhs:.6e} < g*/2={rhs:.6e} ? {lhs < rhs}  (ratio {lhs/rhs:.3e})")
print("direct: min |grad F_(u,v)| on annulus at rectangle corners (must be > 0 in either coordinate system)")
for (u, v) in corners:
    print(f"  u={u:+.3e} v={v:+.3e}: chart {minFz[(u,v)]:.6e}  physical {minFy[(u,v)]:.6e}  "
          f">= g*_z - (B1_z u0 + D1_z v0) = {gz-(B1z*u0+D1z*v0):.6e}: {minFz[(u,v)] >= gz-(B1z*u0+D1z*v0)-1e-15}")
