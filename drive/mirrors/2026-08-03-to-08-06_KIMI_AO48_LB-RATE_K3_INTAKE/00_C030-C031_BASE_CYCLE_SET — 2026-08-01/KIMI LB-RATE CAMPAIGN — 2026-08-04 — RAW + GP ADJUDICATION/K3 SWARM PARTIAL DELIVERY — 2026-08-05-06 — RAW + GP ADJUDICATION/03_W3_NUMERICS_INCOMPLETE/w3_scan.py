# W3 prototype scan: rho(y) landscape over the rigidity zone, hot spot, crude I.
import mpmath as mp
import importlib.util, sys, math

mp.mp.prec = 120

spec = importlib.util.spec_from_file_location("w3p", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_proto.py")
w3p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w3p)

b_mp, ell = w3p.b_mp, w3p.ell

def g_fast(m, V):
    return w3p.g_quad(m, V)

def rho(y, umode="mid"):
    muJ, SigJ = w3p.jetlaw(y)
    muf = muJ[0]; mug = mp.matrix([muJ[1], muJ[2]]); muH = mp.matrix([muJ[3], muJ[4], muJ[5]])
    Sff = SigJ[0, 0]
    Sfg = mp.matrix([[SigJ[0, 1], SigJ[0, 2]]])
    Sgg = mp.matrix([[SigJ[1, 1], SigJ[1, 2]], [SigJ[2, 1], SigJ[2, 2]]])
    SfH = mp.matrix([[SigJ[0, 3], SigJ[0, 4], SigJ[0, 5]]])
    SgH = mp.matrix([[SigJ[1, 3], SigJ[1, 4], SigJ[1, 5]],
                     [SigJ[2, 3], SigJ[2, 4], SigJ[2, 5]]])
    SHH = mp.matrix([[SigJ[3, 3], SigJ[3, 4], SigJ[3, 5]],
                     [SigJ[4, 3], SigJ[4, 4], SigJ[4, 5]],
                     [SigJ[5, 3], SigJ[5, 4], SigJ[5, 5]]])
    detSg = mp.det(Sgg)
    Sggi = Sgg ** -1
    pgrad = mp.exp(-(mug.T * Sggi * mug)[0, 0] / 2) / (2 * mp.pi * mp.sqrt(detSg))
    mu_f2 = muf - (Sfg * Sggi * mug)[0, 0]
    mu_H2 = muH - SgH.T * Sggi * mug
    S_ff2 = Sff - (Sfg * Sggi * Sfg.T)[0, 0]
    S_fH2 = SfH - Sfg * Sggi * SgH
    S_HH2 = SHH - SgH.T * Sggi * SgH
    V = S_HH2 - S_fH2.T * S_fH2 / S_ff2
    beta = S_fH2.T / S_ff2
    def F(u):
        m = mu_H2 + beta * (u - mu_f2)
        return mp.exp(-(u - mu_f2) ** 2 / (2 * S_ff2)) / mp.sqrt(2 * mp.pi * S_ff2) * g_fast(m, V)
    a = b_mp - ell
    umid = b_mp - ell / 2
    Iu_mid = F(umid) * ell
    out = dict(pgrad=float(pgrad), muf2=float(mu_f2), Sff2=float(S_ff2), z=float((umid-mu_f2)/mp.sqrt(S_ff2)),
               muH2=[float(v) for v in mu_H2], Vdiag=[float(V[i,i]) for i in range(3)],
               g=float(g_fast(mu_H2 + beta*(umid-mu_f2), V)))
    return pgrad * Iu_mid, out

if __name__ == "__main__":
    # coarse scan on a grid, report landscape
    pts = []
    grid = [i * 0.1 for i in range(-15, 16)]
    vals = {}
    for gx in grid:
        for gy in grid:
            if gx*gx + gy*gy > 1.5*1.5: continue
            y = (mp.mpf(gx), mp.mpf(gy))
            rv, info = rho(y)
            vals[(gx, gy)] = float(rv)
    top = sorted(vals.items(), key=lambda kv: -kv[1])[:12]
    print("TOP 12 grid points (rho):")
    for k, v in top:
        print("   ", k, mp.nstr(v, 10))
    tot = sum(vals.values()) * 0.01  # dx*dy = 0.01
    print("crude I over |y|<=1.5 (0.1 grid) ~", mp.nstr(tot, 10))
    # diagnostics at hot spot and near pin
    for y in [(mp.mpf(0), mp.mpf('0.6')), (mp.mpf(0), mp.mpf('-0.6')), (mp.mpf('-0.04'), mp.mpf('-0.58')),
              (mp.mpf('0.05'), mp.mpf(0)), (mp.mpf('0.1'), mp.mpf(0)), (mp.mpf('0.3'), mp.mpf('0.3')),
              (mp.mpf(0), mp.mpf('0.3')), (mp.mpf(0), mp.mpf('1.0'))]:
        rv, info = rho(y)
        print("y =", [float(v) for v in y], " rho =", mp.nstr(rv, 8))
        print("     ", {k: (round(v,6) if isinstance(v,float) else [round(t,6) for t in v]) for k,v in info.items()})
