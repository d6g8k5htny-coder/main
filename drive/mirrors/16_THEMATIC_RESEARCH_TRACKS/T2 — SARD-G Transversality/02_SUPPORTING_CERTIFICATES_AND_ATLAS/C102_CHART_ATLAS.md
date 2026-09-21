# C102 Qualitative-Rate Chart Atlas

**Grade:** `DERIVED-EXACT-EXISTENCE`  
**Scope:** exact periodized BF field, \(L=24\), \(0<r\le0.025\), \(b=6/5\), typed pair-Palm.

Every Gaussian/Kac–Rice compactness step is assigned a named chart. The atlas proves qualitative finite constants; it does not provide outward-rounded numerical eigenvalue floors.

| chart | consumer | region | boundary transfer | status |
|---|---|---|---|---|
| `PAIR-CORRECTED` | all pair-Palm regressions | 0<r<=0.025 | r=0 represented by limiting six-jet block | `CLOSED-QUALITATIVELY` |
| `MAX-NEAR-GENERIC` | GAMMA maximum-window singular near count | 0<t<=delta; eta=r/t; eta^2<=4/15; c^2+s^2=1; s!=0; alpha in [0,1] | s=0 -> MAX-NEAR-AXIS; c=0 -> MAX-NEAR-TRANSVERSE; eta=0 retained as mesoscopic face; alpha endpoints included | `CLOSED-WITH-AXIS-ATLAS` |
| `MAX-NEAR-AXIS` | GAMMA maximum-window generic-axis overlap | |s|<=1/2; eta^2<=4/15; 0<t<=delta | overlaps MAX-NEAR-GENERIC; generic zero-gradient density has exp[-(41/60)^2/(4s^2)] suppression | `CLOSED-QUALITATIVELY` |
| `MAX-NEAR-TRANSVERSE` | GAMMA maximum-type exclusion at c=0 | c=0; s=+/-1; eta^2<=4/15; alpha in [0,1] | overlaps generic c->0 chart; endpoint degeneracy kept explicit | `CLOSED-EXACT-TYPE-NOGO` |
| `COLLAR-GENERIC` | additional-critical-point collar count | xi=(X,Y) in scaled two-collar domain; Y!=0; away from pin collisions | Y=0 away from pins -> COLLAR-AXIS; X=0 -> COLLAR-TRANSVERSE; pin faces -> COLLAR-PIN-M/S | `CLOSED-WITH-BOUNDARY-ATLAS` |
| `COLLAR-PIN-M` | collision integrability near configured maximum | xi=M+rho(c,s); 0<rho<=rho0 | rho=rho0 -> COLLAR-GENERIC; s=0 handled by angular Gaussian suppression and COLLAR-AXIS | `CLOSED-QUALITATIVELY` |
| `COLLAR-PIN-S` | collision integrability near configured saddle | xi=S+rho(c,s); 0<rho<=rho0 | rho=rho0 -> COLLAR-GENERIC; s=0 -> COLLAR-AXIS with suppression | `CLOSED-QUALITATIVELY` |
| `COLLAR-TRANSVERSE` | collar chart at X=0 | X=0; 0<|Y|<=Y0 | overlaps COLLAR-GENERIC for X->0; Y=0 is axis face | `CLOSED-QUALITATIVELY` |
| `COLLAR-AXIS` | pair-axis collar boundary | Y=0 excluding M,S | pin endpoints covered by COLLAR-PIN-M/S | `CLOSED-QUALITATIVELY` |
| `INTERCEPTOR-NEAR-GENERIC` | global window-critical singular near count | outside two collars; t<=delta; eta=r/t; eta^2<=4/15; alpha in [0,1] | axis/transverse via C099 charts; lower radius joins C100 collar; eta=0 joins fixed annulus | `CLOSED-QUALITATIVELY` |
| `FIXED-ANNULUS` | Gamma and global interceptor fixed-near regions | delta<=distance from pair midpoint<=3 | distance=delta -> singular atlas; distance=3 -> EXTERIOR | `CLOSED-QUALITATIVELY` |
| `EXTERIOR` | Gamma and global interceptor exterior counts | T_24^2 minus B3 | distance=3 -> FIXED-ANNULUS; torus seams identified periodically | `CLOSED-QUALITATIVELY` |
| `TYPE-BOUNDARY` | all typed pair/third-point Kac-Rice moments | every compact positive-definite scaled Gaussian chart | rank-loss faces must transfer to separate scaled chart | `CLOSED` |

## Full-\(r\) transfer rule

For each chart, the proof is split into two ranges.

1. **Small \(r\).** After the displayed divided differences, the mean, covariance, and determinant-weighted moment are analytic in the chart parameters and extend to the boundary \(r=0\). A positive limiting Gram matrix gives an existential neighborhood on which its least eigenvalue remains positive.
2. **Away from zero.** At every finite distinct configuration, the exact Fourier nondegeneracy theorem gives positive definiteness. Continuity on the compact remainder of the chart gives a positive minimum.

Their union covers the entire interval \(0<r\le0.025\). No sampled-rung argument is used.

## Boundary discipline

A formula that loses rank at a boundary is never extended through that boundary by assertion. The atlas redirects it to a separate axis, transverse, or collision chart. Type-indicator continuity is used only after the scaled covariance is positive definite.

## Coverage limitation found during atlas construction

The atlas covers the **Gaussian critical-point counting layer**. It does not itself prove that every deterministic loop/crater or high-level reconnection mismatch is represented by one of those critical-point counts. That obligation is adjudicated separately in the C102 deterministic-reduction audit.
