# C096 BR-MARK Limit Adjudication

**Jet reduction:** `DERIVED-EXACT`  
**Scaling evidence:** `MEASURED-DIAGNOSTIC`  
**BR-MARK:** `OPEN-RESHAPED`

## Exact limit

For candidate critical points at midpoint $m$, separation direction $e$, transverse direction $n$, and separation $d\downarrow0$, the two gradient constraints converge to
$D_e f=D_e^2 f=D_n f=D_eD_n f=0$. The marks converge to
$A_d\to f(m)$ and $Z_d\to-D_e^3f(m)/12$.

## Two-rung scaling

| scaled midpoint | angle | Var(A) exponent | Var(Z) exponent | density blow-up exponent |
|---|---:|---:|---:|---:|
| ['0', '2.25'] | 0.0000 | 6.001 | 1.998 | 4.000 |
| ['0', '2.25'] | 1.5708 | 6.111 | 0.932 | 4.001 |
| ['2', '2'] | 0.0000 | 6.003 | 1.988 | 3.994 |
| ['2', '2'] | 1.5708 | 7.998 | 1.996 | 4.999 |
| ['3', '1'] | 0.0000 | 6.279 | 1.855 | 3.931 |
| ['3', '1'] | 1.5708 | 7.093 | 0.940 | 4.473 |

The tested pair-scaled charts exhibit Gaussian density growth roughly $r^{-4}$, with a special chart near $r^{-5}$. The largest tested density was `1.372007214e9`. Fixed-physical controls were stable across rungs.

## Adjudication

The uniform planar-style hypothesis $\sup_{r,m,e}\|p_{A,Z}^{(6\mathrm{pin})}\|_\infty<\infty$ is false on the tested near-pair charts. This does not kill Bonferroni. It kills the factorization `spatial repulsion constant × globally bounded mark-density constant` as a uniform architecture.

The replacement is a regional and joint bound: exterior bounded-density chart; pair-scaled joint typed spatial-marked Kac--Rice chart; dedicated pin collars. Determinant Palm depletion and spatial repulsion must be allowed to cancel the mark-density singularity before extracting a scalar constant.