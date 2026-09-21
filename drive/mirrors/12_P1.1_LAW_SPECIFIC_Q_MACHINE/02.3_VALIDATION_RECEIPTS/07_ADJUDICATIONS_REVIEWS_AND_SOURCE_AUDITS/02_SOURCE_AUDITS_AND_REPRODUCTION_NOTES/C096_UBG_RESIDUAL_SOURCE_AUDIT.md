# C096 UB-G Residual Source Audit

**Grade:** `DERIVED-EXACT-IMPLICATION-AUDIT`  
**Source status affected:** fixed-volume upper-rate display and the q₀ corollary  
**Frozen source files:** unchanged

## 1. What the source actually proves as written

The upper-side source explicitly assembles

\[
D(r)
\le
C_{\rm base}r^3
+
e^{-92}
+
\operatorname{Collar}(r),
\]

where

\[
C_{\rm base}
=
\frac{0.657+
2.8185310984873743106223995278378565957688812716057680797457}
{0.80}
=
4.344413873109217888277999409797320744711101589507210099682125.
\]

The source describes the Gamma contribution as \(e^{-92}\), not as
\(C_\Gamma r^3\).

Therefore the printed inequality implies only

\[
\limsup_{r\downarrow0}D(r)
\le
e^{-92}
+
\limsup_{r\downarrow0}\operatorname{Collar}(r).
\]

A positive constant does not vanish when \(r\to0\). The step

\[
C_{\rm base}r^3+e^{-92}+\operatorname{Collar}(r)
\Longrightarrow
4.3r^3
\]

or \(4.35r^3\) on the entire punctured interval \(0<r\le r_0\) is not valid
without an additional \(r\)-dependent Gamma theorem.

## 2. Exact crossover

\[
e^{-92}
=
1.108939019312136379459597534352117145641980176873064965893702126853638866722979505118611312e-40.
\]

Relative to the exact unrounded margin

\[
4.35-C_{\rm base}
=
0.00558612689078211172200059020267925528889841049278990031787500000000000000000000000000000013,
\]

the absolute Gamma ceiling fits only when

\[
r
\ge
0.0000000000002707690093883541437041039891222756067232764080067681042304164283779642167156826243612075574.
\]

Thus even ignoring the collar, the absolute station ceiling cannot be absorbed
uniformly on a neighborhood punctured at zero.

The source's \(e^{-92}\) remains useful as a station/rung-scale diagnostic.
It is not a proof of \(O(r^3)\).

## 3. Collar reduction

The source's analytic backstop is

\[
\lambda_2(y,z)
\le
C_{\rm nd}|y-z|.
\]

For one disk of radius \(2r\),

\[
\int_0^{2r}
C_{\rm nd}t\,(2\pi t)\,dt
=
\frac{16\pi}3 C_{\rm nd}r^3.
\]

For two collars, the conservative coefficient is

\[
\frac{32\pi}3 C_{\rm nd}.
\]

Numerically,

\[
\frac{16\pi}3
=
16.755160819145563938467431377490682049051570130001,
\qquad
\frac{32\pi}3
=
33.510321638291127876934862754981364098103140260001.
\]

The source establishes the **shape** \(O(r^3)\), but the retrieved record does
not provide an explicit \(C_{\rm nd}\) that fits the narrow \(4.35\) budget.

Even with \(C_\Gamma=0\), the two-collar version would require

\[
C_{\rm nd}
\le
0.00016669869513872497927699661528163731222277464685051.
\]

No such certificate has been located.

## 4. Minimal Gamma repair

Let \(G_r\) be the gain of the maximum reached by the second branch of the
canonical saddle under the typed pair-Palm law.

The event is

\[
\Gamma_r=\{0<G_r<\ell\},
\qquad
\ell=\frac{r^3}6.
\]

A uniform density theorem

\[
\sup_{0<r\le r_0}
\sup_{u\in[0,\ell_0]}
p_{G_r}(u)
\le
M_\Gamma
<\infty
\]

implies

\[
P(\Gamma_r)
\le
M_\Gamma\ell
=
\frac{M_\Gamma}6r^3.
\]

Together with the collar backstop,

\[
D(r)
\le
\left[
C_{\rm base}
+
\frac{M_\Gamma}6
+
N_{\rm collar}\frac{16\pi}3C_{\rm nd}
\right]r^3.
\]

This is the exact missing bridge from a tiny fixed-r station probability to a
uniform cubic rate.

## 5. Final disposition

```text
structural UB-G decomposition:
    DERIVED

absolute Gamma station ceiling e^-92:
    STATION/RUNG-SCALE CERTIFICATE

uniform Gamma O(r^3):
    OPEN — GAMMA-DENSITY

collar O(r^3):
    DERIVED STRUCTURE, explicit coefficient unresolved

uniform decimal 4.35:
    BLOCKED

q(r)->1 from UB-G:
    PROVEN-MODULO GAMMA-DENSITY and explicit collar backstop
```

The q₀ conclusion may still be true. The printed \(e^{-92}\) ceiling is not
the theorem that proves it.
