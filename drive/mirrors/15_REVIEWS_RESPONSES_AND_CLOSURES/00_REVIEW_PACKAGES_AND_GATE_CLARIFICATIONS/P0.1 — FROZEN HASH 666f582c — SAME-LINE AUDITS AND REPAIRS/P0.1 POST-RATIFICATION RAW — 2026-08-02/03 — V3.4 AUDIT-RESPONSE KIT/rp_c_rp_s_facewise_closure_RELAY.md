RELAY HEADER (AO48): Operator-relayed from the Drive-unreachable sandbox line
(SIDE24 track), received via chat paste 2026-08-01 and transcribed verbatim
below. Byte identity with the sandbox original is NOT certified — the
authoritative copy is drafts/rp_c_rp_s_facewise_closure.md in their pinned
tree. Relay purpose: make the V3.4 successor draft visible to the corpus for
the independent audit its own promotion gate requires. Relayed by AO48
without endorsement; see AO48-REC-035 for the reviewing context.

---

SIDE24 facewise closure of RP-C and RP-S

Date: 2026-08-01
Status: proof completed in this successor draft; independent audit pending
Scope: the normalized periodized Bargmann–Fock field on the side-24
three-torus, uniformly for the pair direction and for ((b,\kappa)) in a
compact subset of (\mathbb R\times(0,\infty)).

This note completes the seven-part work order left open by V3.3.  It is a
new layer: it does not alter the frozen V3.3 checkpoint.  The symbols
(M=x-rt/2), (S=x+rt/2), and

[
f(M)=b,\qquad f(S)=h=b-\frac{\kappa r^3}{6},\qquad
\nabla f(M)=\nabla f(S)=0                                      \tag{1.1}
]

are fixed throughout.  Conditional expectation before determinant weighting
is denoted by (\mathbb E^0),

[
W_r=|\det H_M\det H_S|
\mathbf1_{{H_M\prec0,\ \operatorname{ind}(H_S)=2}},
\qquad Z_r=\mathbb E^0W_r.                                    \tag{1.2}
]

The V3.3 Palm repair proved, uniformly on the stated compact sets,

[
c r^2\le Z_r\le Cr^2.                         \tag{1.3}
]

The canonical all-witness counts used here are exactly those in (A.35):

[
N_j=#{y\ne M,S:\nabla f(y)=0,\ h<f(y)<b,\ y\in\mathcal R_j}.
\tag{1.4}
]

Thus both the collar and the singular-near Kac–Rice integrals contain the
single, legitimate value-window integration

[
\int_h^bdu,\qquad b-h=\frac{\kappa r^3}{6}. \tag{1.5}
]

2. The common analytic mechanism

The side-24 spectral weight is strictly positive at every torus character.
A finite linear combination of derivative evaluations has zero variance
only if the Fourier coefficients of the corresponding finite distribution
vanish at every character.  Fourier uniqueness for distributions on the
torus then makes that distribution zero.  Distinct derivative distributions
at the same point, and distributions supported at distinct points, are
linearly independent.

Every normalized family below is made from exact divided differences, not
from a formal Taylor truncation.  At a boundary face it converges in (L^2)
to the displayed derivative-distribution family.  Once that family is shown
independent, the preceding Fourier argument gives a positive Gram matrix.
Continuity supplies uniform Schur constants only on the compact chart on
which the correct normalization has already been installed.  No compactness
argument is taken across the unnormalized angular axis.

The conditional means and covariances of any fixed additional normalized
jet are obtained by Gaussian regression.  On a chart with a covariance
eigenfloor, all of its polynomial moments are uniformly bounded.  On an ESIP
chart the inverse covariance can introduce only a fixed projective power;
the nonzero target-mean mismatch supplies an exponential which absorbs every
such fixed power.  This observation will be used after the deterministic
Hessian lemma in Section 6.

3. Generic transverse collar face

Choose an orthonormal frame ((t,e_2,e_3)) and write

[
y=x+r(\xi_1t+\xi_2e_2+\xi_3e_3),\qquad
\rho_\xi=(\xi_2^2+\xi_3^2)^{1/2},\quad
X=r\xi_1,\quad\varrho=r\rho_\xi.                          \tag{3.1}
]

Use the corrected pin basis

[
V_r=\left(f(M),\nabla f(M),
\frac{\nabla f(S)-\nabla f(M)}r,
\frac{f(S)-f(M)-\frac r2(\partial_tf(S)+\partial_tf(M))}{r^3}
\right).                                                   \tag{3.2}
]

Put (d_r=(\nabla f(S)-\nabla f(M))/r), let
(u=(\xi_1+1/2,\xi_2,\xi_3)) in the chosen frame, and denote the last
scalar component of (V_r) by (v_r).  Exact subtraction of pair-pin
combinations gives

[
\begin{aligned}
R_{2,r}&=r^{-1}{D_{e_2}f(y)-D_{e_2}f(M)
-r(\xi_1+\tfrac12)(d_r){e_2}},\
R{3,r}&=r^{-1}{D_{e_3}f(y)-D_{e_3}f(M)
-r(\xi_1+\tfrac12)(d_r){e_3}},\
R{1,r}&=r^{-2}{D_tf(y)-D_tf(M)-r u\mathbin\cdot d_r
+r^2(6\xi_1^2-\tfrac32)v_r}.
\tag{3.3}
\end{aligned}
]

Consequently, if (C_{24}=\operatorname{Cov}(\nabla f(y)\mid V_r)),

[
\det C_{24}=r^8\det\operatorname{Cov}(R_{1,r},R_{2,r},R_{3,r}\mid V_r).
\tag{3.4}
]

At (r=0), modulo the contact-pin symbols, the three residual symbols are

[
TV(\xi_1T+V/2),\qquad VE_2,\qquad VE_3,
\quad V=\xi_2E_2+\xi_3E_3.                                  \tag{3.5}
]

They are independent whenever (V\ne0): they are finite polynomials, and
a finite polynomial vanishing on the full frequency lattice is zero (apply
the one-variable fact successively in the three coordinates).  Hence, on
each compact normalized subchart (|\xi|\le C), (\rho_\xi\ge\epsilon>0),

[
c_{\epsilon,C}\varrho^6(4X^2+\varrho^2)
\le \det C_{24}\le
C_{\epsilon,C}\varrho^6(4X^2+\varrho^2).                 \tag{3.6}
]

This is an exact side-24 comparison.  It is deliberately not extended to
(\varrho=0), where its reference monomial vanishes but the finite-(r)
conditional covariance need not vanish.

For the witness count one also needs its value.  With
(u=(\xi_1+\tfrac12)t+\xi_2e_2+\xi_3e_3), take the exact trapezoidal
Hermite residual

[
R_{0,r}=r^{-3}\left[f(y)-f(M)
-\frac r2u\mathbin\cdot{\nabla f(y)+\nabla f(M)}\right]. \tag{3.6a}
]

Conditional on the pair pins, the raw-to-corrected map
((f(y),\nabla f(y))\mapsto(R_{0,r},R_{1,r},R_{2,r},R_{3,r})) is triangular
up to shears and has diagonal scales

[
(r^3,r^2,r,r),                    \tag{3.7}
]

and hence corrected-over-raw Jacobian (r^{-7}).  At contact its value
symbol is (-D_u^3/12).  Put (V=\xi_2E_2+\xi_3E_3).  The (V^3) term
first isolates this row from the pair-pin space and (3.5); the (TV^2)
term then isolates the tangential gradient row, and the transverse
quadratic terms isolate the last two.  Fourier support and compactness give
a uniform joint Gram floor on every strict transverse chart.  Its joint
density is therefore (O(r^{-7})); integrating the single value window
(1.5) leaves the effective (O(r^{-4})) density used in Section 8.

4. Axial, midpoint, and endpoint collar charts

4.1 Exact axial Hermite factorization

Center the conditional field, so its values and gradients vanish at the pair
endpoints, and put an axial witness at (y=M+qt).  With

[
a=q(q-r),\qquad \chi=2q-r,
]

the exact Hermite divided differences (A,B,C^v,C^w) satisfy

[
\widetilde f(y)=a^2A,\qquad
\partial_t\widetilde f(y)=a(2\chi A+aB),\qquad
\partial_e\widetilde f(y)=aC^e.                          \tag{4.1}
]

Define (d_0=(4\chi^2+a^2)^{1/2}) and
(U=(2\chi A+aB)/d_0).  Then, exactly,

[
\det\operatorname{Cov}(\nabla f(y)\mid V_r)
=a^6d_0^2\det\operatorname{Cov}(U,C^v,C^w\mid V_r).      \tag{4.2}
]

The limiting symbols are a nonzero combination of (T^4,T^5), together
with (T^2E_2,T^2E_3).  Their distinct degrees and transverse monomials make
them independent modulo the pair pins.  Therefore the last determinant in
(4.2) is bounded above and below on the axial compactification.  In
particular the scales are

[
q=o(r):q^6r^8,\qquad q/r\to\theta\notin{0,1,1/2}:r^{14},
\qquad q=r/2:r^{16}.                                     \tag{4.3}
]

The last scale uses the fifth-order residual and is not inferred from the
cancelled fourth-order term.

The conditional axial mean has the exact factor

[
\mathbb E[\partial_tf(M+qt)\mid V_r]=a,m_{r,q},\qquad
m_{r,q}\longrightarrow\kappa.                            \tag{4.4}
]

It follows from (4.2) that

[
p_{\nabla f(y)\mid V_r}(0)
\le C|a|^{-3}d_0^{-1}
\exp!\left(-\frac{c\kappa^2}{d_0^2}\right).       \tag{4.5}
]

Thus the generic axial interior has (e^{-c/r^2}) suppression and the
midpoint has the stronger (e^{-c/r^4}) suppression.

4.1A Axial-interior projective crossover

The exact-axis calculation must be joined to the transverse face before
compactness is used.  Write

[
y=x+r(\zeta t+\rho v),\qquad
A_\zeta=\zeta^2-\frac14,\qquad
\epsilon_a=(r^2+\rho^2)^{1/2},\quad
R=\frac r{\epsilon_a},\quad P=\frac\rho{\epsilon_a},
]

and take (\zeta) in a compact set separated from
(0,\pm\tfrac12).  Let (w\perp{t,v}), and, for the centered
conditional field (\widetilde f=f-\mathbb E[f\mid V_r]), put

[
U_t=\frac{D_t\widetilde f(y)}{r^2\epsilon_a},\qquad
U_v=\frac{D_v\widetilde f(y)}{r\epsilon_a},\qquad
U_w=\frac{D_w\widetilde f(y)}{r\epsilon_a}.               \tag{AI.1}
]

Modulo the contact-pin span
(\mathcal P=\operatorname{span}{1,T,E,W,T^2,TE,TW,T^3}), exact
Taylor–Hermite division gives the uniform boundary rows

[
\begin{aligned}
U_t&\longrightarrow P\zeta T^2E
+R\frac{\zeta A_\zeta}{6}T^4,\
U_v&\longrightarrow PE^2+R\frac{A_\zeta}{2}T^2E,\
U_w&\longrightarrow PEW+R\frac{A_\zeta}{2}T^2W.
\tag{AI.2}
\end{aligned}
]

The constants in (AI.2) are the exact fourth-order axial derivative
remainder (a\chi/12) and the exact two-point transverse remainder
(A_\zeta/2).  The last row is isolated by its (W)-monomials.  If
(P>0), the (E^2) monomial then isolates the second row, and the first
row is nonzero (its (T^4) coefficient is nonzero when (R>0), and its
(T^2E) coefficient is nonzero when (R=0)).  If (P=0), the three
distinct symbols are (T^4,T^2E,T^2W).  Fourier uniqueness and compactness
after this normalization therefore give

[
\det\operatorname{Cov}(\nabla f(y)\mid V_r)
=r^8\epsilon_a^6 A_{\rm AI},\qquad
0<c\le A_{\rm AI}\le C.                                  \tag{AI.3}
]

Analytic regression of the pair target gives

[
D_t\mathbb E[f(y)\mid V_r]
=\kappa r^2A_\zeta+O(r^2\rho+r^3).
]

After shrinking the chart, the normalized target mismatch is separated
from zero.  A covariance ceiling in (AI.3) yields

[
p_{\nabla f(y)\mid V_r}(0)
\le Cr^{-4}\epsilon_a^{-3}
\exp!\left(-\frac{c\kappa^2}{\epsilon_a^2}\right). \tag{AI.4}
]

The face (P=0) is (4.1)–(4.5).  The face (R=0) is the secondary
(\rho\downarrow0) blow-up of (3.5), with rows
((\zeta T^2E,E^2,EW)).  The excluded (\zeta)-neighborhoods overlap the
midpoint and endpoint charts below.  Thus no sequence with
(r,\rho\to0) and axial coordinate separated from
(0,\pm\tfrac12) lies between the transverse and axial charts.

4.2 Off-axis midpoint

In the midpoint chart write
(y=x+r(\zeta t+\rho v)), with
(|\zeta|\le\zeta_m<1/4) for a sufficiently small fixed (\zeta_m), and
with (\rho) the normalized transverse coordinate.  For the centered conditional field put
(R_t=D_t\widetilde f(y)).  Its exact Hermite expansion has contact limit
(r^{-2}R_t\to L_1), where

[
\operatorname{Var}(L_1\mid V_0)
=\frac{\rho^2(4\zeta^2+\rho^2)}2.
]

The finite-(r) remainder is (O_{L^2}(r)), while analytic regression
from the corrected pair target gives

[
\mu_t=\kappa r^2(\zeta^2-1/4)+O(r^2\rho+r^3),\qquad
\operatorname{Var}(R_t\mid V_r)
\le Cr^4{\rho^2(4\zeta^2+\rho^2)+r^2}
\le Cr^4(\rho^2+r^2).                                    \tag{4.6}
]

After making the chart smaller if needed,

[
\mu^TC^{-1}\mu\ge\frac{\mu_t^2}{C_{tt}}
\ge\frac{c\kappa^2}{\rho^2+r^2}.                         \tag{4.7}
]

For the full three-row normalization at the simultaneous midpoint corner,
put

[
\begin{aligned}
\epsilon_m&=(r^2+\rho^2)^{1/2},\
\delta_m^2&=(\rho\zeta)^2+\rho^4+(r\rho)^2
+(r\zeta)^2+r^4,\
(a_1,\ldots,a_5)&=\delta_m^{-1}
(\rho\zeta,\rho^2,r\rho,r\zeta,r^2),\
R_m&=r/\epsilon_m,\qquad P_m=\rho/\epsilon_m.
\tag{MP.1}
\end{aligned}
]

Thus (\sum a_j^2=1) and
(\delta_m\asymp\epsilon_m(\zeta^2+\rho^2+r^2)^{1/2}).  In the quotient by
the contact pins, expansion of the exact corrected residuals gives

[
\begin{aligned}
r^{-2}D_t\widetilde f(y)
={}&\rho\zeta T^2E+\frac{\rho^2}{2}TE^2
-\frac{r\rho}{24}T^3E-\frac{r\zeta}{24}T^4
+\frac{r^2}{1920}T^5+\mathcal R_t,\
r^{-1}D_v\widetilde f(y)
={}&\rho E^2-\frac r8T^2E+\mathcal R_v,\
r^{-1}D_w\widetilde f(y)
={}&\rho EW-\frac r8T^2W+\mathcal R_w,                  \tag{MP.2}
\end{aligned}
]

where, after reducing (\zeta_m) and the chart radius if necessary,

[
|\mathcal R_t|_2\le
C(|\zeta|+\rho+r)\delta_m,\qquad
|\mathcal R_v|_2+|\mathcal R_w|_2
\le C(|\zeta|+\rho+r)\epsilon_m.                         \tag{MP.3}
]

The limiting normalized rows are therefore

[
\begin{aligned}
M_t={D_t\widetilde f\over r^2\delta_m}
&\longrightarrow a_1T^2E+\frac{a_2}{2}TE^2
-\frac{a_3}{24}T^3E-\frac{a_4}{24}T^4
+\frac{a_5}{1920}T^5,\
M_v={D_v\widetilde f\over r\epsilon_m}
&\longrightarrow P_mE^2-\frac{R_m}{8}T^2E,\
M_w={D_w\widetilde f\over r\epsilon_m}
&\longrightarrow P_mEW-\frac{R_m}{8}T^2W.                \tag{MP.4}
\end{aligned}
]

The (W)-monomials first isolate (M_w).  If (P_m>0), the (E^2)
monomial isolates (M_v), and (M_t) is nonzero because its five displayed
monomials are distinct and (\sum a_j^2=1).  If (P_m=0), the projective
relations force (a_1=a_2=a_3=0) and
((a_4,a_5)\ne(0,0)); hence (M_t) is a nonzero (T^4,T^5) combination,
whereas (M_v,M_w) have symbols (T^2E,T^2W).  The small remainders in
(MP.3) preserve this Gram floor.  Consequently

[
\det\operatorname{Cov}(\nabla f(y)\mid V_r)
=r^8\delta_m^2\epsilon_m^4 A_{\rm MP},\qquad
0<c\le A_{\rm MP}\le C,                                  \tag{MP.5}
]

and the same separated mean in (4.6) gives

[
p_{\nabla f(y)\mid V_r}(0)
\le Cr^{-4}\delta_m^{-1}\epsilon_m^{-2}
\exp!\left(-\frac{c\kappa^2}{\delta_m^2}\right).  \tag{MP.6}
]

At the exact midpoint, (\delta_m=r^2) and (\epsilon_m=r), so (MP.5)
has scale (r^{16}) and (MP.6) has the sharp
(e^{-c/r^4}) penalty.  For (|\zeta|) bounded below relative to the
midpoint cutoff this chart overlaps (AI.1)–(AI.4); when (r\ll\rho), it
overlaps the contact rows.  This full frame, rather than the scalar ceiling
alone, controls every remaining normalized jet.

4.3 Endpoint collision charts

Write

[
y=M+du,\qquad u=c t+\rho v,\qquad c^2+\rho^2=1,qquad
\epsilon=(r^2+\rho^2)^{1/2}.                             \tag{4.8}
]

Let (\widetilde f=f-\mathbb E[f\mid V_r]), and define the exact residuals

[
E_1=\frac{D_t\widetilde f(y)}{dr\epsilon},\qquad
E_2=\frac{D_v\widetilde f(y)}{d\epsilon},\qquad
E_3=\frac{D_w\widetilde f(y)}{d\epsilon}.                 \tag{4.9}
]

At (d=0), write (R=r/\epsilon), (P=\rho/\epsilon).  Modulo the
corrected pair-pin span, their limiting Fourier symbols are

[
cR\frac{T^4}{12}-P\frac{T^2V}{2},\qquad
-cR\frac{T^2V}{2}+PV^2,\qquad
-cR\frac{T^2W}{2}+PVW.                                  \tag{4.10}
]

If (P=0), these reduce to (T^4,T^2V,T^2W).  If (P>0), the
monomials (V^2,VW) first isolate the last two rows, and the first row is
then nonzero.  Fourier independence and compactness of (R^2+P^2=1)
give a uniform conditional Gram floor.  Choose the endpoint radius
(d/r\le\delta_{\rm end}) so that the exact (O(d/r))
divided-difference perturbation is less than half that boundary floor.
Exact endpoint Hermite division then gives

[
\det\operatorname{Cov}(\nabla f(y)\mid V_r)
=d^6r^2(r^2+\rho^2)^3 A_{\rm end},\qquad
0<c\le A_{\rm end}\le C.                                \tag{4.11}
]

The corrected pair target also gives

[
\frac{\mathbb E[H_{tt}(M)\mid V_r]}r=-\kappa+O(r),
\qquad \mathbb E E_1=-\frac{c\kappa}{\epsilon}+O(1).     \tag{4.12}
]

On the small-(\epsilon) subchart, (|c|) is bounded below and a covariance
ceiling gives the endpoint ESIP.  On the complementary subchart
(\epsilon\ge\epsilon_0>0), the displayed exponential is bounded below by
a positive constant and can be absorbed into (C).  Thus, on the whole
endpoint chart,

[
p_{\nabla f(y)\mid V_r}(0)
\le Cd^{-3}r^{-1}\epsilon^{-3}
e^{-c\kappa^2/\epsilon^2}.                          \tag{4.13}
]

If both endpoint gradients are zero, Taylor's formula gives the exact soft
columns

[
H_Mu=dR_M,\qquad H_yu=dR_y.               \tag{4.14}
]

The reflected chart at (S) has the same formulas.  These charts cover
(q/r\to0,1) and do not borrow a constant from the transverse chart.

4.4 Axial overlaps

At the (M)-endpoint, (4.1) gives

[
\frac{D_t\widetilde f}{qr^2}\longrightarrow2A,\qquad
\frac{D_v\widetilde f}{qr}\longrightarrow-C^v,           \tag{4.15}
]

exactly the endpoint residuals at ((R,P)=(1,0)); reflection gives the
(S)-overlap.  At the midpoint use

[
\zeta_0=\frac{\chi}{|a|},\qquad
U=\frac{2\zeta_0A+\operatorname{sgn}(a)B}
{\sqrt{4\zeta_0^2+1}}.                          \tag{4.16}
]

Thus (\zeta_0=0) is the (T^5) face and
(|\zeta_0|\to\infty) is the (T^4) axial face.  Those symbols are
independent modulo the pair pins.  Equations (MP.1)–(MP.6) are the full
deeper blow-up: the projective coordinates record simultaneously
(\rho\zeta,\rho^2,r\rho,r\zeta,r^2), so no cancellation between the
(T^4) and (T^5) rows is hidden.  Its (P_m=0) face is (4.16), its
(R_m=0) face is the contact family, and the region in which (|\zeta|)
is bounded below overlaps (AI.1)–(AI.4).  Together with the two reflected
endpoint charts, these are compact projective overlaps and cover the whole
collar axial tube.

5. Singular-near anisotropy and its mixed axis blow-up

Put (y=x+s\omega), (\eta=r/s\le\eta_0<1), and
(\omega=(c,\varrho,0)) in a pair-adapted frame.  At the Euclidean contact
face, direct Schur elimination gives

[
\det C_{\rm sing}^{\rm BF}
=\frac{\varrho^{10}(c^2+\varrho^2)(3c^2+\varrho^2)}{24}.  \tag{5.1}
]

Both quadratic factors are retained.  For the exact side-24 field the
constant (1/24) is replaced by a continuous positive factor (A_{24});
it is not asserted to remain literally Euclidean.

Near the collinear axis put

[
s=\Lambda T,\qquad\varrho=\Lambda R,qquad
T,R\ge0,\quad T^2+R^2=1,\qquad a=\eta^2/4,\quad D=1-a. \tag{5.2}
]

Analytic division by the exact pair pins gives

[
F(u,0,0)=(u^2-r^2/4)^2H(u),\quad
F_v(u,0,0)=(u^2-r^2/4)A(u),\quad
F_w(u,0,0)=(u^2-r^2/4)B(u).                              \tag{5.3}
]

The value row must be a function of the raw witness block and the pair pins.
In particular, subtracting the unpinned random midpoint value (F(x))
would not define a four-variable density transform.  Use instead the exact
pair-pin-measurable frame

[
\begin{aligned}
\bar V_0={}&s^{-3}{F(y)-\tfrac{s\varrho}{2}F_v(y)
-\tfrac{sc}{3}F_u(y)},\
V_1={}&s^{-2}F_u(y),\qquad
V_2=s^{-1}F_v(y),\qquad V_3=s^{-1}F_w(y).                \tag{5.3a}
\end{aligned}
]

The map from ((F(y),F_u(y),F_v(y),F_w(y))) to this frame is triangular
up to shears, with diagonal ((s^{-3},s^{-2},s^{-1},s^{-1})), so its
determinant has magnitude exactly (s^{-7}).

On the positive axial chart (the negative chart follows by reflection),
put (D=1-a).  Taylor expansion of (5.3) through the second nonzero order
gives

[
\begin{aligned}
\bar V_0={}&-\frac{D(1+3a)}3sH_0-\frac{1+3a}{6}\varrho A_0
-\frac{2D(1+a)}3s^2H_1-\frac{3+a}{6}s\varrho A_1
-\frac{\varrho^2}{6}D_0+O_3,\
V_1={}&4DsH_0+2\varrho A_0+D(4+D)s^2H_1
+(2+D)s\varrho A_1+\frac{\varrho^2}{2}D_0+O_3,\
V_2={}&sDA_0+\varrho C_0+O_2,\qquad
V_3=sDB_0+\varrho E_0+O_2,                               \tag{5.3b}
\end{aligned}
]

where (O_j=O_{L^2}((s+\varrho)^j)), uniformly in
(0\le a\le\eta_0^2/4).  This follows from analytic Hermite division in
the spectral Hilbert space; exponential Fourier decay makes the Taylor
remainders and their parameter derivatives uniform for the exact side-24
field.

The first-order rows (\bar V_0) and (V_1) are proportional.  Set

[
\lambda=\frac{12c}{c^2+3a},\qquad
\bar W=V_1+\lambda\bar V_0.                              \tag{5.4}
]

Since (c=1+O(\varrho^2)) on this chart, (5.3b) yields

[
\bar W=-\frac{3D}{1+3a}
\left(s^2D^2H_1+s\varrho DA_1+\frac{\varrho^2}{2}D_0\right)
+O_3.                                                    \tag{5.5}
]

After (s=\Lambda T,\ \varrho=\Lambda R), normalize the four rows as
((\bar V_0/\Lambda,\bar W/\Lambda^2,V_2/\Lambda,V_3/\Lambda)).  Their
limits are

[
\begin{aligned}
U_0&=-\frac{1+3a}{12}(4TDH_0+2RA_0),\
U_1&=-\frac{3D}{1+3a}
(T^2D^2H_1+TRDA_1+R^2D_0/2),\
U_2&=TDA_0+RC_0,\
U_3&=TDB_0+RE_0.                                          \tag{5.6}
\end{aligned}
]

The second row lies in the new ((H_1,A_1,D_0)) symbol sector and never
vanishes; the fourth lies in the distinct ((B_0,E_0)) sector.  If
(R>0), the (C_0) term isolates the third row and the first is nonzero;
if (R=0), the remaining symbols are (H_0,A_0).  Thus the rank is four
on the entire compact quarter-circle, including

[
T=0:(A_0,D_0,C_0,E_0),\qquad
R=0:(H_0,H_1,A_0,B_0).                                  \tag{5.7}
]

As an exact algebra check, the sum of the squared nonzero coefficient
minors factors as

[
\frac{D^2}{16}(R^2+T^2D^2)(R^2+2T^2D^2)^4>0.            \tag{5.8}
]

Fourier independence modulo the pair pins and uniform spectral-Hilbert
convergence therefore give

[
\det\operatorname{Cov}(\bar V_0,V_1,V_2,V_3\mid V_r)
\ge c\Lambda^{10}                                       \tag{5.12}
]

uniformly across the mixed, contact-dominant, and collinear-dominant faces.
No unpinned midpoint coarea integral and no second ((\eta,\Lambda))
blow-up are used.

The axial cubic Hermite interpolant for the two pair values has

[
s^{-2}q'(sc)=\kappa(c^2-\eta^2/4).                        \tag{5.14}
]

Analytic regression and exact Hermite division give, on the whole axis
projective chart,

[
|(V_1)_{\rm target}-\mathbb EV_1|
\ge c_K\kappa,\qquad
\operatorname{Var}(V_1\mid V_r)
\le C(s^2+\varrho^2)\le C\Lambda^2,                    \tag{5.15}
]

after shrinking the axis chart; here (5.14) and
(c^2-a\ge1-\eta_0^2/4+O(\Lambda^2)) were used.  Equations
(5.12) and (5.14)–(5.15), the exact row scales
((\Lambda,\Lambda^2,\Lambda,\Lambda)), and the raw Jacobian imply

[
p_{(f(y),\nabla f(y))\mid V_r}(u,0)
\le Cs^{-7}\Lambda^{-5}
\exp!\left(-\frac{c_K}{\Lambda^2}\right)           \tag{5.16}
]

on the axis charts.  Away from them the normalized density is bounded.

6. Three-Hessian conditional moment envelope

Define

[
\mathcal D(r,s)=[r(r+s^2)]^2(r^2+s^3).                   \tag{6.1}
]

There are two statements, and they must not be conflated.

Noncollinear deterministic statement.  On every chart
(|\sigma|\ge\theta>0), including the perpendicular face (c=0), exact
mixed-curvature interpolation gives

[
\begin{aligned}
|\det H_M|+|\det H_S|
&\le Cr(r+s^2)P(J)\theta^{-N},\
|\det H_y|
&\le C(r^2+s^3)P(J)\theta^{-N}.                         \tag{6.2}
\end{aligned}
]

To fix the scaling, put
[
F(X,Y)=s^{-3}{f(sX,sY)-b},\qquad
Q(z)=b+s^3p(z/s),
]
where (p) is the normalized interpolating cubic.  Then
(H_Q=sH_p) and (\det H_Q=s^2\det H_p).  The cubic (p), equivalently
the physical cubic (Q), matches all three values and plane gradients and
the normalized endpoint mixed-curvature difference

[ \frac{D_{te}^2Q(S)-D_{te}^2Q(M)}r

\frac{D_{te}^2f(S)-D_{te}^2f(M)}r.                       \tag{6.3}
]

In (s)-scaled coordinates, a cubic killed by the endpoint pins has form

[
A(X^2-\eta^2/4)Y+Y^2(BX+C)+DY^3.                        \tag{6.4}
]

The row (6.3) gives (2A); after (A=0), the witness value-gradient
matrix has determinant (\sigma^6).  Thus this interpolation remains
uniform through (\eta=0) and (c=0).  More precisely, replace the raw
endpoint rows by the value, (X)-gradient, and (Y)-gradient
average/divided-difference rows through order three, and use
([p_{XY}(a)-p_{XY}(-a)]/(2a)) as the seventh confluent row.  With the
three witness rows, the normalized ten-by-ten determinant is exactly
(-24\sigma^6), independent of (\eta,c).  The raw determinant's
(\eta^5) factor is therefore only the removed confluent-row Jacobian, not
a covariance loss.

For (g=f-Q), the matched mixed-curvature row supplies the exact endpoint
factors

[
H_g^\Pi(M/S)=
\begin{pmatrix}r^2A_{M/S}&r^2B\r^2B&s^2C_{M/S}\end{pmatrix},
\qquad H_g^\Pi(y)=s^2A_y.                               \tag{6.5}
]

The (r^2) mixed entry follows from the confluent identity
[
h[-r/2,-r/2,r/2,r/2]=2h'(r/2)/r^2,
]
applied to (h(q)=D_eg(x+qt)), for which
(h(\pm r/2)=0) and (h'(r/2)=h'(-r/2)).
The exact value-pinned cubic square completions and type on the exact
(H_f(M)), combined with the determinant perturbation identity, give the
two endpoint bounds in (6.2).  Subtracting the two exact cubic Taylor
identities at (y) before dividing by (r) gives
(|H_Q^\Pi(y)|\le CsP(J)\theta^{-N}), hence the witness bound.
The scalar function (D_nf|_\Pi) vanishes at all three nodes; its divided
differences give hard-mixed columns (O(r),O(s)), and the Schur expansion
preserves the same two weights.  Full details are in
three_hessian_conditional_moment_envelope.md.

Axial density-weighted statement.  Put

[
\begin{aligned}
\mathcal K_r(y,u)
:={}&p_{(f(y),\nabla f(y))\mid V_r}(u,0)\
&\times\mathbb E!\left[
|\det H_M\det H_S\det H_y|\mathbf1_{\rm types}
\mid V_r,f(y)=u,\nabla f(y)=0\right].                   \tag{6.6}
]

On the singular mixed/collinear charts,

[
\mathcal K_r(y,u)
\le Cs^{-7}\mathcal D(r,s),
\Theta^{-N}e^{-c/\Theta^2},
\qquad \Theta^2=s^2+\varrho^2.                          \tag{6.7}
]

On the collar axial charts separated from the endpoint-collision divisor,
after the one legitimate value integration,

[
\int_h^b\mathcal K_r(y,u),du
\le Cr^{-4}\mathcal D(r,s),
\Theta^{-N}e^{-c/\Theta^2},                        \tag{6.8}
]

with (\Theta=\epsilon_a) on the axial-interior projective chart and
(\Theta=\delta_m) on the midpoint projective chart.  Equation (6.8) is not
asserted on the axial endpoint itself; that face has the separate
measure-level ledger (8.4).

To prove (6.7)–(6.8), first use the universal soft-column estimates

[
|\det H_M\det H_S\det H_y|\le Cr^2sP(J)
\quad\hbox{(singular axis)},\qquad
\le Cr^3P(J)\quad\hbox{(collar axis)}.                   \tag{6.9}
]

Because (\mathcal D(r,s)\ge r^2s^7) in the singular chart, the first
bound costs only (s^{-6}); the collar bound costs only (r^{-3}).
Gaussian regression and Wick's formula introduce at most another fixed
projective power.  The mismatches in (AI.4), (MP.6), and (5.15) supply the
displayed exponentials, which absorb all these fixed powers.  On the
angular-dominant overlaps, (6.2) applies with a fixed angular loss and the
same exponential.  This proves one density-weighted monomial
(\mathcal D) on every non-endpoint face without asserting a false
pathwise bound on the exact axis.  On the endpoint-axis face the additional
collision radius must be retained through the radial integration.

Endpoint collision refinement.  If (y=M+du), let
(\rho=|t\wedge u|).  Gradient-zero Taylor identities give

[
\begin{array}{lll}
|H_Mt|\le CrP(J),&|H_Mu|\le CdP(J),\
|H_St|\le CrP(J),&dH_Su-rH_St=O(r^2P(J)),\
|H_yt|\le CrP(J),&|H_yu|\le CdP(J).
\end{array}                                                \tag{6.10}
]

Hadamard's inequality in the domain frame ((t,u,n)), whose volume is
(\rho), yields

[
|\det H_M|\le C\frac{rd}{\rho}P,\quad
|\det H_S|\le C\min!\left(r,\frac{r^3}{d\rho}\right)P,\quad
|\det H_y|\le C\frac{rd}{\rho}P.                         \tag{6.11}
]

Splitting at (d=r^2) gives

[
|\det H_M\det H_S\det H_y|
\le Cr^5d,\rho^{-3}P(J).                                \tag{6.12}
]

On the axial cap (\rho\le2r), use orthonormal domain frames instead.
The vector (u) is a soft column at both (M) and (y), while (t) is
a soft column at (S).  Hence

[
|\det H_M|+|\det H_y|\le CdP(J),\qquad
|\det H_S|\le CrP(J),\qquad
|\det H_M\det H_S\det H_y|\le Crd^2P(J).                 \tag{EP.1}
]

Estimate (6.12) is used only on the oblique endpoint chart (\rho\ge r).
The axial endpoint chart (\rho\le2r) uses the exact density and soft-column
ledger (8.4), with its (e^{-c/r^2}) penalty, so neither (6.8) nor (6.12)
is extrapolated through (\rho=0).

6X. Withdrawn all-face pathwise attempt

The text below is retained only to document the rejected route.  Its
statement (formerly Lemma 6.1) is false at (\sigma=0), and its
(ee)-curvature interpolant has a perpendicular remainder failure.  It is
not used in Sections 7–10.  The exact counterexamples and regression checks
are in quartic_remainder_audit_note.md and
scripts/verify_quartic_endpoint_counterexample.py.

[WITHDRAWN ARGUMENT BLOCK — retained verbatim in the sandbox source for
audit history; omitted here at relay to avoid any reader mistaking it for
live text. See the sandbox original for the full withdrawn §6.1 proof and
its counterexample scripts.]

7. Conditional Gaussian moment form

Apply Gaussian regression to the finite divided-difference vector (J).
On non-endpoint axial faces this paragraph is applied to the crude
soft-column bounds (6.9), with their (s^{-6}) or (r^{-3}) deficit
included in the fixed projective power; it is not a claim that the
determinant moment alone is (O(\mathcal D)).  The axial-endpoint divisor
uses the measure-level estimate (8.4), because a genuine (d^{-1}) factor
remains pointwise.
The controls are face-specific:

|Face                            |covariance or ESIP control                                                                   |
|--------------------------------|---------------------------------------------------------------------------------------------|
|shape-regular transverse collar |normalized eigenfloor; no ESIP                                                               |
|axial interior                  |normalized floor (AI.3) and (e^{-c/\epsilon_a^2}); exact axis (e^{-c/r^2})                   |
|midpoint/off-axis               |normalized floor (MP.5) and (e^{-c/\delta_m^2}); exact midpoint (e^{-c/r^4})                 |
|oblique endpoint, (\rho\ge r)   |(e^{-c/(r^2+\rho^2)}), for angular losses only                                               |
|axial endpoint, (\rho\le2r)     |exact (d)-dependent density/product ledger (8.4) and (e^{-c/r^2})                            |
|singular transverse/contact     |normalized eigenfloor; no ESIP                                                               |
|singular mixed/collinear        |uniform four-row frame (5.6), (e^{-c/(s^2+\varrho^2)}), and variance (O(s^2+\varrho^2))      |
|(\eta\to0) away from the axis   |no ESIP; the pair-pin-measurable frame and value-pinned algebra remove every (\eta^{-N}) loss|
|coefficient-degeneracy/collision|alternate regular chart; no automatic ESIP                                                   |

On a face with an eigenfloor, normalized conditional moments are bounded
directly.  On an ESIP face, for some fixed (N_0),

[
\mathbb E[P(J)\mid V_r,f(y)=u,\nabla f(y)=0]
\le C\theta^{-N_0}.                                      \tag{7.1}
]

The same normalized coordinate which creates the stated loss has squared
Mahalanobis distance at least (c/\theta^2).  Therefore the
density-weighted moment is
bounded by

[
C\mathcal D(r,s)\theta^{-N_1}e^{-c/\theta^2},             \tag{7.2}
]

with the identical monomial (\mathcal D) on every face except the axial
endpoint, where the radial measure is retained as in (8.4).  Since
(\sup_{0<\theta\le1}\theta^{-m}e^{-c/\theta^2}<\infty) for every fixed
(m), no angular power changes the radial ledger.

8. Collar integration, including endpoints and midpoint

On a regular collar chart the joint value-gradient density is (O(r^{-7})).
Using (1.5) leaves (O(r^{-4})); physical collar volume is (O(r^3)), and
the noncollinear part of Section 6 gives (O(r^6)).  Thus the
unnormalized numerator is

[
O(r^{-4})O(r^3)O(r^6)=O(r^5).           \tag{8.1}
]

At an endpoint, value marginalization is performed before estimating the
moment.  For every nonnegative determinant polynomial (Q),

[
\int_h^b p_{F,G}(u,0)\mathbb E[Q\mid F=u,G=0],du
\le p_G(0)\mathbb E[Q\mid G=0].                           \tag{8.2}
]

Thus no conditional moment growing with an unnatural endpoint value is
discarded.  Combining (8.2), (4.13), (6.12), (1.3), and
(dy\asymp d^2,dd,d\omega) gives, before the integrable angular factor,

[
r^{-2},d^{-3}r^{-1},(r^5d),d^2dd
= r^2,dd.                       \tag{8.3}
]

Integration over (0<d<\epsilon_0r) is (O(r^3)).  On (\rho\ge r), every
fixed factor (\epsilon^{-3}\rho^{-N}) is absorbed by
(e^{-c/\epsilon^2}).  On (\rho\le2r), the axial-endpoint chart has no
(\rho^{-N}) singularity.  On this cap,
(\det\operatorname{Cov}(\nabla f(y)\mid V_r)\asymp d^6r^8), while the
three soft columns give (EP.1).  After value
marginalization, volume, and Palm normalization its radial ledger is

[
r^{-2}(d^{-3}r^{-4})(rd^2)d^2,dd
=r^{-5}d,dd.                                           \tag{8.4}
]

Including the angular-cap area (O(r^2)), and allowing a fixed additional
regression power (r^{-N}), gives the complete estimate

[
Z_r^{-1}\int_{\rho\le2r}\int_0^{\delta_{\rm end}r}
d^2\left[d^{-1}r^{-3-N}e^{-c/r^2}\right],dd,d\omega
\le Cr^{-1-N}e^{-c/r^2}=O(r^3).                           \tag{EP.2}
]

The axial-interior and midpoint charts are similarly smaller because of
(AI.4) and (MP.6).  Explicitly, their physical Jacobian is

[
dy=r^3\rho,d\zeta,d\rho,d\phi,
]

and every fixed axial-interior projective loss is bounded by

[
\int_0^{\rho_0}\rho(\rho^2+r^2)^{-N/2}
e^{-c/(\rho^2+r^2)},d\rho\le C_N.                       \tag{8.5}
]

For the midpoint chart, (\delta_m\le C\epsilon_m), so for every fixed
(p,q\ge0), after decreasing (c) if necessary,

[
\sup \delta_m^{-p}\epsilon_m^{-q}
e^{-c/\delta_m^2}\le C_{p,q}.                      \tag{MP.7}
]

Thus all additional midpoint projective powers are absorbed pointwise by
the stronger exponential in (MP.6).

Thus the same (r^{-4}r^6r^3=r^5) pre-Palm ledger holds.  Dividing
(8.1) by (1.3) therefore proves

[
\mathbb E^{MS}N_{\rm col}\le Cr^3. \tag{8.6}
]

9. Singular-near integration, including the angular axis

The corrected joint density contributes (s^{-7}), the physical volume is
(s^2ds,d\omega), and (1.5) contributes (O(r^3)).  On the axis charts,
the remaining angular integral has the form

[
\int_0^{\varrho_0}\varrho
(s^2+\varrho^2)^{-m/2}
e^{-c/(s^2+\varrho^2)},d\varrho\le C_m,                 \tag{9.1}
]

uniformly in (s); use (z=s^2+\varrho^2).  Thus, by (1.3),

[
\mathbb E^{MS}N_{\rm sing}
\le Cr\int_{Cr}^{\delta}s^{-5}
[r(r+s^2)]^2(r^2+s^3),ds.                         \tag{9.2}
]

The six terms in the integrand are

[
r^7s^{-5},\quad2r^6s^{-3},\quad r^5s^{-2},\quad
r^5s^{-1},\quad2r^4,\quad r^3s^2.                        \tag{9.3}
]

Their integrals from (Cr) to (\delta) are respectively

[
O(r^3),\quad O(r^4),\quad O(r^4),\quad
O(r^5\log(1/r)),\quad O(r^4),\quad O(r^3).               \tag{9.4}
]

Therefore

[
\mathbb E^{MS}N_{\rm sing}\le Cr^3. \tag{9.5}
]

10. Disposition

Items 1–5 and 7 of the requested work order are proved in Sections 3–5
and 8–9.  The literal pointwise reading of item 6 is disproved on the axial
endpoint; Sections 6–8 prove the exact hybrid replacement needed by
Kac–Rice, including the measure-level endpoint ledger (EP.2).  Within this
successor draft, that corrected statement proves RP-C and RP-S.  Combining
(8.6), (9.5) with the already proved RP-A, RP-L, and RP-F bounds gives the
compact-positive-mark estimate

[
\sup_{t,b,\kappa}{1-p_r(t,b,\kappa)}\le Cr^3.          \tag{10.1}
]

The package-level theorem remains marked HOLD / ready for independent
audit because the V3.3 promotion protocol requires independent review of
RP-C and RP-S before changing the controlling theorem status.  This label is
procedural: within this draft the mathematical residual-premise list for the
regional estimates is empty after replacing the impossible pointwise form
of item 6 by its exact measure-level form, subject to the stated
independent-audit gate.
