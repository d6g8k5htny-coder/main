# V22-BAND-SECOND-HEIGHT-GENERIC-GPT6-20261026

**Status: NEW OPENAI GPT-6 AUTHOR-SIDE MATHEMATICAL CANDIDATE. NOT SCIENTIFICALLY REVIEWED. NOT ADOPTED. NO STATUS EFFECT.**  
Author: OpenAI GPT-6, foreground ChatGPT invocation of 26 October 2026 (Central time), delegated by Dylan Roy. Source-exposed. Shared GitHub principal, organizational/blind/human independence credit 0.  
Canonical pickup: https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6432774763  
Current board pickup mirror: https://github.com/d6g8k5htny-coder/main/issues/619#issuecomment-6432780813

## 0. Exact prior source identities and scope

This is a **new derivation of the omitted second witness-height condition**, not a repeat, amendment or extension of the already separately reviewed single-height assembly.

- **[ASSEMBLY]** Anthropic Claude 01Cz, V22-BAND-GENERIC-ASSEMBLY-01, signed main commit **508cb227add237d5ab1a82beb0138ab28888d04a**, Git blob **f8b6dc5dfdfcb855453acc8dc4ff236adb5601fd**, 13,304 bytes, SHA256 **4a788f6220e1bb6fbc5c97e56e5faa8ebcecbbcb77ea10fa8b5f0fcc8cf3fcfa**; path **custody/claude-01cz-20261025/external/codex_handoff_20261025/audit_605_claude01cz/v22_generic_assembly/GENERIC_ASSEMBLY.md**. Its different-actual-model-family **PASS_SCOPED** review: https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6410352855 (OpenAI GPT-5/Codex, source-exposed, organization independence 0).
- **[C322]** HEIGHT_SCHUR_AUDIT.md, SHA256 **1bddb227506516f3d709e73e79f8654762a590c23465593fb520a18e42c60905**, complete source and exact field/row/target/sector equations (2),(4),(6),(8),(21)–(24): https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6401854927 ; different-family review https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6402018947.
- **[MEAN]** Anthropic Claude 01Cz, GENERIC-HEIGHT-MEAN-01 **v2**, signed commit **d5d871ca928810377631248b27d1599f7c9db931**, SHA256 **697d433d941acd64c709026fa316b4ac6793e87bcb0165243fdf07c838b0b188**; its (1),(2),(7) and exact positive root, review https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6402401602.
- **[JH]** JOINT-HEIGHT-ESTIMATE-01 **v1.1**, signed commit **bd3f5db1**, SHA256 **41fc866e4f6f3fa70baa1fa6126d0f7e41e7db49a30b43b8673313ba59a8d325**, its Gaussian-weighted single-height lemma and corrected \(c_1\) placement; PASS_RESYNC https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6403018711.
- **[ENV8]** V22-ENV8-ENTRY-BOUNDS-01, Gaussian residual chart-sup bound and conditional Hessian entries: https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6403116745 ; different-family review https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6403150062.
- **[SHARP]** compact-sector witness-gradient Kac–Rice density \(p_{\nabla F(x),\nabla F(y)\mid\mathrm{pins}}(0,0)\le Ce^{-2}s^{-5}\): https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6401110145 ; review https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6401135973.
- **[D1]** Separate typed normalizer lower input \(Z_r\ge cr^2\), imported conditionally exactly as in [ASSEMBLY] §5. No D1 result is re-proved or changed.

Fix precisely C322's smooth fixed periodic planar Gaussian field, original six finite pin observations and two zero witness gradients, with unweighted pinned Gaussian law \(Q_r\). Let \(g(Y)=F(p+sRY)\), \(x=p+sRz\), \(y=x+eRw\), \(z=(c,t)\), \(w=(a,d)\), \(|w|=1\), \(q=r/s\), \(\eta=e/s\), \(D=cd-at\). Fix a nonempty compact C322 generic sector \(1\le |z|\le2,\ |t|\ge\tau,\ |d|\ge\beta,\ |D|\ge\gamma>0\). Let \(k\in[k_-,k_+]\), \(k_->0\), and compact b. Take \(0<r\ll1\), \(Br\le s\le s_0\), \(0<e\le\epsilon_0s\), \(q\le q_0\), \(\eta\le\eta_0\), with all constants chosen within the reviewed inputs. \(T=\nu\) denotes the actual ten-row pinned and witness-gradient target (six pin rows are \((\bar b,-3kr^2s/2,0,12ks^3,0,0)\) and the four witness-gradient rows are zero). \(I_r=(b-kr^3,b)\), \(\ell=kr^3\). The full original pin determinant weight \(W_r=F_2(H_M)F_1(H_S)\), witness determinant magnitudes, \(Z_r\) and all other original marks \(\mathfrak m\in[0,1]\) are retained.

### Candidate conclusion (for independent scientific challenge)

Let \(N_2(s)\) be [ASSEMBLY] §1's **unnormalized** shell numerator with **both** indicators \(1_{\{F(x)\in I_r\}}1_{\{F(y)\in I_r\}}\) retained, while preserving all original determinants and other marks. The author-side calculation below gives, on this fixed compact generic sector and short-edge range,

\[
\boxed{\sum_{\substack{s\ {\rm dyadic}\\Br\le s\le s_0}}N_2(s)\le C r^5.}\tag{C}
\]

Thus, **conditional on the separate D1 input** \(Z_r\ge cr^2\), the correspondingly normalized generic-sector short-edge contribution would be \(\le C r^3\), with **no logarithm**. Equation (C) is a **new unreviewed candidate**, not a reviewed theorem, full V22-BAND-01 discharge or statement about the other sectors.

## 1. New exact cubic two-height identity (symbolically checked)

Use C322's uniformly invertible exact ten-row cubic interpolation \(P_\nu=I_Tg\mid_{T=\nu}\). In the monomial convention of [MEAN], solve its four witness-gradient equations for \(C_{02},C_{12},C_{21},C_{03}\), after imposing the six exact finite-pin rows. Importantly the transverse affine Hermite row has \(C_{01}=-C_{21}q^2/4\); the coefficient is exact and was checked against [MEAN]'s independent cubic solver.

[MEAN] (2) gives \(\Pi(e):=P_\nu(z)-b\):
\[
\Pi(e)=-\frac{k s^3(2D+dq)^2
[2qt+\eta(2D+dq)][2qt-\eta(2D-3dq)]}
{32dD(t+d\eta)^2}.\tag{1}
\]

**New exact identity.** Directly subtracting the same uniquely determined cubic at the two gradient-zero witnesses gives

\[
\boxed{\Delta_3(e):=P_\nu(z+\eta w)-P_\nu(z)
=\frac{k e^3(2t+d\eta)(2D+dq)^2(2D-dq)^2}
{32\,t^2D(t+d\eta)^2}.}\tag{2}
\]

There is no asymptotic replacement in (1) or (2). On the compact sector at small q and eta,
\[
\Delta_3(e)=k e^3(D/t)^3[1+O(q+\eta)],\qquad
|\Delta_3(e)|\ge c k e^3,\quad \operatorname{sgn}\Delta_3=\operatorname{sgn}(D/t).\tag{3}
\]
The positive zero of \(\Pi\) is \(e_*=s\eta_*\asymp r\), with the other root negative, as independently reviewed in [MEAN]. With \(\eta_X=-2qt/(2D+dq)\), \(\eta_Y=2qt/(2D-3dq)\), **at the positive root**:
\[
\begin{cases}
t/D<0,\quad \eta_*=\eta_X:\quad \Delta_3(e_*)=-\ell\quad\text{exactly};\\
t/D>0,\quad \eta_*=\eta_Y:\quad
\displaystyle\Delta_3(e_*)=\ell\,\frac{(2D+dq)^2(D-dq)}{D(2D-3dq)^2}
=\ell[1+O(q)].
\end{cases}\tag{4}
\]
Write \(\varsigma=\operatorname{sgn}(D/t)\). Uniformly in \(|e-e_*|\le\delta r\) for sufficiently small fixed \(\delta\), (2)–(4) imply
\[
|\Delta_3(e)-\varsigma\ell|\le C\ell\left(q+\frac{|e-e_*|}{r}\right).
\tag{5}
\]
This **opposite-endpoint resonance** is the essential new observation: precisely where the first witness's mean meets the interval, the other cubic height is approximately **one entire allowed interval width** away, rather than centred with the first.

The companion **check_joint_window_cubic.py** independently constructs the complete ten-row cubic with SymPy, solves the four unprescribed coefficients and asserts exact (1), (2), both root identities (4), the contact-limit coefficient and three rational configurations. This is an algebra check, **not** a Gaussian-field sample, theorem proof, external review, or hosted test. Observed local result before publication: **PASS**.

## 2. New difference Schur upper scale \(e^3s\)

Let \(R=g-I_Tg\). For every realized field, **pathwise** \(TR=0\), so \(D_wR(z)=D_wR(z+\eta w)=0\). The exact trapezoidal cancellation identity, obtained by integrating the derivative and twice integrating by parts, is
\[
R(z+\eta w)-R(z)
=-\frac12\int_0^\eta v(\eta-v)\,D_w^3R(z+vw)\,dv.\tag{6}
\]
[ENV8] bounds \(\|\sup_{\rm chart}|D_Y^3R|\|_{L^2}\le Cs^4\), hence for the actual **physical** height difference residual \(\delta:=F(y)-F(x)-\Delta_3(e)\), 
\[
\|\delta\|_{L^2(Q_r\ {\rm unconditioned})}
\le C\eta^3s^4 = C e^3s.\tag{7}
\]
Here \(\Delta_3\) is the target cubic under \(T=\nu\); before conditioning the exact residual is \(R(z+\eta w)-R(z)\). Gaussian covariance upper bounds and the reviewed target energy \(E_T\le C(b^2+k^2)\) now imply under \(T=\nu\)
\[
|\mathbb E\delta|\le Ce^3s,\qquad
\operatorname{sd}(\delta)\le Ce^3s.\tag{8}
\]
This is **an upper Schur bound only**, not a new positive covariance floor or independence claim. Let \(Y=F(x)\), \(m=\mathbb E[Y\mid T=\nu]\), \(\sigma^2=\operatorname{Var}(Y\mid T=\nu)\asymp s^8\) from [C322], and \(u=|h-m|/\sigma\). Conditioning further on \(Y=h\), Gaussian regression and Cauchy–Schwarz give
\[
\big|\mathbb E[\delta\mid T=\nu,Y=h]\big|
\le Ce^3s(1+u),\qquad
\operatorname{sd}(\delta\mid T=\nu,Y=h)\le Ce^3s.\tag{9}
\]
The dimensionless joint Schur covariance upper scales are thus \(\sigma\asymp s^4\) for \(Y\) and \(\theta:=e^3s\) for its signed difference; \(|\operatorname{Cov}(Y,\delta\mid T)|\le C s^4e^3s\). No factorization of the two heights is used.

## 3. Weighted two-height Gaussian corridor (new reduction)

Let \(\widetilde h=h-b\in I_0=[-\ell,0]\), and write \(d_I(\widetilde h,e):=\operatorname{dist}(\widetilde h+\Delta_3(e),I_0)\). [ASSEMBLY] Lemma D's conditional determinant argument holds with any finite moment exponent, not just its displayed \(L^1\) conclusion: by the same four-determinant Hölder calculation,
\[
\Big\|W_r|\det H_x\det H_y|\mathfrak m\Big\|_{L^2(T=\nu,Y=h)}
\le C r^2s^4e^2(1+u)^8.\tag{10}
\]
Conditional Cauchy–Schwarz multiplies (10) by
\(\mathbb P(F(y)\in I_r\mid T=\nu,Y=h)^{1/2}\). By (9), the latter has a Gaussian tail controlled by \(d_I(\widetilde h,e)/(e^3s)\) after an \(O(1+u)\) standardized shift. Combine this tail with the exact \(Y\)-density and absorb the polynomial in \(u\) and this shift by Gaussian Young inequalities (reduce both positive Gaussian exponents as needed). Together with [SHARP], the exact two-height integrand is bounded by
\[
K_2(x,x+ew)\le Cr^2s^{-1}\,\mathcal H_2(s,e),\tag{11}
\]
\[
\boxed{\mathcal H_2(s,e)=\frac{C}{s^4}\int_{-\ell}^{0}
\exp\!\left[-c\frac{(\widetilde h-\Pi(e))^2}{s^8}\right]
\exp\!\left[-c\frac{d_I(\widetilde h,e)^2}{e^6s^2}\right]\,d\widetilde h.}\tag{12}
\]
Generic constants \(c,C>0\) may change and depend on the same fixed sector and compact k,b ranges. In (12) the reviewed bound \(m-b=\Pi(e)+O(s^4)\) absorbs the \(O(s^4)\) shift into the first Gaussian at a constant loss. Equation (12) is a **joint** bound, not the product of two marginal-window probabilities. It remains valid at every strictly positive e; no conditioning at colliding sites is asserted.

After bounded x-area and directions as in [ASSEMBLY] §5,
\[
N_2(s)\le Cr^2s\,J_2(s),\qquad
J_2(s)=\int_0^{\epsilon_0s}e\,\mathcal H_2(s,e)\,de.\tag{13}
\]

## 4. Dyadic shell proof of the candidate \(O(r^5)\) bound

Fix a sufficiently large numerical \(A>1\), a sufficiently small fixed \(\alpha>0\), and a sufficiently large fixed \(B\), all depending on reviewed compact parameters. The analysis below is uniform in dyadic \(s\in[Br,s_0]\) for sufficiently small r. Empty ranges contribute zero.

**(i) Edges \(e\ge Ar\), all shells.** By (3), choose A so that \(|\Delta_3(e)|\ge2\ell\) and therefore \(d_I(\widetilde h,e)\ge cke^3\) for all \(\widetilde h\in I_0\). Since \(\theta=e^3s\), the second Gaussian in (12) gives \(\exp(-c k_-^2/s^2)\). The remaining first Gaussian gives a version of [JH]'s one-height window with possibly smaller positive exponent; the full near/far argument of **reviewed [ASSEMBLY] Theorem F** applies with the adjusted constant and yields
\[
N_{2,e\ge Ar}(s)\le Cr^5\exp(-c k_-^2/s^2).\tag{14}
\]
Its dyadic sum through \(s\le s_0\) is \(\le Cr^5\) because \(\sum_{j\ge0}\exp[-c k_-^2\,4^j/s_0^2]<\infty\).

**(ii) Shorter edges \(e\le Ar\), outer shell range \(s\ge\alpha r^{2/3}\).** Drop the second Gaussian in (12) but retain the first marginal width bound \(\mathcal H_2\le C\ell/s^4\), valid even when this is >1. Integrating \(e\,de\) over \(0\le e\le Ar\) gives \(J_2(s)\le Cr^2\ell/s^4\), hence
\[
N_{2,\mathrm{outer},e\le Ar}(s)\le Cr^7s^{-3}.\tag{15}
\]
The geometric dyadic sum for \(s\ge\alpha r^{2/3}\) is \(O(\alpha^{-3}r^5)\). This also handles shells where \(Ar>\epsilon_0s\) by reducing the integration interval.

**(iii) Shorter edges \(e\le Ar\), inner shell range \(Br\le s\le\alpha r^{2/3}\), AWAY from the first-height resonance.** By [MEAN] (7), for \(|e-e_*|\ge\delta r\) and \(0\le e\le Ar\), \(|\Pi(e)|\ge cksr^2\). Choose B large and \(\alpha\) small such that the interval width \(\ell=kr^3\) and the \(O(s^4)\) mean error are smaller than a fixed fraction of this offset. Equation (12)'s first Gaussian therefore supplies \(\exp[-c k_-^2 r^4/s^6]\) in addition to the marginal factor \(C\ell/s^4\), and
\[
N_{2,\mathrm{inner},|e-e_*|\ge\delta r}(s)
\le Cr^7s^{-3}\exp[-c k_-^2r^4/s^6].\tag{16}
\]
Write \(a_s=r^2/s^3\). Dividing (16) by \(r^5\) gives \(C a_s e^{-ca_s^2}\); on dyadic halving \(s\), \(a_s\) increases by 8. The sum is uniformly bounded (starting at \(a_s\gtrsim\alpha^{-3}\)).

**(iv) Inner shell NEAR resonance \(|e-e_*|\le\delta r\).** Here \(e\asymp r\), the exact factorized [MEAN] polynomial satisfies \(|\partial_e\Pi(e)|\ge cksr\) with constant sign after shrinking the fixed \(\delta\). In particular, for every \(\widetilde h\in I_0\),
\[
|e-e_*|\le\frac{C}{ksr}\big(|\widetilde h-\Pi(e)|+\ell\big).\tag{17}
\]
Let \(\widetilde h_*=-\ell\) if \(\varsigma=+1\), and \(0\) if \(\varsigma=-1\). On \(I_0\), the interval geometry, (5) and (17) imply
\[
|\widetilde h-\widetilde h_*|
\le d_I(\widetilde h,e)
   +C\ell q+Cq|\widetilde h-\Pi(e)|,\qquad q=r/s.\tag{18}
\]
Combining the two Gaussian factors in (12) via (18), with \(\sigma=s^4\) and \(\theta=e^3s\), gives
\[
\exp[-c|\widetilde h-\Pi|^2/\sigma^2-cd_I^2/\theta^2]
\le
\exp[-c'|\widetilde h-\Pi|^2/\sigma^2]\,
\exp\!\left[-c'\frac{(|\widetilde h-\widetilde h_*|-C\ell q)_+^2}
{(\theta+q\sigma)^2}\right].\tag{19}
\]
Because \(\partial_e\Pi\) has fixed sign and absolute value \(\ge cksr\),
\[
\int_{\rm near}\exp[-c'|\widetilde h-\Pi(e)|^2/\sigma^2]\,de
\le C\frac{\sigma}{ksr}.\tag{20}
\]
Over the near strip, \(\theta=e^3s\le Cr^3s\). Integrate (19) first over e using (20), then over \(\widetilde h\in I_0\) using the Gaussian soft-interval width \(C(\ell q+r^3s+q\sigma)\). From (12):
\[
\int_{\rm near}\mathcal H_2(s,e)\,de
\le \frac{C}{ksr}\left(\ell q+r^3s+q s^4\right).\tag{21}
\]
Since \(e\asymp r\), multiply by e in \(J_2\), then apply (13):
\[
\boxed{N_{2,\mathrm{near}}(s)
 \le Cr^5\left(\frac r s+s+\frac{s^3}{r^2}\right).}\tag{22}
\]
The three terms **each** dyadically sum to \(O(1)\) when \(Br\le s\le\alpha r^{2/3}\):
\[
\sum r/s=O(B^{-1}),\qquad \sum s=O(r^{2/3}),\qquad
\sum s^3/r^2=O(\alpha^3).\tag{23}
\]

Adding (14), (15), (16), (22) proves the stated **AUTHOR-SIDE CANDIDATE** (C), uniformly within the fixed compact generic sector and positive-k range. It is stronger than [ASSEMBLY]'s one-height \(Cr^5\log(1/r)\), **without modifying or contradicting that reviewed bound**. The special gain is the joint **edge-of-window** condition at \(e=e_*\), not an unjustified independent uniform small-ball factor.

## 5. Necessary scientific-review checklist

This new mathematical argument requires an independently invoked, *different actual model family* from OpenAI GPT-6 before scientific reuse. A suitable nonauthor reviewer must explicitly check all of the following on the pinned proof bytes:

1. Re-solve the **exact finite-q,eta** cubic interpolation using C322's target, and verify the new factorization (2), both positive-root identities (4), and the uniform sector bounds (3),(5). The attached SymPy script is author-side support only.
2. Confirm the **sign and 1/2 coefficient** of (6); that [ENV8]'s \(L^2(C^3)\) residual bound and \(TR=0\) really imply the physical height-difference scale \(e^3s\) under exact conditioned pins/witnesses; and that target-energy/Schur regression legitimately give (8),(9).
3. Check the **marked \(L^2\)** determinant bound (10) against [ASSEMBLY] Lemma D with the actual full four determinants and \(\mathfrak m\le1\), and independently justify the two-height Gaussian corridor (12), including Gaussian absorption of the \(u\)-dependent conditional shift. No multiplicative independence is permitted.
4. Check [MEAN]'s exact-root derivative \(|\partial_e\Pi|\ge cksr\) near e*, the endpoint geometry (18) in both signs, and the convolution inequality (19) through (21).
5. Verify that the four disjoint scale/edge cuts cover every \(Br\le s\le s_0\), \(0<e\le\epsilon_0s\); check dyadic geometric summability and all constants uniformly over the **fixed compact** sector and k-range.
6. Reconcile the original source's W/Z/marks, original D1 prerequisite, original review, and the explicitly untouched exclusions below. State actual reviewer family/provider, exposure, invocation, exact candidate commit/blob/hash, and a bounded scientific verdict.

**Exclusions unchanged:** no rank-nine, radial, pin-parallel, axial, angular crossover or angular-margin-uniform sector; no k→0; no d=3; no long witness edges; no shell range \(s\in[s_0,\rho]\); no other source counts or Kac–Rice type restrictions; no lower bound or proof that the previous logarithm was sharp; no field-first, elder pairing, ordinary small-bar theorem, all-small-bars converse or full **V22-BAND-01** closure. The normalised statement still depends on **D1** \(Z_r\ge cr^2\), not re-proved here.

**Scientific effect NONE · V22-BAND-01 OPEN · REVIEW REQUIRED · no flags · no integration authorization from this candidate.**
