# SIDE24 THEOREM PACKAGE v1.0

**Artifact ID:** GP-SIDE24-THM-001-v1.0  
**Date:** 2026-08-02  
**Owner and operator:** Dylan Roy  
**Track:** three-dimensional SIDE24 compact-positive-mark theorem  
**Controlling disposition:** **RATIFIED-AT-STATED-SCOPE**  
**Purpose:** self-contained export surface for external human specialist review

## 1. Controlling statement

Let \(f\) be the normalized periodized Bargmann--Fock Gaussian field on the
side-24 three-torus. Use the SIDE24 pair chart, Palm normalization, critical
point typing, elder selection, and finite nonessential superlevel
\(H_0\)-lifetime convention fixed by the cited supplement. For pair separation
\(r\), pair direction \(t\), level parameter \(b\), and positive gap parameter
\(\kappa\), let \(p_r(t,b,\kappa)\) be the conditional probability that the
canonical compact-positive-mark configuration survives the collar,
singular-near, far-witness, and elder-selection exclusions.

For every compact \(K\subset\mathbb R\times(0,\infty)\), there are constants
\(C_K<\infty\) and \(r_K>0\) such that

\[
\sup_{t\in\mathbb S^2,\,(b,\kappa)\in K}
\bigl(1-p_r(t,b,\kappa)\bigr)\le C_K r^3,
\qquad 0<r<r_K.                                           \tag{T1}
\]

Consequently, the first-moment lifetime density \(\nu_{3,24}\) of finite
nonessential superlevel \(H_0\) bars obeys

\[
\nu_{3,24}(\ell)=c_{3,24}\ell^{-1/3}(1+o(1)),
\qquad \ell\downarrow0.                                  \tag{T2}
\]

The acceptance is exactly at this scope. It is not a statement about another
torus side length, a different Gaussian law, essential bars, higher homology,
an unbounded \((b,\kappa)\) range, or the separate two-dimensional q0/P0.1
program.

The operator's signed controlling language is:

> I, the human owner of this work, ratify the closure of RP-C and RP-S per the
> KIMI-AUD-006/006b audit chain and accept the theorem sup(1−p_r) ≤ Cr³ and
> the statement ν₃,₂₄(ℓ) = c₃,₂₄ℓ^(−1/3)(1+o(1)) at their stated scopes, with
> the carried dependencies recorded in AO48-AUD-044 §C.

The signed record is AO48-OPR-045-v1.0: Drive
`1MfA94SaoYpnnAs7HYLNvowG9EHIR3Opk`, 3,397 bytes, SHA-256
`e48d7c271dff9c120316ee2129a303aed6f78b518c2e0c08337eb357bd60e2e8`.

## 2. Constant and local parameterization

The two contact values are pinned as

\[
f(M)=b,\qquad f(S)=b-\frac{\kappa r^3}{6},\qquad
\nabla f(M)=\nabla f(S)=0,                                \tag{2.1}
\]

with \(M=x-rt/2\) and \(S=x+rt/2\). Thus the height gap is
\(\ell=\kappa r^3/6\), and the gap-coordinate Jacobian contributes
\(r^3/6\). The pushforward \(r=(6\ell/\kappa)^{1/3}\), together with the
contact intensity and the \(O(r^3)\) failure estimate, produces the exponent
\(-1/3\) in (T2).

Write \(G^{\mathrm{all}}_0(t,b,\kappa)\) for the limiting all-typed,
determinant-weighted contact integrand after the pair blow-up and let
\(G^{\mathrm{all}}_0(\kappa)\) denote its \((t,b)\)-marginal over
\(\mathbb S^2\times\mathbb R\). The finite-side constant is represented by

\[
c_{3,24}=\frac{6^{2/3}}{18}
\int_0^\infty \kappa^{-2/3}G^{\mathrm{all}}_0(\kappa)\,d\kappa. \tag{2.2}
\]

Its infinite-volume reference value is

\[
c_{3,\infty}=
2^{-23/6}3^{-8/3}(29\sqrt6-36)\Gamma(1/6)\pi^{-5/2},       \tag{2.3}
\]

numerically

\[
c_{3,\infty}=
0.041775931840598343342936665428575556466681519661\ldots. \tag{2.4}
\]

The certified side-24 correction is

\[
\frac{c_{3,24}}{c_{3,\infty}}-1
=-\frac{620813376}{35}e^{-288}+\varepsilon_{24},
\qquad |\varepsilon_{24}|<10^{-180}.                      \tag{2.5}
\]

The arithmetic route uses

\[
P_3(L)=-\frac{L^2(10L^4-147L^2+315)}{105},
\qquad P_3(24)=-\frac{620813376}{35},                     \tag{2.6}
\]

and the cone coefficient

\[
D_2=\frac{29}{6}-\sqrt6.                                  \tag{2.7}
\]

These identities belong to the constant chain. They do not replace the
probabilistic and facewise estimates needed for (T1).

## 3. Proof architecture

The proof is organized as one local-to-global implication rather than as a
collection of script outputs:

1. The contact Kac--Rice/Palm formula writes the small-lifetime first moment
   in the pair variables \((x,t,r,b,\kappa)\), with determinant weight and
   critical-point typing.
2. The Palm-repair normalizer satisfies \(c r^2\le Z_r\le C r^2\).
3. RP-A/RP-L control the local same-pair failures, capture/escape events, and
   the weighted Palm/coarea passage by \(O(r^3)\).
4. RP-F controls a fixed-distance or far witness by \(O(r^3)\).
5. RP-C/RP-S control every collar and singular-near face, including the
   angular axis, by \(O(r^3)\). This is the V3.4/FACEWISE closure.
6. The elder-selection recombination joins the mutually compatible local and
   far exclusions without changing the \(r^3\) order.
7. The change of variables \(\ell=\kappa r^3/6\), the full-mark integrable
   majorant of Theorem A.3.3, and the constant ledger give (T2) and
   (2.2)--(2.5). This is a genuine truncation-and-dominated-convergence
   argument on \(\mathbb S^2\times\mathbb R\times(0,\infty)\), not an
   inference from compact convergence alone.

The role of each closing artifact is made explicit below.

## 4. Premise-by-premise proof-chain map

### 4.1 RP-A and RP-L: Palm/coarea and local same-pair control

Theorem G.7.1 supplies the local Palm/coarea interface and the same-pair
failure estimate. Its proof is stratified into:

- covariance eigenfloors and Gaussian regression on the normalized contact
  variables;
- soft Hessian columns and determinant weights;
- the \(Z_r\asymp r^2\) Palm normalizer and the required weighted moments;
- shallow value layers and the unique value-window integral of length
  \(\kappa r^3/6\);
- Gaussian tail budgets;
- the exact gap parameter and the deterministic capture/escape geometry.

The conclusion used downstream is an \(O(r^3)\) bound under the exact weighted
Palm law, uniform at the stated compact-positive-mark scope. The closing
carrier is the recovered V5 `SIDE24_GAP_FILL_SUPPLEMENT.md`: Drive
`1D5eRshQWyiugIrIikBUhItasLyiyV1FX`, 105,980 bytes, SHA-256
`05967f14a8b8d6a9e2637210807872ba4f30810987cb7f3017e10b186d07f6da`.

### 4.2 RP-F: fixed-distance endpoint and far-witness control

Section G.8 uses the same normalized pair pins and determinant-weighted Palm
law to bound a witness outside the shrinking local charts. The joint density
is nonsingular at fixed separation; its conditional mean/covariance and the
Hessian moment envelope are uniform on the compact complement. The single
value window contributes \(O(r^3)\), while polynomial factors in
\((|b|+\kappa)\) are controlled on the stated compact set. The result is the
required \(O(r^3)\) far-witness contribution.

G.8 is frozen in the same 105,980-byte supplement and therefore has the same
Drive ID and SHA-256 as §4.1.

### 4.3 RP-C: collar faces

The collar proof does not extend a transverse compactness argument through the
axis. It uses separate normalized charts:

**Generic transverse face.** With axial coordinate \(X\) and transverse
radius \(\varrho\), exact residualization and Fourier independence give

\[
\det\operatorname{Cov}(\nabla f(y)\mid V_r)
\asymp \varrho^6(4X^2+\varrho^2)                         \tag{4.1}
\]

on normalized strict-transverse compact charts, with uniform side-24 Schur
constants.

**Axial face.** Exact Hermite divided differences factor the centered value
and gradient. Positivity of the residual covariance is proved by Fourier
independence. The exact axis has its own projective compactification; it is not
obtained from (4.1) by compactness. The target mean yields an exponential
penalty on the axial cone.

**Midpoint face.** The secondary expansion retains the fourth/fifth-order
rows that survive the midpoint cancellation. Equivalently, the separated
target-mean mismatch supplies an integrable exponential penalty. The full
projective frame records the simultaneous \((r,\rho,\zeta)\) directions.

**Endpoint faces.** Reflected endpoint charts cover \(q/r\to0\) and
\(q/r\to1\). Their exact Hermite residuals give the determinant and mean
mismatch at the endpoint collision scale without borrowing a transverse
constant.

**Confluent row.** In the \(s\)-scaled interpolating plane, put
\(\eta=r/s\), take the endpoint nodes at \(X=\pm a\) with
\(a=\eta/2\), let \(c\) be the axial component of the normalized witness
direction, and let \(\sigma\) be its signed noncollinearity coordinate (so
the witness value-gradient minor is \(\sigma^6\)). FACEWISE v1.1 makes the
third-order row explicit:

\[
T_4=\frac{3}{a^2}\left[
\frac{p_X(a)+p_X(-a)}2-\frac{p(a)-p(-a)}{2a}
\right],                                                  \tag{4.2}
\]

whose unit limit is \(p_{XXX}(0)\). This distinguishes the literal
average/divided-difference normalization, whose ten-by-ten determinant is
\(2\eta^2\sigma^6\), from the confluent normalization, whose determinant is
\(-24\sigma^6\), independent of \((\eta,c)\).

### 4.4 RP-S: singular-near faces

The singular-near proof preserves both anisotropic quadratic factors. At the
Euclidean contact face its covariance determinant has the form

\[
\det C_{\mathrm{sing}}^{\mathrm{BF}}
=\frac{\varrho^{10}(c^2+\varrho^2)(3c^2+\varrho^2)}{24}.   \tag{4.3}
\]

For the exact side-24 field the numerical factor is replaced by a continuous
positive factor. A mixed axial blow-up is then used near the collinear face.
The already established collinear target-mean mismatch produces the
exponential penalty needed to absorb every fixed projective power.

The conditional three-Hessian moment bound is face-compatible. It retains the
same monomial

\[
D(r,s)=[r(r+s^2)]^2(r^2+s^3)                              \tag{4.4}
\]

on the transverse, axial, endpoint, and singular-near faces. On noncollinear
charts it is a pathwise polynomial-jet envelope; on the exact axis it is a
density-weighted replacement. The false universal pointwise assertion is not
used.

Here \(s=|y-x|\) is the third-station radius, \(\varrho\) is the transverse
distance from the pair axis, and \(\Theta^2=s^2+\varrho^2\) on the
singular-near chart. On the collar charts \(\Theta\) denotes the corresponding
projective scale \(\epsilon_a\) or \(\delta_m\). The collar integrations split
the near-axis and angular-dominant overlap:

- near the axis, \(\Theta\asymp s\), and the exponential absorbs the
  \(s^{-6}/r^{-3}\) deficit;
- in the angular-dominant overlap, the deficit-free transverse estimate
  applies.

A single-chart pointwise absorption for \(\Theta\gg s\) is not asserted.
This explicit two-case display is the second FACEWISE v1.1 amendment.

The angular-axis integral has the uniform model bound

\[
\int_0^{\rho_0}\rho(s^2+\rho^2)^{-m/2}
e^{-c/(s^2+\rho^2)}\,d\rho\le C_m.                         \tag{4.5}
\]

After the value window, radial volume, determinant weights, and
\(Z_r^{-1}=O(r^{-2})\), the remaining monomials are

\[
r^7s^{-5},\;2r^6s^{-3},\;r^5s^{-2},\;r^5s^{-1},\;2r^4,
\;r^3s^2,                                                 \tag{4.6}
\]

with respective integrated orders

\[
O(r^3),\ O(r^4),\ O(r^4),\ O(r^5\log(1/r)),\ O(r^4),\ O(r^3). \tag{4.7}
\]

Thus every collar and singular-near face, including the angular axis,
contributes \(O(r^3)\). In (4.5)--(4.7), \(d\rho\) is the transverse radial
measure after the angular variables have been separated, and the subsequent
\(s\)-integration runs over the singular-near radial chart specified in
FACEWISE Sections 5 and 9.

### 4.5 Elder selection and recombination

The canonical all-witness count uses one legitimate value integration
\(\int_{b-\kappa r^3/6}^bdu\). The local collar, singular-near, and far
regions form the prescribed cover; endpoint overlaps are counted through the
reflected endpoint charts, and the elder-selection rule is applied to the
same critical-point typing as the Palm formula. Union and first-moment bounds
therefore preserve the common \(O(r^3)\) order. No extra value-window factor
is introduced, and no chart is promoted from a diagnostic grid alone.

The measurable elder mark, the disjoint elder-failure exhaustion, and the
selected/all-typed recombination are carried by Parts A.1, F.1--F.2, and
Theorem A.3.3 of the recovered 105,980-byte supplement, Drive
`1D5eRshQWyiugIrIikBUhItasLyiyV1FX`, SHA-256
`05967f14a8b8d6a9e2637210807872ba4f30810987cb7f3017e10b186d07f6da`.

### 4.6 Full-mark integration and constant identification

The compact-positive-mark estimate (T1) supplies pointwise convergence of the
selected integrand on each compact mark set, but the constant in (2.2) uses
the whole mark space. Theorem A.3.3 closes that interface with the uniform
majorant

\[
C\,[1+(|b|+\kappa)^N]e^{-c(b^2+\kappa^2)}\kappa^{-2/3}.    \tag{4.8}
\]

This is integrable on
\(\mathbb S^2\times\mathbb R\times(0,\infty)\): the
\(\kappa^{-2/3}\) singularity is integrable at zero, while the Gaussian
factor controls both noncompact tails. Truncate in \((b,\kappa)\), apply the
compact facewise limit on each truncation, and then remove the truncation by
(4.8). The off-diagonal lifetime density is uniformly \(O(1)\), hence
\(o(\ell^{-1/3})\). The selected and all-typed limits agree because
\(0\le p_r\le1\), \(p_r\to1\), and the same majorant applies. Theorem A.3.3
and the exact constant recombination in Part B of the recovered supplement
therefore close the global passage to (T2) and (2.2)--(2.7), under the same
Drive ID and SHA-256 cited above.

## 5. Exact V3.4 and FACEWISE evidence

| Role | Artifact | Drive ID | Bytes | SHA-256 / identity |
|---|---|---:|---:|---|
| Audited facewise input | `rp_c_rp_s_facewise_closure_RELAY.md` | `1rF9YAVw19fEG_QlRvHIesXev4nsegA7w` | 35,310 | `40faad08824e4e77520143078ed606c8681532cc77ae9c9d42551baa5277b400` |
| Pre-audit disposition/check record | V3.4 closure disposition | `1QJ_H5A-xg2TINB7CqHBopjvY4-AVTJdJ` | 8,157 | `d262021d7c0819ba09a6680c29379e3d731e982705a939bd60b8361e3382fbb1` |
| Full audit-replay archive | `SIDE24_V3_4_AUDIT_REPLAY_FULL_ARCHIVE.zip` | `1TSH5V0f-7BwdlHlY6-VarfuDFEQT-TIf` | 12,465,983 | `ba2a296b6a9c438eaf9f30aeaa40d4162195b256a9b2516259cd30a2468dd480` |
| Conditional-moment companion | `three_hessian_conditional_moment_envelope.md` | `1y0ZFl-S4BU7ZmBg7gDBy7ts3J7oJDduu` | 12,145 | `f19132b47dd58339d632ffe8ef5c9466dbcf9b353b406a31e22d3e35b5e5d0fb` |
| Withdrawn-route audit note | `quartic_remainder_audit_note.md` | `17pL94ArcdpMGm89kwSGui9DnJWnlH41l` | 5,830 | `e344ccec3945ecd1c931e7e84eb25ef2fe2a0c9b41f5768eb03619400b2b38db` |
| Endpoint counterexample regression | `verify_quartic_endpoint_counterexample.py` | `1EzOG80ErsPf785utTcBq7x_DIp_ZSPV1` | 13,329 | `461959368dfc7759c78ee105e4b82ff97659ddfc3a28837bbf7e8092da2f39d9` |
| Post-006b authoring-conformance carrier | `SIDE24_FACEWISE_v1.1_SHA_d5677359.md` | `1Yky0N-Y5M4HzST8-1EfPgsvYPKksnYGL` | 36,542 | `d5677359e72243a89450c14d324051948d721aa031bb729f2ce9f76fb392804e` |

The full V3.4 archive is the source for the §6X withdrawal-completeness check;
the shorter relay intentionally omits the withdrawn argument block.

FACEWISE v1.1 was authored after the 006b reconciliation and before
AO48-OPR-045. Its `d567…` hash is therefore the post-audit
authoring-conformance carrier covered by the later operator ratification. It
must not be described as the byte object Kimi originally read.

## 6. Audit and operator chain

| Stage | Record | Drive ID | Bytes | SHA-256 / qualification |
|---|---|---:|---:|---|
| Initial third-family audit reconciliation | AO48-AUD-043-v1.0 | `1wDuGvUxFnWTYe8PEIgjo6upwzjdBMV4-` | 4,818 | `b41b1e1ff37632cd3034b4bcdf73cc30b999745db2ef68b3ef28a3a1306ef0cc`; records KIMI-AUD-006 only as `4b66f85e…5a38` |
| APPROVE-conversion reconciliation and ratification package | AO48-AUD-044-v1.0 | `1qX_KJz2KreWDy7BXNmCBesUmbkJCoPBa` | 5,245 | `be6b53a21370984650733d9f00c60d0648b452fe65f32bf3ce93846f5c22737c` |
| Operator ratification | AO48-OPR-045-v1.0 | `1MfA94SaoYpnnAs7HYLNvowG9EHIR3Opk` | 3,397 | `e48d7c271dff9c120316ee2129a303aed6f78b518c2e0c08337eb357bd60e2e8` |
| Controlling status fold | GP-SIDE24-CTL-001-v1.0 | `1U8tFvVjqXjXO8Eih_SEUgy8AGBOJC_wp` | 5,117 | `ed688cb64d8ff5b6d8bb09d6d03df0668009ecbbff9bdc83e770a38a3cb86491` |
| Native status mirror | SIDE24 post-ratification status fold | `1my81FaIbVIuMhzyxg34kXuBPCGOwJah_1abJv4mJWwo` | native | mirrors the `ed688cb6…` controller |
| Native closure register | SIDE24 Closure Register R2.8 | `1Yjre5iYWJDHgxask_cuZoEFX4hXn9uCJUnVFc478K9A` | native | RP-C/RP-S/theorem state and carried dependencies |

The exact raw texts of KIMI-AUD-006 and KIMI-AUD-006b have not yet been
supplied to this Drive line. Their current corpus representations are the
AO48-AUD-043 and AO48-AUD-044 reconciliation records and the operator's signed
ratification. This package does not invent full Kimi hashes or quote text not
present in those records. When Dylan Roy supplies the raw report texts, they
should be landed as numbered, whole-file-frozen audit carriers and appended to
this chain without changing the theorem statement.

## 7. Carried dependencies — verbatim

The following is reproduced verbatim from AO48-AUD-044 §C:

> (a) The frozen V3.3 eigenfloor tables — the O(r⁻⁷) collar density origin and
> s⁻⁷ frame bounds — consistent with, but not re-derived in, the audit; same
> object class as Kimi's independently proved G.3/G.7.1 normalizer. (b) Three
> V3.4 diagnostic scripts absent from Kimi's corpus — non-blocking; every
> analytic claim they corroborate was independently recomputed. (c) The
> Palm-repair normalizer cr² ≤ Z_r ≤ Cr² (V3.3, previously audited).

Dependency (b)'s connected-Drive availability issue is now resolved; delivery
of these carriers into Kimi's corpus remains pending. The three scripts are
present byte-for-byte in Drive:

| Script | Drive ID | Bytes | SHA-256 | Replay |
|---|---:|---:|---|---|
| `verify_hybrid_mixed_curvature_and_axis_absorption.py` | `1VPYlprjcWgNmoIMiVaTL5sa6wUirseVr` | 31,533 | `0ba99085cf5b1655deb95859a63f02478057491fe1d394124d853e93bea8c481` | 202/202 PASS; normal/optimized byte-identical |
| `verify_facewise_collar_axis_integration.py` | `1a67DJHnF8NKPxDdhdKvIDpOHi0FWCeQR` | 19,538 | `bb4fdb739636e504d2c2eeafd8d8582dbf1dba12225bca88788927b049d58e5f` | 62/62 PASS in both modes |
| `diagnose_side24_generic_collar.py` | `1iiyLpIDkYR0YDiwv9vGRVQ-Rel8uh0sD` | 6,551 | `abc5b8cd3f03c251a8bf47115a9d0097ef03ae62dd65debcb23082ba973176c0` | `ALL_CHECKS_PASS` on its declared grid |

Replay environment: Python 3.12.13; SymPy 1.14.0; mpmath 1.3.0; NumPy
2.4.4; SciPy 1.17.1. These scripts are regression and ledger evidence within
their own scope statements. They do not replace the analytic
Fourier-independence, compact-atlas, regression, conditional-moment, or
Kac--Rice arguments.

The specific historical V3.3 eigenfloor-table carrier and its individual
whole-file hash are not exposed by the current connected-Drive index. The
tables are carried inside the full V3.4 replay archive, Drive
`1TSH5V0f-7BwdlHlY6-VarfuDFEQT-TIf`, whose envelope is frozen at 12,465,983
bytes and SHA-256
`ba2a296b6a9c438eaf9f30aeaa40d4162195b256a9b2516259cd30a2468dd480`.
No standalone table hash is invented here. The mathematical
\(c r^2\le Z_r\le C r^2\) normalizer is also stated in the historical V3.3
checkpoint; its independently citable proof is Theorem G.3.1 in the recovered
105,980-byte supplement, Drive `1D5eRshQWyiugIrIikBUhItasLyiyV1FX`, SHA-256
`05967f14a8b8d6a9e2637210807872ba4f30810987cb7f3017e10b186d07f6da`.
The absence of a separately indexed V3.3 checkpoint hash is a carrier-level
limitation preserved by this package, not a new mathematical claim.

## 8. Failed approaches and correction ledger

The package preserves the unsuccessful routes because they define the proof's
current safety boundaries.

1. **Mirror-sign error in the triple-contact display.** The printed
   \(-\det H_S\) cross-term originally had the wrong sign. After the third
   witness-value pin is restored, that sign controls the \(\eta^2\)
   cancellation and is load-bearing. The corrected negative sign is used in
   V3 and later.
2. **Raw degree and collision overstatement.** The three-factor determinant
   product has generic degree six in \(\kappa\), but the degree drops on
   coefficient-degeneracy faces \(2c=\pm\eta\). Those faces are not geometric
   collisions unless the transverse coordinate also vanishes.
3. **Far-density order.** A claimed extra \(O(\ell)\) density factor was
   withdrawn; the correct fixed-distance density is \(O(1)\). The sole value
   window supplies the needed \(O(r^3)\).
4. **Capture/escape threshold and tube cost.** An earlier deterministic
   threshold and a tube-cost count were insufficient. The G.7.1 route uses
   the corrected geometry and weighted Palm budgets.
5. **Unpinned midpoint subtraction.** A singular-near value row subtracted a
   random midpoint value not contained in the pin block. It was replaced by a
   pair-measurable residual and the \(s^{-7}\) normalized frame.
6. **Universal pointwise Hessian envelope.** The literal all-face
   pointwise curvature route fails on the exact axis. It was withdrawn and
   replaced by the hybrid monomial envelope: pathwise off-axis and
   density-weighted on-axis.
7. **Quartic endpoint remainder.** A purported uniform quartic remainder has
   an endpoint counterexample. The withdrawn §6X block is retained only as
   provenance; the live proof uses EP.1/EP.2 and the endpoint measure ledger.
8. **Confluent-row ambiguity.** The literal 10-by-10 average/divided-difference
   reading yields \(2\eta^2\sigma^6\); it cannot be cited for the
   parameter-free
   coefficient. FACEWISE v1.1 writes the confluent row (4.2) explicitly.
9. **Single-chart axial absorption.** Pointwise absorption fails when
   \(\Theta\gg s\). FACEWISE v1.1 displays the near-axis and
   angular-dominant regimes separately.
10. **Independent-audit overreach.** Early local checks supported only the
    components they exercised; they did not close RP-A/RP-L, RP-C/RP-S,
    RP-F, or elder recombination. The final status rests on the complete
    facewise proof, the third-family audit/reconciliation chain, and operator
    ratification.
11. **Fail-open reproduction scripts.** Bare Python `assert` statements and
    scripts that printed FAIL while returning status zero were treated as
    non-evidentiary for closure. Later verifiers use explicit fail-closed
    checks and optimized-mode transcript comparisons.
12. **G.9.1 pin-block factorization.** The historical polynomial
    factorization of the pin determinant was false. The corrected planar
    determinant was independently reconstructed as
    \((P+gU)(2gP^2+2vUP+gvU^2)\), where
    \(q=e^{-d^2/2}\), \(U=u_1^2\), \(P=u_2^2+u_3^2\),
    \(g=\operatorname{Var}(f_{xy}\mid\text{pins})\), and
    \(v=\operatorname{Var}(f_{xx}\mid\text{pins})\). The explicitly labeled
    reconstruction and its 26/26 verifier are indexed by the SIDE24 G.9/R2
    Recovery Index, Drive
    `1yQQfid0m0U7Wwu0_e0zr663hrAflZy0srRZGkT0tPyg`. That recovery is
    supporting local covariance evidence, not a substitute for the facewise
    side-24 theorem.

## 9. Reopening rule

The ratified result reopens only upon one of the following:

1. an exact counterexample to an audited display;
2. failure of a carried V3.3 eigenfloor table; or
3. a landed diagnostic script contradicting a claim it was said to
   corroborate.

An unavailable raw report, a stale historical HOLD banner, or a diagnostic
script's absence from a reviewer's corpus is a preservation or routing issue,
not by itself a mathematical counterexample. Such issues must nevertheless be
recorded honestly, as the Kimi raw-carrier gap is recorded here.

## 10. Track firewall

Nothing in this package changes the q0/P0.1 program. GP-DER-118 and
LS-CTL-003-v1.1 remain controlling there; PZ0/N1/E0/E2/D0/I0 and the
independence lanes remain open, LB-RATE remains measured-grade, and P0.1
remains HOLD. No q0 sealing, release, or cross-track promotion is authorized.
The blind \(d^3/48\) midpoint-reflection result remains frozen and untouched
pending the pre-registered Kimi Task-6 comparison.

## 11. External-review route

An external specialist can review the result in this order:

1. verify the operator scope and status in AO48-OPR-045 and this package's
   separate freeze receipt (a file cannot contain its own whole-file hash);
2. read G.7.1 and G.8 in the recovered 105,980-byte supplement;
3. read the audited 35,310-byte V3.4 facewise input and the
   12,145-byte three-Hessian envelope;
4. inspect FACEWISE v1.1 only for the two authoring-conformance amendments,
   using the provenance distinction in §5;
5. audit the §6X withdrawal against the full V3.4 archive and the quartic
   counterexample pair;
6. replay the three diagnostic carriers and the endpoint counterexample under
   the pinned environment;
7. inspect AO48-AUD-043, AO48-AUD-044, and AO48-OPR-045; then append the
   direct KIMI-AUD-006/006b carriers when supplied;
8. keep the three carried dependencies in §7 visible when assessing the
   theorem's stated scope.

This is an export and routing surface. The frozen mathematical sources remain
the authoritative proof bodies.

## 12. Package-local controllers

- Raw post-ratification status fold: Drive
  `1U8tFvVjqXjXO8Eih_SEUgy8AGBOJC_wp`, SHA-256
  `ed688cb64d8ff5b6d8bb09d6d03df0668009ecbbff9bdc83e770a38a3cb86491`.
- Native status mirror: Drive
  `1my81FaIbVIuMhzyxg34kXuBPCGOwJah_1abJv4mJWwo`.
- Diagnostic handoff: Drive `1dolk9RcGWt7vRNopLmbfDXZi0yv8upjI`,
  SHA-256
  `6aef6bd213876245e74040700370c8dc632fab9efef9296143739a8b55c0984e`.
- Raw closure register R2.8: Drive
  `1isOnrX0VG9QeLOTAIrinArHPyzZZu2Sa`, SHA-256
  `fbfa7f8f095a82e9a8a0c62401d73dd6101252db93411149a788ac6b4751e43f`.
- Native closure register R2.8: Drive
  `1Yjre5iYWJDHgxask_cuZoEFX4hXn9uCJUnVFc478K9A`.
- Coupled research registers: Drive
  `1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no`; SIDE24 append rows are
  recorded without changing any q0 row.

**End of SIDE24 THEOREM PACKAGE v1.0.**
