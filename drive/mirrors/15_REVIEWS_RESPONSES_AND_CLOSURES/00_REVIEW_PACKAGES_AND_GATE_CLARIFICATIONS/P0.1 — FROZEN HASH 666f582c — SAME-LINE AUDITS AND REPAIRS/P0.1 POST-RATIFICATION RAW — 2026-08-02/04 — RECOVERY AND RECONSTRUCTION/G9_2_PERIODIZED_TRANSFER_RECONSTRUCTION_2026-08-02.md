# G.9.2 side-24 periodized transfer — reconstructed replacement

**Date:** 2026-08-02  
**Artifact class:** RECONSTRUCTED replacement; not a recovered byte-copy  
**Authority:** none  
**Canonical impact:** none; RP-C/RP-S and the package HOLD are unchanged

## 1. Provenance and scope firewall

The original sandbox carrier and its source
`verify_g9_periodized_transfer_v1.py` have not been recovered.  This note and
its companion verifier reconstruct the transfer from the mathematical
definition, the corrected planar G.9.1 carrier, and the finite-image mechanism
recorded in AO48-DER-037.  They do **not** claim the original filename, bytes,
hash, wording, run number, or R2.5 provenance.

Sources used:

- `G9_1_CORRECTED_RECONSTRUCTION.md` and its fail-closed verifier;
- AO48-DER-037-v1.0 (Drive ID `123Tug63LVAWEYTJQqwBToj8DNAlqgDxR`),
  especially its nearest-image distance (23.5) and (10^{-100}) reserve;
- the general Peano/divided-difference image-bound mechanism in the non-Kimi
  GP-DATA-214 source carrier (Drive ID
  `1ohTtnu7o2sbbcwTfoNTBte9TPLbC2vFlPd9raoDutE0`).

No Kimi implementation, Kimi Task 5/6 output, or Kimi G.9.2 carrier was read or
used.  This is a blind reconstruction relative to that line.

The declared scope of this replacement is

\[
  0<d\le \frac12,\qquad |u|=1,
\]

for the two first-jet pin sites (M=(-d/2,0,0)) and
(S=(d/2,0,0)).  The endpoint (d=1/2) is exactly what yields the
AO48-DER-037 nearest-image distance (24-1/2=23.5).  The result transfers the
contact covariance

\[
 \Gamma_d(u)=\operatorname{Cov}(Hf(M)u\mid(f,\nabla f)(M),(f,\nabla f)(S))
\]

from the planar Bargmann--Fock kernel to the normalized side-24 periodized
kernel.  It does not address the finite-ρ remainder, the axial mean-mismatch
penalty, the conditional three-Hessian moment envelope, or regional
integration.

## 2. Kernels and the stable pin frame

Write

\[
 g(t)=e^{-t^2/2},\qquad
 k_{24}(t)=\frac{\sum_{n\in\mathbb Z}g(t+24n)}
                  {\sum_{n\in\mathbb Z}g(24n)}.
\]

The normalized three-dimensional covariance factorizes as

\[
 K_{24}(z)=k_{24}(z_1)k_{24}(z_2)k_{24}(z_3).
\tag{G9.2.1}
\]

Raw endpoint pins are ill-conditioned as (d\downarrow0), so no inverse bound
is taken in that frame.  Put (h=d/2) and replace the axial pins by

\[
\begin{aligned}
 L_0&=\tfrac12(f(-h)+f(h)),\\
 L_1&=\frac{f(h)-f(-h)}d,\\
 L_2&=\frac{f_x(h)-f_x(-h)}d,\\
 L_3&=\frac{12}{d^2}\left[
       \tfrac12(f_x(-h)+f_x(h))-\frac{f(h)-f(-h)}d\right].
\end{aligned}
\tag{G9.2.2}
\]

For each transverse coordinate (a\in\{y,z\}), use

\[
 T_{a,0}=\tfrac12(f_a(-h)+f_a(h)),\qquad
 T_{a,1}=\frac{f_a(h)-f_a(-h)}d.
\tag{G9.2.3}
\]

For every (d>0) this is an invertible change of pin coordinates.  At
contact it tends to the nonsingular jet frame

\[
 (f,f_x,f_{xx},f_{xxx},f_y,f_{xy},f_z,f_{xz})(0).
\]

The Peano orders of (G9.2.2)--(G9.2.3) in the axial variable are
(0,1,2,3,0,1,0,1), and their constants are
(1,1,1,2,1,1,1,1).  Thus a pin--pin covariance uses at most six axial and
four transverse derivatives, with Peano-constant product at most four.
For an output row (Hf(M)u), ℓ1 coefficients are at most
√3<2, so the same product bound four covers pin--output and
output--output entries.

## 3. Exact rational all-image envelope

For |t|≤1/2 and n≠0,

\[
 |t+24n|\ge23.5|n|,\qquad |t+24n|\le25|n|.
\tag{G9.2.4}
\]

For every probabilists' Hermite polynomial of order (m\le6), direct exact
coefficient comparison at (23.5=47/2) gives

\[
 |\operatorname{He}_m(x)|\le2|x|^6\quad(|x|\ge23.5).
\tag{G9.2.5}
\]

The exact partial sum

\[
 e>\sum_{j=0}^{6}\frac1{j!}=\frac{1957}{720}
\]

and one integer-power comparison give

\[
 e^{-2209/8}<\frac5{4\cdot10^{120}}=:A.
\tag{G9.2.6}
\]

Since (n^6A^{n^2}) has a geometrically negligible tail after (n=1), the
verifier proves exactly

\[
 \sum_{n\ge1}n^6e^{-(23.5n)^2/2}<2A.
\]

Equations (G9.2.4)--(G9.2.6) therefore give the one-dimensional unnormalized
image bound

\[
 E_1=8\,25^6A
 =2.44140625\times10^{-111}.
\tag{G9.2.7}
\]

On |t|≤1/2 the planar derivatives through order six are bounded by 15;
this is checked exactly from the Hermite polynomials and
(\operatorname{He}_m'=m\operatorname{He}_{m-1}).  Normalization at zero
therefore changes a one-dimensional derivative by at most

\[
 \delta_1=16E_1=3.90625\times10^{-110}.
\tag{G9.2.8}
\]

Expanding the product (G9.2.1) one factor at a time gives a three-dimensional
derivative-entry error at most

\[
 E_3=721\delta_1=2.81640625\times10^{-107}.
\]

After the Peano/output coefficient product four, every entry in the joint
stable-pin/output covariance differs from its planar counterpart by at most

\[
 \eta=4E_3=1.1265625\times10^{-106}.
\tag{G9.2.9}
\]

All numbers in this section are exact rational upper bounds, not rounded
lattice truncations.

## 4. Uniform Schur transfer

The planar stable-pin Gram matrix is block diagonal.  Its axial even and odd
blocks have determinants and traces satisfying, uniformly for
(x=d^2/2\in(0,1/8]),

\[
 \det P_{\rm even}\ge2-2x\ge\frac74,
 \qquad \operatorname{tr}P_{\rm even}\le4,
\]

\[
 \det P_{\rm odd}\ge6-6x\ge\frac{21}{4},
 \qquad \operatorname{tr}P_{\rm odd}\le16.
\]

These are alternating-series bounds with decreasing terms.  The two
transverse blocks have diagonal entries at least (15/16).  Consequently

\[
 \lambda_{\min}(P_{\rm pl}(d))\ge\frac{21}{64},
 \qquad \|P_{\rm pl}(d)^{-1}\|\le\frac{64}{21}.
\tag{G9.2.10}
\]

The verifier checks the exact coefficient formulas and the induction
inequalities that make the alternating estimates uniform.

Let (P,C,S) be the stable pin, pin--output, and output covariance blocks.
Entrywise (G9.2.9) implies

\[
 \|\Delta P\|\le8\eta,
 \qquad \|\Delta C\|\le5\eta,
 \qquad \|\Delta S\|\le3\eta.
\]

The planar output covariance has norm 3.  Positivity of the joint covariance,
(G9.2.10), and a Neumann estimate give the deliberately rounded bounds

\[
 \|P_{24}^{-1}\|<4,
 \quad \|C_{\rm pl}\|<7,
 \quad \|C_{24}\|<9,
 \quad \|P_{24}^{-1}-P_{\rm pl}^{-1}\|<128\eta.
\]

Substitution in

\[
 \Gamma=S-C^TP^{-1}C
\]

yields

\[
 \boxed{
 \|\Gamma_{24,d}(u)-\Gamma_{\infty,d}(u)\|
 <8400\eta
 =9.463125\times10^{-103}.
 }
\tag{G9.2.11}
\]

Both conditional covariances have norm below four.  The 3×3 determinant
Lipschitz bound then gives

\[
\boxed{
 |\det\Gamma_{24,d}(u)-\det\Gamma_{\infty,d}(u)|
 <403200\eta
 =4.5423\times10^{-101}<10^{-100}.
}
\tag{G9.2.12}
\]

This proves the AO48-DER-037 transfer threshold on the declared collar range,
including the confluent endpoint (d\downarrow0); no raw-pin inverse or
compactness shortcut is used.

## 5. Independent numerical regression

The companion verifier also reconstructs both fields directly at 180 decimal
digits.  It evaluates the side-24 kernel from image indices
(-2\le n\le2) in each one-dimensional factor; (G9.2.7) dominates all
omitted images.  It conditions in the stable frame, independently evaluates
the corrected planar G.9.1 closed form, and tests axial, transverse, generic,
and skew directions at (d=0.5,0.25,0.1,0.01,0.001).

At the diagnostic probes, the largest observed covariance-entry discrepancy
is in the (10^{-116}) class and the largest observed determinant discrepancy
is in the (10^{-116}) class.  Those observations are regression evidence;
the all-(d), all-(u) conclusion is carried by (G9.2.7)--(G9.2.12).

## 6. What this reconstruction does and does not close

Established here:

- exact all-image derivative control on (0<d\le1/2);
- a confluent-stable pin frame and uniform planar pin floor;
- side-24 conditional covariance transfer;
- side-24 determinant transfer below (10^{-100});
- a fail-closed, optimization-safe, dependency-free replay.

Not established here:

- byte recovery of the original G.9.2 note or
  `verify_g9_periodized_transfer_v1.py`;
- any assertion about runs 15--17 or an R2.5 addendum;
- the finite-ρ G.9.x remainder, axial suppression, endpoint charts,
  singular-near estimates, Hessian moment envelope, or (O(r^3)) integration;
- promotion of RP-C/RP-S or release of the package HOLD.


