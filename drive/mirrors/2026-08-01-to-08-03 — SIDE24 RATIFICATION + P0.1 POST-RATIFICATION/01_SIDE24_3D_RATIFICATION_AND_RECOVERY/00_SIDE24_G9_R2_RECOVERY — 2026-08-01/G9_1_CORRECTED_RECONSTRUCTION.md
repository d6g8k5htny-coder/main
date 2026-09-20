# G.9.1 corrected planar collar determinant — independent reconstruction

**Artifact class:** reconstruction, not a byte-copy of the missing sandbox excerpt  
**Scope:** planar three-dimensional Bargmann–Fock field with covariance
\(K(x-y)=\exp(-|x-y|^2/2)\)  
**Purpose:** restore the mathematical content reviewed in AO48-AUD-033 and
replace the refuted \(f_{xx}\)-sector formula with a compact exact identity

## 1. Conditioning problem

Let
\[
M=(0,0,0),\qquad S=(d,0,0),\qquad d>0,\qquad q=e^{-d^2/2}.
\]
Condition on the eight pins
\[
\bigl(f,\nabla f\bigr)(M)=0,
\qquad
\bigl(f,\nabla f\bigr)(S)=0.
\]
The numerical values of the pins do not affect the conditional covariance.
For \(u=(u_1,u_2,u_3)\ne0\), put
\[
\Gamma_d(u)=\operatorname{Cov}\!\left(Hf(M)u\mid
 (f,\nabla f)(M),(f,\nabla f)(S)\right),
\]
and abbreviate
\[
U=u_1^2,\qquad P=u_2^2+u_3^2.
\]

## 2. Exact scalar coefficients

Define
\[
g(d)=\frac{d^2q^2+q^2-1}{q^2-1}
\tag{G9.1a}
\]
and
\[
v(d)=
\frac{
-d^6q^2+d^4q^4+d^4q^2+4d^2q^4-4d^2q^2
+2q^4-4q^2+2
}{
(-d^2q+q^2-1)(d^2q+q^2-1)
}.
\tag{G9.1b}
\]
Then
\[
g(d)=\operatorname{Var}(f_{xy}(M)\mid\text{pins}),
\qquad
v(d)=\operatorname{Var}(f_{xx}(M)\mid\text{pins}).
\]
Both are strictly positive for \(d>0\). Strict positivity follows directly
from linear independence of the corresponding derivative evaluation
functionals in the Bargmann–Fock reproducing-kernel Hilbert space; it is also
visible as positivity of the relevant Schur complements.

The four-dimensional axial pin block, ordered as
\((f_M,f_S,f_{x,M},f_{x,S})\), is
\[
\Sigma_x=
\begin{pmatrix}
1&q&0&-dq\\
q&1&dq&0\\
0&dq&1&(1-d^2)q\\
-dq&0&(1-d^2)q&1
\end{pmatrix},
\]
with the correct determinant factorization
\[
\det\Sigma_x=(-d^2q+q^2-1)(d^2q+q^2-1).
\tag{G9.1c}
\]

For
\[
c_\perp=(-1,-q,0,dq)^T,
\]
fraction-free elimination gives the exact polynomial identity
\[
c_\perp^T\operatorname{adj}(\Sigma_x)c_\perp=\det\Sigma_x.
\tag{G9.1d}
\]
Consequently, after conditioning, the transverse Hessian block
\((f_{yy},f_{yz},f_{zz})\) has covariance
\(\operatorname{diag}(2,1,2)\), independently of \(d\), and
\[
\operatorname{Cov}(f_{xx},f_{yy}\mid\text{pins})
=\operatorname{Cov}(f_{xx},f_{zz}\mid\text{pins})=0.
\tag{G9.1e}
\]

## 3. Correct determinant theorem

The exact determinant is
\[
\boxed{
\det\Gamma_d(u)
=(P+gU)\bigl(2gP^2+2vUP+gvU^2\bigr).
}
\tag{G9.1f}
\]
This identity is rotationally invariant about the pin axis and is strictly
positive for every \(d>0\) and \(u\ne0\).

To derive it, write the conditioned Hessian entries as
\(A=f_{xx}, B=f_{xy}, C=f_{xz}, D=f_{yy}, E=f_{yz}, F=f_{zz}\).
Parity and (G9.1d)–(G9.1e) give
\[
\operatorname{Var}(A)=v,
\quad
\operatorname{Var}(B)=\operatorname{Var}(C)=g,
\quad
\operatorname{Var}(D,E,F)=(2,1,2),
\]
with all cross-covariances zero. Hence
\[
\Gamma_d(u)=
\begin{pmatrix}
vU+gP & gu_1u_2 & gu_1u_3\\
gu_1u_2 & gU+2u_2^2+u_3^2 & u_2u_3\\
gu_1u_3 & u_2u_3 & gU+u_2^2+2u_3^2
\end{pmatrix},
\]
whose determinant is (G9.1f).

On the transverse face \(U=0\), this reduces to the previously verified
formula
\[
\det\Gamma_d(u)=2g(d)P^3.
\tag{G9.1g}
\]
On the pin axis \(P=0\), it becomes
\[
\det\Gamma_d(u)=g(d)^2v(d)U^3.
\tag{G9.1h}
\]

## 4. Coalescence stratification

As \(d\downarrow0\),
\[
g(d)=\frac{d^2}{2}-\frac{d^4}{12}+\frac{d^8}{720}+O(d^{10}),
\qquad
v(d)=\frac{d^4}{6}-\frac{d^6}{30}+\frac{d^8}{360}+O(d^{10}).
\tag{G9.1i}
\]
Therefore every direction has \(\det\Gamma_d(u)\to0\). More precisely,
transverse and generic directions carry a leading \(d^2\) scale, while the
axis carries
\[
g(d)^2v(d)=\frac{d^8}{24}-\frac{d^{10}}{45}+O(d^{12}).
\tag{G9.1j}
\]
This confirms AO48-AUD-033's diagnosis: the missing divided-difference pins
capture the full axial Hessian row in the confluent limit, so an \(O(1)\)
generic-direction limit is impossible.

## 5. Acceptance fixtures

The reconstructed formula reproduces the independent 60-digit conditioning
fixtures:

| \(d\) | \(u\) | \(\det\Gamma_d(u)\) |
|---:|:---:|:---|
| 0.1 | (1,0,0) | 4.14449071554625894308856970184054374284558536881426e-10 |
| 1 | (1,1,1) | 9.53941731866226576797363743769172724627000946554358636 |
| 2 | (0.9,0.1,0.4) | 0.976024399862822005117691369544350648555032721656323201 |
| 1 | (0,1,0) | 0.836046586261347151229995989781976882906261397849207727 |

The companion `verify_g9_1_reconstruction.py` derives the pin block and Schur
complements independently, proves the polynomial identities symbolically,
recomputes all four fixtures at 80 decimal digits, checks the small-\(d\)
series, and exits nonzero on any failure. Its normal and optimized-mode
transcripts must be byte-identical.

## 6. Lineage and limits

- The original V5 G.9 source and `verifier/v8` tree were recovered byte-for-byte
  from the operator-provided ZIP with SHA-256
  `119ffbfae2d11cc2d3de60f15c06a653c48b17d06ab0a2e125260ab0f1458c5a`.
- The later sandbox G.9.1 excerpt itself was relayed into AO48-AUD-033 but is not
  present as a byte-recoverable standalone file. This document is therefore a
  reconstruction and does not claim the original sandbox hash or wording.
- The refuted closed form in AO48-AUD-033 is not reinstated. Equation (G9.1f)
  is the corrected replacement derived independently from the covariance
  definition.
- This planar identity is a local model. It does not, by itself, prove the
  uniform side-24 collar/singular-near estimates. Those are treated separately
  in the V3.4 RP-C/RP-S facewise closure checkpoint, which remains on HOLD for
  independent expert review.

