# PR-SP-001 — Fixed finite trace moments cannot certify positivity in arbitrary dimension

Date: 2026-09-16. Complete elementary author-side proof; standard moment-nonuniqueness phenomenon, not a new Riemann Hypothesis result and not a claimed barrier for every RH method.

## The theorem

For every integer d>=1 there are real diagonal matrices A_+ and A_- of the SAME finite dimension with

tr(A_+^j)=tr(A_-^j), 0<=j<=d,

such that A_+ is positive definite and A_- has a negative eigenvalue.

Proof. Let m=d+1, nodes i=1,...,m, and define integer Lagrange weights

ell_i=(-1)^(i-1) i binom(m+1,i+1).

Interpolation of polynomials of degree at most m-1 at x=-1 gives

sum_(i=1)^m ell_i i^j=(-1)^j, 0<=j<=d.

This weight formula follows directly from product_(h != i)(-1-h)/(i-h). Let L=max_i |ell_i|. In A_+, give each positive eigenvalue i multiplicity 2L. In A_-, give i multiplicity 2L-ell_i and add one eigenvalue -1. All multiplicities are positive integers. Since sum ell_i=1, both dimensions are 2mL. Subtracting the traces and using the displayed identity gives equality for every j<=d. Their positivity differs. QED.

## Small explicit example

For d=2, ell=(6,-8,3), L=8, and dimension=48:

A_+: eigenvalues 1,2,3 each 16 times.
A_-: eigenvalue -1 once, 1 ten times, 2 twenty-four times, 3 thirteen times.

Both have trace 96 and squared trace moment tr(A^2)=224, but only A_+ is positive definite.

## What this rules out

A proposed positivity certificate based only on a fixed number of ordinary spectral moments, with no extra structure and arbitrary matrix dimension, is insufficient. The theorem does not construct two actual zeta/Weil operators and does not disprove any RH-equivalent criterion. Dimension grows with the number of matched moments; for a fixed known dimension enough power sums can determine a characteristic polynomial.

The 2026 Lamzouri paper's own Remark 3.4 discusses a DIFFERENT, method-specific test-function ceiling. That sourced ceiling and this elementary diagnostic must not be conflated. No RH progress percentage or improved critical-line proportion was proved in this campaign.
