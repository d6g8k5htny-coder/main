# External Review Package  
## Elder-rule pairing, Gaussian transversality, and near-diagonal persistence for the Bargmann–Fock field

### Purpose

This document set asks for an independent mathematical review of a candidate body of results about superlevel-set \(H_0\) persistence for the two-dimensional Bargmann–Fock Gaussian field.

The package has been rewritten for a mathematician. It does not assume access to a research archive, software repository, conversation history, or any prior project-specific terminology. Definitions, hypotheses, proof arguments, numerical observations, and known limitations are included in the files themselves.

The principal questions are not “is the exposition impressive?” or “does the computational evidence look plausible?” They are:

1. **Does the conditional finite-\(r\) pairing theorem follow from the stated Kac–Rice and local-fold estimates?**
2. **Does the Cameron–Martin transversality argument really prove that saddle–saddle heteroclinic connections have probability zero?**
3. **Does the proposed first-moment argument really imply the \(\ell^{-1/3}\) near-diagonal lifetime density?**
4. **Does the local-Palm/percolation argument justify the claimed thermodynamic stabilization?**
5. **Which statements are theorems as written, which are repairable, and which should remain conditional?**

A useful review should identify the **first unjustified or false step**, not merely state general unease.

---

## Files and review tracks

### `01_CORE_PAIRING_THEOREM.md`

A self-contained candidate manuscript for the fixed-torus elder-rule selection problem.

It defines:

- the exact normalized periodized Bargmann–Fock field on \(\mathbb T_{24}^2\);
- the typed and adjacent maximum–saddle Palm laws;
- the selection probability \(q(r)\);
- the deterministic elder-rule reduction;
- the Kac–Rice count objects;
- the corrected collapsing-pair coordinates;
- the local cubic identities used in the singular regions;
- the claimed \(O(r^3)\) bound.

**Best suited to:** Gaussian-field, Kac–Rice, Palm, random Morse theory, and persistence specialists.

**Highest-priority questions:** 1–8 in `06_REVIEW_RESPONSE_FORM.md`.

---

### `02_GAUSSIAN_TRANSVERSALITY.md`

A self-contained proposed proof that the Bargmann–Fock gradient flow is almost surely Morse–Smale in the only sense needed here: no trajectory connects two distinct index-one saddles.

The proof uses:

- a section mismatch functional;
- its Cameron–Martin derivative;
- an RKHS Riesz representative;
- injectivity of the Bargmann–Fock kernel embedding;
- a countable family of shift directions;
- one-dimensional Gaussian disintegration.

The functional derivative and support-separation argument are written explicitly. The charting, measurability, and slicing step is the most delicate part and is presented with its exact intended logic.

**Best suited to:** dynamical systems, transversality, Gaussian measures, Malliavin/Cameron–Martin methods.

**Highest-priority questions:** 9–14 in the response form.

---

### `03_NEAR_DIAGONAL_LIFETIME_LAW.md`

A self-contained derivation of the proposed law

\[
\nu_B(\ell)\sim C_B\,\ell^{-1/3}.
\]

It separates:

- an exact first-moment identity;
- an exact fold Jacobian;
- the candidate-pair contact limit;
- the adjacency mark;
- the elder-rule selection factor;
- modulus-tail domination;
- the numerical validation, including a failed first experiment.

Some of the contact-adjacency and uniformity arguments are proof sketches rather than fully expanded estimates. They are labeled as such.

**Best suited to:** random topology, Kac–Rice point processes, Palm theory, persistence asymptotics.

**Highest-priority questions:** 15–19.

---

### `04_THERMODYNAMIC_AND_PERCOLATION_EXTENSION.md`

A proposed extension from the fixed torus to the planar field and large tori.

It defines a merge-tree influence region, proves the deterministic information-localization statement, writes an exact same-field/cross-fit Campbell identity, and sketches a finite-rank Gaussian transfer from known subcritical Bargmann–Fock percolation estimates to the pair-Palm law.

**Best suited to:** Gaussian percolation, continuum percolation, Palm methods, stabilization theory.

**Highest-priority questions:** 20–24.

---

### `05_RELIABILITY_AND_EVIDENCE.md`

A reliability dossier containing:

- a claim-by-claim status table;
- exact versus measured quantities;
- computational checks;
- failed methods and withdrawn claims;
- common-mode limitations;
- a bibliography.

This file is intended to help the reviewer decide how much weight to place on each part of the work.

---

### `06_REVIEW_RESPONSE_FORM.md`

A response template designed to produce an actionable review in an afternoon.

A reviewer need not read every track. A useful response can focus on one specialty, provided it states which claims were checked.

---

## Main claims, in conservative language

### Claim A: fixed-torus selection rate, conditional form

For the exact normalized periodized Bargmann–Fock field on \(\mathbb T_{24}^2\), fix

\[
b=\frac65,\qquad
M=(-r/2,0),\qquad
S=(r/2,0),\qquad
f(M)-f(S)=\frac{r^3}{6}.
\]

Under an adjacent typed maximum–saddle Palm law, and assuming the field is Morse–Smale with distinct critical values, the proposed theorem is that there exists a finite constant \(C\) such that

\[
1-q(r)\le Cr^3,\qquad 0<r\le0.025.
\]

Consequently \(q(r)\to1\).

The package does **not** claim a numerical value of \(C\).

### Claim B: Gaussian transversality

For the Bargmann–Fock field, the proposed transversality theorem is that almost surely no gradient trajectory has two distinct index-one saddles as its endpoints.

If valid, this removes the Morse–Smale assumption in Claim A.

### Claim C: near-diagonal lifetime density

For a compact birth-height interval \(B\), the proposed first-moment density of non-essential superlevel \(H_0\) lifetimes satisfies

\[
\nu_B(\ell)=C_B\ell^{-1/3}(1+o(1)),
\qquad 0<C_B<\infty.
\]

The exponent follows from an exact polar-coordinate Jacobian once the candidate-pair contact intensity and uniform selection statements are justified.

No numerical value of \(C_B\) is claimed.

### Claim D: fixed-positive-height stabilization

For fixed positive birth heights and normalized gaps, the proposed thermodynamic statement is that the configured-pair selection probability on \(\mathbb T_L^2\) converges to a planar value as \(L\to\infty\), with an exponentially small finite-size error, and the limiting defect probability remains \(O(r^3)\).

This claim uses published subcritical Bargmann–Fock percolation results plus a new local-Palm transfer argument.

---

## Statements deliberately not claimed

The package does **not** claim:

- any certified finite numerical upper or lower coefficient;
- universality over all smooth Gaussian fields;
- a numerical value of the near-diagonal coefficient;
- an exact power law at finite lifetime;
- an exact critical susceptibility exponent;
- that the numerical experiments prove any theorem;
- that any LLM application has been demonstrated on a real model.

Several earlier numerical constants were withdrawn after their uncertainty or dependence structure could not be reconstructed. The reliability file lists them explicitly.

---

## The requested deliverable from the reviewer

Please return one of the following for each track reviewed:

1. **Valid as written.**
2. **Valid after a local repair**, with the repair stated.
3. **Only valid conditionally**, with the missing hypothesis stated.
4. **Not valid**, identifying the first false or unjustified step.
5. **Outside my specialty**, but with any obvious concerns noted.

The most valuable response is a short mathematical memorandum answering the numbered questions in `06_REVIEW_RESPONSE_FORM.md`.

A reviewer who has only two or three hours should prioritize:

- the adjacent-versus-typed Palm measure distinction in File 01;
- the countable Cameron–Martin slicing step in File 02;
- the contact-adjacency limit in File 03;
- the conditional-to-unconditioned percolation transfer in File 04.
