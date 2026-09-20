# C092 PUBLICATION AND CLAIM-LANGUAGE STANDARD

## 1. Canonical theorem wording

### Internal technical wording

> For the exact normalized periodized Bargmann–Fock field on
> \(\mathbb T_{24}^2\), at \(b=6/5\), under the typed maximum–saddle
> pair-Palm law with \(\ell=r^3/6\), the q₀ Rate Program establishes at
> program/verification grade
> \[
> 0.8411r^3\le1-q(r,6/5)\le4.3r^3,
> \qquad0<r\le0.025.
> \]
> Consequently \(q(r,6/5)\to1\).

### External conservative wording

> We give a verification-grade theorem package for a two-sided cubic
> pairing-defect rate in a fixed-volume periodized Bargmann–Fock model. The
> package combines exact structural reductions, independent implementation
> checks, station-certified constants, and measured-modulus uniformity
> controls. Source-level marked-density details, interval replacement of
> measured moduli, and the SARD-G appendix are separately identified for
> external specialist review.

### Measured wording

> At the reference rung \(r=0.025\), the direct measured normalized defect
> coefficient is \(0.946\), with the reported interval \([0.941,0.951]\).

Do not call this value an asymptotic or scale-free truth constant.

---

## 2. Mandatory qualifiers

Any use of the numerical theorem must retain:

```text
model:
    exact normalized periodized Bargmann-Fock field

volume:
    fixed L=24

height:
    b=6/5

law:
    typed maximum-saddle pair-Palm

window:
    ell=r^3/6

radius:
    0<r<=0.025

grade:
    program/verification grade
```

---

## 3. Statements not licensed by the core theorem

The core theorem does not establish:

- all smooth stationary Gaussian fields;
- anisotropic universality with the same constants;
- infinite-volume uniformity;
- a joint \(L,r\) limit;
- a critical-height scaling exponent;
- the full near-diagonal \(\ell^{-1/3}\) density theorem without its separate
  contact-neutrality and normalization conditions;
- the C089 \(0.97\) upper coefficient through \(r=0.05\);
- a finite \(0.8501\) lower coefficient through \(r=0.05\);
- a production LLM factuality guarantee;
- exponential hallucination reduction without coverage and selection terms.

---

## 4. Novelty language

Allowed:

> We found no close match in the targeted prior-art search for the specific
> quantitative elder-rule pairing-defect law under the pinned fold-pair Palm
> conditioning used here.

Not allowed:

> This is the first theorem of its kind.

unless a new, explicit, exhaustive-enough prior-art protocol supports that
stronger statement.

The program does not claim novelty for:

- Morse theory;
- elder-rule persistence;
- Kac–Rice formulas;
- Gaussian regression;
- generic critical-point repulsion;
- divided differences;
- RKHS kernel embeddings;
- Cameron–Martin quasi-invariance.

The contribution is the bridge and its audited quantitative assembly.

---

## 5. LLM claim language

Allowed:

> Under explicit chart coverage, density, multiplicity, Jacobian, decoder,
> selection, and shift assumptions, the false-acceptance tube contribution
> decays polynomially in the acceptance radius with exponent equal to the
> certified transverse rank.

Allowed:

> For fixed \(\varepsilon/\sigma<1\), the covered Q-SARD term decays
> exponentially in certified transverse rank.

Not allowed:

> The framework exponentially eliminates hallucinations.

The full risk ledger is

\[
R_{\mathrm{accept}}
\le
\delta_{\mathrm{cov}}
+
\lambda\theta\chi
+
R_{\mathrm{Q\text{-}SARD}}
+
R_{\mathrm{decoder}}
+
R_{\mathrm{approx}}
+
R_{\mathrm{shift}}.
\]

Only the Q-SARD term receives the transverse-rank exponent.

---

## 6. Correction language

Use:

- “killed” for a mathematically invalid mechanism;
- “superseded” for a historically valid but no longer operative item;
- “open extension” for work outside the finished core;
- “external review pending” for a complete internal argument not yet
  independently accepted;
- “program-grade closed” for a claim whose project gates pass;
- “measured” for empirical values.

Do not use “proved” without specifying the grade when numerical or
station-certified inputs are load bearing.

---

## 7. Abstract template

> We study elder-rule pairing for a nearby maximum–saddle fold pair in a
> periodized Bargmann–Fock field on a fixed two-torus. Under typed pair-Palm
> conditioning at height \(6/5\) and fold gap \(r^3/6\), a
> verification-grade theorem package gives a two-sided cubic defect rate on
> the certified radius interval. The upper rate implies that the local fold
> companion is selected with probability tending to one. The package includes
> an exact torus model, a dependency graph, independent numerical
> reimplementation, append-only correction ledger, and machine-checkable
> theorem contracts. We distinguish the completed fixed-volume result from
> external-referee conversion, sharpened constants, infinite-volume scaling,
> and persistence-density extensions.

**END OF CLAIM-LANGUAGE STANDARD**
