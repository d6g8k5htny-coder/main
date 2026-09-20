# C104 E-LEDGER

**Cycle:** C104  
**Policy:** append-only; supersede-never-overwrite.  
**Scope:** continuum-consistent validation and Q0-B project close.

---

## E-C104-1 — The first full execution exceeded the notebook wrapper limit

The initial C104 implementation used pure-Python union-find and a
field-by-field bootstrap likelihood. The visible execution wrapper interrupted
the run before completion.

**Diagnosis:** computational architecture, not mathematical failure.

**Repair:**

- compiled periodic Freudenthal persistence with Numba;
- replaced repeated likelihood scans with field-level sufficient statistics;
- retained the frozen field counts, grids, seeds, cutoffs, and pass criteria.

**Disposition:** `INSTRUMENT-OPTIMIZATION`. The experimental specification was
not changed.

---

## E-C104-2 — The frozen primary continuum-consistent gate survives

For the frozen primary pair \(256\to512\) and \(U=0.2\):

\[
\widehat\alpha=-0.48144,
\]

with 95% field-bootstrap interval

\[
[-0.71150,-0.24375].
\]

The interval contains

\[
-\frac13.
\]

There were 543 refinement-stable bars and zero failed bootstrap replicates.

**Disposition:** `SURVIVES`.

---

## E-C104-3 — The frozen kill signal does not trigger

Not every valid confidence interval excludes \(-1/3\), and the finest primary
estimate moves toward \(-1/3\) relative to the preceding resolution pair.

**Disposition:** `KILL-SIGNAL-FALSE`.

---

## E-C104-4 — Refinement behavior is directionally consistent

Mean interpolation error decreased:

```text
128->256: 0.05482
192->384: 0.02495
256->512: 0.01418
```

The refinement-stable bar fraction increased:

```text
128->256: 0.5555
192->384: 0.6804
256->512: 0.7336
```

**Disposition:** `MEASURED-SUPPORT`.

---

## E-C104-5 — A secondary finite-window tension is preserved

At \(256\to512\) with \(U=0.3\),

\[
\widehat\alpha=-0.50047
\]

with interval

\[
[-0.63855,-0.37905],
\]

which excludes \(-1/3\).

**Disposition:** `PRESERVED-LIMITATION`.

This does not replace the frozen primary gate, but it forbids describing the
experiment as a precise confirmation of the exponent.

---

## E-C104-6 — The C103 failure remains valid for its proxy

C103's four-neighbor raw cumulative-count test failed and triggered its kill
signal. C104 does not relabel that result.

The C104 replacement used a different pre-registered measurement object:

- Freudenthal PL persistence;
- nested-grid coupling;
- interpolation-error bounds;
- diagram matching;
- refinement-stability filtering;
- left-truncated likelihood.

**Disposition:** `SUPERSEDED-AS-VALIDATION-INSTRUMENT`, not erased.

---

## E-C104-7 — Continuum-persistence validation obligation discharged

The pre-registered primary gate survives and the kill signal is false.

**Disposition:** `CONTINUUM_PERSISTENCE_VALIDATION CLOSED`.

The result is diagnostic support, not a theorem proof.

---

## E-C104-8 — Q0-B reaches terminal external-review disposition

The mathematical core is internally closed at program grade, the held-out
primary validation survives, and the only named terminal dependency is
independent SARD-G specialist review.

New roots:

```text
THEOREM_B_FULL_KAPPA_C104
6f9d5b233ec7551e25ff7c51a0e49af6de7a318837488b3174cdfcdb6b07a6ca

Q0_B_PROJECT_CLOSE_C104
5771999a8d61dfa3c933b2131e6dbdc919c69465949371cc9327a8fa564bcc8c
```

**Disposition:** `EXTERNAL-REVIEW-TRACK`.

---

## E-C104-9 — Numerical constant remains unclaimed

No value of

\[
C_*
\]

is extracted from the simulation or promoted into the theorem.

**Disposition:** `NOT-CLAIMED`.

---

**Entries:** 9  
**Homeless entries:** 0
