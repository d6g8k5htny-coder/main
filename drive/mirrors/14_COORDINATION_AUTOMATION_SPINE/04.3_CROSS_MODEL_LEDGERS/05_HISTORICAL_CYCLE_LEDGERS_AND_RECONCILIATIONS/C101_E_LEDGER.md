# C101 E-LEDGER

**Cycle:** C101  
**Policy:** append-only; supersede-never-overwrite.  
**Scope:** `GLOBAL_INTERCEPTOR_CUBIC_UNIFORM` and qualitative \(q_0\).

---

## E-C101-1 — The old typing division was unnecessary for a qualitative rate

**Finding.** The original numerical UB-G assembly estimated a Gaussian-pinned
saddle count and divided by an approximately \(0.80\) typing probability.

**Repair.** C101 works directly with the exact determinant-weighted typed
pair-Palm Kac–Rice intensity. Pair typing remains inside the numerator and
normalizer.

**Disposition.** `SUPERSEDED-FOR-QUALITATIVE-RATE`. No independent typing
residue remains in this route.

---

## E-C101-2 — Exact \(\eta^6\) three-determinant factor

Under the third value and gradient constraints, each leading Hessian
determinant contains an exact \(\eta^2\) factor. Therefore

\[
\det D^2P(M)\det D^2P(S)\det D^2P(x)
=
\eta^6R.
\]

**Disposition.** `DERIVED-EXACT`.

---

## E-C101-3 — Singular window-critical count is integrable

The C099 value/gradient density has scale \(t^{-6}\). The physical triple
determinant product is \(t^6\eta^6=r^6\), and the pair-Palm normalizer is
\(r^2z_r\).

Thus the per-height intensity is bounded by

\[
Cr^4t^{-6}.
\]

After the \(r^3/6\) height window and polar integration,

\[
Cr^7\int_{(\sqrt{15}/2)r}^{\delta}t^{-5}\,dt
\le C'r^3.
\]

**Disposition.** `DERIVED-EXACT-EXISTENCE`.

---

## E-C101-4 — The predicted pointwise upper scale was not sharp

The diagnostic fixed-\(\eta\) saddle intensities decreased under rung
refinement rather than approaching the conservative \(r^{-2}\) pointwise
upper scale.

**Cause.** Pair typing and geometry impose additional depletion on the sampled
charts.

**Impact.** None on the upper theorem; the analytic scale was only an upper
bound.

**Disposition.** `PRESERVED-NONSHARPNESS`. No asymptotic equality is claimed.

---

## E-C101-5 — Exterior stationary benchmark reproduced

At physical distance \(5\), the measured window-saddle coefficient divided by
\(\rho_{\rm sad}(1.2)/6\) lay in approximately

```text
[0.99742, 1.00972]
```

over three rungs.

**Disposition.** `MEASURED-DIAGNOSTIC`.

---

## E-C101-6 — Independent exact arithmetic checker passed

Nine rational generic cases were solved from the raw affine station equations,
without importing the production symbolic derivation. Every case verified:

```text
pair station/value equations
eta^2 factor in each determinant
eta^6 factor in the product
nonpositive third determinant
polar integral identity
```

**Disposition.** `CERTIFIED-ARITHMETIC`.

---

## E-C101-7 — Final successor residual condition closed

The global interceptor count, Gamma count, and collar residues now all have
finite cubic coefficients under the exact typed pair-Palm law.

**Disposition.** `CORE-CLOSED-INTERNAL-PROGRAM-GRADE`.

---

## E-C101-8 — Qualitative rate and limit receive new roots

```text
Q0_CUBIC_RATE_EXISTENCE_C101
f92e4aae4c5b6cf7a3a34c41a43642f14f129543d03eceb82d73a4e77f49100f

Q0_LIMIT_C101
5ed20eafac7cffa923518fce06cae56822c0a2a62b2d8061e5cd90163ca47996
```

Both roots are shell-valid, unconditionally promotable within the internal Q0
program-grade scope, and have empty successor condition sets.

---

## E-C101-9 — Numerical constants remain withdrawn

The existence theorem does not restore:

```text
4.35 upper
0.8411 finite lower
0.84 finite lower
0.99 / 1.01 sharpened upper
```

**Disposition.** `NOT-CLAIMED`.

---

## E-C101-10 — Initial fresh extraction omitted transitive executable dependencies

**Finding.** The first C101 ZIP passed manifest verification, but its disposable
fresh-extraction audit failed because the release inventory omitted:

```text
C095_SIX_PIN_COVARIANCE_FACTORIZATION.py
q0 registry.json
q0_registry_v2_0.json
```

The C098 Kac–Rice instrument imports the C095 covariance implementation, and
the Gate Kernel 2.0 validation battery exercises both registry fixtures.

**Cause.** The initial inventory followed theorem dependencies but did not walk
the complete executable import/fixture dependency graph.

**Impact.** The first ZIP was not release-closed and is not authoritative.

**Disposition.** `CORRECTION`. The three transitive dependencies were added,
the manifest and ZIP were rebuilt, and true fresh extraction then passed.

---

## E-C101-11 — Audit initially required its own not-yet-created output

**Finding.** The first deep-audit preflight listed
`C101_RELEASE_AUDIT.json` as a required input before the audit had written it.

**Impact.** A valid first execution was incorrectly reported as missing a
required file.

**Disposition.** `CORRECTION`. The audit output remains in the release
inventory but is excluded from the audit's pre-execution required-input list.

---

## E-C101-12 — Claim-language checker used brittle exact strings

**Finding.** The first checker searched for wording that differed from the
actual, stronger not-claimed paragraph, even though the manuscript explicitly
withdrew the 4.3/4.35 upper displays and finite lower displays.

**Disposition.** `CORRECTION`. The checker now tests the semantic withdrawal
phrases appearing in the canonical theorem file.

---

## E-C101-13 — Large diagnostics are frozen falsifiers, not theorem-grade audit dependencies

**Finding.** Re-executing every large Monte Carlo diagnostic inside every deep
audit caused environment/runtime instability while adding no theorem-grade
warrant; the exact symbolic derivations and independent arithmetic checker are
the load-bearing computations.

**Disposition.** `PROTOCOL-CORRECTION`. The deep audit re-executes all exact
proof instruments and the compact C098 anchor, and validates the schema and
integrity of the larger frozen C099–C101 diagnostics. Full diagnostic reruns
remain available as optional reproductions under their frozen seeds.

---

**Entries:** 13  
**Homeless entries:** 0
