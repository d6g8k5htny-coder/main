# THE q₀ RATE PROGRAM — FINAL MASTER C092

## Program-grade completion, exact boundary of validity, final correction ledger, and proof-carrying LLM transfer

**Freeze date:** 2026-07-16  
**Canonical contract:** `Q0-C092-FINAL`  
**Finished scope:** fixed torus side \(L=24\), birth height \(b=6/5\), fold window \(\ell=r^3/6\), typed maximum–saddle pair-Palm law, and the program's declared verification/program-grade standard.  
**Core roots:** `RATE_PROGRAM_GRADE` and `Q0_LIMIT`.  
**Core closure:** zero open nodes.  
**External referee conversion:** deliberately separate.  
**Future extensions:** deliberately outside the core dependency cone.

---

# 0. Final declaration

The q₀ Rate Program is finished in the following precise sense.

1. The mathematical object is now exactly defined on the torus rather than by an informal planar-kernel shorthand.
2. The fixed-\(L\) quantitative theorem has one frozen source-of-truth statement, one explicit epistemic grade, one exact parameter domain, and one closed program dependency graph.
3. The qualitative selection theorem
   \[
   q_0=1
   \]
   is a closed corollary of the fixed-volume upper rate.
4. Every measured number is tagged by rung, measure, estimand, and uncertainty class.
5. Every correction remains in an append-only ledger.
6. The C089 sharpened theorem, infinite-volume theory, Theorem B normalization, external referee conversion, and empirical LLM deployment are extensions. They do not sit upstream of the finished core.
7. A machine verifier rejects the semantic failure classes found during the program: missing mark variables, invalid probability products, endpoint violations, measure changes, scale-free measurements, wrong uncertainty polarity, source-precedence violations, and open-node leakage into a finished theorem.
8. A cold-start checker reconstructs the final theorem graph, recomputes both root hashes, and verifies the artifact set.

“Finished” does **not** mean that every mathematically interesting extension has been solved. It means that the program has a stable theorem, a stable standard of evidence, a stable boundary of validity, and a stable distinction between completed core work and future research.

The final program-grade theorem is:

> **Theorem A-RATE-PG.**  
> For the exact normalized periodized Bargmann–Fock field on
> \(\mathbb T_{24}^2\), at \(b=6/5\), under the typed maximum–saddle
> pair-Palm law with height gap \(\ell=r^3/6\), and under the program
> condition classes recorded below,
> \[
> \boxed{
> 0.8411\,r^3
> \le
> 1-q(r,6/5)
> \le
> 4.3\,r^3,
> \qquad
> 0<r\le0.025.
> }
> \]
> The lower and upper constants are verification/program-grade quantities:
> exact structural reductions combined with derived formulas, independent
> implementation checks, station-certified inputs, and measured-modulus
> uniformity certificates.

Its immediate corollary is:

> **Theorem A-Q0.**
> \[
> \boxed{
> \lim_{r\downarrow0}q(r,6/5)=1
> }
> \]
> at fixed \(L=24\).

The direct measured coefficient at the reference rung remains:

\[
\boxed{
\frac{1-q(0.025,6/5)}{(0.025)^3}
=
0.946,
\qquad
\text{reported interval }[0.941,0.951].
}
\]

It is named `PAIR_DEFECT_DIRECT_R0025_B1200`, never `TRUTH_CONST`.

---

# 1. The exact mathematical object

## 1.1 Torus

The base space is

\[
\mathbb T_L^2
=
\mathbb R^2/(L\mathbb Z^2),
\qquad L=24.
\]

The metric is the quotient Euclidean metric and the total area is

\[
|\mathbb T_{24}^2|=24^2=576.
\]

## 1.2 Exact covariance

The planar Gaussian kernel

\[
K_\infty(x)=e^{-|x|^2/2}
\]

is a reference kernel. It is not periodic and therefore is not literally a
covariance on a torus.

The exact unit-variance periodized covariance is

\[
\boxed{
K_L(x)
=
\frac{
\displaystyle\sum_{n\in\mathbb Z^2}
e^{-\frac12|x+nL|^2}
}{
\displaystyle\sum_{n\in\mathbb Z^2}
e^{-\frac12|nL|^2}
}.
}
\]

Poisson summation gives

\[
\boxed{
K_L(x)
=
\frac{
\displaystyle\sum_{k\in\mathbb Z^2}
e^{-2\pi^2|k|^2/L^2}
e^{2\pi i k\cdot x/L}
}{
\displaystyle\sum_{k\in\mathbb Z^2}
e^{-2\pi^2|k|^2/L^2}
}.
}
\]

Every spectral weight is strictly positive. The associated field is centered,
stationary, unit variance, real valued, and almost surely real analytic.

The derivative covariance convention is

\[
\operatorname{Cov}
\left(
\partial^\alpha f(P),
\partial^\beta f(Q)
\right)
=
(-1)^{|\beta|}
\partial^{\alpha+\beta}K_L(P-Q).
\]

## 1.3 Planar-reference transfer

The exact local comparison certificate gives

\[
Z_{24}-1
\le
3.3515770135277476\times10^{-125}.
\]

On \(|x|\le3\), through total derivative order eight,

\[
\max_{|\alpha|\le8}
\sup_{|x|\le3}
\left|
\partial^\alpha K_{24}(x)
-
\partial^\alpha K_\infty(x)
\right|
\le
1.0152\times10^{-84}.
\]

C091 propagated these entry errors through covariance matrices, inversion, and
Schur complementation in the dimensions used by the program.

Using the smallest retrieved raw nine-pin floor

\[
3.094564589647658\times10^{-14},
\]

the local conditional-covariance perturbation is bounded by

\[
1.7924\times10^{-56}.
\]

For far station blocks at distances \(5\le d\le12\), the Schur perturbation is

\[
1.8608\times10^{-12},
\]

against a retrieved floor

\[
1.4\times10^{-8}.
\]

Thus the far floor remains positive with relative loss approximately

\[
1.33\times10^{-4}.
\]

These transfer certificates close the exact-model issue for derivative order
at most four, pin dimension at most nine, target dimension at most six,
local radius three, and far distances five through twelve. Any future
calculation outside that declared scope must add its own transfer row.

---

# 2. Pair-Palm object and q

Place the maximum and saddle at separation \(r\) along the pair axis. The
canonical height pins are

\[
f(M)=b,
\qquad
f(S)=b-\ell,
\qquad
b=\frac65,
\qquad
\ell=\frac{r^3}{6},
\]

with

\[
\nabla f(M)=\nabla f(S)=0.
\]

The typed event requires

\[
\det H_M>0,
\qquad
\operatorname{tr}H_M<0,
\qquad
\det H_S<0.
\]

Conditioning on the six linear value and gradient functionals gives a Gaussian
conditional law. The critical-pair Palm law is the determinant-weighted law

\[
\mathbb E_{\mathrm{Palm}}[F]
=
\frac{
\mathbb E\!\left[
F
|\det H_M|
|\det H_S|
\mathbf1_{\mathrm{typed}}
\mid
\text{six pins}
\right]
}{
\mathbb E\!\left[
|\det H_M|
|\det H_S|
\mathbf1_{\mathrm{typed}}
\mid
\text{six pins}
\right]
},
\]

with any exact adjacency conditioning included according to the estimand's
declared object card.

Let \(C_M(v)\) be the superlevel component born at \(M\). Under elder-rule
persistence, \(M\) dies when \(C_M(v)\) first merges with a component
containing an older maximum.

The canonical partner is \(S\). Define

\[
q(r,b)
=
P_{\mathrm{Palm}}
\left(
\text{death partner of }M\text{ is }S
\right).
\]

The defect probability is

\[
D(r,b)=1-q(r,b).
\]

---

# 3. Grade and status lattice

The final master separates mathematical truth conditions from project
management.

## 3.1 Grades

- **Established:** externally standard mathematics with an appropriate source.
- **Derived exact:** follows from exact algebra or analysis within the stated
  hypotheses.
- **Certified:** accompanied by an analytic, interval, symbolic, or
  machine-checkable certificate.
- **Program-grade closed:** every project gate in the declared program
  standard is discharged. The proof may still contain station-certified or
  measured-modulus inputs.
- **Measured:** empirical or numerical estimate with sample/tolerance metadata.
- **Conditional:** a valid implication whose named hypotheses remain outside
  the finished core.
- **Open extension:** future work that is not a dependency of a finished core
  theorem.
- **Killed:** a false mechanism or invalid inference retained for provenance.
- **Superseded:** a historically valid record no longer governing live status.

## 3.2 Tiers are not synonyms for grades

The archive uses tiers such as “rigorized,” “Richardson,” and “measured.” A
rigorized program tier may still inherit measured derivative moduli. It is not
silently relabeled “classical unconditional theorem.”

## 3.3 Internal closure and external review

For SARD-G/R0:

```text
internal dependency status:
    PROGRAM-GRADE-CLOSED

external publication status:
    SPECIALIST-REVIEW-PENDING
```

For the quantitative rate:

```text
internal fixed-L theorem:
    PROGRAM-GRADE-CLOSED

external referee conversion:
    SEPARATE EXTENSION
```

This dual status is deliberate and stable.

---

# 4. Exact lower architecture

## 4.1 Qualifying count

Let \(N_{\mathrm{qual}}\) count window saddles satisfying the spatial, index,
flow, and terminal conditions that force a genuine elder-rule defect.

Then

\[
D(r,b)
\ge
P_{\mathrm{Palm}}(N_{\mathrm{qual}}\ge1).
\]

## 4.2 Bonferroni identity

For an integer-valued \(N\ge0\),

\[
\mathbf1_{\{N\ge1\}}
\ge
N-\binom N2.
\]

Therefore

\[
D(r,b)
\ge
E[N_{\mathrm{qual}}]
-
\frac12
E[N_{\mathrm{qual}}(N_{\mathrm{qual}}-1)].
\]

If \(N_w\) counts all saddles in the relevant height window, then
\(N_{\mathrm{qual}}\le N_w\), so

\[
\boxed{
D(r,b)
\ge
E[N_{\mathrm{qual}}]
-
\frac12E[N_w(N_w-1)].
}
\]

Define

\[
\mathfrak B_2(r)
=
\frac{
E[N_w(N_w-1)]
}{
2r^6
}.
\]

Then

\[
\boxed{
\frac{D(r,b)}{r^3}
\ge
\frac{E[N_{\mathrm{qual}}]}{r^3}
-
\mathfrak B_2(r)r^3.
}
\]

This exact reduction is independent of every numerical constant.

## 4.3 First moment

The first moment is represented by the station intensity

\[
\Lambda_r(y)
=
\phi_2\!\left(
\nabla f(y)=0
\mid
\text{six pins}
\right)
P_{\mathrm{win}}(y)
\frac{
E[W_3\mathbf1_{\mathrm{saddle\ at}\ y}\mid\text{nine pins}]
}{
E[W_2\mathbf1_{\mathrm{typed\ pair}}\mid\text{six pins}]
}.
\]

The independent C024 implementation reproduced the conditioning and estimator
stack without shared code and passed all fourteen registered anchors.

The exponent ledger at the fold station is

\[
\phi_2\sim r^{-3},
\]

\[
\frac{E[W_3]}{E[W_2]}\sim r^4,
\]

and the spatial area is

\[
r^2.
\]

Thus

\[
-3+4+2=3.
\]

The qualifying first moment has the form

\[
E[N_{\mathrm{qual}}]
\ge
C_\Lambda(r)AO(r)r^3.
\]

Here \(AO(r)\) is the adjacency and other-branch terminal-survival factor.

## 4.4 Program-grade marked near-diagonal certificate

The live Master-v3.2 source records a program-specific near-diagonal condition
\((ND')\), including the moment-invariant frame

\[
\operatorname{Var}\Xi=6,
\qquad
\operatorname{Var}T=4,
\qquad
\operatorname{Cov}(\Xi,T)=0,
\qquad
\det G_{12}=\frac1{6480}.
\]

The compiled lower source states that the conditioned-field Schur analysis
provides a marked two-point law of the form

\[
\lambda_2(y_1,y_2)
\le
C_{\mathrm{nd}}|y_1-y_2|
\]

in the precise Kac–Rice object used by the program, and uses this together
with far decorrelation to obtain

\[
E[N_w(N_w-1)]=O(r^6).
\]

C091 correctly observed that **unmarked critical-point repulsion in the
literature does not by itself imply a per-height-squared marked estimate**.
That correction remains.

C092 narrows its scope:

> The C091 quarantine applies to a literature-only or externally promoted
> justification. It does not erase the live internal \((ND')\) certificate
> recorded by the source-of-truth master.

Consequently:

```text
ND_PRIME_INTERNAL:
    PROGRAM-GRADE-CLOSED

ND_MARKED_EXTERNAL_AUDIT:
    OPEN EXTENSION
```

This is the decisive status correction that closes the core without
overclaiming external proof provenance.

## 4.5 Lower coefficient

The live rigorized coefficient is

\[
0.8411.
\]

The lower theorem is therefore

\[
\boxed{
D(r,6/5)
\ge
0.8411\,r^3,
\qquad
0<r\le0.025,
}
\]

at program grade.

The C089 positivity observation removes an earlier symmetric shell
subtraction and produces a first-moment core

\[
0.9091-0.0590=0.8501.
\]

That value is retained as an extension candidate. It is not the final finite
coefficient because the AO and Bonferroni losses must be propagated.

---

# 5. Exact upper architecture

## 5.1 Failure decomposition

A pairing defect is reduced to:

- \(\Pi\): a window-valued saddle preempts the canonical pair;
- \(\Gamma\): the second branch of \(S\) reaches a maximum with gain in the
  narrow interval;
- collar and typing residues.

The live UB-G method bounds \(\Pi\) by **global interceptor counting**. It is
deliberately conservative: every exterior saddle in the height window is
counted, whether or not it is actually merge-tree adjacent to \(M\).

## 5.2 Near mass

The six-pin window-saddle mass on the near global region is recorded as

\[
0.657\,r^3
\quad
(r=0.025),
\]

and

\[
0.661\,r^3
\quad
(r=0.05),
\]

with an approximately unit rung ratio.

The older per-candidate decomposition that appeared to diverge like
\(r^{-1.88}\) is superseded. The divergence belonged to the decomposition,
not the global count.

## 5.3 Exterior mass

The stationary saddle density at \(b=1.2\) is

\[
\rho_{\mathrm{sad}}(1.2)=0.030449.
\]

The exterior window count is

\[
\rho_{\mathrm{sad}}(1.2)
\frac{r^3}{6}
\left(
L^2-9\pi
\right)
E_{\mathrm{sup6}},
\]

where

\[
E_{\mathrm{sup6}}=1.014.
\]

This is the six-pin exterior saddle enhancement. It is not the terminal-count
constant \(E_{\mathrm{sup}}=1.0235\).

At \(L=24\), the raw exterior coefficient is approximately

\[
2.82.
\]

## 5.4 Typing and direct-gain terms

The typed probability is stable near

\[
0.80.
\]

The direct-gain coincidence is suppressed at an exit-station exponent near

\[
91.6,
\]

so it contributes at the \(e^{-92}\) class.

The collar has both a station rigidity calculation and an \((ND')\)
backstop.

## 5.5 Upper assembly

The program-grade assembly is

\[
D(r,6/5)
\le
\frac{0.66+2.82}{0.80}r^3
+
e^{-92}
+
\text{collar residue}.
\]

With the program's collar class and rounding convention, this is frozen as

\[
\boxed{
D(r,6/5)
\le
4.3\,r^3,
\qquad
0<r\le0.025.
}
\]

The large constant is intentional. Its purpose is to prove the rate and
selection limit without depending on adjacency-aware exterior geometry.

---

# 6. The q₀ theorem

From the upper bound,

\[
0\le1-q(r,6/5)\le4.3r^3.
\]

As \(r\downarrow0\),

\[
4.3r^3\to0.
\]

Therefore

\[
\boxed{
q(r,6/5)\to1.
}
\]

No lower bound is needed for this conclusion.

Thus, even if a future external referee requests changes to the quantitative
lower coefficient, the qualitative q₀ theorem remains supported by the
separate UB-G upper cone.

At \(r=0.025\),

\[
r^3=1.5625\times10^{-5}.
\]

The final program-grade interval is

\[
1.31421875\times10^{-5}
\le
D(0.025,6/5)
\le
6.71875\times10^{-5},
\]

equivalently,

\[
0.9999328125
\le
q(0.025,6/5)
\le
0.9999868578125.
\]

The direct measured coefficient \(0.946\) gives

\[
D(0.025,6/5)
\approx
1.478125\times10^{-5},
\]

or

\[
q(0.025,6/5)
\approx
0.99998521875.
\]

---

# 7. Morse–Smale/R0 status

The source chain is:

```text
C015:
    reduction and open architecture

C041:
    SARD-G proof

C042:
    audit

C043:
    closeout

C046:
    independent internal review;
    RF1–RF5 incorporated

Master v3.2:
    resolved at program grade
```

The canonical internal status is

```text
PROGRAM-GRADE-CLOSED
```

and the publication status is

```text
SPECIALIST-REVIEW-PENDING
```

A later prose label cannot reopen R0. Reopening requires a node that names:

- the failing SARD-G step;
- a mathematical reason;
- a counterexample or unclosed obligation;
- the source hash;
- downstream claims invalidated.

The exact periodized Fourier representation strengthens the kernel-injectivity
step. If the RKHS representative of a compactly supported distribution
vanishes, strict positivity of every Fourier coefficient forces every Fourier
coefficient of the distribution to vanish, hence the distribution itself to
vanish.

The field is almost surely Morse under finite-jet nondegeneracy. SARD-G
supplies the global Morse–Smale conclusion. Those statements are not
conflated.

---

# 8. Final correction and supersession decisions

## 8.1 C089 lower \(0.8501\)

Preserved:

\[
0.8501
\]

as the shell-positivity first-moment/asymptotic candidate.

Not preserved:

\[
D(r)\ge0.8501r^3
\]

as a finite statement through \(r=0.05\) before AO and Bonferroni
propagation.

## 8.2 C089 upper \(0.97\)

Killed as a theorem through \(r=0.05\), because the registered endpoint
assembly is

\[
0.985>0.97.
\]

The first-interceptor research route remains an extension.

## 8.3 H4 \(q_{\mathrm{step}}^n\)

Killed.

The exact one-step probability is valid. Multiplying it along a correlated
Gaussian corridor without independence, a Markov property, negative
dependence, comparison theorem, or joint certificate is invalid.

The replacement is a joint Gaussian or Palm-weighted Gaussian Chernoff
certificate.

This correction affects only the C089 sharpened first-interceptor upper
route. It does not affect UB-G.

## 8.4 C091 marked-law quarantine

Narrowed.

Correct statement:

- literature-only spatial repulsion is insufficient for a marked window
  estimate;
- the archive nevertheless records an internal Schur/ND certificate as a
  live program condition;
- external source-level audit remains an extension;
- the internal fixed-\(L\) rate theorem is not reopened.

## 8.5 `TRUTH_CONST`

Retired.

Use:

```text
PAIR_DEFECT_DIRECT_R0025_B1200
```

with the rung and confidence interval.

## 8.6 Exact torus definition

The planar kernel is retained as a reference and local computational
coordinate. The exact theorem object is the normalized periodized kernel.

---

# 9. Closed core dependency graph

```text
RATE_PROGRAM_GRADE
│
├── LOWER_RATE_PG
│   ├── LAMBDA_SIDE
│   │   └── TORUS_TRANSFER
│   │       └── MODEL_EXACT
│   ├── AO_SIDE
│   │   ├── R0_SARD_G
│   │   │   └── TORUS_TRANSFER
│   │   └── TORUS_TRANSFER
│   └── BONFERRONI_INTERNAL
│       └── ND_PRIME_INTERNAL
│           └── TORUS_TRANSFER
│
└── UB_G
    ├── TORUS_TRANSFER
    └── R0_SARD_G

Q0_LIMIT
└── UB_G
```

The machine validator reaches eleven nodes from the two roots.

It reports:

```text
missing dependencies:    0
cycles:                  0
open core nodes:         0
killed reachable nodes:  0
superseded reachable:    0
domain violations:       0
measure violations:      0
mark violations:         0
endpoint violations:     0
polarity violations:     0
composition violations:  0
model violations:        0
rung violations:         0
```

Root hashes:

```text
RATE_PROGRAM_GRADE:
e24569bfedfac0d02a30ea606bcf271ad9a06650b2e935f8bdd81aa917ef5d48

Q0_LIMIT:
0c643c9d26fdda9e87065b040c6d55d91c932a8748f4a5eb223f1babf10cb4c6
```

Any change to a dependency changes the corresponding root hash.

---

# 10. Extension cone

The following are scientifically valuable and explicitly **not** blockers of
the finished core.

## 10.1 External referee conversion

- source-level audit of the marked \((ND')\) law;
- interval/analytic replacement of measured GRID moduli;
- one analytic nine-pin frame atlas over the full target radius;
- independent SARD-G specialist review.

## 10.2 C089 sharpening

- asymptotic \(0.8501\) lower coefficient;
- finite lower coefficient after exact AO and Bonferroni propagation;
- first-interceptor upper coefficient after a valid joint Palm-Gaussian path
  bound;
- interval monotonicity or negative-curvature certificate;
- one-sided uncertainty calibration.

## 10.3 Infinite volume and critical height

- pair-Palm influence localization;
- selected-set Campbell control;
- first-interceptor density;
- fixed-\(b\) thermodynamic limit;
- critical crossover as \(b\downarrow0\);
- susceptibility and finite-size scaling.

## 10.4 Theorem B

The near-diagonal persistence-density theorem is retained as a conditional
extension:

\[
\nu(\ell)
\sim
C_*\ell^{-1/3}
\]

conditional on candidate-pair contact neutrality, the required modulus
integration, and the selection factor.

It is not silently bundled into the fixed-\(L\) q₀ Rate theorem.

## 10.5 LLM deployment

The proof-carrying verifier is complete as a research prototype. Training,
model integration, chart coverage, benchmark calibration, and distribution
shift remain empirical work.

---

# 11. Final LLM transfer

The q₀ program suggests a verifier architecture, but mathematical analogy is
not empirical validation.

## 11.1 Proof object

Every externally visible claim carries:

- atomic statement;
- dependency IDs;
- evidence IDs;
- model and measure tags;
- parameter domain;
- required and supplied marks;
- quantifier;
- direction;
- epistemic grade;
- uncertainty side;
- scale/rung tag;
- composition witness;
- source-precedence ID;
- content hash;
- supersession status.

## 11.2 Structural gates

The final v4 verifier implements:

### DOMAIN

A dependency must cover the complete parameter domain of its parent.

### ENDPOINT

Every registered endpoint must satisfy a claimed finite-range bound.

### POLARITY

A lower theorem consumes lower one-sided inputs. An upper theorem consumes
upper one-sided inputs.

### MEASURE

Unconditioned, Gaussian-pinned, typed-pinned, and pair-Palm claims cannot be
silently interchanged.

### MARK

A spatial estimate does not prove a height-marked or type-marked estimate
unless those variables appear in the dependency cone.

### COMPOSITION

A product of probabilities requires a witness:

- independence;
- conditional independence;
- Markov property;
- negative dependence;
- comparison theorem;
- joint certificate;
- exact algebra.

### RUNG

A measured value requires an explicit scale and parameter tag.

### PRECEDENCE

Killed or superseded claims may not remain reachable from a live root.

### CORE CLOSURE

A finished core cannot depend on an open extension node.

### MODEL

Every covariance-dependent claim identifies the exact model or an explicit
transfer certificate.

### COVERAGE

No exponential rank argument can suppress wrong outputs that activate no
registered mismatch chart.

## 11.3 Exact H₀ grounding compression

For support strengths \(s_e\), define

\[
d(v)
=
\max_{v\leadsto E}
\min_{e\in\text{path}}s_e.
\]

A widest-path predecessor forest rooted at evidence nodes preserves every
\(d(v)\). The full \(H_0\) grounding structure therefore needs at most

\[
|V|-|E|
\]

edges.

## 11.4 Anchored RKHS mismatch

For signed residual

\[
T=\sum_i a_i\delta_{z_i},
\]

use

\[
K_{\mathrm{safe}}
=
\lambda K_0+K_\theta,
\qquad
\lambda>0,
\]

where \(K_0\) is a fixed universal kernel and \(K_\theta\) is learned
positive semidefinite.

Then

\[
\|T\|_{K_{\mathrm{safe}}}^2
=
\lambda\|T\|_{K_0}^2
+
\|T\|_{K_\theta}^2.
\]

Training cannot remove the universal-anchor floor.

## 11.5 Quantitative SARD tube bound

If a \(k\)-dimensional mismatch map has singular-value floors
\(s_1,\ldots,s_k\), density bound \(M\), and chart multiplicity \(N\), then

\[
P(\|D\|\le\varepsilon)
\le
NM\operatorname{Vol}(B_k)
\frac{\varepsilon^k}{s_1\cdots s_k}.
\]

The exponent follows certified transverse rank, not nominal detector count.

## 11.6 Coverage-first risk theorem

Let

- \(\delta_{\mathrm{cov}}\): wrong outputs activating no chart;
- \(\lambda\): endogenous selection amplification;
- \(\theta\): baseline hazard rate;
- \(\chi\): expected selected dependency measure;
- \(R_{\mathrm{tube}}\): Q-SARD tube risk;
- \(R_{\mathrm{boundary}}\): decoder boundary risk;
- \(R_{\mathrm{approx}}\): approximation risk;
- \(R_{\mathrm{shift}}\): calibration and distribution-shift risk.

Then

\[
\boxed{
R_{\mathrm{accept}}
\le
\delta_{\mathrm{cov}}
+
\lambda\theta\chi
+
R_{\mathrm{tube}}
+
R_{\mathrm{boundary}}
+
R_{\mathrm{approx}}
+
R_{\mathrm{shift}}.
}
\]

As transverse rank tends to infinity, only \(R_{\mathrm{tube}}\) vanishes.
The limiting floor is

\[
\delta_{\mathrm{cov}}
+
\lambda\theta\chi
+
R_{\mathrm{boundary}}
+
R_{\mathrm{approx}}
+
R_{\mathrm{shift}}.
\]

This prevents the phrase “exponential hallucination reduction” from being
used without a coverage and selection qualification.

## 11.7 v4 regression results

The final validator confirms:

- finished q₀ core graph: valid;
- missing marked variables: rejected by `MARK`;
- \(q_{\mathrm{step}}^n\) without dependence theorem: rejected by
  `COMPOSITION`;
- C089 \(0.97\) endpoint failure: rejected by `ENDPOINT`;
- scale-free measured \(0.946\): rejected by `RUNG`;
- symmetric measured input in an upper theorem: rejected by `POLARITY`;
- anchored RKHS residual: positive;
- rank-deficient Q-SARD example: effective rank two;
- target below coverage floor: rejected;
- exact H₀ widest-path compression: preserved.

---

# 12. Publication claim language

## 12.1 Approved internal statement

> We establish, at the program's verification grade, a two-sided cubic
> pairing-defect rate for the exact periodized Bargmann–Fock field on
> \(\mathbb T_{24}^2\) at height \(6/5\):
> \[
> 0.8411r^3\le1-q(r,6/5)\le4.3r^3
> \]
> for \(0<r\le0.025\). In particular \(q(r,6/5)\to1\).

## 12.2 Approved external conservative statement

> We provide a verification-grade theorem package for a cubic
> pairing-defect rate in the periodized Bargmann–Fock model, with exact
> structural reductions and reproducible numerical certificates. Several
> uniform constants and the SARD-G appendix are separately flagged for
> external specialist review.

## 12.3 Prohibited statements

Do not write:

- “unconditional classical theorem”;
- “first ever” without the prior-art qualifier;
- “all smooth Gaussian fields”;
- “infinite volume”;
- “\(0.97r^3\) through \(r=0.05\)”;
- “\(0.8501r^3\) finite-range lower bound” without AO and Bonferroni;
- “exponential hallucination reduction” without coverage and selection
  terms;
- “truth constant \(0.946\)”;
- “independent corridor cells”;
- “Morse modulo R0.”

---

# 13. Reproduction and cold-start verification

The final bundle contains:

```text
Q0_C092_FINAL_MASTER.md
C092_FINAL_CORRECTION_LEDGER.md
C092_PUBLICATION_CLAIM_LANGUAGE.md
C092_REPRODUCTION_README.md
q0_c092_final_contract.json
q0_c092_contract_checker.py
q0_c092_contract_check_report.json
q0_llm_verifier_v3.py
q0_llm_verifier_v4.py
validate_q0_llm_verifier_v4.py
q0_llm_verifier_v4_validation.json
periodized_bf_contract.py
periodized_bf_contract_report.json
periodized_bf_matrix_transfer.py
periodized_bf_matrix_transfer_report.json
Q0_C091_PROOF_GATE_REDUCTION.md
C091_CORRECTION_LEDGER.md
```

Cold start:

```bash
python3 validate_q0_llm_verifier_v4.py
python3 q0_c092_contract_checker.py
python3 verify_q0_c092_bundle.py
```

A passing final state requires:

```text
semantic core valid:                    true
core open nodes:                        0
bad-contract regression gates:          all detected
root hashes:                            match
required artifacts:                     present
artifact hashes:                        match manifest
```

---

# 14. Final constitutional rules

These rules are frozen.

1. **Definitions precede calculations.**
2. **The exact model is not an informal approximation.**
3. **A measured number is never scale free unless a limit theorem makes it
   so.**
4. **A mark cannot disappear from a dependency graph.**
5. **A probability product requires a dependence theorem.**
6. **A finite interval requires domain coverage, not sampled rungs.**
7. **A lower coefficient carries every multiplicative loss.**
8. **An upper coefficient carries every positive residual and one-sided
   uncertainty.**
9. **A source status changes only through an explicit supersession or
   reopening node.**
10. **Open extensions do not reopen a completed core.**
11. **Numerical validation estimates constants and falsifies mechanisms; it
    is not mislabeled exact analysis.**
12. **External review status is distinct from internal dependency status.**
13. **Coverage failure is a first-class risk term.**
14. **Corrections are appended, not erased.**

---

# 15. Final status table

| Result | Final status |
|---|---|
| Exact periodized BF field | Closed |
| Pair-Palm definition | Closed |
| Cubic fold normalization | Closed |
| GCJA architecture | Program-grade closed |
| Independent one-point estimator | Verified 14/14 |
| \(\Lambda\)-side lower coefficient | Program-grade closed |
| AO survival factor | Program-grade closed |
| Internal \((ND')\) Bonferroni certificate | Program-grade closed |
| External marked-density audit | Open extension |
| UB-G upper \(4.3r^3\) | Program-grade closed |
| R0/SARD-G | Program-grade closed; external review pending |
| Fixed-\(L\) rate theorem | Program-grade closed |
| \(q_0=1\) | Program-grade closed |
| Measured coefficient \(0.946\) at \(r=.025\) | Measured |
| C089 finite lower \(0.8501\) | Not canonical |
| C089 upper \(0.97\) through \(r=.05\) | Killed by endpoint |
| H4 \(q_{\mathrm{step}}^n\) | Killed |
| First-interceptor sharpening | Open extension |
| Infinite-volume theory | Open extension |
| Theorem B normalization | Conditional extension |
| LLM verifier v4 | Completed research prototype |
| Production LLM guarantee | Not claimed |

---

# 16. End state

The project no longer needs another master document to decide what its theorem
is.

Its theorem is:

\[
\boxed{
0.8411r^3
\le
1-q(r,6/5)
\le
4.3r^3,
\qquad
0<r\le0.025,
\quad
L=24,
}
\]

at the explicitly named program/verification grade.

Its qualitative conclusion is:

\[
\boxed{q_0=1.}
\]

Its measured reference value is:

\[
\boxed{
(1-q(0.025,6/5))/(0.025)^3=0.946.
}
\]

Its external-review boundary is explicit.

Its sharpened, infinite-volume, persistence-density, and LLM-deployment
questions are separate modules.

Its proof graph is closed.

Its root hashes are frozen.

Its corrections are preserved.

Its verifier catches the errors that the project itself discovered.

That is the completed q₀ Rate Program.

**END OF FINAL MASTER C092**
