# C102 Deterministic Elder-Rule Defect Dichotomy

**Grade:** `PROVEN-HERE`

## Hypotheses

Let \(f\) be a Morse function on a compact surface with distinct critical
values. Assume its gradient flow is Morse–Smale, so every ascending branch of
an index-one saddle terminates at a local maximum.

Let \(M\) be a local maximum and \(S\) an index-one saddle such that one
ascending branch of \(S\) terminates at \(M\). Put

\[
f(M)=b,
\qquad
f(S)=b-\ell,
\qquad
\ell>0.
\]

Let \(D(M)\) be the death saddle paired with \(M\) by superlevel
\(H_0\) persistence.

Define:

\[
\Pi
=
\left\{
\begin{array}{c}
\text{there exists an index-one saddle }S'
\text{ with}\\
f(S)<f(S')<f(M)
\text{ at which the component born at }M\\
\text{merges before level }f(S)
\end{array}
\right\},
\]

and let \(\Gamma\) be the event that the other ascending branch of \(S\)
terminates at a maximum \(N\) satisfying

\[
f(S)<f(N)<f(M).
\]

## Theorem

\[
\boxed{
\{D(M)\ne S\}
\subset
\Pi\cup\Gamma.
}
\]

## Proof

Consider the superlevel filtration

\[
E_t=\{x:f(x)\ge t\}.
\]

The component \(C_M(t)\) born at \(M\) exists for \(t<f(M)\) until its elder-rule
death.

### Case 1: \(M\) dies before the level of \(S\)

Then

\[
D(M)=S'
\]

for some index-one saddle \(S'\) with

\[
f(S)<f(S')<f(M).
\]

Therefore \(\Pi\) occurs.

### Case 2: \(M\) is alive immediately above \(f(S)\), but the two local arms
of \(S\) already lie in the same component

The component \(C_M\) does not exist above \(f(M)\). By the time the level has
descended to just above \(f(S)\), it has become connected to the other local
arm of \(S\).

For a Morse function with distinct critical values, superlevel connectivity
changes only at index-one critical values. Therefore there is a first level
at which this connection appears. That level is an index-one saddle value

\[
f(S')\in(f(S),f(M)).
\]

At \(S'\), the component born at \(M\) merges before reaching \(S\).
Consequently \(\Pi\) occurs.

This proves that a loop/crater configuration—two arms already connected above
\(S\)—is not a separate uncounted obstruction. It produces a window-valued
interceptor saddle.

### Case 3: \(M\) is alive immediately above \(f(S)\), and the two local arms
belong to distinct components

Crossing \(S\) merges those two components. Let \(C_{\rm other}\) be the
component attached to the second ascending branch.

If \(D(M)=S\), there is no defect.

Assume instead that \(D(M)\ne S\). Since \(M\) is alive immediately above
\(S\), the only possibility is that its component survives the merge at
\(S\). By the elder rule, its birth value is therefore larger than the elder
birth value of \(C_{\rm other}\).

Let \(N\) be the maximum at which the second ascending branch of \(S\)
terminates. Since \(N\in C_{\rm other}\),

\[
f(N)
\le
\operatorname{birth}(C_{\rm other})
<
f(M).
\]

The branch ascends strictly from \(S\) to \(N\), so

\[
f(S)<f(N).
\]

Hence

\[
f(S)<f(N)<f(M),
\]

which is exactly \(\Gamma\).

The three cases exhaust the possibilities. \(\square\)

## Consequences

### Loop/crater closure

A high-level path joining the two arms above \(S\) cannot appear after the
birth of \(M\) without an index-one critical event. Its first appearance is a
saddle whose value lies inside the open persistence window. Therefore the
global window-saddle interceptor count already covers loop/crater failures.

### Exact relation to the C101 decomposition

Under the Morse–Smale and distinct-critical-value hypotheses,

\[
\{D(M)\ne S\}
\subset
\Pi\cup\Gamma.
\]

The additional-critical-point collar term retained by C101 is a conservative
analytic/local-normal-form residue. It is not needed to repair an omitted
topological loop class.

### Boundary of the theorem

The argument fails without the named hypotheses:

- a saddle branch may terminate at another saddle if Morse–Smale fails;
- simultaneous critical values can destroy the strict open-window ordering;
- on a noncompact domain, a global merge tree needs a windowed or intensity
  formulation.

Thus publication may either:

1. state the result conditional on Morse–Smale; or
2. consume the program-grade SARD-G theorem and retain independent specialist
   review as an external status qualifier.

## Status

```text
defect decomposition:
    CLOSED-HERE

loop/crater -> window interceptor:
    CLOSED-HERE UNDER R0

unregistered global path class:
    NONE UNDER THE HYPOTHESES

external SARD-G specialist acceptance:
    PENDING
```
