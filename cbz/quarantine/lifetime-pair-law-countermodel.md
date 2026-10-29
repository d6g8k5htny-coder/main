# Conditional countermodels to universal pair-law conclusions

The constant-zero argument below is source-level and has not been kernel-verified.
The two constant-one declarations were compiled and kernel-rechecked at the
recorded branch cut; they retain the supplied-interface hypotheses. It concerns the precise abstract definitions in
[`Specifications/Lifetime.lean`](https://github.com/d6g8k5htny-coder/Math-/blob/8151f592918a845bf19eb8eafaf8444540a32734/specifications/lean_statements_20261010/Specifications/Lifetime.lean)
(Git blob `f2508f7f2e3fd42f22f5aa52bc5ead5ff9e73b9a`), especially
`PairLawInterface`, `ParentUniformSelectionA` and
`PlanarTwoSidedRateE4`.

Fix \(L>0\), objects \(G:\mathrm{GaussianFieldInterface}(2,L)\) and
\(I:\mathrm{SuperlevelBarInterface}(2,L)\), and an orthonormal frame \(R\).
The argument is conditional on these objects: it does not construct the
elder interface or claim that all its hypotheses are satisfiable.

## Constant zero: the upper predicate can fail

Define \(p(r,b,k,R')=0\) for every input. This satisfies the two
probability fields of `PairLawInterface`. For \(B=[0,0]\), \(K=[1,1]\),
suppose `ParentUniformSelectionA` held with \(r_*>0\) and \(C\).
Put
\[
 r=\min\!\left\{r_*/2,\ \frac1{2(|C|+1)},\ 1\right\}.
\]
Then \(0<r\le r_*\), and the predicate, evaluated at \(b=0,k=1,R\),
requires \(1\le Cr^3\). But \(r\le1\), so
\[
 Cr^3\le |C|r^3\le |C|r
 \le \frac{|C|}{2(|C|+1)}<\frac12,
\]
a contradiction. This is a conditional counterexample to universal
validity over all interface inhabitants, with the same supplied-object
caveat as above. It is not a counterexample to the intended geometric law
and is not a kernel receipt.

## Constant one: the upper predicate holds but the lower rate fails

Define \(p(r,b,k,R')=1\) for every input. This is a
`PairLawInterface` over the given \(G,I\), since \(0\le 1\le 1\).
The interface has no field binding this probability to the actual elder
event. For every allowed pair of windows, `ParentUniformSelectionA`
holds for this artificial law: choose \(r_*=1\), \(C=0\). Both sides
of its stated inequality are then zero.

For the windows \(B=[0,0]\) and \(K=[1,1]\), the remaining hypotheses of
`PlanarTwoSidedRateE4` are satisfied. If its conclusion held, it would
provide \(c>0\), a constant \(C\), and \(r_*>0\) such that, in particular,
\[
  c r^3 \le 1-p(r,0,1,R)=0
  \quad (0<r\le r_*).
\]
Taking \(r=r_*/2\) makes the left side strictly positive, a contradiction.
Thus, for any supplied \(L,G,I,R\) satisfying the displayed assumptions,
the original upper predicate holds for this law but the original E4
predicate fails. No exponent, constant requirement, or other source bound
has been relaxed.

This does **not** refute the source theorem about its intended geometric
probability: no identification of the constant law with that probability
has been made. Nor does it establish an unconditional inhabitant of the
elder interface. The intended mathematical bridge remains quarantined as
Q-1048; Q-1011 and Q-1045 retain the related upper-interface boundaries.

The separate Lean source at Math branch commit
`5a470bfd739aba22e32b91587d5dfc031a7de3fc` formalizes the constant-one
result. Hosted run `40957238766` built the specification and source, printed
only the three approved axioms for each theorem, and ran `leanchecker`
successfully. [Source-bound observation](../evidence/lifetime-countermodel/observation.json).
The artifact ZIP has not independently been downloaded; the source has not yet
been merged to current Math main.

A separate [constant-zero Lean candidate](https://github.com/d6g8k5htny-coder/Math-/tree/cb1fe3d189d94ee207750fc005eadf0596fb9cef) is prepared at
`cb1fe3d189d94ee207750fc005eadf0596fb9cef`. Its compilation, axiom
report and kernel recheck have not yet been observed. The source-level
zero-law argument above remains separate from the two verified constant-one
declarations.
