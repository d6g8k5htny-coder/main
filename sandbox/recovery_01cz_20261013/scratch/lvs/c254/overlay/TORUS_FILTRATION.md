# C254 torus and closed-superlevel source candidate

Status: proposed, uncompiled formal source. Local execution is forbidden by the
current HOSTED_ONLY boundary. Compilation, kernel audit and executed controls are
NOT_RUN. Scientific effect: NONE.

## Controls specified before implementation

1. Closed endpoint: for a constant function of value b, the unique point belongs
   to F(-b), while the strict inequality at that same height is false. This
   detects replacement of the closed filtration by a strict one.
2. Parameter sign: height 0 belongs to F(1). It would fail for the wrong threshold
   +t. An explicit witness also refutes inclusion from F(1) into F(-1).
3. Essential maximum: if every point has height at most g(M), there is no
   strictly higher endpoint. A real-valued interface death cannot create a finite
   bar without a nonessentiality hypothesis.
4. Representatives: translation by a lattice vector has the same quotient
   image. A lifted path ends at an existential representative determined by its
   specified starting lift. It must not be required to end at an arbitrarily
   chosen representative of its projected endpoint.
5. Endpoint attainment remains separate: strict thresholds h < 0 have supremum
   0 but contain no witness at 0. The quotient/path correspondence does not
   convert strict path inequalities to closed critical-level merger attainment.
6. Component equivalence and functor laws do not give an interval decomposition,
   elder labels or a barcode. No theorem below is described as those results.

## Constructed objects and provided proof text

`TorusTopology.lean` uses `EuclideanSpace ℝ (Fin d)` and the literal source-bound
Field formula `∑ i, EuclideanSpace.single i (L * (n i : ℝ))`. A named formula
identity and the integer-basis-sum theorem bind the coordinate convention. The
period lattice is the integer span of the L-scaled orthonormal standard basis;
membership is characterized by exactly those lattice vectors. The module uses
its actual additive quotient. Its proof text supplies compactness, path
connectedness, the covering projection, periodic continuous descent, and path
projection/lifting from a prescribed Euclidean representative. The key
higher-endpoint theorem works for an arbitrary scalar predicate P; its two
corollaries use h < g and h ≤ g without changing the final existential endpoint.

The construction needs only finite dimension d and L > 0. It includes d = 0;
the later C253 hypothesis d ≥ 2 is not needed for this prerequisite. Periodic
descent itself needs periodicity; its continuity theorem additionally needs
continuity of g. The path-predicate transfer does not require g to be continuous.
No quotient property or path-lifting theorem is added as an interface assumption.

`ClosedSuperlevel.lean` defines F(t) = {x | -t ≤ g(x)} for any height function.
Continuity of g supplies closedness; inclusions and their continuous maps do not
need it. Actual `ZerothHomotopy` quotient maps obey identity and composition.
Equality of their classes is equivalent to `JoinedIn` the ambient closed level.
The actual singular-H0 filtration is the topological filtration composed with
`AlgebraicTopology.singularHomologyFunctor` at degree zero and a coefficient
object in a category with the explicitly declared coproduct, preadditive and
homology structure. H0 identity/composition theorem targets consume this
definition for transitive axiom auditing; it is not an unreferenced placeholder.

To instantiate, take X to be `torus_quotient d L hL`, and g in the filtration to
be `torus_descend d L hL f hf`. `torus_descend_continuous` supplies its continuity
from continuous periodic f. `torus_closed_higher_path_iff` transfers C253's
closed higher-endpoint path predicate with the specified lift M; the component
equality theorem interprets paths as equal classes at parameter -h. This remains
an interpretation of a predicate, not an existence theorem at h = δ.

The source module deliberately does not import or alter `Specifications.Field`:
the standalone package receives the exact literal formula, bound to Field SHA-256
`f22c91e082cd7e738ba47c9c3534e47ac687d066f0b78586d15103cf1132535e`.
No claim of an elaborated theorem directly naming that external module is made.

## Explicit open dependencies

- Smooth quotient-manifold charts, smooth descended functions, derivative and
  Hessian invariance, and intrinsic Morse-index compatibility.
- Critical-free band invariance and the critical-level Morse local/handle
  theorem, including merger versus attachment inside a single component.
- Attainment at the critical endpoint, finite critical events, and label tracking.
- Naturality comparing the displayed component maps with H0 generators.
- Any finiteness/tameness claim, interval decomposition, endpoint-preserving bar
  correspondence, elder event, once-counting, or full theorem completion.

Compactness and closedness alone do not imply finitely many path components for
an arbitrary continuous height function. Membership at the birth height is not
a theorem that a new H0 class is born. An essential maximum has no higher
endpoint and acquires no finite death from this construction.

## Evidence boundary

All Lean declarations are proposed proof text. There was no Lean/Lake execution,
local test execution, hosted execution, kernel audit or RED/GREEN result in Task
1. The assigned task report records the exact primary source exposure, inert
checks, hashes and likely elaboration risks. API contributor reports are author
inputs, not independent review; root coordinates a separate source review.
