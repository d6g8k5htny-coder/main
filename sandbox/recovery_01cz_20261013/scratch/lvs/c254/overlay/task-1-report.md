# C254 Task 1 implementation report

Disposition: **PROPOSED FORMAL SOURCE / NOT_COMPILED**. Compilation, kernel audit,
local tests, hosted tests and executed negative controls: **NOT_RUN**. Scientific
effect: **NONE**. This report does not claim RED/GREEN, a verified theorem, C253
completion, tameness, a barcode, a source-owner branch update or external custody.

Author: OpenAI/Codex `/root/c254_source_implementation`, delegated AI work for
Dylan Roy. Organizational-independence credit: 0. The two API briefs are author
inputs; neither they nor this author report supplies nonauthor review. Root owns
separate review, integration, hosted admission, publication and custody.

Parent supplied claim 8a5c7a24-784b-4aaa-b951-cd9f37b0dc64, R17 Work Events 1559,
2026-10-11T15:57:44Z–17:57:44Z. This author did not reread the register.

## Delivered exact files

All paths below are relative to
`/Users/dylanroy/Documents/Codex/2026-09-19/connect-research-repository/work/continuation254/`.

| File | Bytes | Lines | SHA-256 |
|---|---:|---:|---|
| candidate/formal/ResearchFormalCoreR1/TorusTopology.lean | 11542 | 244 | efa80d2879758b6d6473a0024a308ff758c308cdc8e4b5f400f39dde63a7f6f4 |
| candidate/formal/ResearchFormalCoreR1/ClosedSuperlevel.lean | 7827 | 195 | 1fed3edada12b4dc7f5fcf874dae7ccde5d8f65af63cf103159debe58ba6789e |
| candidate/formal/TORUS_FILTRATION.md | 5611 | 90 | 971e70d69a96661e5be077eb7a2991dfc69288be73b1ebcaa62415bb943ccc8b |

There are 21 `torus_` and 20 `superlevel_` theorem declarations at column 1,
all in the sole namespace `ResearchFormalCoreR1`. Every exported definition is
used in a theorem statement or proof. In particular, the actual H0 composition
is consumed by identity and composition theorem targets for transitive auditing.
No target list, integration manifest, import root, predecessor source, workflow,
source-owner file or branch was changed by this task.

## Provided construction and proof text

The torus uses the actual additive quotient of `EuclideanSpace ℝ (Fin d)` by
the integer span of an L-scaled standard orthonormal basis. It needs L > 0 and
permits every finite d. The basis-sum identity converts integer coefficients to
the exact Field lattice formula; the membership theorem characterizes the
lattice by those vectors. The quotient equality theorem fixes the representative
orientation to y = x + latticeVec n, avoiding an unnoticed sign convention.

The concrete quotient's compactness uses the pinned lattice periodic-range
theorem, with the quotient projection as the periodic map; its covering property
uses the discrete-subgroup quotient covering theorem. Path connectedness comes
from the quotient instance. Periodic functions descend by the actual quotient
eliminator; continuity is separately proved from continuity of the original
function. A named literal-periodicity identity makes the Field convention exact.

The specified-start lift uses the pinned covering-map path-lift construction.
The endpoint is the endpoint of that lift, existentially quantified. Projection
and lifting prove a higher-endpoint path equivalence for any scalar predicate;
strict and closed threshold corollaries are separate. Essential/global-maximum
compatibility follows by projection surjectivity. These are topology and scalar
predicate statements, not critical-level attainment results.

The filtration uses F(t) = {x | -t ≤ g(x)}. It provides closedness for continuous
g, actual continuous inclusions and their laws, actual path-component quotient
maps and their laws, and equality of component classes iff ambient `JoinedIn`.
The topological functor is composed with the actual singular homology functor
in degree zero. Its coefficient-category assumptions are stated explicitly.

`Field.lean` is deliberately not imported by this standalone package, and its
owner's file remains untouched. The local ambient-space/lattice/periodicity
definitions use its literal formulas. Thus the source supplies exact formula
identities, not a compiled theorem naming `UniversalLaw.Spec.Field.latticeVec`.
The independently rehashed Field source matches the required
`f22c91e082cd7e738ba47c9c3534e47ac687d066f0b78586d15103cf1132535e`.

## Controls and their exact limits

Controls were specified in TORUS_FILTRATION.md and the first five closed-level
control statements before the construction bodies were added.

| Control | Wrong change or inference detected |
|---|---|
| Closed endpoint and strict endpoint | Replacing `≤` by `<`, or claiming the strict witness exists at equality |
| Parameter sign | Replacing `-t` by `t` |
| Inclusion direction | Mapping F(1) into F(-1) for the constant-zero function |
| Essential maximum | Supplying a strictly higher endpoint under global maximality |
| Strict threshold witnesses | Inferring a closed endpoint witness from witnesses strictly below it; the formal statement gives intermediate negative witnesses and rejects a nonnegative negative witness, not a compact Morse counterexample |
| Lattice translation | Treating translated representatives as distinct quotient points |
| Constant projected path endpoint | Ending a specified-start lift of the constant path at a different representative |
| No arbitrary endpoint | Combines the same quotient image with the impossibility of the fixed different endpoint whenever the chosen lattice vector is nonzero |

The fixed-endpoint countercontrol does not add a circle-loop parametrization.
It uses uniqueness of the lift of the constant path. Its nonzero-vector premise
is explicit; it is a reusable rejection statement, not a claim that d = 0 has
distinct representatives. None of these formal controls has been executed.

## Actual source exposure

Read locally: the task brief, PLAN global constraints/Task 1 and surrounding
scope, full QUOTIENT_API.md and FILTRATION_API.md, full C253 TASK.md, Field lines
1–200 and 285–326 plus a bounded declaration search, and the authored source
files. The C253 Field bytes were independently rehashed with `shasum`.

The following exact Mathlib primary files were fetched as text through raw
GitHub at commit `d13f23b723b8a846827a245b89c10fc7d3f11612`. Only the displayed
bounded passages and declaration searches were substantively read. No full
repository inventory or dependency checkout was performed. The toolchain pin
Lean 4.34.1 is inherited from the task and contributor evidence; this author did
not execute it or independently fetch lean-toolchain.

| Path below Mathlib/ | Direct read extent |
|---|---|
| LinearAlgebra/Basis/SMul.lean | 95–128 |
| Analysis/InnerProductSpace/PiL2.lean | 796–824; declaration search for toBasis/single/scalar operations |
| Algebra/Module/ZLattice/Basic.lean | 290–340, 405–440, 780–805 |
| GroupTheory/Coset/Defs.lean | 148–232 |
| Topology/Covering/Quotient.lean | 275–320 |
| Topology/Homotopy/Lifting.lean | 22–65, 192–213, 244–290 |
| Topology/Constructions.lean | 732–750, 793–812 |
| LinearAlgebra/Finsupp/LinearCombination.lean | 368–386 |
| Analysis/Normed/Lp/PiLp.lean | 110–138, 168–181; bounded declaration search |
| Topology/Covering/Basic.lean | declaration context 365–380 for eq_of_comp_eq |
| AlgebraicTopology/SingularHomology/Basic.lean | 20–57 |
| Topology/Connected/PathConnected.lean | 110–165, 206–220, 618–637 |

The URLs have the fixed prefix
`https://raw.githubusercontent.com/leanprover-community/mathlib4/d13f23b723b8a846827a245b89c10fc7d3f11612/Mathlib/`.
Contributor reports supply their independently stated Git blob metadata; this
author's raw text fetches were not independently Git-blob hashed. One raw PiLp
read timed out (`curl: (35) Recv failure: Operation timed out`); a subsequent
bounded fetch with a 20-second timeout returned the required lines. This was
an inert source-read failure, not Lean execution or a failed proof attempt.

## Verification performed and remaining elaboration concerns

Performed: AI static reconstruction against the listed primary APIs,
plain-text namespace/declaration/prohibited-token inspection, theorem counts,
byte/line counts and SHA-256 hashing. These are inert checks, not project tests,
compilation, kernel checking, theorem acceptance or proof correctness evidence.
The source text contains no prohibited proof escape declaration or tactic.

Most likely first-build risks, in order:

1. The scaled-basis coordinate simplifier must reduce the orthonormal-to-basis
   coercion, unit action, integer scalar action and PiLp evaluation as expected.
   The individual APIs are inspected; their combination has not elaborated.
2. Quotient projection/left-relation definitional reductions and typeclass
   inference for the concrete integer-span lattice have not elaborated.
3. The pinned Path/ContinuousMap coercions in specified-start lifting and the
   forward predicate-transfer term rely on definitional computation; no build
   has confirmed those exact terms.
4. Category universes, preorder morphisms, H0 coefficient-category inference and
   pointwise TopCat extensionality require hosted elaboration confirmation.

These are compilation risks, not known counterexamples to the mathematical
arguments. Root should preserve any first hosted error exactly and repair only
within the admitted scope. No successful compilation is inferred from inspection.

## Explicit remaining mathematical scope

Smooth quotient charts, smooth descent, derivative/Hessian invariance and Morse
index compatibility remain OPEN. Continuous descent does not discharge them.
Critical-free band and critical-level handle theorems, closed endpoint
attainment, merging versus nonmerging attachments, finite critical events,
elder label tracking, H0/component generator naturality, tameness, interval
decomposition and a finite-bar correspondence remain OPEN. Compact closed
superlevels can still have infinitely many path components for a general
continuous height function. No density, Kac–Rice, contact integral, once-counting,
regional collision, or scientific acceptance claim follows from these drafts.
