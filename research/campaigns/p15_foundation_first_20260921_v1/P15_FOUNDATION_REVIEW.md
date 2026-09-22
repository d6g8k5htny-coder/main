# Nonauthor technical review of the P15 finite-rank successor and foundations

Date: 2026-09-21. Reviewer: `/root/architecture_triage`, a nonauthor of the
successor and a source-exposed OpenAI reviewer. Organizational independence
credit: **0**. This is a written mathematical review, not a proof-assistant
verification, an external acceptance decision, or a novelty determination.

**Finding: no substantive mathematical gap found in the reviewed successor
or the consumed arguments listed below.** One intermediate zero-probability
clarification was requested and is present in the exact final proof reviewed.
The finite executable companion and its tests are outside this review; the
parent is checking those separately. This finding does not depend on their
passing, and finite examples would not prove the analytic foundations.

## Exact successor reviewed

`p15_hypergraph/PROOF.md`, 12,060 bytes, SHA-256
`f7994948d7f3b81d73dd700296cb93184c6f5a21b8664d1dd96ace59eed80f1c`.

This review applies to those bytes. It checks the fixed-finite-rank
complete-transversal composition and its stated structural hypotheses. It
does not assert a decomposition for arbitrary downsets, a palette independent
of macro rank, incomplete transversal lifting, or safety of overlapping child
supports. No q0, canonical scientific status, original-prize result, or review
independence requirement is changed.

## Exact foundation bodies read

The source directory is the eligible repository mirror
`/Users/dylanroy/Documents/Codex/research-main/drive/mirrors/2026-09-16 — PRIZE PROBLEM RECONNAISSANCE — INDEPENDENT TRACK/06_TALAGRAND_DISCRETE — RESTRICTED_PROOF_CANDIDATES/`.
The first six rows below were also compared with the successor's selected
source copies where present. Hashes authenticate bytes; the mathematical
checks in subsequent sections are separate from that authentication.

| Source | Bytes | SHA-256 |
|---|---:|---|
| `P10-A_01_COMMON_PALETTE_SUBSTITUTION.md` | 3,631 | `af4b470824dd31ae645f51c8a286baee281487e890102fe99526b1e6573ced87` |
| `P11_ATTACHED_ee4a16e1_01_GRAPH_POTENTIAL_AND_CORE_RECOLORING.md` | 5,045 | `21c3d944ccc5dc1de3146b10ee7ddd1b5a869a0f618359e906415786b11eec24` |
| `P11_ATTACHED_ee4a16e1_02_RANK_THREE_COMPOSABLE_POTENTIAL.md` | 5,984 | `af8bf4813391c67ec148d08d12c4618703d9460ff4ec5e3787444a211c448fc4` |
| `P11_ATTACHED_ee4a16e1_04_ALL_FINITE_PRIMITIVE_RANKS_HAZARD_CLOSURE.md` | 8,583 | `0ce1352b67899d3cd805def6ea25cc7cee8a33ac6dfbb5e73bd897565d60e1ea` |
| `P14-A_SCALAR_SANDWICH_THEOREM.md` | 6,004 | `b3c7ea04ee7970f6fdc148bf71268d00b1482ea098343dccb5905f238532db05` |
| `P15-D_COMPLETE_TRANSVERSAL_GRAPH_COMPOSITION.md` | 7,710 | `ad2c55b8609781d8c748cc93db1f297bf13c534667f60aac86ab59e8230ca31c` |
| `P11_D_ZERO_Q_ENDPOINT_ADDENDUM_a7f03b3f.md` | 1,391 | `7935e89f9d95ec31d2b13cd73cdf293962de45db393a466c1e4dc56d41ac4489` |

The exact attached P11 snapshot matters. The unattached rank-three source has
a different palette and a hazard-based coefficient; its constants were not
substituted into this review. The reviewed attached rank-three argument uses
368640 colors and the coefficient `3761/4608` times the failure probability.

## P14-A: local scalar foundation

The normalized seed-count identity is valid: setting `t_i=p_i/(1+p_i)`
turns the finite subset sum into exact configuration probabilities times
`product(1+p_i)`. Monotonicity then bounds its bad-event probability by the
original product-law failure probability. The chosen bad seed and remaining
coordinates determine a generator; dropping the disjointness condition in
the extension sum enlarges a nonnegative upper bound.

For the filtered-prefix step, avoiding a heavy generator forces
`P_(u_j)>j/R` after the first heavy prefix. The decreasing step function on
original-probability intervals therefore contributes at least `a_(u_j)/R`
on the stated disjoint intervals. Zero-probability intervals have zero
length, but the generator definition still covers zero-probability subsets.
This establishes the required setwise bound, not a sampled implication.

The seed has size `ceil(k/4)`, the remaining size is `floor(3k/4)>=3`, and
each remaining size occurs at most twice. These facts give the stated `Q/2`
sum after the exponential/factorial bound. The Cantelli step gives `M<4*kappa`
from good probability at least one third. Strict next-fit is also sound:
adjacent bins have combined load at least one; if `K+1` bins open, pairing
bins and accounting for the final positive item gives total load strictly
greater than `K/2`. Thus equality and zero coefficients do not break coverage.

The scalar-sandwich conditions are substantive hypotheses for every subset,
including zero-probability configurations. This review does not prove such a
sandwich exists for arbitrary local families.

## Attached P11-A and P11-B: graph and rank-three foundations

For the graph second moment, ordered intersecting edges contribute at most
`2*c*mu`, giving `E Y^2<=mu^2+(1+2c)*mu`. The first-selected-core events are
disjoint and use only remaining neighbor coordinates, so the extraction
probability estimate is legitimate. Core and residual palettes are disjoint.

In the robust graph variant, `Q<=3/4` gives
`sum_C p <= (4Q/3)log(8) < 3Q`. The two palette budgets yield exactly
`27/256+5/32=67/256`. At inflation four the palette is
`512+2048=2560`, as required by the attached rank-three source.

P11-B uses two independent samples of probability `p/2`; their union is
stochastically dominated by the original p law, not asserted equal to it.
The high pair is chosen from the first sample, leaving its completion event
in the second sample independent of that choice. Weighted extension mass,
not unweighted codegree, drives that estimate. The residual triple
dependency load `3*100+3*5=315`, price inflation factor eight, and separate
core/residual colors give the stated low-triple budget. The exact sum is
`335/1152+269/512=3761/4608`.

Both covers are certificates for fixed colorings with generators of size at
least two. Consequently an independent binary refinement before any lifting
retains each generator with probability at most one half and still covers
every monochromatic original macro edge. This justifies the successor's
737280 colors and coefficient `3761/9216`; it is not a refinement of an
arbitrary obstruction cover containing singleton generators.

## Attached P11-D: fixed-rank foundation and endpoint repair

The graph base case has budget `305Q/768<Q/2`. The proper-link induction
groups ordered intersecting pairs by their exact nonempty intersection;
the remaining edge parts belong to lower-rank links. Bounding those links
by the previously obtained moments gives the recurrence as written, without
assuming independent overlapping link events.

The high-singleton peeling uses first-selected events and the current links,
so previously removed coordinates cannot corrupt its independence argument.
In the residual, minimizing the collected cores preserves the completion
witnesses and ensures every removed edge contains an actual retained core.
There are no singleton cores after peeling. The q/t sprinkling identity is
exact, and `q>=a*p` proves the required price inflation bound. Three-color
refinement acts on generators of size at least two, before product assembly.

For the remaining uniform r-edges, every proper link has the required hit
cap. The full `(A*d)^r` price factor remains in the residual palette. Separate
core colors and product colors on the residual give the total
`15/128+5/28+1/8=377/896<1/2`. I found no missing mixed-rank or incidence
assumption in this induction.

The original strict core estimate is not literally valid at Q=0. The exact
addendum repairs this by retaining every zero-price original edge as a
one-color certificate. This issue is explicitly resolved in the successor,
including the intermediate case `Q_H=0` while the global Q is positive:
each macro edge has product q zero, hence product c zero under `c_i<=2q_i`.
The successor then uses all those edges and a constant macro coloring.
No strict `0<0` argument is consumed.

## Successor composition, singleton handling and empty generators

The occupancy vector is taken under the original product law. The actual
occupancy-cover cost is `c_i=min(1,sum z_v)`, and the hazard identity gives
`c_i<=phi(q_i)<=2q_i`. It is not replaced by q_i. This bound remains valid
at q_i equal to zero or one.

If an original set avoids every lifted generator, its occupied index set
avoids every macro generator: for each occupied index, choose an occupancy
generator contained in that block's selected coordinates, then take their
union. This argument remains valid when an occupancy generator is empty.
Disjoint block supports justify multiplication of description costs;
collisions of lifted descriptions can only reduce the resulting family cost.
Any color refinement must precede lifting, since lifted generators can have
size one or zero. The successor does so.

Local good events are independent across blocks; local goodness and macro
goodness are not assumed independent. Their two separate bounds yield
`(sum Q_i)/2+Q_H/2<=H_global`, or the stated rank-three coefficient
`1/2+3761/9216=8369/9216`. The weaker all-probability primitive bound alone
would not justify that budget. The successor consumes the stronger low-Q
certificate instead.

All genuine original singleton forbidders are removed first, including whole
blocks forbidden by macro singleton edges. If a block becomes empty, every
incident macro edge is deleted rather than shortened. This preserves the
actual occupancy event. Original removed coordinates are independent of the
remainder, so singleton hazards add exactly. The stronger rank-three
coefficient is correctly restricted to the singleton-free low-Q case.

The zero-failure, empty-generator, empty-family, empty local family and empty
macro-edge cases are distinguished. The complete-transversal condition is
load-bearing: it is precisely what makes macro failure a subset of actual
global failure. The proof does not apply to an arbitrary sparse cross-block
lifting.

## P10-A: optional disjoint substitution

The exact common-palette identity is valid in both directions. Hard child
blocks are assigned wholly to their parent color; soft blocks are decomposed
inside every color. Downward closure and empty-set admissibility justify the
subset and padding steps. Generator-cost multiplication is valid on disjoint
supports, and child failure probabilities remain separate from child cover
costs. The successor's all-probability, transformed-price conclusion has the
required primitive interface. This does not license reusing original
coordinates among different child supports.

## Review disposition

The reviewed mathematical implication can proceed to the parent's separate
companion validation and delivery checks. The exact source arguments above
are written proofs checked in this review, rather than numerical examples
standing in for proofs. Their scope and structural hypotheses remain attached.
External organizational review and historical novelty are still unresolved;
this review supplies neither. No repository file was edited during review.
