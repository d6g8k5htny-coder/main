# Exact H0 endpoints from the dyadic samples

[Experiment](README.md) · [Spatial certificate](HESSIAN_GRID.md) ·
[Interpolation and bin comparison](APPROXIMATION.md)

This note justifies the integer algorithm in `exact_h0.py` and its independent
connectivity checker. It computes ordinary H0 of a specified finite periodic
vertex cubical filtration. Applied to the regenerated samples identified by
`hessian1`, it supplies the previously missing identification of computed
endpoints with that abstract filtration. It does not change any historical
NumPy or GUDHI result, the frozen C38 receipt, or a Gaussian or infinite-field
acceptance condition. The claim is deterministic and uses no sampling law.

## Object, units and graph

Let `n >= 3`, `V = (Z/nZ)^2`, and let each supplied `a_v` be a strict Python
integer. The layout is `v = x*n+y`, with both coordinates ranging from zero
through `n-1`. Let `S` be a specified positive integer and set `s_v = a_v/S`.
For C38, `S = 2^96`; the integers are the real centers of the unchanged
`nodal_core.evaluate` computation, bound to the ordered digest, stage records
and nodal error in the frozen `hessian1` certificate.

The mathematical superlevel cubical complex `C_t` contains a grid cell if
every one of its vertices has supplied value at least `t`. Vertices and
edges are cells too. Its H0 is exactly that of the graph whose vertices have
`s_v >= t` and whose edges are the four periodic axis neighbors, with an edge
present when both endpoints are present. Indeed, each included square has
its connected boundary already in the graph. Attaching the square cannot
join two graph components or split one. This identification respects the
inclusions as thresholds decrease, so it identifies H0 persistence modules.

There are no diagonal edges in this graph. A fixed-diagonal PL interpolant
generally gives a different H0 module and is not computed here. The adaptive
minimum-corner triangulation in `APPROXIMATION.md` identifies this cubical
module with a suitable PL module for the existing approximation theorem;
the algorithm does not need to construct that triangulation. Periodic seams
are graph edges, with no duplicate boundary vertices or extra boundary strip.

We work over any one fixed field. A reported finite pair `(b,d)` consists of
integer numerators with `b > d`; it represents the superlevel interval

```text
(d/S, b/S],
```

or the ordinary sublevel interval `[-b/S,-d/S)` of `-s`. At the death level
the dying class has already merged. Its exact lifetime is `(b-d)/S`. Zero
pairs are omitted from the diagram. The connected torus graph has one
essential class with birth `max_v a_v/S`, representing `(-infinity,b/S]`
in superlevel coordinates. No longest finite interval is removed.

## Descending integer sweep

Give each vertex the elder priority `(a_v,-v)`: larger values are older, and
among equal values the smaller row-major vertex id is older. Process vertices
in descending value, with increasing vertex id within a tie. When activating
a vertex, inspect all its already active axis neighbors. Join different
components; retain their older elder and pair their younger elder's birth
with the current integer value. Keep a pair only when birth exceeds death.

The disjoint-set data structure has two separate roles. Its parent and size
arrays implement union by size and path compression. A separate elder array
stores the mathematical elder of each component. A large data-structure tree
is not allowed to overrule the older birth. An inactive parent is marked -1;
all active vertices, including negative-valued ones, are processed.

After each whole equal-value block at level `c`, the active vertices are
exactly `{v:a_v>=c}`. Each edge of their induced graph was inspected when its
later endpoint activated, and unions connected exactly its components. The
elder metadata is the maximum priority in each component, by induction on
the unions. Cyclic edges whose endpoints already share a root cause no merge.

To justify the persistence pairs, refine the finite filtration by its actual
vertex and edge insertion order. Each new vertex initially creates one H0
basis vector. Adding an edge within a component acts identically on H0. An
edge joining two components identifies their two H0 vectors. If their elder
births are `b_old >= b_young`, the older class already exists at the younger
birth. From that birth until the merge, replace the younger generator by the
younger vector minus the older vector, transported by the inclusions. At the
merge this difference becomes zero, while the older generator continues.
Thus this change of basis splits off an interval from `b_young` to the merge
level. Inducting over all insertions proves the elder pairing; subtraction
also works in characteristic two.

This refined insertion order is only a way of computing the original
filtration: every insertion within a block has the same real parameter.
Restricting to the states after complete blocks identifies those equal
parameters. Pairs with identical birth and death become zero modules and
are discarded; positive pairs retain their exact endpoints. This explains
why equality comparisons are exact and why no generic-position assumption
or artificial value perturbation is needed. Other tie orders may choose
different representative labels but cannot change the positive interval
multiset of this module. The declared priority fixes the labels used here.

There are `N=n^2` vertex insertions and exactly `N-1` successful component
unions, since the final periodic graph is connected. Consequently

```text
positive_pair_count + zero_count + 1 = N.
```

The final elder is the smallest vertex id attaining the largest value.
Every finite endpoint is an input integer; division by `S` is exact rational
arithmetic. Hence this computation contributes zero endpoint rounding error
relative to the supplied cubical filtration. It does not set the separate
nodal or spatial approximation error to zero.

## Independent verification by connectivity

`verify_by_connectivity` does not rerun the union-find sweep. It first finds
all connected equal-value plateaus by graph searches. A plateau is a regional
maximum if it has no neighbor with greater value. Its representative is its
smallest vertex id. At a threshold equal to a plateau's value, a plateau with
a higher neighbor is attached to an already present component; a regional
maximum plateau creates one new component. Thus the positive and essential
births, counted with multiplicity, are exactly the regional maximum plateaus.

Order those representatives by the same elder priority. The oldest one is
essential. For every other representative `p` of birth value `b=a_p`, define

```text
d_p = max over paths from p to an older regional maximum
          of (the minimum vertex value along the path).
```

There is at least one such path because the full periodic graph is connected.
Only finitely many vertex values can be a path minimum, so the maximum is
attained. At any threshold `t`, the component of `p` contains an older
maximum exactly when a qualifying path has minimum at least `t`. The elder
invariant above therefore makes `d_p` the first merge level at which `p`
dies. Distinct regional maximum plateaus are separate at their common birth
level, and a regional maximum has no higher neighbor; hence `d_p < b`.

For a BFS started at `p`, testing for an older regional maximum can be
replaced by testing whether it reaches any vertex `u` with

```text
a_u > b, or (a_u = b and u < p).
```

An older maximum representative itself satisfies this condition. Conversely,
a reached vertex above `b` has a nondecreasing path to a regional maximum
above `b`: traverse its equal plateau, move to a higher neighbor if one
exists, and repeat; the finite graph makes this process terminate. This path
stays above the current threshold. A reached vertex at value `b` with id
smaller than `p` cannot be in `p`'s plateau, since `p` is its minimum id.
Its own plateau either leads to a higher maximum by the same argument or is
a maximum with an even smaller representative. In either case an older
maximum is connected. This proves the equivalence used by the checker.

For each proposed finite pair `(b,d)` labelled by `p`, the checker proves:

1. `p` is the representative of the asserted regional maximum and `b=a_p`.
2. At the inclusive threshold `d`, a BFS from `p` reaches an older vertex.
3. At the threshold `d+1`, a complete BFS from `p` reaches no older vertex.

All vertex values are integers, so the graph at `d+1` is exactly the graph
with values strictly greater than `d`. The last two conditions establish
`d_p >= d` and `d_p < d+1`, respectively, and the integer-valued `d_p` must
therefore equal `d`. They test both sides of the endpoint, rather than
checking only one nearby threshold or using a rounded value.

The checker also requires that finite labels exhaust every nonessential
regional maximum exactly once, the one essential value and label are
correct, the zero count satisfies the conservation equation, and the entire
output has the declared strict types and canonical order. Thus a missing bar
cannot hide between the tested endpoint levels. This completeness step is
essential: checking only the Betti numbers at selected thresholds would not
exclude a missing bar born and killed between them.

## Exact bin transfer to the finite polynomial

Let `Q` be this exact sample diagram, and let `P` be a target diagram with an
independently justified epsilon-matching to `Q`. For the specified finite
rounded-coefficient polynomial and the C38 samples, the existing theorem
supplies `epsilon=eta+B`, with its exact rational value from `hessian1`.
This paragraph invokes that existing matching and does not infer it from
the graph computation or a hash.

Set `delta=2*epsilon`, and let `N` count only finite positive bars. Matched
finite lifetimes differ by at most `delta`. A bar matched to the diagonal
has lifetime at most `delta`, including equality. For a half-open bin
`[a,b)`, `0<a<b`, the algorithm evaluates exactly

```text
N_Q([a+delta,b-delta)) <= N_P([a,b)),
N_P([a,b)) <= N_Q([a-delta,b+delta))  if a > delta.
```

For the lower inequality, a counted sample lifetime is strictly above
`delta` and therefore has a finite partner. Its partner is at least `a`
and strictly less than `b`. Injectivity of the matching proves the count
bound. For the upper inequality, `a>delta` ensures every counted target bar
has a finite sample partner; its lifetime is at least `a-delta` and strictly
less than `b+delta`. This proves the other injection. Essential classes and
diagonal points are never counted. If the contracted lower endpoint is not
less than its upper endpoint, its count is zero.

The clean upper bound is deliberately absent (`None`) when `a<=delta`.
At equality, a target bar of lifetime exactly `delta>0` can be matched to
the diagonal in an empty sample diagram, so a clean upper bound zero would
be false. The contraction is nonempty exactly when `b-a>2*delta`; equality
leaves an empty half-open interval. All comparisons use `Fraction`, so bars
on the lower endpoint are included and bars on the upper endpoint excluded
even at an exact rational equality. Dividing endpoints or lifetimes through
float64 would not supply this guarantee.

These bounds concern the specified finite polynomial's realization counts.
They do not supply a confidence interval, an ensemble expectation, an
asymptotic lifetime range, a remainder constant, or a uniform spectral tail.

## Interfaces, cost and checks

`compute(values,n)` accepts a flat list or tuple of strict integers. It returns
integer `intervals`, aligned `birth_vertices`, the one-entry `essential`
and `essential_vertices` lists, and `zero_count`. Finite triples are sorted
by birth, then death, then vertex id. Boolean or floating-point values and
dimensions are rejected. The grid has no numerical-resolution limit beyond
the signed index capacity of the array implementation and available memory.

`verify_by_connectivity(values,n,result)` returns `True` only after all the
checks above, otherwise raises `ValueError`. Its topology is implemented
with explicit boundary branches rather than the sweep's modular neighbor
expressions. The tests use a third construction: undirected edges generated
in the two positive coordinate directions.

`bin_transfer(barcode,scale,epsilon,edges)` requires a positive strict integer
scale and exact rational error and bin edges. It returns the supplied bin,
the exact sample count, the target lower count, the conditional target upper
count, and the two strict-condition flags. It validates the diagram schema
but cannot establish its provenance or the caller's claimed matching.

For `N=n^2`, sorting costs `O(N log N)`. Four neighbor probes per vertex and
union by size with path compression cost `O(N alpha(N))`; integer comparison
and arithmetic have their ordinary bit costs. Storage is `O(N)` plus the
input integers and returned positive pairs. No floating point, numerical
package, simplex list or list of all edges is needed. On a million nodes the
three disjoint-set arrays occupy about 12 MiB with 32-bit array entries;
the Python integer samples and sorted index list are additional storage.

The independent verifier uses `O(N)` storage and at most `O(N M)` work for
`M` regional maxima. Searches often finish much sooner, but this is not
assumed for correctness. The wrapper records actual reproduction and
verification evidence; algorithmic feasibility is not a substitute for
executing that check on the identified input.

The focused tests compare every source-to-target H0 rank at all sample levels
on all 512 binary 3-by-3 grids and seeded tied grids. At thresholds `p>=q`, an
independent BFS counts components of the graph at `q` that meet a vertex of
value at least `p`. A proposed barcode must give that same rank:

```text
number of finite (b,d) with b>=p and d<q
  + number of essential births b>=p.
```

Literal controls distinguish periodic seams, fixed diagonals, wrong elders,
plateau ties, the essential class and finite lifetimes of one integer unit
above a `2^96` offset. Deliberately omitted, shifted and mislabelled bars are
rejected by connectivity verification. Exact bin-boundary and malformed-type
controls cover the transfer calculation. These are engineering and finite
mathematical checks; review lineage, scientific acceptance and any formal
verification remain separate evidence.
