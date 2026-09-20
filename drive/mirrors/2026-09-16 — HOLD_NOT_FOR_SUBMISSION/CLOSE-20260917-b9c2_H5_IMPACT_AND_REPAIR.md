# H5-REPAIR-002 — Downstream impact and complete scalar-kernel correction

Task DQ-CLOSE-20260917-b9c2, 2026-09-17. Author-side source repair and executable counterexamples. Frozen source files remain unchanged. This concerns the H5 numerical integral engine; it is distinct from P02-LM-007's analytic fifth-derivative supremum theorem.

## Disposition

The earlier odd-spectral-moment defect reaches both numerical lower and upper bounds. Its downstream impact is now traced, and actual degree-six CoarseBox kernel factors are falsified at the archived 40-digit working precision as well as at 100 digits. A second defect—omitting the infinite normalization denominator's tail—is reproduced at high precision and repaired in the same additive successor.

The repaired kernel passes the actual archived dominant-box calculation, giving

\[
I_{box,hi}=0.000013285433832093048058366549502018472,
\]

inside the archived acceptance window (10⁻⁶,10⁻⁴). This is a scoped replay. It does not revalidate the complete cover, the remaining legacy arithmetic, H5 totals, D1, or all-small-r interpolation. The affected numerical values have not all been shown false; their inherited certification argument needs replay and review.

## 1. Exact source identities

- Frozen archive: Drive `1vSI-evINWskhXVyiZ0slt-rLT74sPpXH`, SHA-256 `a2136bc033f349382f9896896347da7a6dabde3334103276ad04db9205aa2b5b`.
- Frozen h5_kernel.py: SHA-256 `b2e6014c2fd6eb9e72b2104cc828a764e280f76b8396a02edd37654b6ad83ad7`.
- New kernel candidate: SHA-256 `60c63257e949fd4d5311900d2963ca1722f41d456179d50c0567f9706702dd83`.

H5_AFFECTED_OBJECTS.json binds the relevant archived drivers and totals by exact path, size and SHA-256. H5_SOURCE_DEPENDENCY_SCAN.json preserves the broader archive scan, including historical snapshots and mutation copies; import hits alone are not treated as proof that an output was affected.

## 2. Actual call chain

The concrete chain is:

1. `Lattice.lambda_q` supplies the derivative majorant used by `KernelSeries.tm_factor`.
2. `cross_tm` multiplies those one-dimensional Taylor models.
3. Both `StationBox` and `CoarseBox` consume these products. CoarseBox explicitly sets Taylor degree six and builds its cross-covariance blocks with `cross_tm`; its description as having “no TM inverse” does not remove its Taylor-model dependence.
4. `h5_run.Ctx.box`, `bound_cell_fine`, near/rim probe calculations and lower patches feed the numerical shards. Upper calculations use `CoarseBox` and `coarse_fine_hi`; lower patches use `StationBox` and `GBounds`.
5. `h5_merge.assemble` consumes those recorded shard values into versioned totals. `h5_promote` consumes the banked records and rung totals. Arithmetic recomputation of their sum does not re-establish the validity of an input enclosure.

Consequently a repair/replay decision is required for H5-derived numerical claims consumed by D1 and rung-promotion documents. Unrelated analytic certificates and separately proved kernel bounds do not lose their status merely because they also use the name H5.

## 3. Odd absolute-moment repair and actual consumer witnesses

For frequencies kj=jπ/12 and weights wj=exp(-kj²/2), the derivative majorant must use Σwj |kj|^q. The frozen implementation uses Σwj kj^q. Odd q cancels, although odd derivatives away from zero do not vanish.

At the exact archived dominant-box setup r=.05, angular center 165°, distance .040 from M and half-width .0015, CoarseBox degree six requests odd remainder orders q=n+7 for even kernel orders n=0,2,4. At 100 digits, the endpoint discrepancies and repaired remainder bounds are approximately:

| Kernel order n | Actual endpoint error, lower estimate | Repaired remainder, upper estimate |
|---:|---:|---:|
| 0 | 1.36560×10⁻²³ | 1.29835×10⁻²² |
| 2 | 1.22843×10⁻²² | 1.03868×10⁻²¹ |
| 4 | 1.35061×10⁻²¹ | 1.03868×10⁻²⁰ |

The old enclosures are disjoint from independently evaluated image-space endpoint intervals. All repaired enclosures contain those intervals. The same three disjointness/containment checks pass with the driver's 40-digit working precision. The exact receipts retain all six witnesses and their precision.

## 4. Missing normalizer tail

Write the truncated normalizer ZJ, true omitted mass t0≥0, truncated derivative numerator NJ and omitted derivative numerator en. For the exact truncated value v=NJ/ZJ,

\[
\frac{N_J+e_n}{Z_J+t_0}-v=
\frac{e_n-vt_0}{Z_J+t_0}.
\]

If `tail(n)` bounds |en|/ZJ and `tail(0)` bounds t0/ZJ, a sufficient error radius is

\[
\operatorname{tail}(n)+|v|\operatorname{tail}(0).
\]

The candidate adds exactly this outward interval correction. The old code added only the first term. At s=12/117 the first omitted spectral frequency has phase π. A 340-digit independent image-space enclosure of k(s) lies below the old interval by more than 3.83588×10⁻²⁰⁵. The repaired interval contains it. This demonstrates a mathematical defect in the precision-independent kernel-enclosure claim; the effect is far below the archived 40-digit arithmetic resolution. It is not used to claim a visible 40-digit totals error.

## 5. Reproduction and scope

Run `python3 closure_round2/h5_kernel_repair.py` to reconstruct the candidate from the pinned source, `python3 closure_round2/h5_downstream.py` for the seven counterexample/repair checks, and `python3 closure_round2/h5_downstream.py --replay` for the archived dominant box. Normal and optimized Python are checked. The candidate changes two mathematical ingredients and the misleading majorant docstring; it does not silently replace a frozen dependency.

The source defect and affected call paths are settled by these witnesses. The repaired scalar-kernel obligation is discharged author-side. Complete downstream certificate replay, other arithmetic soundness issues, cross-provider review and theorem promotion remain open.
