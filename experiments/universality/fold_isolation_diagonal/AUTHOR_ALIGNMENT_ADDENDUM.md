# Author alignment addendum to the restricted collinear calculation

16 October 2026 UTC. Author: OpenAI/Codex `/root/core_theorem_audit_20261015`. This is an append-only clarification for handoff. It leaves the two frozen objects below unchanged and makes no additional mathematical or process claim.

| Frozen object | SHA-256 |
| --- | --- |
| [COALESCING_CONTACT_CALCULATION.md](COALESCING_CONTACT_CALCULATION.md) | `39a63e1b00393b49e7b6d049c0c9ea7be82436352eed64014155a63aa8c7fb36` |
| [NONAUTHOR_COALESCENCE_REVIEW.md](NONAUTHOR_COALESCENCE_REVIEW.md) | `2f766d8aa5541602ace7bbc70fd360e3a342d2cd258c56c44a35503e3e8624d5` |

The complete review was read. I accept both nonblocking wording corrections. I found no mathematical disagreement with the review's restricted interpretation. The following paragraphs give the precise reading of the original report's basepoint and simplex sentences; they do not rewrite its historical bytes.

## Basepoint convention

In each four-site configuration, replace the original coordinates
`z_i=z+x_i u` by

```text
z' = z+x_1 u = z_1,
x_i' = x_i-x_1,
z_i = z'+x_i' u,          x_1'=0.
```

Then rename z' as z and x_i' as x_i. Thus §3's limiting gradient jets are at the **first physical site z=z_1**, and the segment is `0=x_1<x_2<x_3<x_4`. In the original unrecentered notation they are at `z+x_1 u`, rather than necessarily at the original z. Small diameter alone does not make the original numerical x_i approach zero.

This is a change of coordinates for the same physical sites and same field. The anchor may vary with the configuration. Finite-Fourier mean-square convergence is uniform in the torus basepoint and frame; stationarity makes the covariance independent of the basepoint. No isotropic replacement, alteration of spectral weights, field resampling or independence of contacts is involved. Differences, radii, eta, V and D_i are unchanged.

## One simplex normalization

Use **ordinary simplex-volume measure**, not probability measure, in the divided-difference integral. For j>=1 define

```text
Delta_j = {(t_1,...,t_j): t_i>=0, sum_i t_i<=1},
t_0=1-sum_i t_i,
volume(Delta_j)=1/j!.
```

The convention is

```text
a[x_0,...,x_j]
 = integral_(Delta_j)
     a^(j)(t_0 x_0+...+t_j x_j) dt_1...dt_j.
```

There is **no additional factor 1/j! in the integrand or outside this integral**. The j=0 convention is point evaluation with mass one. At coincident nodes, this gives `a^(j)(x)/j!`. If probability-simplex averaging is used instead, one multiplies that expectation by 1/j! exactly once; that is only an equivalent alternative description.

Accordingly, the original §3 limit is read literally as

```text
(F_u, F_uu, F_uuu/2!, F_uuuu/3!,
 F_v, F_uv, F_uuv/2!, F_uuuv/3!)(z_1).
```

The phrase in the frozen report about an “average … divided by j!” together with a measure of mass 1/j! must not be read as applying the factorial twice. Its intended corrected interpretation is the unscaled-derivative/simplex-volume formula above. The displayed limiting jets already use that convention. The same convention applies to the Hermite divided differences in the twelve-row discussion, including orders through seven. No covariance determinant power or displayed limiting-jet factorial is changed by this clarification.

## Review lineage and preserved limits

The reviewer is OpenAI/Codex `/root/coalescence_review_20261016`, a separate nonauthor agent within the same provider and parent thread. Its frozen review records a complete read of the report, diagnostic program and JSON, pinned source checks, reconstruction of the restricted estimates, diagnostic replay and independent finite algebra. Those are the reviewer's performed checks, not new checks attributed to this author addendum. Organizational-independence credit is **0**; there is no human or blind review.

Its verdict is **ACCEPT, restricted conventional mathematics only**, with the two wording corrections incorporated by this addendum. That verdict covers the stated collinear bounds (8)/(11), coordinate-aligned ranks and K2 obstruction (16), and the K>=4 exact-contact bound (20). This addendum neither extends the verdict nor converts it into source-owner adoption, formal verification, a graph transition or scientific promotion.

The fixed-law, planar, positive-full-square spectrum, short embedded chart, four distinct sites, declared pair-separation ratio, canonical zero-gradient regression and fixed positive lower cubic marks remain essential. Constants are not made effective or uniform over vanishing spectral weights, unbounded K or varying L. The twelve-row K2/K3 coordinate strata remain singular. The eight-gradient argument retains its stated all-frame fixed-law scope; the twelve-row obstruction is not a blanket claim about every K2 orientation.

Actual H0 selectors are consumed only as Borel marks bounded by one. No global pairing or selected-bar counting identity is assumed or proved here. The collinear stratum still has zero full spatial measure. Transverse collision geometry, k_i=O(eta²), uniform finite-radius/contact-limit interchange, other incidence patterns, increasing-domain estimates, global factorial control, Lean interfaces and spatial point-process convergence remain unresolved. Decimal arithmetic remains diagnostic and uncertified; no sampler, seeds or numerical H0/process experiment were added.

This addendum records the author's accepted interpretation for handoff alongside the frozen report and review. No native write, source claim, PR, merge, source-owner transfer or theorem-status change occurred.
