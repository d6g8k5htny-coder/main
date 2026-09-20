# Additive minimal repair of the H5 derivative-majorant defect

Task DQ-MATH-20260917-b9c2; 2026-09-17 UTC. This repair continues the
reconnaissance-bound author-side work documented in H5_ANALYTIC_ADVANCE.md.
No frozen source, theorem status or existing receipt is changed.

The source patch changes exactly one expression inside Lattice.lambda_q:

    wj*(kj**q)  ->  wj*(abs(kj)**q)

All other bytes in h5_kernel.py remain unchanged. The build script requires
the original source SHA256
b2e6014c2fd6eb9e72b2104cc828a764e280f76b8396a02edd37654b6ad83ad7,
and verifies the companion H2 source hashes before making a separate
repair directory. The repaired kernel SHA256 is
effa5994d2dcfaa25eedac37c79b8c5b1cc686fe84a79aefc9c734d89a9d5e9a.
The directory includes the unchanged hash-bound H2 dependencies, a unified
patch, and a build manifest.

## Why this specific patch is sufficient for lambda_q

For every derivative order q, the correct global majorant is

    L_q = sum_j w_j |k_j|^q / Z_full.

The repaired finite numerator is nonnegative. The existing lambda_q
denominator uses the lower endpoint of Z_finite, and its existing tail
adds an upper bound for the omitted absolute numerator divided by that
same lower denominator. Since Z_full >= Z_finite >= Z_finite.lower,
the upper endpoint returned by the repaired method bounds L_q from above.
Thus the numerator-tail implementation already present in this majorant
method suffices after the signed-power cancellation is removed. This
argument concerns the positive majorant method specifically; it is not a
new audit of every signed kernel-evaluation path or downstream certificate.

The derivative-majorant comparison, and every use as a remainder bound,
must consume its upper endpoint. An interval returned here represents
a conservative computed bound; it need not be a tight enclosure of L_q.

## Actual-consumer verification

verify_lambda_repair.py loads both the original and repaired modules,
then calls their actual KernelSeries.tm_factor method at n=1, TD=5,
r=0.05, yc=(0.025,0), eta1=eta2=0.001. Direct evaluation of the exact
normalized torus kernel at the positive endpoint establishes:

    actual Taylor error approximately 7.28495881102e-21;
    original claimed radius upper approximately 1.68559496223e-119;
    repaired radius upper approximately 5.31923041621e-20.

The original consumer interval is disjoint from the exact endpoint
interval. The repaired consumer interval contains it. The Taylor
polynomial itself is unchanged by the repair.

The verifier also checks positive repaired majorants and exact-torus
derivative witnesses for q=0,...,18; reproduces the original q=1 and
q=7 failures; rejects a source-byte mutation by its hash guard; and
reinstalls the defective signed-power method to show that the actual
consumer fails again. Both ordinary Python and python -O pass all six
named checks and produce byte-identical receipts. The controls use
explicit exceptions and retain their force under optimization.

Rebuild and replay with mpmath==1.3.0 and sympy==1.14.0:

    python build_lambda_repair.py --source /path/to/K3_SIDE24_LB/UPPER2D
    python verify_lambda_repair.py --source /path/to/K3_SIDE24_LB/UPPER2D
    python -O verify_lambda_repair.py --source /path/to/K3_SIDE24_LB/UPPER2D

The scripts suppress bytecode writes and never write into the input tree.
The destination must be outside it. A reviewed new source version and
new downstream consumer replays are still required before the repaired
implementation can support new program-level certificates. No previously
frozen totals are retroactively blessed by this local source repair.
