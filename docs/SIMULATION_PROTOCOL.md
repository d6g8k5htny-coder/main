# Bargmann-Fock short-bar histogram

Scientific effect: NONE. A figure is a check, not a proof. This summary does not replace the [experiment note](research-translation/20260930/EXPERIMENT.md). If they disagree, the experiment note is the authority.

## Generator

Variance-one periodized Bargmann-Fock on the flat torus of side 24, dimension 2. Covariance is the normalized image sum

K_24(z) = sum_n exp(-|z+24n|^2/2) / sum_n exp(-|24n|^2/2).

A different width, a minimum-image kernel, or a differently normalized truncation is a different field. The SIDE24 enclosure does not apply to it.

## Filtration

Cubical superlevel sets on a periodic cubical complex: opposite faces are glued. An ordinary square with boundary is not the torus. Record strictly positive finite nonessential H0 bars only. Count zero-length and tied pairs separately. Do not put them in the logarithm.

## What is being estimated

Let mu(A) = 24^{-2} E[number of those bars with lifetime in A]. The candidate law is about the density dmu/dell ~ c ell^{-1/3}. The count of bars shorter than ell is the cumulative mu((0, ell]), which is order ell^{2/3} if the density law holds.

Normalize a bin (a, b) by the number of fields, the area 24^2, and the bin width. Or compare the bin count with the integrated prediction (3c/2)(b^{2/3} - a^{2/3}). Raw counts are not comparable to c.

## Fit

Fit log(density) against log(ell). The candidate slope is -1/3. Do not fit against (-1/3) log(ell) and then expect slope -1/3; that transformed predictor has slope 1 if the law holds.

A lifetime is a field value. Grid spacing is a length. Convert the mesh to a filtration error before comparing it with a bin. A window that is large against the grid and small against the correlation length is not, by itself, a resolved window.

## What a mismatch means

A stable slope or prefactor discrepancy is a diagnostic. Check sign, volume and amplitude normalization, the exact covariance, essential and zero bars, spectral cutoff, spatial approximation, and bin integration. Report sampling uncertainty across fields separately from spatial and spectral error. If those errors cannot be controlled, write inconclusive at the tested resolutions. A histogram mismatch does not by itself isolate failure of elder pairing or of the continuum asymptotic.

## Reading path

Return to the [public mathematics catalog](PUBLIC_MATHEMATICS.md) for the Gaussian-field proof sources and their stated scopes. The [P15 combinatorics note](P15_SPLIT.md) has a separate reading route.
