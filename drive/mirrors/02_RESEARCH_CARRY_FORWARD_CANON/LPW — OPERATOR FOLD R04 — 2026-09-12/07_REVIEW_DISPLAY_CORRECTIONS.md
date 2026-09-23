# Two ancillary display corrections and a spectral regression target

These are new author-side observations about the incoming review text. They do not silently edit the PDF or refute the qualitative LPW proof.

## D1 — rational comparison in Addendum 2, page 2

The PDF says “31/250 > 1/8”. Exact arithmetic gives

    1/8 - 31/250 = 1/1000 > 0.

Replace with 31/250 < 1/8. The valid reasoning is: the planar covariance has eigenfloor greater than 1/8; an exact-periodic perturbation smaller than 1/1000 preserves a floor greater than 31/250. The later subtraction 31/250 - 32/1600 = 13/125 is correct. This is a repair to an ancillary reviewer sentence, not a defect in the original endpoint construction.

## D2 — average y-gradient in LEAD_ANALYTIC_VERDICT Q2

The displayed intermediate chain adds a positive 11r^3/48 remainder to 11r^2/8 and then bounds the sum by 11r^2/8. That intermediate inequality is false for r>0. The desired bound is nevertheless valid directly, without adding that remainder:

Let g(t)=f_y(t,0) and h=r/2. Then

    (g(h)+g(-h))/2-g(0)
      = (1/2) integral_0^h (h-t)[g''(t)+g''(-t)] dt.

Minkowski and sup_t ||g''(t)||_L2 <=11 give

    ||(g(h)+g(-h))/2-g(0)||_L2
      <= (1/2)*22*(h^2/2) = 11r^2/8.

This is exactly the linear-Taylor integral argument used in the author candidate. It repairs the review's expansion without changing the constant or the proof target. Request additive reviewer acknowledgment.

## D3 — off-origin covariance and the sine regression

For the exact stationary periodic covariance K and multiindices alpha,beta,

    Cov(partial^alpha f(x), partial^beta f(z))
      = (-1)^|beta| partial^(alpha+beta) K(x-z).

Equivalently its spectral summands contain

    Re[(ik)^alpha (-ik)^beta exp(ik.(x-z))].

Odd total derivative order vanishes at zero displacement by even symmetry. It need not vanish at nonzero displacement. In particular

    Cov(f(x), f_x(0)) = -K_x(x),

which is generally nonzero. The added checker tests a pure cosine covariance at x=pi/2: Cov(f(pi/2),f_x(0))=1. The sine-omission mutant returns zero and must be rejected. This is a structural regression test, not a certification of a truncated field calculation. Require both the exact identity and nonzero-offset tests in any new full covariance implementation.

The incoming RB erratum preserves the endpoint proof and invalidates the affected finite-r table. A corrected single density being higher does not establish that every corrupted matrix entry, conditioning inverse or supremum has a conservative sign. Audit actual downstream dependencies.
