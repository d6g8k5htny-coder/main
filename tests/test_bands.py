"""Tests for ``research/bands/`` -- lane A1, the ``OBL-H5-JETMOD`` machinery.

WHAT IS UNDER TEST, in four groups.

1. ``lattice`` CONTAINMENT. The certified band enclosure must contain the
   lattice sum evaluated by direct summation at a much higher truncation, for
   every displacement in the band. That comparison is logically forced, not
   merely numerical: for ``N' >= N`` the difference ``S_{N'}(d) - S_N(d)`` is a
   sub-sum of the terms omitted at radius ``N``, so it is bounded in absolute
   value by the same ``tail_bound``, hence ``S_{N'}(d)`` lies inside
   ``truncated +/- tail`` whenever ``S_N(d)`` does.

2. ``lattice`` TAIL DOMINATION. The proved tail bound must dominate the actual
   omitted mass, at every point of the box, for both envelope branches.

3. ``ladder`` EXACT ARITHMETIC. The published-number analysis reproduced to
   exact rational values, with the certified fractional power cross-checked
   against an independent route (three nested certified square roots at
   ``kappa = 1/8``).

4. ``falsifier`` TRUTH TABLE, including that equality does not falsify and that
   a band with no data is never a ``PASS``.

5. **NEGATIVE CONTROLS** (twenty of them, each named ``test_control_*``). Each
   builds a deliberately weakened variant *locally inside this file* -- never
   by editing the package -- and asserts that the weakening is caught. Each
   names its mutation. The controls were additionally run against deliberately
   broken copies of the package in a scratch directory; the mutations tried and
   the controls that fired are recorded in the docstrings of the controls
   concerned.

6. **CONTROLS ADDED AFTER AN ADVERSARIAL AUDIT (2026-09-18)**, in the last
   section. Each closes a defect the audit demonstrated on the shipped code:

   * a ``DecayEnvelope`` that is wrong by a factor of 2.45e24 could produce a
     record whose ``to_dict()["certified"]`` read ``true``, because
     ``PlaneKernel.certified`` defaulted to ``True`` and covered only the
     evaluator while the envelope was validated by nothing but a non-empty
     justification string. Both flags now default to ``False`` and the record's
     ``certified`` is their conjunction;
   * ``normalized_band_enclosure`` handed back the obligation's own quantity as
     a bare ``Interval`` with no flag, note or caveat attached;
   * ``falsifier.format_report`` rendered two exact ``Fraction``s as ``%.6g``
     floats with no NON-CERTIFYING marker, so two rows with opposite verdicts
     could print byte-identical columns;
   * ``ladder``'s "the published points do not themselves exhibit a single
     constant" was FALSE. A single admissible ``C`` exists at every ``kappa``
     in the sweep; what the two bands differ in is the MINIMUM each forces.

   Two coverage gaps were closed in place rather than in this section: the
   uniformity test sampled only the diagonal of a two-dimensional box, and no
   test came near the truncation edge where ``a = L(N+1) - R`` is small.

   None of this changed a bound. Four of the five were LABELLING defects, and
   the false-envelope control asserts explicitly that the wrong number is still
   exactly as wrong as it was -- labelling it honestly is not repairing it.

WHAT A GREEN RUN OF THIS FILE DOES **NOT** ESTABLISH
----------------------------------------------------
* It does **not** discharge, reduce, close, promote or reclassify
  ``OBL-H5-JETMOD``, which stays **OPEN (display only)**; nor
  ``OBL-H5-ZBAND`` (hi side), ``OBL-H5-REMOTE-THRESHOLD``, ``OBL-D1-PROMOTE``
  (chart side) or either Piece of ``D3-LEMMA-RN-UNIF``.
* The kernels exercised here are **REFERENCE KERNELS**. The program's own
  ``kplane`` -- with its Hermite factors and a certified decay envelope for
  them -- is not bound in this repository, nor are the 24-jet definitions with
  their powers ``p_J``, nor the actual band endpoints ``r_k``. Passing tests on
  a reference kernel say the machinery works; they say nothing about the
  program's jets.
* The published values in ``ladder`` remain **point** certifications. The
  implied-modulus analysis is an observation under a stated reading of a
  DISPLAY and is not a refutation of anything.
* A green build is not a mathematical review. The tail-bound argument is
  written out as a proof in ``lattice.tail_bound``'s docstring and a human must
  read it; these tests check that the code behaves as that proof says on the
  inputs exercised here.
* No result here composes the 2D upper, 2D lower and 3D lifetime tracks, and
  none bears on any prize problem.
"""
import math
import os
import sys
from fractions import Fraction as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest  # noqa: E402

from research.interval import Interval, sqrt  # noqa: E402
from research.bands import falsifier as FZ  # noqa: E402
from research.bands import ladder as LD  # noqa: E402
from research.bands import lattice as LT  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers used by several tests. None of these is part of the package API.
# ---------------------------------------------------------------------------

def _omitted_mass(kernel, dx, dy, n_trunc, n_outer, period, prec):
    """Enclosure of ``sum_{n_trunc < |n|_inf <= n_outer} |kplane(d + L n)|``.

    A *finite* sub-sum of the terms the truncation omits. Any certified tail
    bound must dominate it, because the tail bound covers all of them and each
    term is non-negative in absolute value.
    """
    total = Interval.exact(F(0))
    for i in range(-n_outer, n_outer + 1):
        for j in range(-n_outer, n_outer + 1):
            if max(abs(i), abs(j)) <= n_trunc:
                continue
            v = kernel.evaluate(
                dx + Interval.exact(period * i),
                dy + Interval.exact(period * j),
                prec,
            )
            total = total + abs(v)
    return total


def _direct_point_sum(kernel, r, displacement, n_outer, period, prec):
    """The lattice sum at ONE value of ``r``, summed directly to radius ``n_outer``."""
    dx, dy = displacement(Interval.exact(r))
    return LT.truncated_sum(
        kernel, dx, dy, n_trunc=n_outer, period=period, prec=prec
    )


GAUSS = LT.gaussian_reference()
POW3 = LT.inverse_power_reference(3)


# ===========================================================================
# 1. lattice -- containment against direct summation at high truncation
# ===========================================================================

def test_truncated_sum_evaluates_exactly_the_engine_s_nine_points():
    """``n_trunc = 1`` is the frozen engine's ``_IMG`` shape: nine evaluations."""
    seen = []

    def counting(dx, dy, prec):
        seen.append((dx.mid(), dy.mid()))
        return Interval.exact(F(0))

    kern = LT.PlaneKernel(
        name="counter", evaluate=counting,
        envelope=GAUSS.envelope, certified=False,
    )
    LT.truncated_sum(kern, Interval.exact(F(0)), Interval.exact(F(0)),
                     n_trunc=LT.ENGINE_TRUNCATION, period=F(24), prec=10)
    assert LT.ENGINE_TRUNCATION == 1
    assert len(seen) == 9
    assert set(seen) == {(F(24) * i, F(24) * j)
                         for i in (-1, 0, 1) for j in (-1, 0, 1)}


@pytest.mark.parametrize("period", [F(2), F(3), F(5)])
def test_band_enclosure_contains_direct_high_truncation_sum(period):
    """The certified total contains the directly summed lattice sum at N' = 12.

    Run at small periods, where the omitted mass is a real quantity rather than
    something below any reachable precision, so containment is genuinely being
    exercised and not asserted about a number that is zero to working accuracy.
    """
    band = Interval(F(1, 4), F(3, 4))
    enc = LT.band_enclosure(GAUSS, band, displacement=LT.diagonal_displacement,
                            n_trunc=1, period=period, prec=30)
    for r in (F(1, 4), F(1, 2), F(3, 4)):
        direct = _direct_point_sum(GAUSS, r, LT.diagonal_displacement,
                                   12, period, 40)
        assert direct in enc.total, (r, direct, enc.total)


def test_band_enclosure_contains_direct_sum_on_the_side24_torus():
    """Same containment at the program's own period, ``L = 24``.

    Here the tail is about ``1e-498`` and the band width about ``1.6e-3``, so
    the tail contributes nothing measurable. It is still *proved* rather than
    asserted, which is the whole difference from the frozen engine, and the
    breakdown reports it separately so a reader can see its size.
    """
    band = Interval(F("0.025"), F("0.035355"))
    enc = LT.band_enclosure(GAUSS, band, n_trunc=1, period=LT.SIDE24_PERIOD,
                            prec=30)
    assert enc.tail > 0
    assert enc.tail < F(1, 10 ** 400)
    for r in (F("0.025"), F("0.03"), F("0.035355")):
        direct = _direct_point_sum(GAUSS, r, LT.axial_displacement, 6,
                                   LT.SIDE24_PERIOD, 40)
        assert direct in enc.total, (r, direct, enc.total)


def test_band_enclosure_at_a_point_band_is_not_contained_without_the_tail():
    """A POINT band at a small period: the tail is the only slack there is.

    The wide-box containment tests above pass even with the tail removed,
    because band variation swamps it. This one does not: at a point band the
    truncated enclosure is ~1e-30 wide while the omitted mass is ~5.6e-3, so
    ``band_enclosure`` is contained only because it adds the tail.

    Confirmed against a deliberately broken copy: with ``band_enclosure``
    patched to ``total = trunc``, this test fails and the wide-box ones do not.
    """
    point = Interval.exact(F(1, 2))
    enc = LT.band_enclosure(GAUSS, point, displacement=LT.diagonal_displacement,
                            n_trunc=1, period=F(2), prec=30)
    direct = _direct_point_sum(GAUSS, F(1, 2), LT.diagonal_displacement,
                               14, F(2), 40)
    assert direct in enc.total
    assert enc.truncated.width() < F(1, 10 ** 25)
    assert enc.tail > F(1, 100)
    assert direct not in enc.truncated


def test_shell_counts_are_exactly_8m():
    """STEP 2 of the tail-bound proof, pinned two ways.

    The closed form multiplies by ``8m``. That number is
    ``(2m+1)^2 - (2m-1)^2``, checked here as an identity and again by counting
    the lattice points a truncated sum actually visits, so a coefficient
    mutation in the proof's combinatorics has somewhere to fail.
    """
    for m in range(1, 12):
        assert (2 * m + 1) ** 2 - (2 * m - 1) ** 2 == 8 * m

    counted = []

    def counting(dx, dy, prec):
        counted.append(1)
        return Interval.exact(F(0))

    kern = LT.PlaneKernel(name="counter", evaluate=counting,
                          envelope=GAUSS.envelope, certified=False)
    sizes = []
    for n in range(0, 5):
        counted.clear()
        LT.truncated_sum(kern, Interval.exact(F(0)), Interval.exact(F(0)),
                         n_trunc=n, period=F(24), prec=10)
        sizes.append(len(counted))
    assert sizes == [1, 9, 25, 49, 81]
    for m, (a, b) in enumerate(zip(sizes, sizes[1:]), start=1):
        assert b - a == 8 * m


def test_power_kernel_band_enclosure_contains_direct_sum():
    """The POWER envelope branch, at ``L = 24``, where the tail is ~2e-9.

    The inverse-power reference kernel is exactly rational, so its truncated
    sum carries no transcendental slack at all and the whole enclosure width is
    band variation plus the proved tail.
    """
    band = Interval(F(2, 100), F(6, 100))
    enc = LT.band_enclosure(POW3, band, n_trunc=1, period=LT.SIDE24_PERIOD,
                            prec=30)
    assert enc.tail > F(1, 10 ** 10)
    for r in (F(2, 100), F(4, 100), F(6, 100)):
        direct = _direct_point_sum(POW3, r, LT.axial_displacement, 9,
                                   LT.SIDE24_PERIOD, 30)
        assert direct in enc.total, (r, direct, enc.total)


def test_normalized_band_enclosure_contains_the_pointwise_ratios():
    """``S(B)/r^p`` as an interval contains ``S(r)/r^p`` for every r in the band."""
    band = Interval(F(2, 100), F(6, 100))
    ratio, enc = LT.normalized_band_enclosure(
        POW3, band, 3, n_trunc=1, period=LT.SIDE24_PERIOD, prec=30)
    assert enc.certified
    for r in (F(2, 100), F(4, 100), F(6, 100)):
        direct = _direct_point_sum(POW3, r, LT.axial_displacement, 9,
                                   LT.SIDE24_PERIOD, 30)
        # S_9(r) lies in enc.total (proved: the extra terms are a sub-sum of
        # the omitted ones), and r**3 lies in band**3, so the quotient lies in
        # the quotient interval.
        assert (direct / Interval.exact(r ** 3)) in ratio, r
    # The normalisation is what widens the band: dividing by r^3 over a 3x
    # range of r spreads a 1.6e-3-wide enclosure over four orders of magnitude.
    assert ratio.width() > enc.width()


def test_enclosure_breakdown_is_reported_and_serialisable():
    band = Interval(F("0.025"), F("0.035355"))
    enc = LT.band_enclosure(GAUSS, band, prec=25)
    d = enc.to_dict()
    assert d["terms_summed"] == 9
    assert d["n_trunc"] == 1
    assert d["period"] == "24"
    assert F(d["tail_bound"]) == enc.tail
    assert F(d["total_width"]) == enc.width()
    # The caveats travel with the number.
    assert any("OBL-H5-JETMOD" in c for c in d["caveats"])
    assert "REFERENCE KERNEL" in d["notes"]


# ===========================================================================
# 2. lattice -- the tail bound dominates the omitted terms
# ===========================================================================

@pytest.mark.parametrize("period", [F(2), F(3), F(5)])
def test_gaussian_tail_bound_dominates_the_omitted_terms(period):
    """The proved bound is at least the actual omitted mass, over the box."""
    box = Interval(F(1, 4), F(3, 4))
    dx, dy = LT.diagonal_displacement(box)
    bound = LT.tail_bound(GAUSS.envelope, dx, dy, n_trunc=1,
                          period=period, prec=30)
    actual = _omitted_mass(GAUSS, dx, dy, 1, 10, period, 30)
    assert actual.hi <= bound, (period, float(actual.hi), float(bound))


def test_power_tail_bound_dominates_the_omitted_terms_at_period_24():
    box = Interval(F(2, 100), F(6, 100))
    dx, dy = LT.axial_displacement(box)
    bound = LT.tail_bound(POW3.envelope, dx, dy, n_trunc=1,
                          period=LT.SIDE24_PERIOD, prec=30)
    actual = _omitted_mass(POW3, dx, dy, 1, 9, LT.SIDE24_PERIOD, 30)
    assert actual.hi <= bound
    # The looseness is pinned from BOTH sides. The upper pin says the bound is
    # not absurd; the LOWER pin is a regression guard with teeth -- a bound
    # that became less than ~1.5x the truth would mean a factor had gone
    # missing from the closed form, and a shell undercount does exactly that.
    # Measured factor on this configuration: about 2.38.
    assert actual.hi * F(3, 2) < bound < actual.hi * 4


def test_tail_bound_is_uniform_over_the_box():
    """ONE constant covers EVERY displacement in the box -- the WHOLE box.

    This is the property ``OBL-H5-JETMOD`` asks for in the words "lattice-tail
    constants re-certified uniformly in the band", and the property the frozen
    engine's ``tail_bound`` lacks: it hard-codes ``rho = m*_LT - 17``, a point
    separation.

    AN EARLIER VERSION OF THIS TEST SAMPLED ONLY THE DIAGONAL. ``box`` built by
    ``diagonal_displacement`` is the SQUARE ``[1/4, 3/4] x [1/4, 3/4]``, but the
    five sampled points were ``(r, r)`` -- a one-dimensional slice of the
    two-dimensional object the docstring named. The corners, where the box
    radius argument actually bites, were never touched. The sweep below is the
    full 5 x 5 grid of the square, corners included, plus the four corners
    called out separately so a regression cannot quietly drop them.
    """
    box = Interval(F(1, 4), F(3, 4))
    dx, dy = LT.diagonal_displacement(box)
    band_bound = LT.tail_bound(GAUSS.envelope, dx, dy, n_trunc=1,
                               period=F(2), prec=30)
    grid = (F(1, 4), F(3, 8), F(1, 2), F(5, 8), F(3, 4))
    for a in grid:
        for c in grid:
            here = _omitted_mass(GAUSS, Interval.exact(a), Interval.exact(c),
                                 1, 10, F(2), 30)
            assert here.hi <= band_bound, (a, c, float(here.hi),
                                           float(band_bound))
    # The four corners named explicitly. (1/4, 3/4) and (3/4, 1/4) are the
    # OFF-DIAGONAL corners the old test never reached.
    for a, c in ((F(1, 4), F(1, 4)), (F(1, 4), F(3, 4)),
                 (F(3, 4), F(1, 4)), (F(3, 4), F(3, 4))):
        here = _omitted_mass(GAUSS, Interval.exact(a), Interval.exact(c),
                             1, 10, F(2), 30)
        assert here.hi <= band_bound, ("corner", a, c)


def test_tail_bound_is_uniform_over_an_asymmetric_box():
    """Uniformity must not depend on the box being a square about the diagonal.

    ``diagonal_displacement`` happens to produce a square, so a test built only
    on it cannot tell a genuinely uniform bound from one that accidentally works
    on squares. Here the two factors are different intervals and the grid over
    the resulting rectangle is swept in full.
    """
    dx = Interval(F(1, 5), F(4, 5))
    dy = Interval(F(-1, 2), F(1, 10))
    band_bound = LT.tail_bound(GAUSS.envelope, dx, dy, n_trunc=1,
                               period=F(2), prec=30)
    xs = (F(1, 5), F(2, 5), F(3, 5), F(4, 5))
    ys = (F(-1, 2), F(-1, 4), F(0), F(1, 10))
    for a in xs:
        for c in ys:
            here = _omitted_mass(GAUSS, Interval.exact(a), Interval.exact(c),
                                 1, 10, F(2), 30)
            assert here.hi <= band_bound, (a, c, float(here.hi))


@pytest.mark.parametrize("radius", [F("0.9"), F("0.99"), F("0.999999")])
def test_tail_bound_dominates_just_inside_the_truncation_edge(radius):
    """The NEAR-EDGE regime: ``a = L*(N+1) - R`` small and strictly positive.

    No earlier test came anywhere near this. The refusal test sits at
    ``R = 56.57`` against ``L*M = 48``, comfortably past the boundary, and every
    domination test sits at ``a >= 2.9``. The edge is where the Gaussian
    geometric ratio ``q = exp(-B(2aL+L^2))`` approaches its floor and where the
    closed form is under the most strain, so it is exactly where an auditor
    expects the argument pinned.

    Here ``L = 1`` and ``n_trunc = 0``, so ``M = 1`` and ``a = 1 - R`` runs down
    to ``1e-6``. The bound must still dominate the omitted mass.
    """
    box = Interval(F(0), radius)
    zero = Interval.exact(F(0))
    bound = LT.tail_bound(GAUSS.envelope, box, zero, n_trunc=0,
                          period=F(1), prec=40)
    assert bound > 0
    for a in (F(0), radius / 2, radius):
        here = _omitted_mass(GAUSS, Interval.exact(a), zero, 0, 8, F(1), 40)
        assert here.hi <= bound, (float(radius), float(a), float(here.hi),
                                  float(bound))


def test_tail_bound_near_edge_grows_as_the_box_reaches_the_omitted_image():
    """As ``a -> 0+`` the bound must blow up, not quietly stay small.

    A bound that did not grow as the nearest omitted image approached the
    evaluation box would be ignoring the geometry the whole argument rests on.
    """
    zero = Interval.exact(F(0))
    bounds = [LT.tail_bound(GAUSS.envelope, Interval(F(0), r), zero,
                            n_trunc=0, period=F(1), prec=40)
              for r in (F(1, 2), F("0.9"), F("0.99"), F("0.999999"))]
    assert bounds == sorted(bounds), [float(b) for b in bounds]
    assert bounds[-1] > bounds[0]


def test_the_shrink_factor_guard_cannot_fire_while_a_is_positive():
    """``c.lo <= 0`` in the POWER branch is DEFENSIVE AND UNREACHABLE. Pinned.

    A guard that cannot fire is not a tested guard, and writing a test that
    pretends otherwise would be worse than writing none. ``c`` is built as
    ``1 - Interval.exact(R) / Interval.exact(L*M)`` from two exact rational
    POINTS, so the division is exact and ``c.lo = 1 - R/(L*M) = a/(L*M)``, which
    the earlier ``a > 0`` check has already forced positive.

    This test pins that implication over a sweep, so that a change making either
    operand a non-degenerate interval -- which is the change that would make the
    guard live -- is caught here rather than discovered by a wrong bound.
    """
    for L in (F(1), F(2), F(24)):
        for n_trunc in (0, 1, 2):
            M = n_trunc + 1
            for frac in (F(1, 1000), F(1, 2), F("0.9"), F("0.999999")):
                R = L * M * frac
                a = L * M - R
                assert a > 0
                c = (Interval.exact(F(1))
                     - Interval.exact(R) / Interval.exact(L * M))
                assert c.lo == a / (L * M)
                assert c.lo > 0


def test_the_geometric_ratio_guard_IS_reachable_and_refuses():
    """``q.hi >= 1`` in the GAUSSIAN branch is reachable, and it refuses.

    The true ``q = exp(-B*(2aL+L^2))`` is below 1 for every positive ``B``,
    ``a`` and ``L``, but a certified enclosure of it need not prove that: for a
    tiny exponent the outward-rounded upper endpoint lands at or above 1, and
    then ``1/(1-q)`` and ``q/(1-q)^2`` are bounded by nothing this code
    computes. Refusing is the only honest exit.

    Built with ``B = 1e-60``, so ``B*(2aL+L^2)`` is around ``3e-60`` and the
    ``exp`` enclosure cannot separate it from 1.
    """
    tiny = LT.DecayEnvelope(
        kind=LT.GAUSSIAN, A=F(1), B=F(1, 10 ** 60), valid_from=F(0),
        justification=(
            "CONTROL ONLY. A legitimate envelope shape with an absurdly slow "
            "decay rate, used to drive the geometric-ratio enclosure against "
            "its guard. It is not claimed to dominate any kernel."
        ),
    )
    with pytest.raises(ValueError, match="does not certify q < 1"):
        LT.tail_bound(tiny, Interval.exact(F(0)), Interval.exact(F(0)),
                      n_trunc=0, period=F(1), prec=30)


def test_tail_bound_shrinks_as_the_truncation_radius_grows():
    box = Interval(F(1, 4), F(3, 4))
    dx, dy = LT.diagonal_displacement(box)
    bounds = [LT.tail_bound(GAUSS.envelope, dx, dy, n_trunc=n,
                            period=F(2), prec=30) for n in (1, 2, 3, 4)]
    assert all(b > 0 for b in bounds)
    assert bounds == sorted(bounds, reverse=True)
    assert bounds[-1] < bounds[0]


def test_tail_bound_refuses_a_box_that_reaches_the_first_omitted_image():
    """``a = L*(N+1) - R <= 0`` means an omitted image can sit on the point."""
    big = Interval(F(0), F(40))   # radius sqrt(2)*40 = 56.6 > 48 = L*(N+1)
    with pytest.raises(ValueError, match="tail bound refused"):
        LT.tail_bound(GAUSS.envelope, big, big, n_trunc=1,
                      period=LT.SIDE24_PERIOD, prec=25)


def test_tail_bound_refuses_below_the_envelope_s_validity_radius():
    env = LT.DecayEnvelope(
        kind=LT.GAUSSIAN, A=F(1), B=F(1, 2), valid_from=F(1000),
        justification="control: an envelope claimed only far out",
    )
    p = Interval.exact(F(1, 2))
    with pytest.raises(ValueError, match="valid_from"):
        LT.tail_bound(env, p, p, n_trunc=1, period=F(24), prec=25)


def test_envelope_constructor_refuses_unproved_or_divergent_envelopes():
    with pytest.raises(ValueError, match="justification"):
        LT.DecayEnvelope(kind=LT.GAUSSIAN, A=F(1), B=F(1, 2),
                         valid_from=F(0), justification="   ")
    with pytest.raises(ValueError, match="p > 2"):
        LT.DecayEnvelope(kind=LT.POWER, A=F(1), p=F(2), valid_from=F(0),
                         justification="control: p at the divergence boundary")
    with pytest.raises(ValueError, match="p > 2"):
        LT.DecayEnvelope(kind=LT.POWER, A=F(1), p=F(3, 2), valid_from=F(0),
                         justification="control: p below the divergence boundary")
    with pytest.raises(ValueError, match="positive Fraction B"):
        LT.DecayEnvelope(kind=LT.GAUSSIAN, A=F(1), B=F(0), valid_from=F(0),
                         justification="control: zero decay rate")


def test_band_enclosure_refuses_a_non_positive_band():
    with pytest.raises(ValueError, match="positive separation"):
        LT.band_enclosure(GAUSS, Interval(F(0), F(1, 10)))


def test_uncertified_kernel_propagates_its_flag():
    """An uncertified evaluator must not acquire the word 'certified'."""
    kern = LT.PlaneKernel(
        name="float stand-in", envelope=GAUSS.envelope, certified=False,
        evaluate=lambda dx, dy, prec: Interval.exact(F(0)),
    )
    enc = LT.band_enclosure(kern, Interval(F(1, 100), F(2, 100)), prec=20)
    assert enc.certified is False
    assert enc.to_dict()["certified"] is False


# ===========================================================================
# 3. ladder -- the published arithmetic, to exact rational values
# ===========================================================================

def test_published_points_are_exact_decimal_transcriptions():
    by = {(p.line, p.r): p for p in LD.PUBLISHED_POINTS}
    assert by[(LD.LINE_LIVE_V3, F("0.025"))].value == F(6643979, 10000)
    assert by[(LD.LINE_LIVE_V3, F("0.035355"))].value == F(6614712, 10000)
    assert by[(LD.LINE_LIVE_V3, F("0.05"))].value == F(6478048, 10000)
    assert by[(LD.LINE_FROZEN_V1, F("0.05"))].value == F(7314311, 10000)
    assert by[(LD.LINE_LIVE_V3, F("0.035355"))].r == F(35355, 10 ** 6)
    # digests are quoted only where a rung document quotes one
    assert by[(LD.LINE_LIVE_V3, F("0.025"))].totals_digest == (
        "f7697bcfa0fe32b5c87c8ef8adef5d4384ab1bda7cd4a7c4782fef8c99103fcb")
    assert by[(LD.LINE_LIVE_V3, F("0.035355"))].totals_digest == (
        "808d6901e3254a73181aa696904ed04567d401b6e58300454d489046c36f4f64")
    assert by[(LD.LINE_LIVE_V3, F("0.05"))].totals_digest is None
    assert by[(LD.LINE_FROZEN_V1, F("0.05"))].totals_digest is None


def test_same_r_version_spread_is_exact():
    """731.4311 - 647.8048 = 83.6263, which is 12.9092...% of the live value."""
    sp = LD.same_r_version_spread()
    assert sp["spread"] == F(836263, 10000)
    assert sp["relative"] == F(836263, 10000) / F(6478048, 10000)
    assert sp["relative"] == F(836263, 6478048)
    # 12.91% to two places, computed exactly and only then rounded for display
    pct = sp["relative_percent"]
    assert F("12.9091") < pct < F("12.9092")
    assert math.floor(pct * 10000) == 129091          # 12.90915...%
    assert pct == F(836263, 6478048) * 100


def test_adjacent_band_widths_and_differences_are_exact():
    lo, hi = LD.adjacent_bands()
    assert (lo.r_lo, lo.r_hi) == (F("0.025"), F("0.035355"))
    assert (hi.r_lo, hi.r_hi) == (F("0.035355"), F("0.05"))
    assert lo.delta == F(10355, 10 ** 6)
    assert hi.delta == F(14645, 10 ** 6)
    assert lo.abs_diff == F(29267, 10000)
    assert hi.abs_diff == F(136664, 10000)


def test_implied_modulus_constants_at_the_displayed_kappa():
    """The two adjacent bands force 23.1709... and 5.1818..., a ratio of 4.47.

    Enclosures, not point values: ``delta^(1/8)`` is irrational. The assertions
    bracket each forced constant inside a window narrower than the published
    rounding, so a wrong value could not pass.
    """
    lo, hi = LD.implied_modulus_table(LD.DISPLAYED_KAPPA, prec=40)
    assert lo.kappa == F(1, 8)
    assert F("5.18184") < lo.constant.lo <= lo.constant.hi < F("5.18186")
    assert F("23.17090") < hi.constant.lo <= hi.constant.hi < F("23.17092")
    # each enclosure is a genuine enclosure of |Delta| / delta^kappa
    assert lo.constant.lo * lo.delta_pow_kappa.hi <= lo.band.abs_diff
    assert lo.constant.hi * lo.delta_pow_kappa.lo >= lo.band.abs_diff


def test_implied_modulus_ratio_at_the_displayed_kappa():
    r = LD.implied_modulus_ratio(LD.DISPLAYED_KAPPA, prec=40)
    assert F("4.4715") < r["ratio"].lo <= r["ratio"].hi < F("4.4717")
    assert r["wide_band"] == "[7071/200000, 1/20]"
    assert r["narrow_band"] == "[1/40, 7071/200000]"


def test_delta_pow_kappa_agrees_with_three_nested_square_roots():
    """An independent route to ``delta^(1/8)``: sqrt(sqrt(sqrt(delta))).

    The package computes ``exp(kappa * log(delta))``. Three nested certified
    square roots use neither ``exp`` nor ``log``, so agreement is a real
    cross-check of the fractional-power path rather than a restatement of it.
    Two certified enclosures of the same real number must intersect.
    """
    for band in LD.adjacent_bands():
        viaexp = LD._delta_pow_kappa(band.delta, F(1, 8), 40)
        viasqrt = sqrt(sqrt(sqrt(Interval.exact(band.delta), 45), 45), 45)
        assert viaexp.intersect(viasqrt) is not None
        assert viasqrt.lo < viaexp.hi and viaexp.lo < viasqrt.hi


def test_the_analysis_is_not_hardcoded_to_kappa_one_eighth():
    """At ``kappa = 0`` the forced constants are the bare endpoint differences."""
    lo, hi = LD.implied_modulus_table(F(0), prec=30)
    assert lo.constant == Interval.exact(F(29267, 10000))
    assert hi.constant == Interval.exact(F(136664, 10000))
    r = LD.implied_modulus_ratio(F(0), prec=30)
    assert r["ratio"] == Interval.exact(F(136664, 29267))


def test_the_ratio_is_strictly_decreasing_in_kappa():
    """``ratio(kappa) = (|D_w|/|D_n|) * (delta_n/delta_w)^kappa``, and
    ``delta_n < delta_w``, so the second factor strictly decreases."""
    rows = LD.ratio_table(prec=30)
    vals = [row["ratio"] for row in rows]
    for a, b in zip(vals, vals[1:]):
        assert b.hi < a.lo, (a, b)
    # it crosses 1 somewhere between kappa = 4 and kappa = 5, certified
    assert LD.implied_modulus_ratio(F(4), prec=30)["ratio"].lo > 1
    assert LD.implied_modulus_ratio(F(5), prec=30)["ratio"].hi < 1


def test_every_reported_modulus_carries_its_assumption():
    """The number cannot travel without the reading it depends on."""
    for im in LD.implied_modulus_table():
        assert im.assumption == LD.IMPLIED_MODULUS_ASSUMPTION
        assert "NOT a quotation" in im.to_dict()["assumption"]
        assert "OBL-H5-JETMOD" in im.to_dict()["status_note"]
    assert "ASSUMPTION" in LD.implied_modulus_ratio()["assumption"]
    text = LD.format_report()
    assert "ASSUMPTION" in text
    assert "POINT certifications" in text
    assert "refutes" in text


def test_frozen_v1_line_has_a_single_point_and_no_adjacent_band():
    assert len(LD.points_for_line(LD.LINE_FROZEN_V1)) == 1
    assert LD.adjacent_bands(LD.LINE_FROZEN_V1) == []
    with pytest.raises(ValueError, match="exactly two adjacent bands"):
        LD.implied_modulus_ratio(line=LD.LINE_FROZEN_V1)


# ===========================================================================
# 4. falsifier -- the truth table
# ===========================================================================

def test_falsifier_rule_is_the_source_s_sentence():
    assert FZ.FALSIFIER_RULE == (
        "A band enclosure whose width exceeds the claimed modulus.")


@pytest.mark.parametrize("width,modulus,expected", [
    (F(1, 10), F(1, 2), False),      # comfortably inside
    (F(1, 2), F(1, 2), False),       # EQUAL: does not exceed, does not falsify
    (F(1, 2) + F(1, 10 ** 9), F(1, 2), True),   # exceeds by a hair
    (F(3, 2), F(1, 2), True),
    (F(0), F(0), False),
    (F(1, 10 ** 9), F(0), True),
])
def test_falsifies_truth_table(width, modulus, expected):
    assert FZ.falsifies(width, modulus) is expected


def test_band_verdict_truth_table():
    assert FZ.band_verdict(F(1, 10), F(1, 2)) == FZ.PASS
    assert FZ.band_verdict(F(1, 2), F(1, 2)) == FZ.PASS
    assert FZ.band_verdict(F(9, 10), F(1, 2)) == FZ.FALSIFIED
    assert FZ.band_verdict(None, F(1, 2)) == FZ.INSUFFICIENT_DATA
    assert FZ.band_verdict(F(1, 2), None) == FZ.INSUFFICIENT_DATA
    assert FZ.band_verdict(None, None) == FZ.INSUFFICIENT_DATA
    assert set(FZ.OUTCOMES) == {FZ.PASS, FZ.FALSIFIED, FZ.INSUFFICIENT_DATA}


def test_falsifies_refuses_none_and_float():
    with pytest.raises(TypeError, match="INSUFFICIENT_DATA"):
        FZ.falsifies(None, F(1, 2))
    with pytest.raises(TypeError, match="float"):
        FZ.falsifies(0.5, F(1, 2))
    with pytest.raises(TypeError, match="float"):
        FZ.falsifies(F(1, 2), 0.5)
    with pytest.raises(ValueError, match="non-negative"):
        FZ.falsifies(F(-1, 2), F(1, 2))


def test_band_report_emits_exactly_one_row_per_band_with_reasons():
    rows = FZ.band_report([
        FZ.BandCheck("[0.025, 0.035355]", F(1, 100), F(1, 10), "worked example"),
        FZ.BandCheck("[0.035355, 0.05]", F(9, 10), F(1, 10), "worked example"),
        FZ.BandCheck("[0.0125, 0.0177]", None, F(1, 10), "worked example"),
        FZ.BandCheck("[0.0177, 0.025]", F(1, 100), None, "worked example"),
    ])
    assert [r.verdict for r in rows] == [
        FZ.PASS, FZ.FALSIFIED, FZ.INSUFFICIENT_DATA, FZ.INSUFFICIENT_DATA]
    assert FZ.report_summary(rows) == {
        FZ.PASS: 1, FZ.FALSIFIED: 1, FZ.INSUFFICIENT_DATA: 2}
    assert rows[2].reason == "no enclosure width"
    assert rows[3].reason == "no claimed modulus"
    assert FZ.report_is_clean(rows) is False
    text = FZ.format_report(rows)
    assert "INSUFFICIENT_DATA" in text
    assert "NOT a discharge of OBL-H5-JETMOD" in text.replace("\n  ", " ")


def test_report_is_clean_requires_every_band_to_pass():
    ok = FZ.band_report([FZ.BandCheck("b", F(1, 100), F(1, 10))])
    assert FZ.report_is_clean(ok) is True
    assert FZ.report_is_clean([]) is False   # nothing checked is not clean


def test_end_to_end_lattice_width_through_the_falsifier():
    """A real certified width fed to the real falsifier, both outcomes.

    This is the intended wiring and nothing more: it discharges nothing,
    because the kernel is a reference kernel and the modulus is invented for
    the test.
    """
    enc = LT.band_enclosure(GAUSS, Interval(F(2, 100), F(6, 100)), prec=30)
    w = enc.width()
    assert FZ.band_verdict(w, w * 2) == FZ.PASS
    assert FZ.band_verdict(w, w / 2) == FZ.FALSIFIED


# ===========================================================================
# 5. NEGATIVE CONTROLS
#
# Each control builds the weakened variant locally. None edits the package.
# ===========================================================================

def test_control_dropping_the_tail_bound_breaks_containment():
    """MUTATION: ``total = truncated`` -- the frozen engine's shape.

    ``kdcov`` returns its nine-point sum and never adds ``tail_bound`` to
    anything. Reproduced here: at period 2 with a point displacement, the
    truncated-only interval does NOT contain the directly summed value, so the
    tail is load-bearing rather than decorative.

    Confirmed against a deliberately broken copy: with ``band_enclosure``
    patched to ``total = trunc`` in a scratch copy of ``lattice.py``, the
    containment tests in group 1 fail at every small period.
    """
    r = F(1, 2)
    p = Interval.exact(r)
    trunc = LT.truncated_sum(GAUSS, p, p, n_trunc=1, period=F(2), prec=30)
    tail = LT.tail_bound(GAUSS.envelope, p, p, n_trunc=1, period=F(2), prec=30)
    direct = LT.truncated_sum(GAUSS, p, p, n_trunc=14, period=F(2), prec=40)

    sound = Interval(trunc.lo - tail, trunc.hi + tail)
    assert direct in sound                       # with the tail: contained
    assert direct not in trunc                   # WITHOUT it: not contained
    assert trunc.width() < F(1, 10 ** 25)        # and the gap is not slack


def test_control_a_weakened_tail_bound_breaks_containment():
    """MUTATION: the tail bound scaled down by 1/100 (an inward rounding of it)."""
    p = Interval.exact(F(1, 2))
    trunc = LT.truncated_sum(GAUSS, p, p, n_trunc=1, period=F(2), prec=30)
    tail = LT.tail_bound(GAUSS.envelope, p, p, n_trunc=1, period=F(2), prec=30)
    direct = LT.truncated_sum(GAUSS, p, p, n_trunc=14, period=F(2), prec=40)
    weakened = Interval(trunc.lo - tail / 100, trunc.hi + tail / 100)
    assert direct not in weakened


def test_control_a_false_decay_envelope_loses_domination():
    """MUTATION: claim ``B = 1`` for a kernel that only decays at ``B = 1/2``.

    A decay rate twice the truth makes the tail bound about 200x too small at
    period 2, and it then fails to dominate the actual omitted mass. This is
    the control that protects the ``DecayEnvelope`` contract: the tail bound is
    only as good as the envelope handed to it, which is why the envelope must
    carry a justification and why ``PlaneKernel.certified`` exists.
    """
    liar = LT.DecayEnvelope(
        kind=LT.GAUSSIAN, A=F(1), B=F(1), valid_from=F(0),
        justification="DELIBERATELY FALSE. Negative control only. The unit "
                      "Gaussian decays at B = 1/2, not B = 1.",
    )
    p = Interval.exact(F(1, 2))
    false_bound = LT.tail_bound(liar, p, p, n_trunc=1, period=F(2), prec=30)
    true_bound = LT.tail_bound(GAUSS.envelope, p, p, n_trunc=1,
                               period=F(2), prec=30)
    actual = _omitted_mass(GAUSS, p, p, 1, 12, F(2), 40)
    assert actual.hi <= true_bound            # the honest envelope dominates
    assert false_bound < actual.lo            # the false one does NOT


def _power_tail_closed_form(A, p, radius, period, n_trunc, prec,
                            *, shell_factor=8, radius_sign=-1):
    """A local reimplementation of the power closed form (P), for mutating.

    Faithful at the default keyword values; the faithfulness is asserted before
    any mutation uses it.
    """
    M = n_trunc + 1
    L = F(period)
    c = Interval.exact(F(1)) + radius_sign * Interval.exact(radius) / Interval.exact(L * M)
    cl_pow = (c * Interval.exact(L)) ** int(-p)
    m_int = Interval.exact(F(M))
    term_head = m_int ** int(1 - p)
    term_int = m_int ** int(2 - p) / Interval.exact(p - 2)
    return (Interval.exact(shell_factor * A) * cl_pow
            * (term_head + term_int)).round_out(4 * max(prec, 1) + 64).hi


def test_power_tail_bound_matches_an_independent_closed_form():
    """The POWER branch reproduced term by term, coefficient by coefficient.

    Confirmed against a deliberately broken copy: replacing ``8 * envelope.A``
    with ``4 * envelope.A`` in a scratch ``lattice.py`` fails this test and the
    Gaussian faithfulness test, which is how a shell-count mutation is caught
    even where the domination margin is too small to catch it.
    """
    box = Interval(F(2, 100), F(6, 100))
    dx, dy = LT.axial_displacement(box)
    radius = LT._box_radius(dx, dy, 30)
    for m in (2, 3, 4):
        kern = LT.inverse_power_reference(m)
        for n_trunc in (1, 2, 3):
            mine = _power_tail_closed_form(F(1), F(2 * m), radius,
                                           LT.SIDE24_PERIOD, n_trunc, 30)
            theirs = LT.tail_bound(kern.envelope, dx, dy, n_trunc=n_trunc,
                                   period=LT.SIDE24_PERIOD, prec=30)
            assert mine == theirs, (m, n_trunc)


def test_control_flipping_the_sign_in_the_power_branch_loses_domination():
    """MUTATION: ``c = 1 + R/(L*M)`` instead of ``1 - R/(L*M)``.

    The same geometric sign as the Gaussian control, on the other branch: it
    assumes every omitted image is farther than it can be shown to be. The
    power branch is the tight one (bound ~2x the truth), so the margin here is
    small and the assertion says so by comparing against the actual mass rather
    than against a round factor.
    """
    kern = LT.inverse_power_reference(2)
    box = Interval(F(30, 100), F(90, 100))       # a wide band: R matters
    dx, dy = LT.axial_displacement(box)
    radius = LT._box_radius(dx, dy, 30)
    honest = _power_tail_closed_form(F(1), F(4), radius, LT.SIDE24_PERIOD, 1, 30)
    flipped = _power_tail_closed_form(F(1), F(4), radius, LT.SIDE24_PERIOD, 1, 30,
                                      radius_sign=+1)
    assert flipped < honest
    undercount = _power_tail_closed_form(F(1), F(4), radius, LT.SIDE24_PERIOD, 1,
                                         30, shell_factor=2)
    actual = _omitted_mass(kern, dx, dy, 1, 9, LT.SIDE24_PERIOD, 30)
    assert actual.hi <= honest
    assert undercount < actual.lo


def _gaussian_tail_closed_form(A, B, radius, period, n_trunc, prec,
                               *, shell_factor=8, radius_sign=-1):
    """A local reimplementation of the Gaussian closed form (G), for mutating.

    Faithful at the default keyword values -- ``test_local_reimplementation_is_
    faithful`` asserts it reproduces ``lattice.tail_bound`` exactly -- so a
    mutation of a keyword is a mutation of the real formula and not of a
    different one.

    ``radius_sign = -1`` is the reverse triangle inequality
    ``|d + Ln| >= L*m - R``. ``+1`` is the mutation that uses ``L*m + R``, i.e.
    assumes every omitted image is FARTHER than it can be shown to be.
    """
    M = n_trunc + 1
    a = F(period) * M + radius_sign * radius
    L = F(period)
    head = LT.exp(Interval.exact(-B * a * a), prec)
    q = LT.exp(Interval.exact(-B * (2 * a * L + L * L)), prec)
    one_minus_q = Interval.exact(F(1)) - q
    series = Interval.exact(F(M)) / one_minus_q + q / (one_minus_q ** 2)
    return (Interval.exact(shell_factor * A) * head * series).round_out(
        4 * max(prec, 1) + 64).hi


def test_local_reimplementation_is_faithful_to_the_package():
    """The mutation harness must reproduce the real bound before mutating it."""
    p = Interval.exact(F(1, 2))
    radius = LT._box_radius(p, p, 30)
    mine = _gaussian_tail_closed_form(F(1), F(1, 2), radius, F(2), 1, 30)
    theirs = LT.tail_bound(GAUSS.envelope, p, p, n_trunc=1,
                           period=F(2), prec=30)
    assert mine == theirs


def test_control_flipping_the_reverse_triangle_inequality_loses_domination():
    """MUTATION: ``a = L*(N+1) + R`` instead of ``L*(N+1) - R``.

    STEP 1 of the proof is ``|d + L n| >= |L n| - |d| >= L*m - R``. Flipping
    that sign assumes every omitted image sits FARTHER away than can be shown,
    which shrinks the Gaussian bound by a factor of about 280 at period 2 --
    and the shrunk bound then fails to dominate the mass it is supposed to
    cover. This is the control on the geometric step; the ``B = 1`` control
    above is the one on the envelope step.
    """
    p = Interval.exact(F(1, 2))
    radius = LT._box_radius(p, p, 30)
    honest = _gaussian_tail_closed_form(F(1), F(1, 2), radius, F(2), 1, 30)
    flipped = _gaussian_tail_closed_form(F(1), F(1, 2), radius, F(2), 1, 30,
                                         radius_sign=+1)
    actual = _omitted_mass(GAUSS, p, p, 1, 12, F(2), 40)
    assert actual.hi <= honest
    assert flipped < actual.lo
    assert flipped * 100 < honest


def test_control_undercounting_the_shell_loses_domination():
    """MUTATION: ``2m`` points per shell instead of the correct ``8m``.

    ``#{n in Z^2 : |n|_inf = m} = (2m+1)^2 - (2m-1)^2 = 8m``. Both closed forms
    are linear in ``8*A``, so an amplitude four times too small is arithmetically
    the same mutation as counting ``2m`` points per shell, and it goes through
    the package's own code path rather than a copy of it.

    Exercised on the inverse-power kernel at the side-24 period, where the
    proved bound is only about 2.0x the actual omitted mass, so a 4x
    undercount is caught with a clear margin. (On the Gaussian the same
    undercount would NOT be caught: the bound there is about 13x the truth
    because every point of a shell is charged at the shell's nearest distance,
    and a control that cannot fire is worse than no control. Said plainly
    rather than quietly dropped.)
    """
    pow2 = LT.inverse_power_reference(2)
    box = Interval(F(2, 100), F(6, 100))
    dx, dy = LT.axial_displacement(box)
    honest = LT.tail_bound(pow2.envelope, dx, dy, n_trunc=1,
                           period=LT.SIDE24_PERIOD, prec=30)
    undercount = LT.DecayEnvelope(
        kind=LT.POWER, A=F(1, 4), p=pow2.envelope.p, valid_from=F(0),
        justification="DELIBERATELY FALSE. Negative control only: amplitude "
                      "A/4, arithmetically identical to counting 2m points "
                      "per shell instead of 8m.",
    )
    weak = LT.tail_bound(undercount, dx, dy, n_trunc=1,
                         period=LT.SIDE24_PERIOD, prec=30)
    actual = _omitted_mass(pow2, dx, dy, 1, 9, LT.SIDE24_PERIOD, 30)
    assert weak * 4 == honest
    assert actual.hi <= honest
    assert weak < actual.lo
    # Pin the looseness from below too: on this configuration the honest bound
    # is about 2.01x the actual mass. If it ever drops under 1.5x, a factor has
    # gone missing -- which is what a shell undercount inside the package (as
    # opposed to this locally-built one) would look like.
    assert actual.hi * F(3, 2) < honest < actual.hi * 3


def test_control_an_inward_rounded_band_enclosure_is_caught():
    """MUTATION: shrink the certified band enclosure toward its midpoint.

    The band enclosure over ``r in [0.02, 0.06]`` attains its upper end at the
    smallest ``r``, so the true value at ``r = 0.02`` sits on the boundary.
    Any inward rounding at all excludes it -- which is why interval libraries
    round outward and why ``Interval.round_out`` is the only rounding this
    package uses.
    """
    band = Interval(F(2, 100), F(6, 100))
    enc = LT.band_enclosure(GAUSS, band, prec=30)
    edge = _direct_point_sum(GAUSS, F(2, 100), LT.axial_displacement, 6,
                             LT.SIDE24_PERIOD, 40)
    assert edge in enc.total
    shrink = enc.width() / 1000
    inward = Interval(enc.total.lo + shrink, enc.total.hi - shrink)
    assert edge not in inward


def test_control_a_band_with_no_data_is_never_a_pass():
    """MUTATION: treat a missing enclosure as satisfying the falsifier.

    This is the control worth the most. Everything that has gone wrong in this
    program's history has gone wrong by an unchecked thing being counted as a
    checked one, so the assertion is made three ways: the verdict is not
    ``PASS``, the report is not clean, and the boolean ``falsifies`` refuses to
    answer at all rather than returning ``False``.

    Confirmed against a deliberately broken copy: changing ``band_verdict`` to
    return ``PASS`` when either input is ``None`` makes this test fail on the
    first assertion, and ``test_band_report_emits_exactly_one_row_per_band``
    fail as well.
    """
    for w, m in ((None, F(1, 10)), (F(1, 10), None), (None, None)):
        v = FZ.band_verdict(w, m)
        assert v != FZ.PASS
        assert v == FZ.INSUFFICIENT_DATA
    rows = FZ.band_report([
        FZ.BandCheck("evaluated", F(1, 100), F(1, 10)),
        FZ.BandCheck("not evaluated", None, F(1, 10), reason="shards resuming"),
    ])
    assert FZ.report_is_clean(rows) is False
    assert FZ.report_summary(rows)[FZ.INSUFFICIENT_DATA] == 1
    with pytest.raises(TypeError):
        FZ.falsifies(None, F(1, 10))


def test_control_shrinking_the_claimed_modulus_flips_the_falsifier():
    """MUTATION: a modulus claimed smaller than a known enclosure width.

    Uses a real certified width from the lattice machinery, not a made-up
    number, so the control exercises the wiring end to end.
    """
    enc = LT.band_enclosure(GAUSS, Interval(F(2, 100), F(6, 100)), prec=30)
    w = enc.width()
    assert w > 0
    assert FZ.band_verdict(w, w * 10) == FZ.PASS
    assert FZ.band_verdict(w, w) == FZ.PASS                 # equality: PASS
    assert FZ.band_verdict(w, w - F(1, 10 ** 12)) == FZ.FALSIFIED
    assert FZ.band_verdict(w, w / 10) == FZ.FALSIFIED
    assert FZ.band_verdict(w, F(0)) == FZ.FALSIFIED


def test_control_a_mistranscribed_published_value_moves_the_forced_constant():
    """MUTATION: one digit changed in a published ``I_hi/r^3`` value.

    Proves the exact-arithmetic assertions in group 3 are sensitive: a
    transcription error of one part in ten thousand moves the forced constant
    far outside the windows those tests assert.

    Confirmed against a deliberately broken copy: editing ``664.3979`` to
    ``664.3978`` in a scratch ``ladder.py`` fails both
    ``test_published_points_are_exact_decimal_transcriptions`` and
    ``test_implied_modulus_constants_at_the_displayed_kappa``.
    """
    good = LD.adjacent_bands()[0]
    bad = LD.AdjacentBand(
        r_lo=good.r_lo, r_hi=good.r_hi,
        value_lo=good.value_lo - F(1, 10000),   # 664.3979 -> 664.3978
        value_hi=good.value_hi,
        package_lo=good.package_lo, package_hi=good.package_hi,
    )
    c_good = LD.implied_modulus_constant(good, F(1, 8), 40).constant
    c_bad = LD.implied_modulus_constant(bad, F(1, 8), 40).constant
    assert c_good.intersect(c_bad) is None
    assert not (F("5.18184") < c_bad.lo <= c_bad.hi < F("5.18186"))


def test_control_a_wrong_sign_in_the_spread_is_caught():
    """MUTATION: reverse the subtraction in the same-r version spread.

    The frozen v1 value is the larger one, so the spread is positive. A flipped
    sign would give -83.6263 and a negative relative figure.
    """
    sp = LD.same_r_version_spread()
    assert sp["spread"] > 0
    assert sp["frozen_v1"] > sp["live_v3_clean"]
    flipped = sp["live_v3_clean"] - sp["frozen_v1"]
    assert flipped == -F(836263, 10000)
    assert flipped != sp["spread"]


def test_control_truncation_alone_is_not_the_infinite_sum_at_period_24():
    """The side-24 case, where the gap is real but below working precision.

    At ``L = 24`` the omitted mass is about ``1e-498``, so no ordinary
    computation would notice it missing -- which is exactly why the frozen
    engine's assertion that it is "< 1e-60" reads as sufficient. It is not
    sufficient as a *bound*, because nothing adds it. Here it is added, and the
    test pins that the added quantity is strictly positive and strictly smaller
    than any width the machinery reports.
    """
    enc = LT.band_enclosure(GAUSS, Interval(F(2, 100), F(6, 100)), prec=30)
    assert enc.tail > 0
    assert enc.tail < enc.width()
    assert enc.total.lo < enc.truncated.lo
    assert enc.total.hi > enc.truncated.hi
    assert enc.total.width() == enc.truncated.width() + 2 * enc.tail


# ===========================================================================
# 6. Regression controls added after an adversarial audit (2026-09-18)
#
# Each of these closes a defect the audit demonstrated on the shipped code.
# They are written so that reverting the corresponding fix makes them fail;
# that was checked against broken copies in the scratchpad and is recorded in
# each docstring.
# ===========================================================================

_HONEST_JUSTIFICATION = (
    "CONTROL ONLY. Shape-valid envelope used to exercise the certification "
    "flags; it is not claimed to dominate any kernel."
)


def test_control_a_false_envelope_cannot_produce_a_certified_record():
    """THE AUDIT'S HEADLINE DEFECT. Reproduced, then closed.

    Before the fix, ``PlaneKernel.certified`` DEFAULTED to ``True`` and
    ``BandEnclosure.certified`` was copied from it alone. A ``DecayEnvelope``
    was validated only by ``justification.strip()`` being non-empty, so a
    record whose ``to_dict()["certified"]`` read ``true`` could be built from an
    envelope claiming ``B = 7`` for a kernel that decays at ``B = 1/2``. The
    audit measured the damage: a tail of 8.687e-26 where the honest bound is
    2.129e-1, a factor of 2.45e24, and the record still said certified.

    Two things now stand between that envelope and the word "certified":
    ``DecayEnvelope.certified`` defaults to ``False``, and
    ``band_enclosure`` writes ``certified`` only as the conjunction of the
    evaluator flag and the envelope flag. This test asserts both, and asserts
    that the arithmetic is still exactly as wrong as it was -- the fix is a
    labelling fix and must not be mistaken for a repair of the number.
    """
    bad = LT.DecayEnvelope(
        kind=LT.GAUSSIAN, A=F(1), B=F(7), valid_from=F(0),
        justification="DELIBERATELY FALSE, control only: the unit Gaussian "
                      "decays at B = 1/2, not B = 7.",
    )
    assert bad.certified is False          # the default, and the point

    kern = LT.PlaneKernel(name="float stand-in", evaluate=GAUSS.evaluate,
                          envelope=bad)    # certified= deliberately not passed
    assert kern.certified is False         # the evaluator default is False too
    assert kern.envelope_certified is False
    assert kern.fully_certified is False

    box = Interval(F(1, 4), F(3, 4))
    enc = LT.band_enclosure(kern, box, displacement=LT.diagonal_displacement,
                            n_trunc=1, period=F(2), prec=20)
    assert enc.certified is False
    assert enc.to_dict()["certified"] is False
    assert enc.to_dict()["envelope_certified"] is False
    assert any("DecayEnvelope.certified is False" in c for c in enc.caveats)
    assert any("envelope" in c.lower() and "INPUT" in c.upper()
               for c in enc.caveats)

    # The NUMBER is still wrong by the factor the audit measured. Labelling it
    # honestly does not make it a bound, and this test says so out loud.
    dx, dy = LT.diagonal_displacement(box)
    honest = LT.tail_bound(GAUSS.envelope, dx, dy, n_trunc=1, period=F(2),
                           prec=20)
    assert enc.tail < honest / 10 ** 20


def test_control_an_honest_envelope_still_needs_a_certified_evaluator():
    """The conjunction must be a conjunction: neither flag alone is enough.

    MUTATION this catches: ``certified=kernel.certified`` (the old code) would
    make the second case below read ``True``, and
    ``certified=kernel.envelope_certified`` would make the first.
    """
    good_env = LT.DecayEnvelope(
        kind=LT.GAUSSIAN, A=F(1), B=F(1, 2), valid_from=F(0),
        justification=_HONEST_JUSTIFICATION, certified=True)
    weak_env = LT.DecayEnvelope(
        kind=LT.GAUSSIAN, A=F(1), B=F(1, 2), valid_from=F(0),
        justification=_HONEST_JUSTIFICATION)          # certified False

    band = Interval(F(1, 100), F(2, 100))

    # good envelope, uncertified evaluator
    k1 = LT.PlaneKernel(name="k1", evaluate=GAUSS.evaluate,
                        envelope=good_env, certified=False)
    e1 = LT.band_enclosure(k1, band, prec=20)
    assert (e1.evaluator_certified, e1.envelope_certified) == (False, True)
    assert e1.certified is False

    # certified evaluator, unflagged envelope
    k2 = LT.PlaneKernel(name="k2", evaluate=GAUSS.evaluate,
                        envelope=weak_env, certified=True)
    e2 = LT.band_enclosure(k2, band, prec=20)
    assert (e2.evaluator_certified, e2.envelope_certified) == (True, False)
    assert e2.certified is False

    # both flagged
    k3 = LT.PlaneKernel(name="k3", evaluate=GAUSS.evaluate,
                        envelope=good_env, certified=True)
    e3 = LT.band_enclosure(k3, band, prec=20)
    assert e3.certified is True


def test_control_both_certification_flags_default_to_false():
    """A flag meaning "this was checked" must default to "it was not".

    MUTATION this catches: restoring either ``certified: bool = True`` default.
    """
    env = LT.DecayEnvelope(kind=LT.GAUSSIAN, A=F(1), B=F(1, 2),
                           valid_from=F(0), justification=_HONEST_JUSTIFICATION)
    assert env.certified is False
    k = LT.PlaneKernel(name="k", evaluate=GAUSS.evaluate, envelope=env)
    assert k.certified is False
    # The two shipped reference kernels are the only things that opt in, and
    # they do so explicitly, with one-line justifications.
    assert GAUSS.certified is True and GAUSS.envelope.certified is True
    assert POW3.certified is True and POW3.envelope.certified is True


def test_control_the_normalized_ratio_cannot_be_separated_from_its_caveats():
    """The quantity the obligation's content line names must carry its flags.

    ``J(B)/r^{p_J}`` is the number most likely to be quoted out of context.
    Before the fix it came back as a bare ``Interval`` with no ``certified``
    flag, no REFERENCE-KERNEL note and none of ``_BAND_CAVEATS``; the companion
    breakdown was in the tuple but nothing forced a caller to keep it.

    MUTATION this catches: returning ``enc.total / (r_band ** power)`` bare.
    """
    k = LT.PlaneKernel(name="x", evaluate=GAUSS.evaluate,
                       envelope=GAUSS.envelope, certified=False,
                       notes=GAUSS.notes)
    rec = LT.normalized_band_enclosure(k, Interval(F(2, 100), F(6, 100)), 3,
                                       prec=20)
    assert isinstance(rec, LT.NormalizedBandEnclosure)
    assert rec.certified is False
    assert rec.evaluator_certified is False
    assert rec.envelope_certified is True
    assert rec.caveats and any("OBL-H5-JETMOD" in c for c in rec.caveats)
    assert "REFERENCE KERNEL" in rec.notes
    d = rec.to_dict()
    assert d["certified"] is False and d["caveats"]
    # and the older tuple call site still works, with the flags intact
    ratio, enc = rec
    assert ratio is rec and enc is rec.enclosure
    assert ratio.lo == rec.ratio.lo and ratio.hi == rec.ratio.hi
    # ``x in ratio`` still asks the containment question, not an equality scan.
    assert Interval.exact(rec.ratio.mid()) in ratio


def test_control_the_falsifier_report_labels_its_float_rendering():
    """Two rows, opposite verdicts, byte-identical decimals -- and the output
    must say the decimals are NON-CERTIFYING and print the exact values.

    The verdict was always exact; the ARTIFACT A HUMAN READS was not, and that
    artifact is what gets pasted into a status discussion. Repository rule:
    every float path is labelled NON-CERTIFYING in code AND in output.

    MUTATION this catches: dropping the ``exact:`` line, or the header note.
    """
    w = F(1, 1000)
    rows = FZ.band_report([
        FZ.BandCheck("band-A", w, w + F(1, 10 ** 15)),
        FZ.BandCheck("band-B", w, w - F(1, 10 ** 15)),
    ])
    assert [r.verdict for r in rows] == [FZ.PASS, FZ.FALSIFIED]

    # The decimal renderings really are identical -- that is the hazard.
    assert f"{float(rows[0].claimed_modulus):.6g}" == \
           f"{float(rows[1].claimed_modulus):.6g}"

    text = FZ.format_report(rows)
    assert "NON-CERTIFYING" in text
    # Every row prints its exact Fractions, and those DO differ.
    assert f"modulus={rows[0].claimed_modulus}" in text
    assert f"modulus={rows[1].claimed_modulus}" in text
    assert str(rows[0].claimed_modulus) != str(rows[1].claimed_modulus)
    # No row is representable only by its lossy columns.
    for r in rows:
        assert f"exact: width={r.width} modulus={r.claimed_modulus}" in text


def test_control_a_single_admissible_modulus_constant_exists():
    """THE SECOND AUDIT DEFECT. 'The published points do not exhibit a single
    constant' was FALSE, and this test pins the true statement.

    A modulus is an INEQUALITY ``|Delta| <= C*delta^kappa``. Two bands forcing
    different MINIMAL constants are compatible with one admissible constant --
    the larger. The check below is deliberately biased against itself: ``C`` is
    the max over bands of the enclosure's UPPER endpoint, and each band is then
    tested against ``C * delta_pow_kappa.lo``, the LOWER endpoint of the power
    enclosure, which under-estimates ``delta^kappa``.

    MUTATION this catches: any reintroduction of a "no single constant" claim,
    since the table below would then have to show a False.
    """
    for kappa in (F(0), F(1, 8), F(1, 2), F(1), F(2), F(4), F(5)):
        d = LD.common_admissible_constant(kappa, prec=40)
        assert d["all_satisfied"] is True, kappa
        assert len(d["bands"]) == 2
        for b in d["bands"]:
            assert b["satisfied_by_common_C"] is True
        assert IS_ASSUMPTION_ATTACHED(d)

    # At the displayed kappa the constant is the larger of the two forced
    # minima, ~23.1709, and it is exhibited rather than asserted.
    d = LD.common_admissible_constant(F(1, 8), prec=40)
    C = d["common_constant"]
    assert F("23.17") < C < F("23.18")
    table = LD.implied_modulus_table(F(1, 8), prec=40)
    assert C == max(im.constant.hi for im in table)
    # And what the two bands really differ in is the MINIMUM each forces.
    ratio = LD.implied_modulus_ratio(F(1, 8), prec=40)["ratio"]
    assert F("4.47") < ratio.lo <= ratio.hi < F("4.48")


def IS_ASSUMPTION_ATTACHED(d) -> bool:
    """The reading these numbers depend on must travel with them."""
    return (d["assumption"] == LD.IMPLIED_MODULUS_ASSUMPTION
            and d["status_note"] == LD.STATUS_NOTE)


def test_control_the_common_constant_report_line_is_printed():
    """The corrected statement has to reach the human-readable report too."""
    text = LD.format_report(F(1, 8), prec=30)
    assert "SINGLE ADMISSIBLE CONSTANT EXISTS" in text
    assert "tightness, not about consistency" in text
    assert "forced MINIMA" in text
    assert LD.IMPLIED_MODULUS_ASSUMPTION.split(". ")[0] in text


def test_control_a_common_constant_below_a_forced_minimum_is_rejected():
    """The satisfaction check must have teeth: shrink C and it must fail.

    MUTATION this catches: ``common_admissible_constant`` reporting
    ``all_satisfied`` unconditionally, or comparing against the wrong endpoint.
    """
    table = LD.implied_modulus_table(F(1, 8), prec=40)
    C = max(im.constant.hi for im in table)
    too_small = min(im.constant.lo for im in table)     # only the narrow band
    assert too_small < C
    ok = [im.band.abs_diff <= C * im.delta_pow_kappa.lo for im in table]
    bad = [im.band.abs_diff <= too_small * im.delta_pow_kappa.hi
           for im in table]
    assert all(ok)
    assert not all(bad)          # the small constant fails the wider band
