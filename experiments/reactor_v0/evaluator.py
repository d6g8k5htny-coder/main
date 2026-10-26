"""Finite amplitude-transform regression control; no sampler or theorem use."""

from fractions import Fraction

from exact_helpers import exact_periodic_h0, rational_power_exact


def short_bar_lifetimes(barcode, threshold):
    """Select positive finite bars with inclusive lifetime threshold."""
    if type(threshold) is not Fraction or threshold < 0:
        raise ValueError("nonnegative exact Fraction threshold required")
    intervals = barcode.get("intervals")
    if type(intervals) is not list:
        raise ValueError("finite intervals list required")
    lifetimes = []
    for pair in intervals:
        if (type(pair) is not list or len(pair) != 2 or
                any(type(endpoint) is not int for endpoint in pair)):
            raise ValueError("strict integer finite-bar endpoints required")
        lifetime = pair[0] - pair[1]
        if lifetime <= 0:
            raise ValueError("positive finite bars required")
        if Fraction(lifetime) <= threshold:
            lifetimes.append(lifetime)
    return tuple(lifetimes)


def _fraction(document):
    return Fraction(document["numerator"], document["denominator"])


def evaluate_package(package):
    fixture = package["fixture"]
    control = package["control"]
    barcode = exact_periodic_h0(fixture["values"], fixture["grid_side"])
    if barcode != fixture["expected_barcode"]:
        raise ValueError("exact barcode differs from frozen regression fixture")
    threshold = _fraction(control["threshold"])
    alpha = _fraction(control["alpha"])
    amplitude_scale = _fraction(control["amplitude_scale"])
    finite_lifetimes = tuple(birth - death for birth, death in barcode["intervals"])
    # The finite transform integrates each declared positive finite lifetime.
    terms = []
    for lifetime in finite_lifetimes:
        base = threshold / (amplitude_scale * lifetime)
        terms.append(min(rational_power_exact(base, alpha), Fraction(1)))
    expected = sum(terms, Fraction(0))
    if expected != _fraction(control["expected_control"]):
        raise ValueError("declared expected control differs from exact evaluation")
    return {
        "finite_bars": tuple(tuple(pair) for pair in barcode["intervals"]),
        "essential_birth": barcode["essential"][0],
        "included_lifetimes": finite_lifetimes,
        "expected_short_bar_count": expected,
    }
