"""Exact finite-barcode dependence control; reference answers are exposed.

The fixed amplitude A=2*U**3, with density 2*u for U in (0,1), has
P(A<=x)=min((x/2)**(2/3),1). Its second moment is one. A shared A
scales a complete barcode; independent per-bar amplitudes instead define
an ABSTRACT barcode surrogate with no established underlying field law
or covariance realization. All lifetimes are supplied positive finite
bars, in the same units as the inclusive threshold; essential bars are
excluded by the input contract.

This uses the finite scaling/count identity in main#334, Sections 2-3/6:
https://github.com/d6g8k5htny-coder/main/blob/
6bd133d591e43e75a9292463bb4bfde7a34984dc/
experiments/universality/spectral_amplitude_crossover/PROOF.md
No Gaussian, continuum, asymptotic or process-limit theorem is consumed.
There is no sampler, blind experiment, novelty or scientific acceptance.
"""
from fractions import Fraction as Q
import json

if __package__:
    from .fold_controls import lifetime_bin_mass
else:
    from fold_controls import lifetime_bin_mass


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rational_sequence(values, name):
    _require(type(values) in (list, tuple), name+' must be a list or tuple')
    _require(all(type(value) is Q for value in values),
             name+' must contain exact Fractions')
    return tuple(values)


def short_bar_probabilities(lifetimes, threshold):
    """Exact P(A*ell<=threshold); refuse every unsupported unsaturated root.

    Support clipping precedes root evaluation, after validating ALL inputs.
    Continuous amplitude mass makes inclusive and strict endpoint CDFs equal.
    """
    lengths = _rational_sequence(lifetimes, 'lifetimes')
    _require(all(length > 0 for length in lengths), 'positive finite lifetimes required')
    _require(type(threshold) is Q and threshold >= 0,
             'nonnegative exact Fraction threshold required')
    probabilities = []
    for length in lengths:
        if threshold == 0:
            probability = Q(0)
        elif threshold >= 2*length:
            probability = Q(1)
        else:
            probability = lifetime_bin_mass(
                Q(0), threshold, gap_coefficient=2*length,
                radial_power=1, gap_order=3, weight=Q(2))
        probabilities.append(probability)
    return tuple(probabilities)


def count_distribution(probabilities, *, coupling):
    """Return P(N=k), k=0,...,m, for nested or independent short-bar events.

    In the shared-amplitude world all events are nested: sorting their CDF
    probabilities gives the count tails. The independent world convolves
    the individual Bernoulli laws. Marginals alone do not choose a world.
    """
    marginals = _rational_sequence(probabilities, 'probabilities')
    _require(all(0 <= p <= 1 for p in marginals), 'probabilities must lie in [0,1]')
    _require(type(coupling) is str and coupling in ('shared', 'independent'),
             'coupling must be shared or independent')
    if not marginals:
        return (Q(1),)
    if coupling == 'shared':
        ordered = sorted(marginals, reverse=True)
        return tuple([1-ordered[0]]
                     + [left-right for left, right in zip(ordered, ordered[1:])]
                     + [ordered[-1]])
    law = [Q(1)]
    for p in marginals:
        following = [Q(0)]*(len(law)+1)
        for count, mass in enumerate(law):
            following[count] += mass*(1-p)
            following[count+1] += mass*p
        law = following
    return tuple(law)


def _probability_law(distribution):
    law = _rational_sequence(distribution, 'distribution')
    _require(bool(law) and all(p >= 0 for p in law) and sum(law) == 1,
             'a complete nonnegative probability law summing to one is required')
    return law


def expected_count(distribution):
    """Exact mean of a count law; this mean does not determine its coupling."""
    law = _probability_law(distribution)
    return sum((count*mass for count, mass in enumerate(law)), Q(0))


def pgf(distribution, z):
    """Exact E[z**N] for a Fraction z in [0,1], including the 0**0 term."""
    law = _probability_law(distribution)
    _require(type(z) is Q and 0 <= z <= 1, 'exact Fraction z in [0,1] required')
    value = Q(0)
    for mass in reversed(law):
        value = mass + z*value
    return value


def report():
    """Evaluate the known-answer control, with no field draws or inference."""
    marginals = short_bar_probabilities([Q(1), Q(8)], Q(1, 4))
    result = {
        'scope': 'EXACT_FINITE_BARCODE_DEPENDENCE_CONTROL',
        'reference_answers_exposed': True,
        'independent_world': 'ABSTRACT_BARCODE_SURROGATE',
        'field_realization_established': False,
        'theorem_conclusion_consumed': False,
        'scientific_effect': 'NONE', 'scientific_acceptance': False,
        'lifetimes': ['1', '8'], 'threshold': '1/4',
        'amplitude': 'A=2*U^3; U has density 2*u on (0,1)',
        'marginal_short_probabilities': [str(p) for p in marginals],
    }
    for coupling in ('shared', 'independent'):
        law = count_distribution(marginals, coupling=coupling)
        result[coupling] = {'count_probabilities': [str(p) for p in law],
                            'mean': str(expected_count(law)),
                            'pgf_at_half': str(pgf(law, Q(1, 2)))}
    return result


if __name__ == '__main__':
    print(json.dumps(report(), sort_keys=True, indent=2))
