#!/usr/bin/env python3
"""Frontier round (docs only) for specifications/lean_statements_20261010 at Math- 8acc4856.

Inserts one "Frontier" section into README.md and SCOPE.md listing the 64 registered Props that have no
proof-packet theorem, each with its class (vocabulary / interface-blocked / Mathlib-blocked) and a one-line
reason anchored to the statement module line. Fail-closed: the base bytes are read from git, every anchor
must occur exactly once, every name must be a registered def_prop without a theorem, and the three counts
must be 36 + 24 + 4 = 64. REGISTER.json and every .lean file are untouched. Scientific effect NONE.
"""
import hashlib, json, re, subprocess, sys
from pathlib import Path

S = Path(sys.argv[1]); REPO = S / 'repos/math_r5'; WT = S / 'lvs/r7/wt'
CUT = '8acc48566de3ae613c1ae320903b66c1d5411797'
PK = 'specifications/lean_statements_20261010'

def show(path):
    return subprocess.run(['git', '-C', str(REPO), 'show', f'{CUT}:{path}'], check=True, capture_output=True).stdout

reg = json.loads(show(f'{PK}/REGISTER.json'))
props = {s['declaration'] for c in reg['claims'] for s in c['statements']
         if s['package'] == PK and s['kind'] == 'def_prop'}
assert len(props) == 141, len(props)
cov = json.load(open(S / 'lvs/r7/coverage_8acc4856.json'))
covered = set(cov['covered']) | {'UniversalLaw.Spec.' + n for n in (
    'P15.CoverCostHazardBound', 'P15.RestrictedLocalBudget', 'P15.RhoTestSufficient', 'P15.LocalRestriction',
    'P15.PaletteNumberAttained', 'P15.LocalPriceChain', 'P15.IndependentBlockHazard', 'P15.GlobalAssemblyHazard',
    'P15.LocalPriceBoundFull', 'P15.FullTransformedPriceBudget', 'P15.LocalDecomposability',
    'P15.LocalObstructionIsFullBlock', 'P15.FullBlockCoverIff', 'P15.WholeGroundChromaticNumber',
    'RN.RareClusterPoissonObstructionO', 'Side24.ConeMomentD1Identity')}
assert len(covered) == 77, len(covered)
remaining = {p.replace('UniversalLaw.Spec.', '') for p in props - covered}
assert len(remaining) == 64, len(remaining)

modules = {m: show(f'{PK}/Specifications/{m}.lean').decode().splitlines() for m in ('Field', 'Lifetime', 'RN', 'Side24', 'P15')}
def defline(short):
    mod, name = short.split('.', 1)
    hits = [i + 1 for i, l in enumerate(modules[mod]) if re.match(r'^def\s+' + re.escape(name) + r'\b', l)]
    assert len(hits) == 1, (short, hits)
    return f'{mod}.lean:{hits[0]}'

VOC = 'Predicate with explicit non-instance arguments, consumed only as a hypothesis; its universal closure is false or empty, so it names an object, not a result.'
vocab = ['Field.IsLPeriodic', 'Field.IsSmooth', 'Field.IsCriticalPoint', 'Field.HessianHasInertia', 'Field.IsNondegMax',
         'Field.IsIndexSaddle', 'Field.IsCandidatePair', 'Field.IsMorseDistinct', 'Field.IsFiniteBar',
         'Field.IsLifetimeDensityVersion', 'Lifetime.IsLPeriodic', 'Lifetime.IsSmooth', 'Lifetime.IsCriticalPoint',
         'Lifetime.HessianHasInertia', 'Lifetime.IsNondegMax', 'Lifetime.IsIndexSaddle', 'Lifetime.IsCandidatePair',
         'Lifetime.IsMorseDistinct', 'Lifetime.IsFiniteBar', 'Lifetime.IsLifetimeDensityVersion',
         'Lifetime.IsUnrestrictedPair', 'Lifetime.IsUnrestrictedLeadingCoefficient', 'Lifetime.IsWindowPair',
         'Lifetime.IsWindowElderPair', 'RN.IsCriticalPoint', 'RN.HessianHasInertia', 'Side24.IsUnitTuple',
         'Side24.NegDefCone', 'P15.IsRealizedFamily', 'P15.IsGood', 'P15.IsKDecomposable', 'P15.IsGeneratorCover',
         'P15.IsPaletteFeasible', 'P15.IsMinimalForbidden', 'P15.IsTransversal', 'P15.IsProperDownset']
PIN = ('`PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins '
       '(`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not '
       'transcribed (RN.lean:325-333). ')
interface = {
 'Side24.CoefficientEnclosure': '`PeriodicCoefficientInterface` (Side24.lean:770-822) leaves `coneMoment` (788) and `c24` (772) free up to `eq_15_2` (812-815): doubling both gives another inhabitant, so a theorem of this type would force the interface to be empty. Missing field: the conditional law of the Hessian block given the gradient under `K24`, which pins `coneMoment`.',
 'Side24.ReferenceComparison': 'Same mechanism as `CoefficientEnclosure`: `|c24 2 / cRef 2 - 1| < 10^-106` and the doubled inhabitant\'s `|2 q - 1| < 10^-106` refute each other. Missing field: the same conditional law; the arithmetic is trivial.',
 'RN.AnnulusBridgeAllHeight2': PIN + 'A Dirac law at a smooth periodic pinned path with one extra nondegenerate minimum at `8 r u` in the annulus has mean count >= 1 against a bound `C r^3 vol(E0)` that vanishes on a point.',
 'RN.D5aPuncturedPinDisks': PIN + 'Extra minimum at `M + r^2 u`, `E0` a single point of volume 0: left side >= 1, right side 0.',
 'RN.D5bMicrodisk': PIN + 'With kappa = 1 and an extra minimum at `M + r^2 u`, the count is >= 1 while `C r^5 -> 0`.',
 'RN.D5cCollarC1': PIN + 'The midpoint 0 lies in every admissible collar; an extra minimum there with `E0 = {0}` gives >= 1 against `C r^3 * 0`.',
 'RN.D5dSummedPinNeighbourhoodC2': PIN + 'No volume factor: an extra minimum at the midpoint gives >= 1 against `C r^3 -> 0`.',
 'RN.D5eIntermediateShellsI3I4': PIN + 'An extra window minimum at `8 r u` (height `b - k r^3/2`) lies in the shell for `A0 = 4`, `rho = s0`; right side `C r^3 (1/16 + s0^2) -> 0`.',
 'RN.D5fGlobalWindowFirstMomentI5': PIN + 'Conjunct (i) fails for the extra-minimum Dirac law (count >= 1 > C r^3); conjunct (ii)\'s lower bound fails for the Dirac law with no window critical point (probability 0 < c r^3) and its upper bound for the extra-minimum law (probability 1).',
 'RN.DimensionLiftCompactCollarC': PIN + 'General-d `D5cCollarC1` (`2 <= d`): same midpoint inhabitant with an index-(d-1) saddle at `S`; the lemmas used are dimension-free.',
 'RN.DimensionLiftGlobalWindowG': PIN + 'General-d conjunct (i) of `D5fGlobalWindowFirstMomentI5`: extra window minimum at the midpoint, count >= 1 > C r^3.',
 'RN.DimensionLiftIntermediateShellsI': PIN + 'Either conjunct fails for the extra minimum at `8 r u`: take `s = 8 r` for (1.5) or `A0 = 4`, `rho = s0` for (1.6); both right sides -> 0.',
 'RN.DimensionLiftPuncturedPinBallP': PIN + 'The second inequality (`C r^3 \\int omegaD <= C\' r^3`) is provable on its own; the first fails as `D5aPuncturedPinDisks` with `E0 = {r e_0}` (set integral over a point is 0).',
 'RN.FixedAnnulusWindowCandidateA2': PIN + 'No absolute-continuity field either: a Dirac law at a pinned path with an extra nondegenerate index-j window critical point in the annulus, `E0` that point, gives >= 1 against `C k r^3 * 0`.',
 'RN.FixedRemoteCountBoundA3': PIN + 'Extra index-j window critical point at `x0 = (L/2) e_0` in the remote region, `A = {x0}` of volume 0: both left sides 1, both right sides 0.',
 'RN.FixedRemoteMeanMeasureA2': 'The field `Lambda` (RN.lean:382) has no defining property in the structure, so any inhabitant can be copied with `Lambda = 0`, and `0 < c_Lambda <= Lambda` fails on the nonempty remote region. ' + PIN,
 'RN.FourierCutoffPlanarSecondFactorialF4': PIN + 'A two-point law `(1 - p_r) delta_g0 + p_r delta_g1` with `n_r >= 1/r` window critical points on `g1` and `p_r = r^3 / n_r` satisfies the first-moment hypothesis `G_2` (`E N = r^3`) while `E N(N-1) >= r^2 - r^3` exceeds `C r^3 log(1/r)` for small `r`.',
 'RN.PalmRouteFactorialMomentsQ': PIN + 'A Dirac law at a pinned path with exactly two extra window critical points has `(N)_2 = 2` for every `r`, against `C_q r^3 -> 0`.',
 'RN.PalmRouteSecondFactorialTheta': PIN + 'Two-sided: the pinned path with no window critical point gives `E (N)_2 = 0 < c r^3`; the two-extra-point path gives `2 > C r^3`.',
 'RN.RemoteCollisionSecondFactorialD': PIN + 'Two extra window critical points at remote `x0`, `x1` and `A = {x0, x1}` of volume 0: `(N)_2 >= 2` and the event `2 <= N` has mass 1, against right sides 0.',
 'RN.WitnessCollisionNearPinPairsTarget': PIN + 'Two extra window critical points between the pins within torus distance `eta` of `pinM` give `nearPinPairCount = 2` for every small `r`, against `C r^3 -> 0`. (The source marks this target OPEN; the interface cannot carry it either way.)',
 'Lifetime.ParentUniformSelectionA': 'The only fields of `PairLawInterface` are `p_nonneg` (Lifetime.lean:463) and `p_le_one` (465); nothing ties `p` to `G` or `I`. The inhabitant `p = 0` breaks `1 - p <= C r^3` (probe Lifetime-B_1 elaborates the refutation). Missing field: the law of `p` itself.',
 'Lifetime.PlanarElderLowerE2': 'Same interface: `p = 1` makes `1 - p = 0`, so `c r^3 <= 1 - p` fails for every `c > 0` (probe Lifetime-B_1).',
 'Lifetime.PlanarTwoSidedRateE4': 'Same interface: `p = 1` satisfies the hypothesis `ParentUniformSelectionA` (with `C = 0`) and breaks the lower half of the conclusion (probe Lifetime-B_1).',
}
mathlib = {
 'Lifetime.ParentCompactWindowDensityB': 'Theorem B (1.2) of the pinned source in full: density versions (a Kac-Rice representation), a.s. finiteness and integrability of the pair counts, the `c l^(-1/3)` asymptotics with `c > 0`, and the `O(l^(2/3))` gap. The `gaussian` field (Lifetime.lean:175-177) fixes the law, so the truth value does not depend on the inhabitant; none of the Gaussian-field machinery exists in Mathlib `d13f23b7`.',
 'Lifetime.ParentUnrestrictedLeadingDensityC': 'Theorem C (1.3): density versions, a.s. Morse genericity (so the elder relation is pinned a.s.), the off-diagonal domination and the small-`l` asymptotics with a positive constant; absent from the pinned Mathlib.',
 'Lifetime.PlanarDensityGapE11': 'The hypothesis is satisfied by `p = 1` (probe Lifetime-B_1), so a theorem would be exactly the unconditional two-sided planar density gap for every `(G, I)` at `d = 2`: the (E2) planar lower argument plus the compact-window `O(l^(2/3))` upper bound, both Gaussian-field analysis absent from the pinned Mathlib.',
 'Lifetime.RemainderBoundedR1': 'Theorem R (R1): the hypothesis gives only `nu ~ c l^(-1/3)`, i.e. an `o(l^(-1/3))` error, strictly weaker than the `O(1)` remainder claimed; density versions are unique a.e., so re-choosing them cannot supply the bound; the remainder analysis of the source has no Mathlib counterpart.',
}
assert len(vocab) == 36 and len(interface) == 24 and len(mathlib) == 4
allnames = set(vocab) | set(interface) | set(mathlib)
assert allnames == remaining, (sorted(allnames - remaining), sorted(remaining - allnames))

def row(name, cls, reason):
    return f'| `{name}` | {defline(name)} | {cls} | {reason} |'
rows = [row(n, 'vocabulary', VOC) for n in vocab] + [row(n, 'interface-blocked', r) for n, r in interface.items()] \
     + [row(n, 'Mathlib-blocked', r) for n, r in mathlib.items()]

SECTION = f"""## Frontier: the 64 registered Props without a theorem

At the Math- cut `{CUT[:8]}` the sibling proof packet proves 77 of the 141 registered Props. This section
classifies the 64 others, so that the remaining formal work is named where the register lives. The
classification is this packet's author's reading of the statement modules (the round-4 triage of 2026-10-11,
re-applied at this cut; the statement modules have not changed since `873c6c0`, so the cited lines hold),
by a single reader, not a reviewed fact; it moves no label: every one of the 64 stays `specified`, nothing is
promoted, reclassified or withdrawn. Scientific effect NONE.

Three classes, by what a proof would need:

- **vocabulary** (36): a `def … : Prop` with explicit non-instance arguments that the modules consume only as a
  hypothesis (`IsCriticalPoint d g x →`, `IsUnitTuple d q u →`, …). Its universal closure is false or empty, so no
  theorem of its type is a target; it is counted among the 141 because the register inventories every `Prop`-valued
  definition. Of the 105 target Props, 77 are proved and 28 remain.
- **interface-blocked** (24): the Prop quantifies over every inhabitant of a pinned interface whose fields do not
  determine the conclusion. For each, a (mathematical, not kernel-checked) inhabitant that satisfies every field and
  violates the conclusion is named below. No cleverer proof exists; the route is a numbered successor interface
  carrying the missing field (a Kac-Rice intensity, the law of `p`, the conditional Hessian law), followed by
  successor Props. Nothing here says the informal source is wrong: the source's hypotheses are exactly what the
  interface does not transcribe.
- **Mathlib-blocked** (4): the Prop's truth value is fixed by the interface, and the statement follows the pinned
  source, but the proof needs Gaussian-field or density theory that Mathlib `d13f23b7` does not contain
  (effort estimated in weeks each).

| Prop (`UniversalLaw.Spec.`) | defined at | class | what a proof would need |
|---|---|---|---|
{chr(10).join(rows)}

Counts: 36 vocabulary + 24 interface-blocked + 4 Mathlib-blocked = 64; 64 + 77 = 141. The three Side24 Props the
round-4 triage filed as Mathlib-blocked (`ImageBound`, `EntrywiseCovarianceBound`, `CovarianceOrdering`) were proved
in round 6 by building the missing lemmas inside the proof packet, which is why they are absent here; the four
Lifetime entries above are not of that kind.

"""
def edit(text, heading_anchor, xref_anchor, xref_add):
    assert text.count(heading_anchor) == 1 and text.count(xref_anchor) == 1, (heading_anchor, xref_anchor)
    text = text.replace(xref_anchor, xref_anchor + xref_add)
    return text.replace(heading_anchor, SECTION + heading_anchor)

readme = show(f'{PK}/README.md').decode()
scope = show(f'{PK}/SCOPE.md').decode()
readme2 = edit(readme, '## What this packet is not\n', '(77 of the 141 Props, 78 theorems)',
               '; the 64 Props without a theorem are classified in the Frontier section below')
scope2 = edit(scope, '## Not established (by construction)\n', '(78 theorems typed exactly as 77 of the 141 Props;',
              ' the 64 others are classified in the Frontier section below;')
assert subprocess.run(['git', '-C', str(WT), 'status', '--short'], check=True, capture_output=True).stdout == b''
assert subprocess.run(['git', '-C', str(WT), 'rev-parse', 'HEAD'], check=True, capture_output=True).stdout.decode().strip() == CUT
for name, before, after in (('README.md', readme, readme2), ('SCOPE.md', scope, scope2)):
    p = WT / PK / name
    assert p.read_bytes() == before.encode(), name
    p.write_bytes(after.encode())
    print(f'{name}: {len(before.encode())} -> {len(after.encode())} bytes, sha256 {hashlib.sha256(after.encode()).hexdigest()[:16]}…, '
          f'+{after.count(chr(10)) - before.count(chr(10))} lines')
print('rows', len(rows), '| vocabulary 36, interface-blocked 24, Mathlib-blocked 4')
