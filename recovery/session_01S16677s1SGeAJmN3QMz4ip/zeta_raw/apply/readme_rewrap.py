import sys, re
from pathlib import Path
P = Path(sys.argv[1]); s = P.read_text()
E = []
def r(old, new): E.append((old, new))
r("""   ℓ^(−1/3) to roughly ℓ⁰. Our numerics, at heights 10⁴–10⁶, are decisive against −1/3 for
   zeta. They are consistent with 0 once the small finite-cutoff bias that the estimator also shows on
   controls with known answers (0.02 to 0.07) is allowed for. The data were inspected before
   any prediction was frozen, so this is exploration, not confirmation.""",
"""   ℓ^(−1/3) to roughly ℓ⁰. Our numerics, at heights 10⁴–10⁶, are decisive against −1/3 for
   zeta. They are consistent with 0 once the small finite-cutoff bias that the estimator also
   shows on controls with known answers (0.02 to 0.07) is allowed for. The data were inspected
   before any prediction was frozen, so this is exploration, not confirmation.""")
r("""    broad kernel classes [de Bruijn 1950; Newman 1976; snippet]. Ki–Kim–Lee and Rodgers–Tao use ζ-specific zero information
    (arithmetic input, not theta structure): Rodgers–Tao use pair-correlation estimates and
    zero counting, per their abstract [snippet]; Ki–Kim–Lee use zero-distribution facts
    [memory].""",
"""    broad kernel classes [de Bruijn 1950; Newman 1976; snippet]. Ki–Kim–Lee and Rodgers–Tao
    use ζ-specific zero information (arithmetic input, not theta structure): Rodgers–Tao use
    pair-correlation estimates and zero counting, per their abstract [snippet]; Ki–Kim–Lee use
    zero-distribution facts [memory].""")
r("""The lobe area |∫_{z₋}^{z₊} H_t| is the critical-value gap of F_t. Its bar pairing is verified only
near the collision (Proposition 1(c); persistence check at τ = 0.01), not at t = 0. A gap need
not be a finite bar: on [−3/2, 3/2], x³/3 − x has critical-value gap 4/3, but its only finite
H0 bar, born at the right end, has length 7/24.""",
"""The lobe area |∫_{z₋}^{z₊} H_t| is the critical-value gap of F_t. Its bar pairing is verified
only near the collision (Proposition 1(c); persistence check at τ = 0.01), not at t = 0. A gap
need not be a finite bar: on [−3/2, 3/2], x³/3 − x has critical-value gap 4/3, but its only
finite H0 bar, born at the right end, has length 7/24.""")
r("""**Reading of these numbers.** The ζ data are **decisive against −1/3** (the point estimate is
0.31 above −1/3, far larger than any error source). They are **consistent with ≈ 0** given the cut-0.01
bias of about −0.02 seen in the Gaussian and CUE N=12 controls (and −0.065 in CUE N=128). The
value −0.020 [−0.041, +0.001] is not a measurement of 0: the intervals are sampling-only, and
these data cannot separate ℓ⁰ from a mark-driven ℓ^q with |q| ≲ 0.04. At cut 0.003 the""",
"""**Reading of these numbers.** The ζ data are **decisive against −1/3** (the point estimate is
0.31 above −1/3, far larger than any error source). They are **consistent with ≈ 0** given the
cut-0.01 bias of about −0.02 seen in the Gaussian and CUE N=12 controls (and −0.065 in CUE
N=128). The value −0.020 [−0.041, +0.001] is not a measurement of 0: the intervals are
sampling-only, and these data cannot separate ℓ⁰ from a mark-driven ℓ^q with |q| ≲ 0.04. At
cut 0.003 the""")
r("""density were exactly flat. For ζ, t ∈ [10⁴, 10⁵] it is 0.844 [0.769, 0.910] with 20000-lobe
blocks (only 7 clusters; [0.786, 0.911] with 2000-lobe blocks); the two windows differ, which is consistent with, but not evidence for, a slowly
drifting coefficient.)""",
"""density were exactly flat. For ζ, t ∈ [10⁴, 10⁵] it is 0.844 [0.769, 0.910] with 20000-lobe
blocks (only 7 clusters; [0.786, 0.911] with 2000-lobe blocks); the two windows differ, which
is consistent with, but not evidence for, a slowly drifting coefficient.)""")
r("""  F_U = ∫ Z_U on the deterministic arc [0, 2π) (cut at angle 0), with bars born at the arc
  ends dropped (N even if a periodic Z_U is wanted). Does the expected short-bar density satisfy
  ν_N(ℓ) → c_N ∈ (0, ∞) as ℓ ↓ 0, with""",
"""  F_U = ∫ Z_U on the deterministic arc [0, 2π) (cut at angle 0), with bars born at the arc
  ends dropped (N even if a periodic Z_U is wanted). Does the expected short-bar density
  satisfy ν_N(ℓ) → c_N ∈ (0, ∞) as ℓ ↓ 0, with""")
r("""  coefficient comparison**, recompute the CUE bars cut at angle 0 under exactly this
  convention, or derive c_N for the cut-at-θ₁ convention used in `stats/cue.py` (§5.3: about a 10% shift at
  N = 12).""",
"""  coefficient comparison**, recompute the CUE bars cut at angle 0 under exactly this
  convention, or derive c_N for the cut-at-θ₁ convention used in `stats/cue.py` (§5.3: about a
  10% shift at N = 12).""")
r("""  merger for t > 0. RH (Λ ≤ 0) implies that statement, and it is equivalent to RH if a positive
  Λ forces a real collision (`CLAIMS.md` 6.2); it is not merely tied to RH. No packet content is progress toward RH, support for RH, or a bound on Λ
  beyond known results.""",
"""  merger for t > 0. RH (Λ ≤ 0) implies that statement, and it is equivalent to RH if a positive
  Λ forces a real collision (`CLAIMS.md` 6.2); it is not merely tied to RH. No packet content
  is progress toward RH, support for RH, or a bound on Λ beyond known results.""")
r("""`track.py 270`, `track.py 103 -40 0.02 low10`, `collision_table`, `fold_check` for both pairs
(CSV, JSON and stdout), `fold_analyze` (including `fold_tau_r_ell.csv`), `validate_H0`, `count_check` (window [1, 103] at t = −1.89),
`check_gue_transfer`,""",
"""`track.py 270`, `track.py 103 -40 0.02 low10`, `collision_table`, `fold_check` for both pairs
(CSV, JSON and stdout), `fold_analyze` (including `fold_tau_r_ell.csv`), `validate_H0`,
`count_check` (window [1, 103] at t = −1.89), `check_gue_transfer`,""")
bad=[]
for o,n in E:
    c=s.count(o)
    if c!=1: bad.append((c,o[:80])); continue
    s=s.replace(o,n)
if bad:
    print(bad); sys.exit(1)
P.write_text(s); print("rewrapped", len(E))
