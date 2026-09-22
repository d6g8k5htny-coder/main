# Reviewer cross-checks for the H3 search-and-certify pilot

**NON-CERTIFYING. Zero organizational independence credit. No gate moved.**

Two scripts backing a nonauthor technical review of
`engine/solver_pilots/h3_20260921/` at commit
`2371f660d90963e558a0d08921c634a36b3be016`. They are reviewer working
artifacts, not part of the engine, not invoked by any workflow, and nothing in
the repository depends on them.

| script | question it answers | method |
|---|---|---|
| `reconstruct_bound.py` | does an implementation written from `PROOF.md` alone land inside the author-side rational enclosure? | mpmath, 120 dps, imports nothing from `engine/solver_pilots/` |
| `sample_true_value.py` | is the claimed inequality consistent with the quantity it is about? | float Monte Carlo of the conditional law at `r=1/20` |

Both guard their imports: `mpmath` and `numpy` are not guaranteed present, and
each script exits 0 with a message when its dependency is missing.

## What was observed

`reconstruct_bound.py` reproduces the four section-1 inequalities
(`transport^2 = 2.40063476562 < 2.4025`, `Q(0) = 2.82666666667 < 17/6`,
`(193/1000)^2 < 3/80`, `Q* = 7.91130850871 < 8`) and arrives at
`1.7472485483566`, inside the author-side enclosure
`[1.747106129274, 1.747994749902]` and above `1747/1000` by `2.49e-4`.

`sample_true_value.py` estimates the true `Z(r)/r^2` at `r=1/20` as
`3.224 +/- 0.005` (3 sigma, 8e6 samples, typed-event rate `0.7938`), roughly
`1.85x` the candidate coefficient — the margin the proof's event and
positive-part truncation losses would predict.

## What these scripts do not establish

They are mpmath and float paths. CLAUDE.md rule 3 names both a high-precision
computation and a Monte Carlo estimate as things that are not certificates, and
that is exactly what these are. Agreement between two implementations is
evidence that both read the same prose the same way; it is not evidence that
the prose is correct, and the sampling found no counterexample at one radius
rather than showing there is none. Neither script bounds anything, verifies a
register, reclassifies a claim, or supplies the organizationally distinct
reviewer any independence-requiring predicate still calls for. The sharper
fixed-radius RN floor `0.0077592917375327855` is untouched by both, and the
pilot they cross-check is NOT DEPLOYED.
