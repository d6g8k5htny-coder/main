# H3 search-and-certify pilot

A bounded executable research experiment, not a new global governance layer.
It turns a 30,351-case NONCERTIFYING parameter search into a separately certified
scalar result supporting the fixed-axis whole-band candidate
`Z(r) >= (1747/1000) r^2`, for `0<r<=1/20`. See PROOF.md for the analytic warrant.
Existing claims, gates, frozen sources and the stronger fixed-r RN floor are unchanged.

From the repository root, using only Python standard library dependencies:

```sh
python -m unittest discover -s tests -p test_h3_solver_pilot.py -v
python -O -m unittest discover -s tests -p test_h3_solver_pilot.py -v
python engine/solver_pilots/h3_20260921/h3_solver.py \
  --source-archive research/campaigns/h3_rn_n6_20260920_v1.zip \
  --context engine/solver_pilots/h3_20260921/CONTEXT.json \
  --search --cold --out /tmp/h3-pilot-new-run.json
```

Outputs are exclusive-create; use a new output filename for each run. Optional
`--cache /tmp/h3-pilot-cache` enables trusted-local content-addressed reuse.
`--cold` recomputes cached arithmetic and compares it. `--verify-result PATH`
always recomputes and compares only the deterministic certificate, not timing.
Do not consume an untrusted cache on the strength of a self-reported hash.

The recipe includes exact source archive and manifest, all three executable
modules, parameters, arithmetic precision, runtime fingerprint and caller context.
This is not a hermetic Nix/Guix/container build. Current admissibility is a separate
check; CONTEXT.json is candidate-use metadata, not an authenticated approval service.
A cache hit is not another experiment, review or theorem acceptance.

The certificate contains a declared formula dependency DAG with recursive hashes.
`affected_nodes` identifies downstream formula stages needing reevaluation when an
input changes. This is formula-level lineage for the actual computation, NOT a
parser or tracer for Google Sheets. No spreadsheet cells feed this H3 proof.

The adversarial component currently challenges sufficient-bound parameters and
implementation mutations; it is NOT an oracle independently integrating Z, and
NOT a continuously running prover/falsifier competition. A wider interval crossing
a bound is inconclusive; a lower-estimate formula below a target does not refute
the true quantity. No number of failed samples proves a theorem.

`diagnostic_mpmath.py` is optional, noncertifying, and records availability/version.
The core and CI use no mpmath dependency. Arb/FLINT dual execution is not implemented.
No new scheduler, broad migration, proof market or automatic budget escalation is
installed. Source-exposed author-side independence credit is zero; review remains open.
