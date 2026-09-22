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

The recipe includes exact source archive and manifest, all four executable
modules, parameters, arithmetic precision, runtime fingerprint and caller context.
This is not a hermetic Nix/Guix/container build. The source gate binds a numbered
local provenance record, its exact retained custody receipt, current exclusions,
and the current inventory/Payloads metadata for the two declared RN5 prerequisites.
CONTEXT.json is an exact eight-field, caller-supplied **deny-only** filter; unknown
fields and missing required fields refuse. It cannot supply source admission.
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

The source-read boundary is local to this H3 pilot. Before opening the source ZIP,
`h3_source_admission.py` validates the pinned
`sources/H3_AUTHORED_NESTED_SOURCE_V1.json` and its custody receipt. The actual
parent is the 2,972,608-byte registered delivery; its pinned manifest identifies
the 571,735-byte nested source archive. The original `SOURCES.json` remains
unchanged historical provenance: its delivery Drive ID does not identify the
raw archive or the 10,423-byte proof as a standalone Drive object. No synthetic
inventory entry is made for these authored objects.

Current exclusions apply to the ancestor ID and the ancestor/archive/proof and
declared upstream hashes and member paths. The RN5 carrier must have one exact
active, research-context, `ARCHIVE_INDEXED` inventory row; each declared upstream
member must have one exact `TEXT_READING_COPY` payload row with empty scope holds.
Links must name the exact HTTPS Drive file ID; only an empty query or the existing
transport parameter `usp=drivesdk` is allowed, with no fragment. Missing, ambiguous,
malformed, held, or unavailable inputs refuse before archive bodies are read.
No upstream proof/code body is opened by this metadata check.

Only `MANIFEST.json` is read as archive metadata and `h3_floor/PROOF.md` as an
analytic body. All 70 manifest entries are metadata; this run does **not** claim
70 verified analytic leaves, execute archived programs, or read unrelated leaves.
The gate rechecks its metadata immediately before and after the allowed proof
read, and the CLI repeats source, context and code checks before output. Warm
cache hits undergo these checks too. Recipe v2 intentionally invalidates old
cache keys; exact scalar certificate bytes and `prove()` are unchanged.

The retained custody observation is dated 2026-09-21T23:52:30.286501+00:00. Offline
execution checks that pinned snapshot and the current local policy files; it
cannot discover an unmirrored remote hold or authenticate live Drive permissions.
The historical registration label PRIVATE CANDIDATE is not a claim about current
GitHub visibility. Admission here permits this scoped candidate reading only:
scientific acceptance, organizational independence and automatic promotion remain
false/zero. A concurrent policy change is caught at the next check, not by an
atomic cross-system transaction. The pure scalar `prove()` API reads no source
archive and remains available when source-dependent CLI admission refuses.

## Frozen evidence for public review

The full [CERTIFICATE.json](CERTIFICATE.json) and the original [31-member author delivery](../../../research/campaigns/h3_solver_pilot_20260921_original.zip) are now stored in GitHub. The [custody record](sources/H3_PUBLIC_EVIDENCE_CUSTODY_V1.json) binds thirteen fresh raw Drive comparisons, the complete bundle manifest and the unchanged canonical certificate hash. The archive preserves the original run JSONs, logs, diagnostic, manifests and delivery records.

The archived runner predates the source-admission repair and is retained as history. Use the current repository command above for execution. The original manifest flag excluding the certificate from its earlier Git publication remains frozen; Dylan's later authorization is recorded in the custody successor. This transfer adds no mathematical claim, independent-review credit or live Drive permission.
