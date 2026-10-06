# Closure evidence: blind reconstruction, adversarial attack and derived profiles

On 6 October 2026 Dylan forwarded a proposal for a "theorem-closure
superstructure" and pre-approved the team's decision about it. The
[decision record](https://github.com/d6g8k5htny-coder/main/issues/275) maps all
thirteen items onto existing machinery. This protocol adds only the parts that
record did not find here already. It extends the
[current workflow](OP-WORKFLOW-20260930.md). **It starts no review, executes no
register transition and changes no scientific status.**

## Evidence profiles are derived, never stored

Show how far a result has been checked as a profile computed from records the
project already keeps:

| Axis | Read from |
|---|---|
| Source-bound argument and nonauthor review | The result's review records and its `PROVED_REVIEWED` graph node |
| Dependency closure | The downstream hard gate: every required dependency satisfied |
| Provider-distinct review | Recorded author and reviewer providers |
| Blind reconstruction | A review record with that basis (below) |
| Adversarial attack | A review record with that basis (below) |
| Formal evidence | Kernel-checked gate receipt and separate accepted alignment review |
| Numerical reproduction | A second implementation recorded in the packet's review record |
| Novelty | The existing discovery/priority ledger row |
| Human reading | Dylan's personal reading, or an independent human review |

An axis without a record reads `not recorded`, never a pass. A profile is a
view: it promotes nothing, is not committed as a status file and adds no status
word. When an axis later fails, report a finding; the existing loss-only
transition rules decide what is held or revalidated. Proof bytes and earlier
verdicts are never edited to match a profile.

## Blind reconstruction

The packet contains the statement, hypotheses, definitions, notation and an
allowed-literature list. It contains no proof steps, intermediate lemma names or
review text. Anyone may prepare it; record its exact identity.

The reconstructor must not have read the proof or its reviews. Shared accounts
and repository access make this hard to guarantee, so state exactly what was
available. The credible form is a fresh session with no access to the project's
repositories or files. Record that exposure in the review form. The
reconstructor publishes a complete derivation, or the precise point where it
fails, **before** seeing the original.

A third participant then compares the two dependency graphs. Record whether the
mechanisms converge, whether the routes differ, and any gap. A reconstruction
that reuses an original lemma must justify it independently. A blind
reconstruction earns no organizational-independence credit by itself; provider
and account rules still apply.

## Adversarial attack

Name the exact target (repository, commit, path and blob) and assume it is
false. Look for the weakest counterexample or failure. Check at least: missing
hypotheses, non-uniform error terms, illegal limit interchange, boundary
contributions, degeneracies, multiple witnesses, candidate/elder confusion,
conditioning errors, hidden independence assumptions, exceptional
configurations, scaling regimes where the asymptotic changes, and
dimension-dependent failure. End the record with:

```text
ATTACK STATUS: counterexample found | counterexample not found
Target: <repository@commit:path, blob>
Attempts: <surface -> method -> result>
Remaining attack surfaces: <list>
```

"Not found" is not acceptance. A counterexample is a finding for the owner; it
is not a register edit. The attacker should not be the target's author, and the
record states the attacker's prior exposure.

## Numerical reproduction for headline constants

For headline constants, such as the SIDE24 coefficients, `C_U(1)` and certified
tail constants, a second implementation uses a different method and is written
by a nonauthor lane. Record input identities, code commit, runtime, precision
and the resulting enclosure in that packet's review record. Disagreement beyond
the stated enclosures is a finding. Running the same program twice is not a
second implementation.

## Novelty and the human referee

Novelty audits amend the existing
[discovery/priority ledger](../docs/DISCOVERY_PRIORITY_LEDGER_20260925.md).
Use two independent searches, each told to find an earlier proof rather than
support. Its wording rule is unchanged: "to our knowledge", never "first".

A referee packet runs to 20 pages at most: the theorem, definitions, the critical
lemmas, a dependency diagram, an assumption ledger, the open interfaces, the
certificates and any blind reconstruction. It asks one question: *identify the
earliest unsupported inference*. Choosing and contacting a human referee is
Dylan's decision. Delegated AI review never fills a human-review requirement.

## Not adopted

No `CLOSED` status, passport file, or new `closure/` or `reproduction/`
directory. These would duplicate the scientific-status register, which
[AGENTS.md](../AGENTS.md) and the current workflow forbid. Results stay in the
existing review, certificate, coefficient, formal and graph locations.
