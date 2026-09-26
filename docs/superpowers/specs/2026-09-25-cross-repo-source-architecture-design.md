# Cross-Repository Source Architecture and Publication Design

**Status:** DESIGN SPECIFICATION — implementation not yet authorized by this document alone  
**Date:** 2026-09-25  
**Owner:** Dylan Roy  
**Design author:** OpenAI / ChatGPT  
**Scope:** all eight repositories in the `d6g8k5htny-coder` research workspace  
**Scientific effect:** NONE. This design changes software/repository architecture only; it does not promote, prove, refute, or close any mathematical claim.

---

## 1. Purpose

The research workspace has evolved from a small collection of AI-assisted repositories into a federated research system with:

- mathematical proofs and counterexamples,
- machine-readable scientific state,
- exact-source custody,
- independent and same-provider review records,
- reproducible computational evidence,
- fail-closed dependency/revalidation controls,
- cross-repository lookup,
- controlled experiments,
- selected Drive replicas,
- multiple concurrent AI contributors.

That evolution exposed a new bottleneck: source code, proof text, review objects, routing metadata, and executable control logic are increasingly useful, but they are not yet organized under a single stable source architecture.

The goal of this redesign is to make the workspace behave like a maintainable research software system **without rewriting its scientific history or breaking proof identities**.

The desired outcome is:

```text
human/model intent
        │
        ▼
canonical scientific objects
        │
        ├── immutable proof / review bodies
        ├── executable source modules
        ├── machine-readable dependencies
        └── source/provenance identities
        │
        ▼
generated navigation + query surfaces
        │
        ▼
CI / falsification / revalidation
        │
        ▼
published source snapshots
```

The architecture must make it easy for a clean researcher to answer:

1. What theorem/lemma/result is this?
2. Where is its complete proof?
3. What exact source code supports it?
4. What has been independently reviewed?
5. What remains open?
6. What breaks if a dependency changes?
7. Which repository owns each piece of state?
8. Can I reproduce this from default-branch GitHub alone?
9. Can I obtain a source archive whose identity is stable?
10. Can old paths still reproduce historical runs during migration?

---

## 2. Design principles

### P1 — One canonical owner per kind of state

No scientific or operational fact may have two independently editable canonical homes.

Examples:

- theorem/proof bodies: **Math-**
- cross-project dependency/revalidation state: **main**
- public artifact routing: **meta-framework**
- lookup implementation: **query-**
- integration experiments: **trial**
- public Drive replicas: **google-drive**
- operating contract: **governance-**
- private exploratory work: **sandbox**

Other repositories may contain generated views or mirrors, but those must be clearly derivative.

### P2 — Proof identity is more important than aesthetic path cleanup

Reviewed proof files are historical scientific objects. They do **not** move simply because a new directory would look cleaner.

A reviewed proof remains at its existing path unless there is a strong scientific reason to move it. New reusable source code may be extracted around it.

### P3 — Code moves additively before old interfaces disappear

Migration pattern:

```text
historical executable path
        │
        ▼
compatibility wrapper
        │
        ▼
canonical src/ module
```

The wrapper is deleted only after:
- all callers are migrated,
- reproducibility documentation is updated,
- no historical test depends on the old interface,
- CI verifies equivalent behavior.

### P4 — Generated views must not become second status databases

README tables, proof indexes, dashboards, navigation files, and public catalogs should be generated or validated against canonical machine state wherever feasible.

Manual duplication of status across multiple files is treated as technical debt.

### P5 — Hashes establish identity, not truth

A SHA, green test, CAS replay, SMT proof object, or successful build is evidence about a precisely defined interface. None automatically promotes a mathematical theorem.

### P6 — Fail closed across repository boundaries

Unknown IDs, missing source objects, mutable references, stale generated views, missing proof paths, ambiguous ownership, incompatible dependency versions, and unresolved controlling sources must block promotion/integration rather than be guessed.

### P7 — Public source publication is explicit

Nothing from private `sandbox`, private Drive material, credentials, Vault99, or unreviewed restricted sources is published merely because another public artifact refers to it.

### P8 — Research remains primary

Architecture is justified only when it:
- lowers duplicate work,
- catches defects,
- improves reproducibility,
- improves source retrieval,
- reduces coordination cost,
- or safely enables more mathematical progress.

The architecture must not become the research program.

---

## 3. Repository responsibility model

### 3.1 `main` — research control plane

**Owns**
- cross-project work queue,
- machine-readable claim/dependency graph,
- source-binding authority,
- reverse-impact/revalidation logic,
- work-order routing,
- scientific-state schemas,
- control-plane CI,
- project-wide navigation into current work.

**Does not own**
- canonical mathematical proof bodies,
- reusable mathematical computation implementations,
- public Drive replicas,
- private experiments.

**New canonical source package**

```text
main/
  src/
    universal_law_control/
      __init__.py
      claims/
      dependencies/
      source_bindings/
      revalidation/
      routing/
      schemas/
      reports/
```

Existing `tools/*.py` entry points should become thin CLI wrappers over this package over time.

**Key rule:** `main/src` may know *where* a theorem is and *what depends on it*; it must not silently reimplement the theorem's mathematics.

---

### 3.2 `Math-` — mathematical source of truth

**Owns**
- theorem and lemma statements,
- proof bodies,
- counterexamples,
- mathematical reviews,
- proof-specific executable checks,
- coefficient calculations,
- imported immutable proof mirrors,
- proof availability index,
- canonical published mathematical claim manifest.

**Existing reviewed directories remain stable**

```text
Math-/
  frontiers/
  reviews/
  imports/
  coefficients/
  claims/
```

These are scientific-history paths. They are not wholesale moved into `src`.

**New reusable mathematical library**

```text
Math-/
  src/
    universal_law_math/
      __init__.py
      gaussian/
        covariance.py
        regression.py
        jets.py
      kac_rice/
        weights.py
        densities.py
        counting.py
      topology/
        persistence_interfaces.py
      divided_differences/
      exact/
        rational.py
        polynomial.py
      geometry/
      certificates/
```

Only general, reused code belongs here.

Proof-specific scripts remain next to the proof until they have at least two genuine consumers or a clear reusable contract.

**Rule:** a theorem may depend on `src/universal_law_math`, but `src/universal_law_math` must not depend on a theorem-specific proof directory.

---

### 3.3 `query-` — public lookup client and source-verification library

This repository should become the cleanest outward-facing software package.

**Target layout**

```text
query-/
  pyproject.toml
  src/
    universal_law_query/
      __init__.py
      catalog.py
      lookup.py
      verify.py
      models.py
      cli.py
  tests/
  research_query.py           # compatibility wrapper
  catalog_entry_helper.py     # compatibility wrapper
  verify_portable_stubs.py    # compatibility wrapper
```

The wrappers import the package and preserve existing commands during migration.

**Publication target**
- installable directly from GitHub source,
- source archive attached to tagged releases,
- no PyPI publication in this phase,
- no network access required for local catalog verification.

---

### 3.4 `meta-framework` — federation catalog and source identities

This repository remains lightweight.

**Owns**
- repository-role registry,
- public artifact catalog,
- schemas for catalog entries,
- cross-repository source identity records,
- optional generated source map.

**Does not own**
- scientific status,
- theorem proofs,
- runtime orchestration,
- duplicated dependency graphs.

Recommended target:

```text
meta-framework/
  registry.json
  schemas/
    registry.schema.json
    source_ref.schema.json
  generated/
    SOURCE_MAP.json
```

`SOURCE_MAP.json` is derivative and reproducible from canonical repository manifests.

---

### 3.5 `trial` — federation and integration laboratory

**Owns**
- cross-repo integration tests,
- compatibility tests,
- migration rehearsals,
- mutation/fuzz experiments,
- release-install smoke tests.

**Does not own**
- production package implementations,
- claim status,
- canonical proofs.

A successful experiment may graduate into the owning repository. The experiment remains preserved as historical evidence.

---

### 3.6 `google-drive` — selected public replica repository

**Owns**
- explicitly approved public replicas,
- exact source identity metadata,
- validation of replica bytes.

No broad Drive synchronization.

No scientific-status authority.

No application/runtime package beyond tiny source-verification helpers if strictly necessary.

---

### 3.7 `governance-` — concise working contract

**Owns**
- repository role contract,
- review topology,
- stable operating principles,
- human-readable policy.

It should intentionally stay small.

Executable enforcement belongs in `main/src/universal_law_control`.

---

### 3.8 `sandbox` — private experimentation

Private by default.

No automatic export.

Any graduation from sandbox requires an explicit source review and a fresh public artifact with provenance that does not reveal private paths or secrets.

---

## 4. Canonical scientific-object model

The workspace already has several useful metadata objects. The redesign should **not** create another competing status database.

The canonical relationship should be:

### 4.1 Math scientific publication manifest

`Math-/claims/LANDING_CLAIMS.json`

Owns published mathematical object metadata:
- stable claim ID,
- statement/proof path,
- exact source identity,
- scope/domain,
- review object,
- disposition at exact scope,
- explicitly required scientific dependencies.

### 4.2 Main dependency/revalidation graph

`main/claims/graph.json`

Owns:
- dependency edges,
- controlling/noncontrolling role,
- operational classification,
- impact propagation,
- HOLD / REVALIDATION proposals,
- source monitoring.

### 4.3 Meta-framework routing catalog

`meta-framework/registry.json`

Owns:
- public lookup keys,
- repository/path/commit/hash,
- artifact type,
- public scope,
- retrieval metadata.

It does **not** own whether a theorem is accepted.

### 4.4 Generated views

Examples:
- Math `PROOF_INDEX.md`
- repository README result tables,
- main research index,
- source maps,
- dashboards.

These should be generated or validated against the canonical objects.

---

## 5. Stable source-reference contract

Every cross-repository dependency should eventually use an explicit structured source reference.

Recommended conceptual schema:

```json
{
  "repository": "d6g8k5htny-coder/Math-",
  "commit": "<40-hex immutable commit>",
  "path": "frontiers/.../PROOF.md",
  "git_blob_sha1": "<blob>",
  "sha256": "<content sha256>",
  "size_bytes": 12345,
  "role": "proof",
  "visibility": "public"
}
```

For non-Git objects:

```json
{
  "provider": "google-drive",
  "source_id": "...",
  "sha256": "...",
  "size_bytes": 12345,
  "monitorability": "external-frozen",
  "role": "historical-source"
}
```

A controlling dependency whose source cannot be monitored must be explicitly represented as unresolved/HOLD rather than hidden inside prose.

### URI convention

For human/debug use, generated tools may display:

```text
gh://d6g8k5htny-coder/Math-@<commit>/frontiers/.../PROOF.md#sha256=<...>
```

This is a display convention, not a replacement for the structured fields.

---

## 6. Package architecture

### 6.1 Distribution names

Initial distributions should be independent to avoid PEP 420 namespace complexity:

- `universal-law-control`
- `universal-law-math`
- `universal-law-query`

Python imports:

- `universal_law_control`
- `universal_law_math`
- `universal_law_query`

This is simpler and safer than splitting one `universal_law` namespace across multiple repositories.

An umbrella package can be considered later only if real users need it.

### 6.2 Dependency direction

**Production runtime rule: the three public packages do not import one another across repositories.**

```text
universal_law_math       standalone runtime package
universal_law_control    standalone runtime package
universal_law_query      standalone runtime package
trial                    may install/test all three together
```

Cross-repository communication uses versioned **data contracts** rather than Python imports:

- JSON / JSON Schema,
- stable source-reference records,
- claim/dependency manifests,
- release/snapshot manifests.

The schema authority is held in `meta-framework`; each consuming repository validates the relevant schema version locally. Generated language bindings are permitted only if they are reproducible from the exact schema and carry the schema identity.

This prevents an apparently small package change in one repository from silently changing another repository's runtime semantics.

If a future cross-package runtime dependency becomes genuinely useful, it requires a separate architecture decision with cycle analysis and a version-compatibility contract.

### 6.3 No circular repository imports

CI should reject:
- Math importing main/control internals,
- meta-framework importing production implementations,
- governance importing runtime code,
- trial becoming a production dependency.

---

## 7. Proof-to-code binding

A proof may rely on reusable code, but that dependency must be explicit.

Recommended proof metadata block or companion JSON:

```json
{
  "proof": "frontiers/example/PROOF.md",
  "code_dependencies": [
    {
      "distribution": "universal-law-math",
      "repository": "d6g8k5htny-coder/Math-",
      "commit": "...",
      "module": "universal_law_math.gaussian.regression",
      "verification_role": "supporting-computation"
    }
  ]
}
```

The role must say whether code:
- proves an exact finite identity,
- reproduces arithmetic,
- numerically explores,
- produces an enclosure,
- or merely illustrates.

The existence of code never silently upgrades proof status.

---

## 8. Compatibility-wrapper policy

During migration, historical entry points remain callable.

Example:

```python
# tools/claims_gate_adapter.py
from universal_law_control.cli.claims_gate import main

if __name__ == "__main__":
    raise SystemExit(main())
```

CI must enforce:
- wrapper contains no business logic beyond argument compatibility,
- wrapper and package command produce equivalent outputs,
- old commands remain documented during deprecation.

Each wrapper receives:
- `introduced_wrapper_at`,
- `canonical_module`,
- `deprecation_not_before`.

No wrapper removal is based solely on elapsed time.

---

## 9. Published source model

### Phase 1 publication

Publish source through the existing public GitHub repositories.

For package-bearing repositories:
- `pyproject.toml`
- `src/`
- explicit license declaration only where already legally established,
- `SOURCE_MANIFEST.json`,
- deterministic source archive attached to tagged GitHub release,
- SHA256 printed in release notes.

### No external package registry yet

Do **not** publish to PyPI/npm in the first migration.

Reasons:
- avoid premature public API promises,
- avoid name squatting/version obligations,
- keep research releases source-bound to Git,
- allow package contracts to stabilize.

A later explicit decision can add PyPI.

### Release classes

Recommended tags:

```text
control-src-v0.x.y
math-src-v0.x.y
query-src-v0.x.y
workspace-snapshot-YYYYMMDD-N
```

A source tag is not a theorem-status tag.

---

## 10. Source manifests

Each package-bearing repository gets a generated `SOURCE_MANIFEST.json` containing:

- distribution/version,
- repository,
- exact commit,
- files included in source distribution,
- SHA256 and byte size,
- package entry points,
- generated-at commit identity,
- excluded private/generated directories.

The manifest should be generated by code and checked in only if reproducibility benefits outweigh drift risk.

Preferred approach:
- generator script canonical,
- CI regenerates and compares,
- release artifact contains manifest,
- repository may keep latest manifest if it helps source custody.

---

## 11. CI architecture

### 11.1 Per-repository local CI

Every package-bearing repository runs:

1. unit tests,
2. package import/install test,
3. compatibility-wrapper parity,
4. source-manifest check,
5. forbidden dependency-direction check,
6. generated-view freshness where applicable,
7. exact proof/source availability checks where applicable.

### 11.2 Main/control CI

Additionally:
- dependency graph schema,
- source-binding monitorability,
- base→head impact comparison,
- fail-closed unknown source tests,
- reverse-closure mutation tests,
- no mathematical promotion from CI.

### 11.3 Math CI

Additionally:
- every REVIEWED_SCOPED object has an in-default-branch proof,
- every proof-review pointer exists,
- open object with no proof says `NO COMPLETE PROOF YET`,
- proof-specific code pins are resolvable,
- generated `PROOF_INDEX.md` matches canonical manifest,
- counterexamples remain reachable,
- superseded proofs remain reachable.

### 11.4 Query CI

Additionally:
- install from clean checkout,
- catalog schema validation,
- exact local verification,
- unknown key fails closed,
- public/private boundary tests.

### 11.5 Trial CI

Cross-repository matrix:
- install control/math/query from pinned commits,
- run public smoke test,
- verify catalog→artifact→proof navigation,
- migration compatibility tests,
- synthetic source mutation,
- deliberately missing proof,
- stale generated view,
- incompatible package version.

---

## 12. Generated navigation

A central source of recurring maintenance cost is handwritten navigation drift.

The redesign should introduce generated or validator-backed pages.

### Generated from Math manifest
- proof index,
- reviewed result table,
- open result table,
- proof/review availability report.

### Generated from main graph
- unresolved dependency report,
- reverse-impact report,
- work queue summary.

### Generated from meta-framework registry
- repository role map,
- public source lookup index.

Generated files must include:

```text
<!-- GENERATED FROM <source> AT <commit>; DO NOT EDIT STATUS HERE -->
```

Where generation is not yet implemented, validators compare handwritten views to canonical state.

---

## 13. Review and provenance classes

Every review object should identify:

- reviewed source commit/blob,
- author provider/model where known,
- reviewer provider/model where known,
- same-provider vs cross-provider,
- organizational independence credit if any,
- interfaces reviewed,
- exact disposition,
- explicit exclusions.

Do not collapse:
- same-author replay,
- same-provider technical review,
- cross-provider technical review,
- organizationally independent human review,
- formal-kernel verification.

These are distinct evidence types.

---

## 14. Verification ladder integration

Retain the existing ladder but represent it as evidence metadata rather than theorem status.

Suggested values:

```text
L0 prose/LLM argument
L1 executable reproduction
L2 exact arithmetic / symbolic CAS
L3 adversarial mutation / property tests
L4 SMT / decision-procedure certificate
L5 proof-assistant kernel checked
```

A theorem can be reviewed with L2 support and still not be L5.

A theorem can be L5-formalized at an algebraic sublemma while the surrounding analytic theorem remains prose-reviewed.

---

## 15. Proof availability invariant

This becomes a workspace-wide hard invariant:

```text
reviewed_or_closed(object)
    =>
proof_available_on_default_branch(object)
AND review_available_on_default_branch(object)
AND source_identity_resolves(object)
```

For open objects:

```text
candidate_complete(object)
    => proof_path exists
```

Otherwise:

```text
NO COMPLETE PROOF YET
+ strongest partial derivation
+ exact missing interface
```

The invariant should eventually be checked across all public research repositories, not just Math.

---

## 16. Source-custody invariant

A proof referenced as load-bearing may not exist only:
- in an issue comment,
- in model memory,
- on an abandoned branch,
- in an inaccessible Drive object,
- or in an unspecified ZIP.

If exact source exists but is not on the owning default branch, migrate exact bytes additively and preserve provenance.

If source is absent, record `BLOCKED_ABSENT`.

Never reconstruct a missing proof and label it original.

---

## 17. Cross-repository release manifest

A workspace snapshot may contain:

```json
{
  "snapshot": "workspace-snapshot-20260925-1",
  "repositories": [
    {
      "repository": "d6g8k5htny-coder/main",
      "commit": "..."
    },
    {
      "repository": "d6g8k5htny-coder/Math-",
      "commit": "..."
    }
  ]
}
```

This makes a federation replayable without pretending all repositories share one commit history.

---

## 18. Migration phases

### Phase 0 — inventory and freeze contracts

No code movement.

Deliver:
- current executable path inventory,
- import graph,
- proof-specific vs reusable classification,
- package candidates,
- legacy entry points,
- source-custody gaps.

### Phase 1 — package skeletons

Add:
- `pyproject.toml`,
- `src/`,
- minimal package imports,
- package-only smoke tests.

No logic move yet.

### Phase 2 — leaf-module extraction

Move lowest-risk reusable utilities first:
- exact arithmetic,
- schema models,
- source-reference helpers,
- read-only query models.

Add wrappers at old paths.

### Phase 3 — control-plane extraction

Move:
- claim graph parsing,
- source bindings,
- impact computation,
- revalidation reporting.

The already-merged #90 gate becomes a high-value migration target because its semantics are well tested.

### Phase 4 — math utility extraction

Only utilities with multiple consumers:
- exact polynomial tools,
- Gaussian regression helpers,
- covariance utilities,
- reusable Kac–Rice helpers.

No wholesale migration of proof-local scripts.

### Phase 5 — query package publication

Make `universal_law_query` installable and publish its first GitHub source release.

### Phase 6 — source/proof generated views

Generate proof/source indexes and source maps from canonical manifests.

### Phase 7 — workspace source snapshot

Produce first cross-repo snapshot with exact commits and source archives.

### Phase 8 — deprecation review

Evaluate wrappers individually.

No bulk deletion.

---

## 19. Migration safety gates

Every migration PR must state:

- canonical old path,
- canonical new module,
- historical consumers,
- compatibility behavior,
- exact tests,
- scientific effect,
- source identities,
- rollback procedure.

Required negative tests:
- old wrapper deleted prematurely,
- canonical module missing,
- duplicated logic changes only one copy,
- source manifest stale,
- proof references old module incorrectly,
- generated view stale,
- private path leaks into public manifest.

---

## 20. Rollback

Every source migration must be reversible by reverting one PR without losing scientific artifacts.

Proof files are not rewritten during source extraction.

Wrappers make rollback inexpensive.

No migration PR may simultaneously:
- move proof bodies,
- change theorem status,
- change mathematical logic,
- and change package architecture.

Those changes must be separable.

---

## 21. API stability

Before v1.0 all Python package APIs are research APIs.

Public functions must document:
- input domain,
- output semantics,
- whether deterministic,
- whether exact vs numerical,
- whether scientifically controlling.

Internal helpers remain under private modules.

No compatibility promise beyond documented wrappers before v1.0.

---

## 22. Naming and versioning

Distribution versions use SemVer for software behavior.

Mathematical object versions continue to use their scientific IDs and source hashes.

Do not derive theorem version numbers from Python package versions.

Example:

```text
universal-law-math 0.3.0
Theorem object D5-FIXED-ANNULUS-... unchanged
```

---

## 23. Dependency locking

For reproducibility:
- Python minimum initially 3.11,
- test 3.11 and current project runtime where feasible,
- dev/test dependencies locked,
- core libraries should prefer the standard library unless a dependency materially improves correctness,
- solver/proof-assistant dependencies explicitly versioned.

Math proofs that rely on numerical third-party code must state the version.

---

## 24. Security and privacy

Never publish:
- tokens,
- credentials,
- device login codes,
- private sandbox source,
- private Drive objects without visibility verification,
- secret repository paths that reveal restricted data.

Public source manifests include only public artifacts.

A public source release is a separate publication action from making a proof review available.

---

## 25. Agent collaboration under the new layout

Agents should be able to determine ownership without chat memory.

Each repository `AGENTS.md` should eventually contain only:
- role,
- canonical paths,
- write boundaries,
- verification command,
- how to claim a task,
- where to report review.

Long historical operational instructions should move to clearly historical docs.

### Work claims

Keep GitHub issue/PR comments as the collaboration lease mechanism for now.

Do not introduce a second live lease database unless measurable evidence shows the comment system is inadequate.

---

## 26. Why not a monorepo?

A monorepo would simplify imports but would harm this project today:

- mathematical source history would be harder to separate from orchestration,
- private/public boundaries become easier to violate,
- repository-specific review provenance becomes blurred,
- large history/custody imports increase checkout cost,
- independent package roles are already useful.

The current federation is a feature; the problem is unclear source contracts, not the number of repositories.

---

## 27. Why not move all mathematics into src/?

Because proof bodies are not ordinary implementation source files.

`src/` is for reusable executable code.

Reviewed theorem text belongs in stable scientific paths where review pointers and hashes remain meaningful.

The architecture intentionally separates:

```text
proof source
from
software source
```

while binding them explicitly.

---

## 28. Success criteria

The architecture is successful when a clean agent can:

1. clone the public repositories,
2. install the three source packages,
3. locate any published theorem via a stable ID,
4. read its complete proof and review,
5. resolve every controlling source dependency,
6. execute the supporting calculation,
7. see exact open gaps without consulting chat history,
8. mutate a dependency and observe the correct revalidation blast radius,
9. generate the same proof/source index,
10. produce a content-addressed workspace source snapshot.

And when doing so requires **less** agent-facing procedural reading than today.

---

## 29. Quantitative architecture metrics

Track before/after:

- time to locate complete proof,
- time to identify next valid task,
- fraction of reviewed results with default-branch proof custody,
- duplicate implementation count,
- stale-navigation incidents,
- source-binding failures,
- reverse-impact false negatives,
- wrapper parity failures,
- number of files an agent must read before contributing,
- CI runtime.

Architecture changes that increase complexity without improving at least one primary metric should be reconsidered.

---

## 30. Explicit non-goals

This redesign does not:

- claim mathematical novelty,
- reopen reviewed theorems,
- close open D5 geometry,
- publish private sandbox material,
- publish packages to PyPI,
- force a monorepo,
- rewrite proof histories,
- replace independent review,
- make CI a theorem prover,
- make hashes truth certificates,
- require every proof script to become a library module,
- require every model to use the same provider.

---

## 31. Improved target state

The final workspace should look conceptually like:

```text
                         d6g8k5htny-coder
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
           main                 Math-            meta-framework
      control plane       scientific source       routing catalog
             │                    │                    │
             │              src/math library            │
             │                    │                    │
             └──────────────┬─────┴──────────────┬─────┘
                            │                    │
                         query-                trial
                     public lookup         federation tests
                            │
                            ▼
                      GitHub Releases
                    source snapshots
                            │
              ┌─────────────┴──────────────┐
              │                            │
        google-drive                  governance-
      approved replicas              human contract

                   sandbox stays private
```

The scientific truth chain is:

```text
Math proof/review
      │
      ▼
Math claim manifest
      │
      ▼
main dependency graph
      │
      ▼
fail-closed impact/revalidation
      │
      ▼
generated navigation + public catalog
```

No reverse arrow is allowed to silently promote a theorem.

---

## 32. First implementation slice after approval

The first implementation plan should deliberately avoid touching mathematics.

Recommended first slice:

1. add package skeletons to `main`, `Math-`, and `query-`;
2. publish no external package;
3. move one low-risk read-only module from `query-` into `src`;
4. add compatibility wrapper and parity test;
5. add repository contract checker;
6. add source manifests;
7. have `trial` install and exercise the packages from exact commits;
8. measure complexity/runtime impact;
9. only then migrate the first control-plane module.

This establishes the pattern before touching the claims gate or mathematical utilities.

---

## 33. Decision

**Recommended architecture:** federated canonical-source migration with compatibility wrappers, immutable proof paths, separate package distributions, generated/validated views, explicit source-reference contracts, and GitHub source releases.

This supersedes the simpler “add src directories” concept by making the source architecture enforceable and migration-safe without collapsing repository roles or rewriting scientific history.


---

## 34. Authority matrix — strengthened self-review decision

The phrase “one canonical owner per kind of state” must be enforceable, not aspirational.

| Object | Canonical authority | Derived consumers |
|---|---|---|
| Mathematical statement/proof/review/disposition at exact scope | `Math-` claim manifest + proof/review bytes | main graph, meta catalog, READMEs |
| Operational dependency state: OPEN/HOLD/REVALIDATION_REQUIRED, reverse impact | `main` claims graph | dashboards, work queues |
| Public artifact lookup identity | `meta-framework` registry | query client |
| Repository role | `meta-framework` registry + governance human contract | generated repo docs |
| Public replica custody | `google-drive` SOURCE metadata | registry/query |
| Experiment outcome | owning experiment repo (`trial` or `sandbox`) | promoted successor only after explicit graduation |

A main-graph classification must never silently override a Math scientific disposition. Conversely, a Math review does not automatically clear operational HOLDs in main. Reconciliation requires an explicit dependency transition.

During legacy migration, fields that duplicate another authority are labeled `legacy_mirror` or `derived` until removed.

---

## 35. Active-branch convergence precondition

The `main` repository currently has a distinction between:
- default branch `main`, and
- the active integrated research/hardening line `chatgpt/drive-github-hardening-20260919` (or its successor).

**No control-plane `src/` migration is published from two independent bases.**

Before Phase 1 for `main`:

1. identify the exact active integrated base;
2. compute default-main ↔ active-base divergence;
3. designate one migration base in a source-bound decision record;
4. land source architecture on that base;
5. before the first public control-package release, converge the intended public/default branch or explicitly designate the release branch and document why.

A green package built from stale default-main must never be presented as the current control plane.

`Math-` and `query-` use their current default branches unless a similar divergence is documented.

---

## 36. Data-contract architecture

Because production packages are runtime-independent, shared meaning travels through schemas.

Required schema families:

```text
meta-framework/schemas/
  source_ref.schema.json
  mathematical_object.schema.json
  dependency_edge.schema.json
  review_ref.schema.json
  repository_role.schema.json
  source_manifest.schema.json
  workspace_snapshot.schema.json
```

Rules:

- schema versions are explicit;
- unknown major schema versions fail closed;
- producers declare the schema version;
- consumers preserve unknown noncritical extension fields but reject unknown controlling semantics;
- migrations are pure transformations with fixtures;
- schema changes receive mutation tests;
- schemas describe data shape, not theorem truth.

This gives the federation a stable protocol without a shared runtime library.

---

## 37. Artifact-class separation

Every file belongs conceptually to one of these classes:

1. **scientific source** — proof, theorem, counterexample, review;
2. **software source** — reusable executable implementation;
3. **proof-local executable** — script/check coupled to one scientific object;
4. **evidence** — logs, solver transcripts, enclosures, receipts;
5. **generated navigation** — indexes, dashboards, source maps;
6. **historical provenance** — frozen/superseded/negative records;
7. **private experiment** — sandbox-only.

The migration must not blur these classes.

In particular:
- evidence does not move into `src/`;
- proof bodies do not move into `src/`;
- generated navigation is never imported as runtime source;
- historical artifacts are never “cleaned up” by rewriting them into current source.

---

## 38. Deterministic source publication

Git tags alone identify commits, but reproducible release **bytes** require a deterministic builder.

Each package release should produce a custom archive with:

- lexicographically sorted paths,
- normalized path separators,
- normalized file mode policy,
- owner/group numeric IDs set to 0 where the archive format permits,
- fixed mtime equal to `SOURCE_DATE_EPOCH` derived from the release commit,
- no VCS metadata,
- no caches/build outputs,
- embedded `SOURCE_MANIFEST.json`,
- embedded `BUILD_INFO.json`,
- SHA256 of the final archive.

The release record stores:
- repository,
- exact commit,
- archive SHA256,
- manifest SHA256,
- build command,
- builder version.

GitHub's automatically generated source ZIP/tar may remain available, but the **project deterministic archive** is the reproducibility object.

---

## 39. Build and package decision

Initial Python packaging choice:

- build backend: `setuptools.build_meta`;
- Python: >=3.11;
- runtime dependencies: **zero by default** for the first package skeletons;
- optional solver/scientific dependencies remain extras or proof-local until justified;
- package versions begin below 1.0;
- no implicit version from Git tags unless the version derivation itself is tested and source-bound.

Reason: maximize portability and minimize new supply-chain surface while the APIs stabilize.

This is a design default, not a prohibition on future justified dependencies.

---

## 40. Public API budget

The first releases intentionally expose very small APIs.

### control
Initially public:
- parse/validate source reference,
- parse/validate dependency graph,
- compute reverse impact,
- render a non-promoting impact report.

### math
Initially public:
- exact arithmetic/polynomial primitives only after two real consumers exist.

Gaussian/Kac-Rice utilities remain internal until their contracts survive reuse across multiple proofs.

### query
Initially public:
- list keys,
- lookup exact object,
- verify local object bytes,
- render source reference.

Everything else is private/internal.

This prevents “publishing src” from accidentally freezing the entire research implementation as a public API.

---

## 41. Compatibility matrix and migration ledger

Every migrated executable receives one row in a machine-readable compatibility ledger:

```json
{
  "legacy_path": "tools/example.py",
  "canonical_module": "universal_law_control.example",
  "introduced_at": "<commit>",
  "wrapper_status": "ACTIVE",
  "behavioral_fixture": "tests/compat/example.json",
  "removal_blockers": [
    "historical replay references legacy path"
  ]
}
```

The compatibility test compares:
- exit status,
- stdout/stderr contract where stable,
- generated file identities where relevant,
- failure behavior on invalid inputs.

Behavioral parity is stronger than “both commands run.”

---

## 42. Supply-chain hardening

Package/publication workflows must:

- pin GitHub Actions by immutable commit SHA;
- use minimal workflow permissions;
- avoid `pull_request_target` for untrusted code execution;
- never expose credentials to fork/untrusted PR code;
- build releases from a verified exact commit;
- verify the working tree is clean;
- verify generated manifest freshness before release;
- fail if private/sandbox paths occur in the release manifest;
- produce a release provenance record.

A later phase may add SLSA-style provenance/signatures if useful; it is not required to start the migration.

---

## 43. Architecture conformance checker

Before moving business logic, create a small checker that answers:

- Does this repository contain forbidden cross-repo runtime imports?
- Are canonical source directories where the role contract says they are?
- Do compatibility wrappers point at resolvable canonical modules?
- Are reviewed proof paths reachable on the owning default/release branch?
- Do generated files declare their authority source?
- Does any public manifest mention private `sandbox` paths?
- Are source references immutable and well formed?

This checker has **no scientific promotion authority**.

Its purpose is structural integrity.

---

## 44. Migration stop conditions

Pause source migration if any of these occur:

- a proof/review path would change identity unexpectedly;
- compatibility tests reveal semantic drift;
- active research agents are editing the same implementation being extracted;
- release packaging exposes a private path/object;
- CI runtime materially worsens without a compensating reliability gain;
- package separation forces circular dependencies;
- the migration requires changing a theorem to fit the software architecture.

When paused, research continues on the existing interfaces.

---

## 45. Improved implementation ordering

The self-review changes the first implementation slice slightly:

**Slice A — contracts only**
1. repository/source inventory;
2. schema files in meta-framework;
3. architecture conformance checker in trial;
4. no production logic movement.

**Slice B — query first**
1. add `query-/pyproject.toml`;
2. create `src/universal_law_query`;
3. migrate read-only lookup code;
4. legacy wrappers;
5. clean-install + parity tests;
6. deterministic source archive dry run.

**Slice C — control skeleton**
1. create `main/src/universal_law_control` on the designated active base;
2. migrate source-reference parsing first;
3. then graph parsing;
4. only later migrate impact/revalidation logic after parity fixtures reproduce #90 behavior.

**Slice D — Math**
1. add package skeleton;
2. migrate exact generic utilities only;
3. do not touch proof-local computation until reuse justifies it.

This ordering reduces risk because `query-` is read-only and already has a clear executable contract.

---

## 46. Spec self-review checklist

Before implementation planning, this design has been checked for:

- no monorepo assumption;
- no proof-body relocation requirement;
- no runtime cross-repo import cycle;
- no duplicate scientific-status authority;
- explicit public/private boundary;
- explicit current-main branch divergence handling;
- deterministic source-release identity;
- rollback path;
- compatibility-wrapper strategy;
- generated-view authority;
- no package-registry publication;
- no theorem promotion from architecture;
- bounded first migration slice.

Remaining decisions intentionally deferred to implementation planning:
- exact filenames for package-internal modules after inventory;
- exact first query function extraction;
- release version numbers;
- eventual wrapper retirement dates;
- whether signatures/SBOM become worthwhile after the first source release.


---

## 47. External standards check

This design was compared against current primary guidance before implementation planning.

### Python packaging

The Python Packaging User Guide documents the `src/` layout as a standard separation between importable packages and repository-root tooling/configuration. Setuptools supports `pyproject.toml` and package discovery from `src`.

Primary references:
- https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
- https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html
- https://packaging.python.org/en/latest/specifications/source-distribution-format/

Design consequence: retain `src/` for importable production modules, while proof bodies, repository tooling, evidence and generated navigation stay outside the import package unless they are true runtime resources.

### Reproducible release timestamps

The reproducible-builds project defines `SOURCE_DATE_EPOCH` as the standard environment variable for reproducible build timestamps and documents deriving it from Git history.

Primary reference:
- https://reproducible-builds.org/docs/source-date-epoch/

Design consequence: deterministic source archives use `SOURCE_DATE_EPOCH` from the exact release commit rather than wall-clock build time.

### GitHub Actions security

GitHub's secure-use guidance recommends pinning actions to full-length commit SHAs for immutable action identity. GitHub also warns against executing untrusted pull-request code with privileged `pull_request_target` workflows.

Primary references:
- https://docs.github.com/en/actions/reference/security/secure-use
- https://docs.github.com/en/actions/reference/security/securely-using-pull_request_target

Design consequence: source/package publication workflows use SHA-pinned actions, minimal permissions and unprivileged pull-request testing. Privileged release jobs operate only on trusted exact commits.

These standards checks support the design choices; they do not replace repository-specific tests or review.
