# CBZ tools — independent engineering review

**Current disposition: AMEND for CBZ-R3. CBZ-R1 and CBZ-R2 are addressed in reviewed successor bytes.** Three Important findings were reproduced; no Critical finding was established. This is a finite engineering review of evidence validation and cleanup safety, not scientific truth, statement alignment or CBZ-0 certification. The previous source-workflows review remains frozen B and is not reopened.

Actual reviewer: `/root/source_workflows_review_1029`, separate OpenAI GPT-6-family agent, neither implementer of these tools nor author of their fixes. Same-family engineering scope only; no family-diversity or organizational-independence credit. Scientific effect NONE. No Git/source index/ref changes, original-file mutations, real remote mutations or subagents were used. Adversarial mutations were confined to disposable fixtures; reports and probe programs remain in this directory.

Scope: `tools/cbz_lean_check.py`, `tools/CBZNonvacuity.lean`, `tools/cbz_refs.py`, `tools/cbz_storage.py` and their three matching tests. Existing `formal_gate_check.py` and exact pinned LeanChecker implementation were read as consumed contracts. `cbz_audit.py` and the in-flight baseline implementation are excluded and were not reviewed. Registry reports and the owner mission were read to assess stated guarantees.

## Source identities

Original reviewed source, recorded before fixes in `REVIEWED_SOURCE.json`:

| Path | SHA-256 |
| --- | --- |
| `tools/cbz_lean_check.py` | `7b716e0507c0963510af5d276d1b900378df1cb630c5888551db3f939b333e75` |
| `tools/CBZNonvacuity.lean` | `b9f29f231bb41ff6ffc896230c5c23b358cf2527ebd33a576d3c6d96a0ded49f` |
| `tools/cbz_refs.py` | `186a090c2dd536f3e21dbaf71d9a11e48281fcb6644bbc74653543774180bc24` |
| `tools/cbz_storage.py` | `a0aee53dec348bf5bc40016b4956fae41b1a672a36942fcb9160398a399379ae` |
| `tests/test_cbz_lean_check.py` | `cabf2ba2f705b3b289482468f55de77e86a727a01f223ce47832ec8dd0ab5150` |
| `tests/test_cbz_refs.py` | `091ce8c4fb16f5dbe79dc97cab52428af9430100548237672f2ef5505e39e6a5` |
| `tests/test_cbz_storage.py` | `8fc7303a983fc626f3aa9a422451812db25e4cbe8ff2b3054b77274127afaefb` |

These matched the supplied worker registry identities. Subsequent fixes were made by the original authors after findings were reported; original findings/evidence remain preserved.

## Strengths and checked boundaries

- Ref archival uses a new standalone bare repository, exact canonical embedded manifest, exact original branch tips as archive-commit parents, object-format matching, original annotated-tag IDs, independent fresh remote archive fetches and full Git fsck. Branch deletion is one atomic push with an explicit expected-old lease for every deleted branch. Default/protected/active exclusions and exact state checks are present; failed operations do not return cleanup success. Tests include actual server-hook rejection and changed-ref races.
- Storage apply verifies a separately supplied digest, canonical plan bytes and a complete freshly re-derived plan before any mutation. Same-device/byte/mode/owner eligibility, no-follow file opening, path checks, content rechecks, temporary hardlink replacement and post-apply verification are meaningful. It does not pretend hardlinks retain independent future-write semantics.
- The Main wrapper calls the existing source-bound pinned gate, captures all selected module declarations rather than just named proofs, rejects transitive forbidden axioms and project axioms, separates theorem constants from definitions, and rejects unexpected proposition premises. Actual compiled controls exercise sorry, native computation, unused axioms, False premises and missing modules. The current four witness statements inspected are the intended finite-domain statements; no original mathematical statement was edited by this work.
- Pinned LeanChecker source was inspected: normal module arguments select a module-name prefix; `--fresh UniversalLaw` selects the one root module and replays all its imported/project constants into an empty kernel environment. The reviewed call does not accidentally select only an empty root. The named witness module has its own kernel replay. This is Lean's own kernel, not an independent external verifier.
- Independent disposable real-Lean controls additionally showed that both ordinary and `module`-header private unused custom axioms are rejected by the census (`private-axiom-probe.json`). No hidden-private-axiom gap was reproduced.

## Important findings

### CBZ-R1 — Original tag retention was not verified before claiming archive completeness

Original location: `tools/cbz_refs.py`, `verify_snapshot`, original lines 229–259. It checked original tag object existence and peeled identity but did not require the original local tag refs to remain present. `git fsck` succeeds for an unreferenced but still present object.

Independent reproduction in `probe_safety.py` / `safety-probe-report.json`: created a disposable remote with a lightweight tag pointing to a unique blob, made a snapshot, deleted only that tag from the local snapshot, and called `verify_snapshot`. It returned `verified: true`. After fixture-only reflog expiry/GC the tag-only blob disappeared. The original disposable remote tag remained intact. The successful verification therefore overstated durable standalone archive coverage. The same risk applies to annotated tag objects and tag-only histories.

**Successor disposition: ADDRESSED.** Reviewed `verify_snapshot` successor lines 232–238 now require exact local head/tag membership equal to original manifest tags plus archive tag, reject any extra heads/tags, and require detached HEAD exactly at the archive commit. Missing original tags cannot pass simply because their objects have not yet been collected.

Independently reran the original tag-only-blob case in `probe_safety_successor.py`: intact archive accepted; removal refused while the object was still present (`safety-successor-report.json`). Read author 26/26 normal and optimized passing tests, including missing annotated tag, blob/tree tag-only objects, unexpected heads/tags and incorrect HEAD. Successor identities:

- Tool `f26c6f8f13aac2b967a8c03cf4ca9643a875a93ebd0855a732576e8fdceaca85`.
- Tests `87894edca7a54974ff53af31e8d3a696643633c148d591d410709b28881916de`.

No live snapshot or remote was mutated by this review or its verification.

### CBZ-R2 — Hardlink grouping discarded distinct extended attributes

Original locations: `tools/cbz_storage.py:54–74` (`file_record`), `make_plan` grouping and `same_content`. File identity/grouping considered device, bytes, mode, uid/gid and byte hash, but did not capture or refuse xattrs. Atomic replacement substitutes the source inode's attributes for the target's.

Independent reproduction in `probe_safety.py` / `safety-probe-report.json`: two equal-byte/equal-mode/equal-owner disposable regular files, with `user.review-note` only on the target, produced one operation. Apply erased the target attribute and reported `content_paths_modes_verified: true`. ACLs/capabilities and other xattr metadata likewise cannot be presumed equivalent from mode/ownership alone. This is a destructive metadata-safety issue, not a claim that a previous real storage operation lost such metadata.

**Successor disposition: ADDRESSED for new uses.** Successor `require_no_xattrs` and `file_record` refuse nonempty descriptor xattrs and enumeration failures before/after hashing; per-operation rechecks use the same function. This is a conservative refusal instead of silently broadening metadata-copy semantics.

Independent successor probe refused the original pair before planning/mutation; target metadata remained present and the inodes stayed distinct. Read author 13/13 normal and optimized passing tests, including xattr enumeration failure and a newly added attribute before replacement. Successor identities:

- Tool `c829d392e00601d7876a2530a8bff02b7f913cff7bed2dec29df69d796b4ff9a`.
- Tests `5efc1289a0d13e266cd1143b7f6b1946efb29a24deba718eb0170093f67b1295`.

Historical apply receipts did not capture pre-apply xattrs. A later clean scan cannot prove attributes were absent before those operations. Their original byte/path/mode/owner claims retain that exact scope; no retroactive all-metadata preservation claim is made.

### CBZ-R3 — Witness names/axioms were accepted without enforcing witness propositions

Original location: `tools/cbz_lean_check.py:162–168` and `validate_census`. The wrapper checks that the four expected names are theorem constants with allowed axioms/no unsupported proposition premises, but does not check that their propositions are the finite-domain witnesses the wrapper relies on. An expected name alone is not a witness contract.

Independent reproduction in `probe_witness_contract.py` / `witness-contract-probe.json`: compiled a disposable real Lean module with the four exact expected witness names, each proving only `True`. Generated the real census using `audit_program`, ran Lean successfully, then passed its real output to the production `validate_census`. All four were accepted with statements `True`. This is an audit-stage reproduction, not an asserted full wrapper replay. The actual original witness source remains correct; the issue is the validator's acceptance of weakened replacements and its resulting nonvacuity claim.

**Disposition at this writing: NOT YET ADDRESSED / native fix in progress.** The author acknowledged the issue and is adding independent generated exact-type ascriptions, elaborated proposition checks and an actual same-name-True rejection control. A source-bound successor disposition will follow observed output; no future result is assumed.

## Receipt binding, observed execution and limits

Read actual `portable-main-receipt.json`, `portable-main-final-run.log/.exit`, the registry and relevant kernel/control logs. The preserved receipt identifies manifest `38eefbfdd84658cd0f6e18bd7d714458a7522b1021f73b1075e7972a63b9bcb0`, commit `f20cc54d4d136e854608853372249882eb4c949f`, original helper/source hashes above, 45 project constants, 31 theorem constants and four witness constants; current original witness types were inspected. Version text identifies Lean 4.34.1 commit `5045d0056413266e57c625dcd7c365b10e377c52`. The wrapper's actual command arrays were inspected: named witness kernel replay and `leanchecker -v --fresh` with the manifest root are explicit; output logs are hashed in the receipt. The receipt does not separately serialize an argv/exit ledger, so this review does not claim one exists.

A read-only observation of mutable live evidence at 17:54:10 UTC found the current CBZ receipt still referring to baseline hash `2f028a…`, while the live formal baseline receipt had hash `2cce1c…`. Details are in `live-receipt-observation.json`. The read was not atomic and other gate runs were in progress, so it is not attributed as a completed-run failure or an invalidation of historical retained evidence. It does establish that those live files cannot presently be combined as one coherent final receipt. Root confirmed the final sequence will run all suites first, then one complete wrapper invocation serially with no Main concurrency and exact final commit/helper binding.

The wrapper reuses fixed output names; source and receipt hashes must be consumed together with their actual run context. Historical observed execution does not automatically rebind to fixes, a new commit, a new binary/version or a concurrently replaced baseline receipt. The author’s original execution reports are evidence consumed, not newly rerun kernel proof by this reviewer. Final R3 full runtime evidence remains pending; no final CBZ/global regression/hosted result is inferred.

## Scope exclusions and retained assumptions

- Mathematical truth, semantic alignment to papers, Gaussian/interface/Palm applicability, Math- closure, whole-federation CBZ status and scientific acceptance are excluded. Literal finite-domain witness validation is an engineering protocol check, not a scientific alignment review.
- Ref scope is advertised heads/tags and default HEAD. Hidden refs, PR objects, stashes, dirty worktrees, protected/active branch discovery and quarantine/disposition indexing require caller-owned inventories. The tool explicitly requires the complete policy inventory; no hosting-policy inference was credited.
- Atomic leases guard each requested deletion; they are not a server-wide lock on unrelated concurrent changes. Final-state refusal after a successful atomic push can require recovery. That documented limit is not a claim that no deletion occurred on every failed post-check.
- Storage is for stable, explicitly selected immutable files. Shared-write semantics, deliberate inode/link-count changes and non-preserved per-path timestamps are documented. No concurrent-writer linearizability or adversarial directory-swap guarantee is inferred from ordinary path checks.
- No false historical metadata restoration is claimed for the xattr fix. No in-flight audit/baseline implementation was reviewed. No unchanged full suite or expensive full kernel replay was repeated solely to restate worker evidence.

This report preserves the initial findings and exact successor dispositions. Pending author output is not a passing gate.
