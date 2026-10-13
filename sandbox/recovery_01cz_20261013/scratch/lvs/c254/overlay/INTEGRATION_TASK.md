# C254-I1 — preserve consumer source pins while admitting the torus prerequisite

Status: **OPEN TASK / ROUTE DECISION REQUIRED / NO PICKUP CLAIMED HERE**.
This is a bounded observed integration dependency, not a failed Lean theorem or
a request to weaken a guard. C254 proof sources remain NOT_COMPILED.

Actual author: OpenAI/Codex root01a0bbb5, delegated for Dylan Roy, source-exposed,
organizational-independence credit 0, scientific effect NONE.

## Exact observation

At Math- main `2bbce4b65a4bccece8f234eeac7efba0f4bcd72b`, three existing companion
drivers require the entire `formal/` tree to equal
`38b6d1a7746075b560349d184a461933745c11ce` and its manifest SHA-256 to equal
`40cb6ffc6776717a26d2a0a66815aeef508b0c8411588fff06ff2cec43a752a1`:

| Current source | Git blob | Relevant lines |
|---|---|---|
| companions/d2_square_schur_20261006/check.py | 4e02c2d939d1f5b19a72c562df6174c2cb4c4dbc | 21–23, 59–61, 120–121 |
| companions/d2_moment_bridge_20261006/check.py | 882130d0a5428799c654b60c3ba01903df970d7d | 16–18, 59–61, 138–141 |
| companions/d2_atom_integral_20261006/check.py | d7e7df7dfc49de50596059db3d11dd3f552cb0c0 | 21–23, 64–66, 122–123 |

Their workflows `d2-square-schur.yml` (blob8ea3f10edd8b73bb7c98af622304c7994e5373ba),
`d2-moment-bridge.yml` (069d028c5080f9f81728d6aca0c557d0fa2abeaf), and
`d2-atom-integral.yml` (bcba3e01b82a8d7194a2a5c10e6dad230b2dc4d6) each trigger on
PR changes to `formal/**`. All require their current exact source identities.

C254's nine-file sparse overlay changes that formal tree and manifest by adding
two modules and their exact target registration. Therefore these current guards
would reject it before accepting a companion replay. This is a direct static
consequence of the equality predicates; no job was run to manufacture a failure.
The guards correctly demand new source review. Historical compiled companion
evidence remains bound to its original package and is not invalidated retroactively.

The current native GitHub connector exposes source publication and readback but
no workflow-dispatch action. Opening a PR would trigger the known conflicting
consumer guards. No C254 commit, branch, PR or workflow was created in this round.
This is a scoped integration/admission limit, not a claim of unavailable GitHub
write permission or a blanket ban on other owners' hosted Math workflows.

## Pickup and acceptance

1. Recheck current main, these three consumers and their actual owners; acquire
   and read back a fresh R17 claim for the exact route work. Preserve the C254
   source hashes and all frozen original companion receipts.
2. Select and record one reviewed route: an isolated companion consuming the
   unchanged formal core, or a coordinated additive core extension with each
   affected companion's new exact source/manifest/receipt bindings reviewed by
   its owner. These are proposals, neither adopted nor authorized on another
   worker's branch by this task. Standing project permission already exists;
   the unresolved issue is technical scope and ownership, not renewed owner consent.
3. Keep the existing 63 theorem statements/proofs, gate, compiler, nine pins,
   axiom checks and rejection controls. Do not remove tree/hash checks, replace
   equality by unchecked acceptance, silently retarget historical receipts, or
   call expected stale-source refusal a theorem failure.
4. Publish the reviewed exact sources through the chosen route; bind actual
   head/tree/tested merge or diagnostic commit, workflow, attempt, logs and
   original artifact bytes. Run the complete applicable master verification on
   final stable inputs, including the new 41 targets and affected consumers.
   Preserve real elaboration failures and repair only changed inputs under review.
5. Fresh nonauthor review and all applicable engineering gates precede any
   integration. Kernel success remains distinct from independent alignment,
   H0/component naturality, critical-level/Morse analysis, bars and scientific
   acceptance. No merge is requested by this heartbeat.

The immediate next useful task is this source-bound execution route. The next
mathematical task after it is smooth quotient/Morse compatibility and the C253
critical-level endpoint argument, with merging distinguished from nonmerging
attachments. Do not substitute further supporting arithmetic.

## Architecture and owner reconciliation applied in this round

The live main#275 Claude table records main#357 landed at `1ae0a1b5`; its review,
integration and bounded deployed-page observation have distinct executors and
scopes. OpenAI's 6110892767 says the Canvas proposal remains unadopted and keeps
metadata presence, source applicability, recorded result and custody separate.
Claude's 6110924400 is a bounded design-only pickup, not implementation ownership;
Muse's 6110945561 is also design input. This round uses source identities, actual
review scopes and recorded result labels; it does not implement their proposed
Canvas, invent live presence, or claim runtime/scientific acceptance.

The active board is main#307 as pointed to by capped main#229. Existing C234F4,
reader, runtime acquisition and controller owners remain undisturbed. Open PR424's
actual control diagnostics were relayed to its active owner, who has already
claimed the repair; C254 does not duplicate that work. This is one bounded audit
delivery, not completion of the entire project audit.
