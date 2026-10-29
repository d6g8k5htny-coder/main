# CBZ addendum adversarial qualification

Owner intent: every active repository participates in empirical falsification and
structural synchronization before CBZ-0. Layer A remains unchanged. This work
qualifies delivered tools; it does not introduce a competing implementation or
edit B-003 canonical surfaces.

The first subject is Main PR #712 at
`15e8f689d00ca03e28e9bc5d97b6aba3f19ec900`. Its existing 43-case suite and
mutation checks are rerun and retained separately from new adversarial cases.
The starting Main inventory suite had 42 passes and one skip.

## Chosen approach

Use real temporary Git repositories and the subject's public crawler/Repo API,
with local file remotes for network-dependent cases. Reuse fixture construction
from its existing tests, but derive expectations from hand-written topology and
Git commit identities. Do not mock crawler decisions. New tests and evidence
remain uniquely named; fixes go to the implementation owner.

Alternatives considered: another crawler implementation would duplicate active
work; documentation-only review would not establish actual failures. A finite
adversarial suite directly tests the delivered behavior.

## Required observations

1. A Lake dependency available at its declared local remote is discovered and
   cloned even when no search-directory clone exists.
2. A newly opened dependency is scanned in the same traversal; a changed shared
   Lean definition produces a blocking drift finding.
3. Pins and workspace records reached in a child repository extend the graph.
4. An explicitly declared, absent canonical registry input cannot silently reduce
   artifact coverage to zero and return success.
5. A recorded Lake revision absent from the available repository blocks pin sync.
6. A command used to establish current tips must refresh cached default refs.
   The last case also tests the freshness limitation of the documented
   missing-object-only fetch implementation; it is not a claim that its README
   already promises remote refresh.

The tests establish engineering behavior only. Lexical equality is not elaborated
AST/type equality, and the tool's declared Lean-only domain is not full addendum
coverage. Python/schema/generated-declaration gaps remain explicit. An empirical
PASS cannot grant mathematical closure. Unsupported or missing coverage blocks
that status; source and kernel checks remain separate.

## Delivery

Retain original failures, exact subject hashes, actual commands, outputs and exits.
Report confirmed findings on PR #712 and the #684 execution queue. Preserve
historical pins and typed active/history boundaries; do not turn ordinary
historical lag into a defect by default. Canonical registration and G0–G11
composition stay with the current B-003 integrator.
