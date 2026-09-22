# Mechanism assimilation: bounded implementation record

Observed 2026-09-22 UTC. Follows architecture audit `ASSIMILATION-20260922-01a0c69f`.
This records engineering decisions and their measured limits. It is not a new
governing protocol, queue, scientific verdict or evidence of deployed enforcement.
The code changes remain a review candidate until integrated; native enforcement
has the separately verified deployment state recorded below.

| Mechanism | Decision | Implemented or measured scope | Remaining gate |
|---|---|---|---|
| Native GitHub required checks and branch rules | **ADOPT, active** | Ruleset 23798639 protects exact `main` and the dedicated test branch: PR required, strict `verify` from Actions app 15368, deletion/force blocked, admin recovery through PR only. | Direct and force updates were rejected on the protected probe; the same credential updated an unprotected control. Protected PR success/failure, deletion and recovery bypass were not exercised. [Exact receipt](https://drive.google.com/file/d/16-cSRzAPfE80D_SYAMEUmlshX2kyvOF2/view). |
| Pinned CI actions and hashed test dependencies | **ADOPT in this candidate** | Full action commit IDs, Python 3.11.16, Ubuntu 24.04 series, and the minimal hashed pytest dependency closure. Withdrawal and rollout checkers run unconditionally in full CI. | Final full-suite and remote CI evidence must bind the published revision. Hosted OS and bootstrap tooling remain outside the lock. |
| SQLite FTS5 metadata candidates | **ADAPT, explicit opt-in** | In-process trigram candidates followed by the original matcher, ordering and current metadata. Default/CLI behavior remains the original scan. | Sustained callers must justify construction/maintenance cost; no relevance or scientific-quality gain has been measured. |
| Existing withdrawal semantics | **ADAPT integration, synthetic CI only** | Reuse the current grounded-support pilot and its synthetic controls; full CI now runs its loss-only transition. | Actual scientific claims remain **NOT_MIGRATED**. Coverage, source authenticity and authorized status transitions remain separate requirements. |
| PaperQA, Aviary/LDP, Sakana loops, durable orchestrators, proof/provenance/TMS adapters | **WATCH** | Mechanism candidates and bounded experiment designs exist. These packages/frameworks were not installed or comparatively benchmarked in this implementation. | Execute the relevant preregistered baseline comparison before selecting an adapter or replacing existing machinery. |

## Why the retrieval adapter is opt-in

The implemented comparison used **16,583 metadata records**: 4,934 reconciled
files/folders and 11,649 archive occurrences, including held metadata for
navigation. It hydrated no source bodies. Eighteen fixed queries had exact
result parity; short/non-ASCII queries retain the original scan behavior.

On local macOS, Python 3.11.16 and SQLite 3.53.1, the sum of warm median query
times was 0.34368 seconds for the original scan and 0.24006 seconds for the
adapter: **1.43× across that workload**, with 15 repetitions per query. The
median individual-query ratio was 1.72×; these are different aggregations.
Building the index and serving the first query took 0.40991 seconds, versus
0.01396 seconds for a first scan. Estimated amortization was about **71 repeated
queries**, using that fixed mix; it is not a production guarantee.

Those results do not support accelerating a one-shot CLI by default. Sustained
callers may explicitly use `keyword_matches(records, query, use_index=True)`.
The index is an in-memory derived view, not a new source of identity or authority.
It validates the current search projection and returns current records; missing
FTS5 support and unsupported inputs fall back to the scan. Eligibility for
accepted evidence remains outside ranking. This is a finite metadata/timing
comparison, not a retrieval-relevance benchmark or mathematical verification.

The task retains the raw result and benchmark script outside the repository:
`retrieval-evidence/results.json` SHA-256
`95dc7aedb633837a23b9cc1d00d2e70fbbcc436a4c76711d1bbf7f6db91ac69a`;
`retrieval-evidence/benchmark.py` SHA-256
`dc1a53dbbf8bf06a82a2a7f898c1d6267ddd83b4a1d9b4003b0e13d915806721`.
The result binds tested source-file hashes because its checkout was uncommitted.
Attach the immutable result pointer to the existing trial record when registered.

## CI evidence and boundaries

The existing tested environment was retained: pytest 9.0.2, iniconfig 2.3.0,
packaging 26.3, pluggy 1.6.0 and Pygments 2.21.0. Official PyPI wheel hashes
are in [requirements-ci.lock](../requirements-ci.lock). A fresh Python 3.11.16
environment installed it with pip's `--require-hashes --only-binary=:all:`;
a dry-run with a deliberately substituted local wheel failed hash validation.

The scoped workflow/bridge/withdrawal/rollout/runner suite passed **760 tests**.
Withdrawal and rollout controls also passed under optimized Python (36 and
46 tests). The subsequent protected-lock change passed 42 relevant bridge tests.
The subsequent complete local run passed all 43 checks and 3,042 tests, with 2
conditional skips, on the source tree published as `105a426`. GitHub then caught
unquoted YAML colon-whitespace in the CI/research dependency install commands
before execution. This successor quotes those scalars; Ruby/Psych confirms all
three workflows parse, and 208 affected tests pass with the same 43 checker
commands. The failed observation is retained. Final remote results and the
source-bound delivery receipt are linked from [PR 9](https://github.com/d6g8k5htny-coder/main/pull/9).
The task retains logs and JUnit output under `ci-evidence/` outside the repository.

Action pins were checked against official refs and their declared interfaces:
[checkout](https://github.com/actions/checkout/commit/11d5960a326750d5838078e36cf38b85af677262),
[setup-python](https://github.com/actions/setup-python/commit/a26af69be951a213d495a4c3e4e4022e16d87065),
[upload-artifact](https://github.com/actions/upload-artifact/commit/ea165f8d65b6e75b540449e92b4886f43607fa02).
This was not an exhaustive audit of bundled JavaScript. Exact Python version,
action source and wheel hashes narrow variability; the hosted runner image,
Python binary distribution selection and bootstrap pip are not a hermetic build.

## Reuse sources and deferred experiments

- [SQLite FTS5 trigram tokenizer](https://www.sqlite.org/fts5.html#the_trigram_tokenizer): candidate lookup, while project scope and identity remain outside the index.
- [GitHub rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets) and [pip hash checking](https://pip.pypa.io/en/stable/topics/secure-installs/): reuse established enforcement and installation mechanisms.
- [PaperQA](https://github.com/Future-House/paper-qa), [Aviary](https://github.com/Future-House/aviary) and [LDP](https://github.com/Future-House/ldp): E1 retrieval/routing comparisons remain NOT_RUN; public components do not reproduce the complete hosted FutureHouse stack.
- [Sakana v1](https://github.com/SakanaAI/AI-Scientist) and [v2](https://github.com/SakanaAI/AI-Scientist-v2): E3 experiment allocation and E4 review comparisons remain NOT_RUN. Their current custom licenses require examination before code reuse.
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence): E2 crash/retry comparison remains NOT_RUN. Checkpoints do not confer evidence validity or cross-service atomicity.
- [Lean validation](https://lean-lang.org/doc/reference/latest/ValidatingProofs/), [in-toto statements](https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md), [Drools truth maintenance](https://kie.apache.org/docs/10.0.x/drools/drools/rule-engine/index.html), and [Debezium outbox](https://debezium.io/documentation/reference/stable/transformations/outbox-event-router.html): V1/P1/W1/R1 remain NOT_RUN. Existing project proof/withdrawal/provenance work is retained.

## Institutional memory

Use existing **Work Events**, **Reusable Operations** and **Operation Trials**
for outcome pointers, applicability and matched-budget evidence. Keep code,
fixtures and immutable receipts in their existing execution/artifact locations.
No second queue, schema or mandated global protocol is introduced by this note.

Each decision should identify the external version/license, extracted mechanism,
incumbent revision, fixture, raw result, total cost, scope, rollback and revisit
trigger. A failed or inconclusive comparison remains useful institutional memory.
Same-provider implementation/review earns **zero organizational-independence
credit**. No claim, premise, gate, grade or original prize status changes here.
