# Reconnaissance — ROLLOUT-R2-20260921

Scope: risk-proportionate governance adoption, typed identity, scoped publication,
not a mathematical theorem. Inspected 21 September 2026. Primary-source reading.

Queries executed included: "site.sre.google workbook canarying releases config
rollback"; "docs.github actions secure pin"; "docsGoogle writeControl";
"site.docs.github.com REST git references update a reference force false fast forward";
"site.docs.github.com REST pulls merge sha head branch compare base". The first
three are summarized query labels from this round's reconnaissance; the final two
are exact literal queries. No private source text was submitted to web search.

Google SRE Canarying Releases (https://sre.google/workbook/canarying-releases/)
describes limited release cohorts and evaluation. We apply that idea to material
behavior changes, not to make every link repair wait through a new governance chain.
GitHub's reference API (https://docs.github.com/en/rest/git/refs), Update a reference,
uses force=false to require a fast-forward update. It is not a distributed lock.
The pull-request API (https://docs.github.com/en/rest/pulls/pulls), Merge a pull
request, binds its sha parameter to the PR head; it does not authenticate a Drive
snapshot or guarantee semantic compatibility of a clean merge. The workflow-events
document (https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows),
schedule section, limits scheduled workflows to the default branch. This matters
because the observed default branch lacks the working branch's workflows.

Google Docs documents.batchUpdate
(https://developers.google.com/workspace/docs/api/reference/rest/v1/documents/batchUpdate)
provides requiredRevisionId guarding. GitHub secure-use guidance
(https://docs.github.com/en/actions/reference/security/secure-use) supports pinned
actions and restricted permissions. Existing read-only CI safeguards are retained.

Findings are application-specific. No source proves our dependency map complete,
verifies all consumers adopted policy, authenticates independent review, or supplies
a theorem verdict. The new tests are author-side regression evidence. This audit
reuses current R17 authority; broader behavior rollout remains scoped and reviewed.
