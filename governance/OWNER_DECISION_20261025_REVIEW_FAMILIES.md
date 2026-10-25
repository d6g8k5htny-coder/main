# Owner decision: review model families, 25 October 2026

Direct instruction from Dylan Roy in the live ChatGPT conversation on
25 October 2026, about 20:13 UTC. Recorded by OpenAI/Codex AI agent
`/root/document_review_family_policy`; this attribution does not claim a human
review. The [public dissemination record](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6402659728)
quotes the same instruction and identifies the documentation stage.

## Exact owner instruction

> For engineering task reviews by same model family’s are allowed. Not for scientific task. Make this update know to all models and agents directed by Dylan Roy

This direct owner instruction is effective for all models and agents directed
by Dylan Roy. No further approval gate is added by this record.

## Current review rule

| Reviewed scope | Model-family requirement |
|---|---|
| Engineering, including UI, code and tooling behavior | A separate reviewer may be from the author's model family. |
| Scientific work, including proof correctness, statement/source alignment, hypotheses, mathematical applicability and scientific acceptance | The reviewer must be from a different model family from the author. |
| Mixed engineering and scientific work | Qualify each scope separately. Engineering review does not qualify the scientific claims. |

A different family is necessary for scientific review; it does not by itself
establish correctness, acceptance or organizational independence. Other
applicable scientific predicates and review qualifications still apply.

Record the actual author and reviewer, their model families and providers,
reviewed source identity, scope and exposure. Distinct sessions, accounts,
agent nicknames or model variants do not by themselves establish distinct
families. Provider identity and family identity are separate facts.

Routine author testing, self-checks and same-family technical reads remain
permitted and useful. Keep their actual attribution and limits: they are
technical or author-side evidence, not qualifying scientific review. A
separate same-family engineering reviewer is still distinct from an author
reviewing their own work.

## Relationship to earlier guidance and records

This later instruction partially supersedes the review qualification in
[25 October's rules-loosening decision, LOOSEN C](OWNER_DECISION_20261025_RULES_LOOSENING.md#c-scientific-status-process),
where its unqualified statement that same-provider review counts could allow
same-family scientific review. Engineering review permissions remain in
effect. A provider label alone does not satisfy the scientific family rule.
The earlier decision remains unchanged as a historical record.

Apply this rule to new and still-pending review decisions. An older review
used for new or pending scientific acceptance must meet the current family
requirement within its actual source and scope. Preserve historical review
text, attribution and outcomes. This decision does not automatically promote
or retract any scientific status or create a missing scientific review.

Required hosted checks, platform-enforced approvals and protections, honest
attribution and existing access limits remain in force. This documentation
does not change GitHub settings, rulesets or workflows, and authorizes no
approval simulation or platform bypass. Scientific effect of this record:
NONE; organizational-independence credit 0.

## Dissemination and onboarding

Current and future models and agents directed by Dylan Roy must carry this
decision with the [owner mandate](OWNER_MANDATE_20261015.md),
[standing delegation](OP-AUTONOMY-20260923-v2.1.md) and
[current workflow](OP-WORKFLOW-20260930.md) when dispatching review work,
handing off tasks or onboarding another agent. The [root agent entry](../AGENTS.md)
routes to this decision before review assignment or reuse.

The linked public communication is a dissemination record, not evidence that
every model or agent has read or acknowledged it.
