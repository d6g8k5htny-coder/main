# Bounded research continuation

Dylan Roy authorized recurring advancement and optimization in the current
research task. The Codex task heartbeat **Advance research and
verification** is enabled hourly, attached to task
`01a0bbb5-2fcb-77f0-b78b-4d220ddd7ab2`. Its scheduler identifier is
`advance-private-research-and-verification`. This document records the setup;
the app scheduler is the live configuration authority. It was created at a
six-hour cadence; a later live configuration changed this to one hour. The
delivery preserves that newer setting rather than overwriting it.

Each run checks the current source/head/claim boundary and attempts at most
one scoped delivery. Actionable CI failures or review findings take priority;
otherwise the worker advances a tractable mathematical or architectural item.
The initial mathematical priority is the RN density/window-to-spatial-cover
path. Priorities follow the live register rather than a stale local status copy.

The heartbeat complements the register's existing **Daily Drive Review Pickup**.
It must reconcile prior verdicts and avoid duplicate reviews or another
worker's active claim. Local scheduler inspection found no prior matching
Codex automation before this setup; the older daily pickup's current runtime
was not independently verified. Neither process is represented as a hard lock:
R17 requires fresh claims, readback, lease renewal and guarded publication.

Runs reuse source bytes only by exact digest and extraction rule. Targeted
Drive reads and modified-time deltas handle changed dependencies; removals or
access loss still require targeted reconciliation. Full-account recrawls and
unchanged full-suite reruns are avoided. A changed final tree receives the
complete master verification once, after focused development checks; failures
or further edits require the relevant fresh verification. A reused receipt
must never be described as a new execution.

Dylan explicitly authorized public repository visibility on 2026-09-20, and
unauthenticated access was verified. Visitors have no write access; Drive
sharing remains separate. Deliveries use the existing draft PR and one immutable,
read-back Drive bundle with a release/handoff event. Tests, hashes, scheduler
runs and same-provider checks do not promote scientific status or manufacture
independence. Frozen sources and historic errors stay intact. No automatic
merge, GitHub Release, visibility or access change, source deletion or third-party
message is authorized by this schedule.

The worker stays quiet when nothing actionable changes and reports substantive
results, meaningful failures or required user action. Local execution requires
the computer to remain available with the desktop app running. The owner can
change or pause the schedule in the app. The repository's runner and this
document do not themselves install a background service.

Scheduler behavior follows the official
[scheduled-task documentation](https://developers.openai.com/pt-BR/docs/automations).
