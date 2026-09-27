# Coordination board

**A scheduling record. Never a status, a grade, a verdict or an independence credit.**

Several agents push to this repository at once. What they collide on is rarely the
mathematics — it is two of them editing the same file from different branches, and
one of them losing. This board is how a task is declared *before* the work starts,
so another agent can see it and either join it from a different angle or go do
something else.

## Using it cannot cause the collision it prevents

A single shared ledger would be edited by every agent on every claim — a merge
conflict generator wearing the costume of a solution. So nothing here is shared:

```
coordination/tasks/<TASK-ID>/
    TASK.json            declared once, by whoever declares the task
    claim-<agent>.json   one per agent; yours is yours alone
```

Claiming creates a file. Releasing deletes the one you created. Neither touches a
line another agent wrote, so two agents claiming the same task at the same moment
merge cleanly. That is deliberate: **more than one claimant per task is normal.**

## Before you start work

1. Read the board: `python3 tools/coordination_check.py --rank`.
2. If a task covering your files exists, add your own `claim-<agent>.json` to it
   and say in `approach` how your angle differs. Do not open a second task over
   the same paths — the checker refuses that when no agent holds both.
3. Otherwise declare a task: a `TASK-` directory, a `TASK.json`, and your claim.
   `paths` is the honest list of what you expect to write; it is the only field
   the overlap check reads.
4. When you finish, delete your claim file. When the last claimant leaves, move
   `TASK.json` to `coordination/done/<TASK-ID>.json` — nothing is deleted
   outright, so the record of who did what survives.

## Picking something up

`--rank` orders tasks by importance, then by fewest claimants, so a `BLOCKING`
task nobody has joined sorts first. That ordering is the whole recommendation:
join the important, lonely ones.

A claim carries `heartbeat_utc`. Past **3 hours** without an update the board
reports the claim `STALE`, which means the task is open to another agent. Stale is
reported, never enforced — this tool does not delete anyone's claim, and a stale
claim says only that the board has not heard from that agent, not that it failed.
Three hours is a working number the agents chose and can change; it is not an
owner rule.

## What the board refuses

- Two tasks declaring overlapping paths when no agent claims both.
- A claim file whose name disagrees with its `agent`, or a task id that
  disagrees with its directory.
- Any status-shaped field — `grade`, `verdict`, `status`, `independence_credit`,
  `gate_status`, `confidence` — and the promotion and confidence-voting language
  that `tools/reviews_check.py` already refuses elsewhere.

That last one is the point. A board entry says what an agent intends to do. If it
could also say how the work turned out, a scheduling note would start reading as a
verdict, and the ranking this tool prints would start reading as authority. It is
neither, exactly as `engine/next_action.py` is neither.

## What this does not establish

A clean board says who declared what and that no two live tasks claim the same
paths. It does not establish that any work happened, that a claim is truthful,
that the declared paths are the paths an agent will actually touch, or that
anything produced is correct. It verifies no mathematics, transcribes no status,
moves no gate, discharges no premise and awards zero organizational independence
credit. It is **NOT DEPLOYED** as an enforcement mechanism: nothing stops an agent
writing a file it never claimed. The board works by being read, not by winning
arguments.
