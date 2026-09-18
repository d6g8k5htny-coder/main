# `engine/bridge/orders/` — committed work orders (empty)

One `<task_id>.json` per work order, schema `q0.bridge.work_order/v1`, validated
by `python3 tools/bridge_check.py`. A file here is a **pinned copy** of a Drive
work-order record; the authority is the Drive record it names in
`authorization.authorizing_record_drive_id`, never this file. An order whose
`authorization.verification_status` is `NOT_VERIFIED` is PREPARED-only. Every
order carries `scientific_status_change_authorized: false`, and the checker
refuses any other value.

No writer in this repository creates files here, the receipt store refuses
this directory as a destination, and no order may name it, the bridge, the
checkers or the agent instructions in its scope. An order is frozen once
committed: the checker fails on any file here that any reachable commit
rewrote or removed (a digest is identity; history is the freeze). An order
must not be created or edited by the PR it authorizes. This directory is empty
because the contract it belongs to is PROPOSED / NOT DEPLOYED; the README
exists so the directory does.

Nothing in this directory establishes, authorizes or changes any mathematical
status.
