# Proposed §11 quarantine records for 52 Math- proof packets (QP-C54)

By `session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0; scientific effect
NONE. These are PROPOSED records for main#695's `cbz/quarantine/catalog.json`. They close and
admit nothing.

- `facts.json`: the mechanical facts for each packet at Math- `8151f592918a845bf19eb8eafaf8444540a32734`
  (files, blobs, sha256, bytes, `PROOF.md` headings, script indicators), computed from Git.
- `assemble.py`: rejects any record whose quoted statement differs from its `PROOF.md` lines, any
  missing, duplicate or extra packet, and any unclassified floating-point field. It fills every
  source binding from `facts.json`.
  Usage: `python3 -I -B assemble.py MATH_REPO facts.json VERDICTS.json OUT.json`.
- `singlepass_unverified/records_singlepass.json`: **UNVERIFIED, single pass. Do not adopt.**
  Retained only so the work survives a session lapse. It passes `assemble.py`'s mechanical
  checks, but no independent adversarial check has been made yet.
