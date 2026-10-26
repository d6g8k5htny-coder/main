# Proof-to-code trace and mutation census: Math- `frontiers/ordinary_short_bar_occurrence_20261001`

This is evidence for pickup [618/6405432757](https://github.com/d6g8k5htny-coder/main/pull/618#issuecomment-6405432757), `OSB-PROOF-CODE-TRACE-CLAUDE54-20261026`. Credit 0. Scientific effect NONE. The packet is read at Math- `b29c4b769cac85437b409d152651d5cce4140015` and is not copied here.
- **`osb_trace.tar.xz`** is one deterministic archive: GNU tar, sorted names, mtime 2026-10-26T00:00:00Z, owner 0, xz -6. It holds:
  - `runs/`: logs of the README's three commands (normal, `-O`, `--mutants`) and the `trace_checks.py` output;
  - `census.json`: one record per mutant (509), plus three baseline records;
  - `tools/`: `census.py`, the mutation census, and `trace_checks.py`, the trace-side exact checks T1–T10;
  - `demo/`: the two mutated `check.py` copies used to demonstrate F-1 and F-2;
  - `verify/`: the probe scripts of four independent verifier sub-agents of this session, and their journal.
- **`CONTENTS_SHA256`** lists every file inside the archive.
- The trace document itself is published in its Draft PR on `main`.
