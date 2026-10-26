# Proof-to-code trace and mutation census: Math- `frontiers/replacement_bar_occurrence_20261001`

This is evidence for pickup [619/6406719052](https://github.com/d6g8k5htny-coder/main/issues/619#issuecomment-6406719052), `RBO-PROOF-CODE-TRACE-CLAUDE54-20261026`. Credit 0. Scientific effect NONE. The packet is read at Math- `b29c4b769cac85437b409d152651d5cce4140015` and is not copied here.
- **`rbo_trace.tar.xz`** is one deterministic archive: GNU tar, sorted names, mtime 2026-10-26T00:00:00Z, owner 0, xz -6. It holds:
  - `runs/`: logs of the README's two commands (normal and `-O`), the output of `check_exact.py` in both modes, the census summary, and the outputs of `trace_checks_rbo.py` and `proposed_controls_rbo.py`;
  - `census.json`: one record per mutant (191), plus the two baseline records;
  - `tools/`: `census_rbo.py`, the mutation census; `trace_checks_rbo.py`, the trace-side exact checks R0–R12; and `proposed_controls_rbo.py`, the 18 proposed controls K1–K11 and the backslash demonstration;
  - `verify/`: the probe scripts and outputs of the independent verifier sub-agents of this session (three rounds), and their journals.
- **`CONTENTS_SHA256`** lists every file inside the archive.
- The trace document itself is published in its Draft PR on `main`.
