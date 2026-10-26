# Proof-to-code trace and mutation census: Math- `frontiers/unique_replacement_bar_intensity_20261001`

This is evidence for pickup [619/6405822096](https://github.com/d6g8k5htny-coder/main/issues/619#issuecomment-6405822096), `URB-PROOF-CODE-TRACE-CLAUDE54-20261026`. Credit 0. Scientific effect NONE. The packet is read at Math- `b29c4b769cac85437b409d152651d5cce4140015` and is not copied here, apart from the mutated `check_exact.py` copies that verifier probes wrote under `verify/`.
- **`urb_trace.tar.xz`** is one deterministic archive: GNU tar, sorted names, mtime 2026-10-26T00:00:00Z, owner 0, xz -6. It holds:
  - `runs/`: logs of the README's four commands (normal and `-O`), with the Git 2.43 shim and without it; the outputs of `trace_checks_urb.py`, `packet_code_offfixture.py` and `proposed_control_demo.py`; and both survivor classifications;
  - `census.json`: one record per mutant (983), plus the baseline records;
  - `tools/`: `census_urb.py`, the mutation census; `survivors_urb.py`, `classify_urb.py` and `classify2_urb.py` (the line-and-kind classes the trace uses); `trace_checks_urb.py`, the trace-side exact checks U1–U8; `packet_code_offfixture.py`; `proposed_control_demo.py`; and `git-2.43-shim`;
  - `verify/`: the probe scripts and outputs of four independent verifier sub-agents of this session, and their journal.
- **`CONTENTS_SHA256`** lists every file inside the archive.
- The trace document itself is published in its Draft PR on `main`.
