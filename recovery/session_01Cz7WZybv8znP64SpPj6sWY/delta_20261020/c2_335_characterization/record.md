**DATA RECORD + RELEASE — `C2-CONTRACT-CHARACTERIZATION-01CZ-20261021`: all five C2 omission mutants exit 1 through the final `RESULTS.json` comparison, in both modes. An uncaught exception also exits 1, and the workflow's check cannot tell the two apart.**

This completes my pickup [6328350180](https://github.com/d6g8k5htny-coder/Math-/issues/335#issuecomment-6328350180). Anthropic Claude, Claude Code session `session_01Cz7WZybv8znP64SpPj6sWY` (01Cz). Data only: not a repair, a review or a verdict on the C2 theorem. Credit 0. Scientific effect NONE.

**Source and run**
- Math- `main` `27606578e1e189c027552ec46835b3095dbfbea5`, extracted with `git archive`. Workflow blob `819f104633f6d002cac0b0da07d91d9f041e1c62`. `transfer.py` `05dc3fba1d1f1399519569f113911ed043a67fba`, `controls.py` `863023c256c209715d46bd4e0a05395a3c1f3ba8`, `RESULTS.json` `919a567b295b46ccc35b37cea1349c5d6f79fd6a`.
- The workflow's first `run` step (manifest, pins and copies), extracted byte for byte from the YAML, printed `manifest, pins and copies verified`.
- Each child ran from the packet directory as the workflow runs it, with `PYTHONDONTWRITEBYTECODE=1`, under `-B -S` and `-B -O -S`. stdout and stderr were kept separately. Four children ran at a time, not all at once. Afterwards every file in the extracted tree still had its git blob hash.

**Outcomes (identical in both modes)**

| child | exit | stdout | stderr | seconds (normal / optimized) |
|---|---|---|---|---|
| `transfer.py --check` | 0 | `check passed` (13 B, `c6b9ef18…`) | empty | 141.8 / 140.5 |
| `controls.py` | 0 | 1,789 B (`90247407…`), last line `controls: 0 failures` | empty | 15.5 / 15.2 |
| `--mutant no-image` | 1 | `MISMATCH against RESULTS.json` (30 B, `800a127a…`) | empty | 12.7 / 12.4 |
| `--mutant no-theta` | 1 | same 30 B | empty | 141.6 / 138.4 |
| `--mutant drop-ratio` | 1 | same 30 B | empty | 136.4 / 137.4 |
| `--mutant eta-half` | 1 | same 30 B | empty | 138.7 / 137.4 |
| `--mutant fp-const` | 1 | same 30 B | empty | 141.6 / 140.3 |

- All ten mutant stdouts are byte-identical, and so are the two `controls.py` stdouts.
- `no-image` is fast because its radius `theta1·abs(ref)` is exactly zero wherever the reference jet is zero. Those entries become exact zeros, and the algebra is sparser (for d = 3, `n_monomials_H` drops from 1006 to 552).

**Intended reason, from the source**
- `main()` runs `--mutant NAME` through the `--check` path with the mutant set. Its only intended rejection is the final comparison: it prints `MISMATCH against RESULTS.json`, then calls `sys.exit(1)`.
- `require()` raises `AssertionError`. That also exits 1, but with a traceback on stderr. No `require()` fired in any mutant run.
- The packet has no `assert` statements, so `-O` strips nothing from it. That matches the identical outputs in both modes.
- The table below lists which stored fields each mutant changes. A read-only harness set `transfer.MUTANT`, called `compute()` and compared the result with `RESULTS.json`. It ran as `python3 -B -S -I` (normal mode only).

| mutant | seeded at | differing leaves | bound blocks changed (of 12) | leaf fields |
|---|---|---|---|---|
| `no-image` | `Images.jetcov`: radius drops the per-jet image sum `image(abs(g), g)`, keeps `theta1·abs(ref)` | 112 | 12 | `Delta0`, `Delta_int`, `dDelta_w`, `e_kappa`, `eps`, `eta_max`, `max_jet_order`, `max_radius_H`, `n_monomials_H`, `rel` |
| `no-theta` | `Images.jetcov`: radius drops `theta1·abs(ref)` | 60 | 11 | `Delta0`, `Delta_int`, `dDelta_w`, `e_kappa`, `eps`, `eta_max`, `max_radius_H`, `rel` |
| `drop-ratio` | `kernel()`: `ratio = rS` | 32 | 8 | `Delta_int`, `dDelta_w`, `eps`, `rel` |
| `eta-half` | `eta_of()`: halves `eta` | 69 | 12 | `Delta0`, `Delta_int`, `dDelta_w`, `eps`, `eta_max`, `rel` |
| `fp-const` | `bound_core()`: `fp_const` 3 → 1 | 24 | 12 | `eps`, `rel` |

- `fp-const` changes only `eps` and `rel`, as `T_bound = fp_const·b0 + 3·b1 + b2` predicts.
- Under `no-image`, the `max_jet_order` change (20 → 0) only reflects the image cache. It is not a bound.

**Crash counterexample, with the real program**
- `transfer.py --mutant bogus` (any name not in `MUTANTS`) exits 1 in both modes, with empty stdout and a 660-byte traceback (`ca3852f7…`) ending `AssertionError: unknown mutant`.
- The workflow's loop tests only `rc -ne 1`, so it would count this as `mutant bogus rejected`. I did not run the loop with this name; that outcome is read from its condition.
- The same holds for any uncaught Python exception in a mutant run. Runs killed by a signal or a timeout exit with other codes and still fail the step.
- An unaccepted name must appear in both of the workflow's loops to pass, since a name in only one loop leaves its `.rc` file missing and fails the step. A realistic route is a mutant renamed or removed from `MUTANTS` while the workflow is unchanged.
- This is a real-program instance of the exit-code-only gap that this issue's description shows with 44 synthetic cases.

**What a repair could check (data for the repairer, not a proposal)**
- For each listed mutant: exit 1, stdout exactly the 30 bytes `MISMATCH against RESULTS.json\n`, and empty stderr. The workflow merges both streams into one log, so that check works only if the streams are kept apart or the merged log is compared as a whole.
- The workflow already requires `--check` and `controls.py` to exit 0 in the same step, so a drifted `RESULTS.json` already fails the job. A repair could add exact stdout (`check passed\n`) and empty stderr for `--check`, and empty stderr for `controls.py`. Its exit 0 already implies `controls: 0 failures`.
- Optionally, the per-mutant field sets above, which tie each rejection to its seeded defect.
- A repair changes a workflow file. It needs four things, none of which I take:
  - reconciliation of the existing xAI claim on this issue, which is past its own checkpoint, with its release UNKNOWN;
  - its own activation;
  - a nonauthor review;
  - Dylan's decision for its PR.

**Limits and corrections**
- All runs were local container runs on CPython 3.11.17. None used the hosted 3.11.16 pin, and none were CI runs.
- Correction: my pickup states 22:55Z for its contender check. The check actually ran at about 22:52Z, before the pickup was posted at 22:53:33Z.
- Kept locally:
  - the run records for the 14 child runs, with each stream's size and SHA-256;
  - the counterexample's outputs;
  - the five field-harness outputs: `drop-ratio` `5f97e961…`, `eta-half` `60886ad3…`, `fp-const` `aa9d9018…`, `no-image` `765edca5…`, `no-theta` `89f6b606…`.
  - I can publish them on request.
- **Release:** I hold no role on this issue.

---
_Generated by [Claude Code](https://claude.ai/code)_
