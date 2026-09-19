# `engine/rn_engine` — the recovered RN uniform engine

Closes `LANE_RN_UNIF.md` caveat 5, which recorded the frozen engine
`d3_rn_unif.py` as "not in the RN_UNIF Drive folder … not downloaded here" and
so left `rnu_ds3.py`, the preferred A5 carrier, un-runnable.

The full report — how the reading copy was inverted, what the engine does line
by line, where the `chi2_grad_bound` ~1e19 slack comes from, which chain-rule
terms `mean_grad_exact` is missing, and what all of that does and does not
mean — is **[`docs/ENGINE_RECOVERY.md`](../../docs/ENGINE_RECOVERY.md)**.

## Layout

```
BINDING.json          eight carrier records, carrier-manifest shape
reconstruct.py        ACCESS READING VOLUME -> archived payload; returns only on a SHA-256 match
verify_recovery.py    checks A-D; --selftest runs the negative controls
frozen/               the eight recovered payloads, byte-identical to the archive
```

## Verify

```bash
python3 engine/rn_engine/verify_recovery.py --selftest
```

Checks: (A) every blob hashes to its recorded digest and byte count; (B) each
digest agrees with `drive/source_map/`; (C) the `_PINS` block read back out of
the recovered `d3_rn_unif.py` matches the six recovered dependencies; (D)
negative controls on the reconstruction rule. Each control was shown to fire
against a deliberately broken copy — see `docs/ENGINE_RECOVERY.md` §2.

## Rules

* **Nothing under `frozen/` may be edited.** These are frozen bodies. A repair
  belongs in a sibling module that rebinds the symbol, the way `rnu_ds3.py`
  already rebinds eight of the engine's names at import.
* **Do not run the engine's certifier and do not quote a bound from it.**
  Nothing here invokes it; `run_certification` is in fact defined and never
  called in the frozen body.
* `d3_perc.py` (`RNENG-03`, `5bc09241…`, 40,337 B) is **bound as bytes and
  logically quarantined** at the scope of `Q-RN5-MOMENT-001` (class
  `DEFECTIVE_SCOPE`, register export 2026-09-18): `envelope_v`,
  `window_cap_env` and consumers relying on the claimed polarity-safe upper
  bound. Other functions are outside that finding. Recovering the file does not
  lift the hold; recording the hold does not unbind the file. The record in
  `BINDING.json` carries the exclusion's key, class and scope under
  `quarantine_exclusions`, and `tools/quarantine_check.py` invariant 5 fails
  closed if that annotation is missing. No value produced by `envelope_v` or
  `window_cap_env`, or by anything that consumes them, may be cited from this
  repository. (`Q-RN5-MOMENT-002` names `d3_amend_v2.py`, a different file from
  the `d3_amend.py` bound here as `RNENG-04`.)
* Merging `BINDING.json` into `engine/carriers/MANIFEST.json` is a follow-up
  step for whoever owns that file; it was deliberately not written there.

## What this directory does not establish

Recovering a carrier byte-for-byte is provenance, not proof. `OBL-H5-JETMOD`,
`OBL-H5-ZBAND`, `OBL-H5-REMOTE-THRESHOLD`, `OBL-D1-PROMOTE` and **both** pieces
of `D3-LEMMA-RN-UNIF` remain OPEN. The recovered engine is `mpmath` binary
floating point at `mp.dps = 100` throughout, with no exact-rational and no
interval arithmetic anywhere in it: **NON-CERTIFYING**. High precision is not a
certified enclosure. No number produced by this engine is quoted as a result
anywhere in this repository. Original prize problems solved: 0.
