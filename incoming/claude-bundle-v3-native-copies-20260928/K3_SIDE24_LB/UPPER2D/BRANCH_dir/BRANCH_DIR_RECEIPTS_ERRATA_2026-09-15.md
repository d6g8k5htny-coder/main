# BRANCH_DIR receipts errata — 2026-09-15 (Stage-E numerics re-review)

Scope: RECEIPTS GRADE ONLY (the reviewer's words: "NOT soundness"). The
certificate passes both modes and its r=0.05 production rung was
independently reproduced by the reviewer (q = 0.999855, 1−q = 1.453e-4,
counts 3112/888); the falsifier catches all 4 mutations. The frozen
BRANCH_DIR.md (8821c8facc7e36635c4b84967ba55037ff5a33fff09359f817032eecaab24286)
is untouched; this errata corrects its §4 receipts paragraph and completes
the receipt set.

## 1. What was wrong at freeze time

(i) §4 claimed shipped `receipt_normal.json` / `receipt_O.json` as
    "byte-identical modes". The files were ABSENT at freeze time, and the
    claim was mis-stated: machine receipts record per-run labels (the mode
    field) and so are never the byte-identity carrier; the charter's
    byte-identity applies to PROGRAM STDOUT.
(ii) `rung025.log` shipped EMPTY: its driver died (3-way CPU contention on a
    4 GB box: the rung025 driver, the certificate, and the falsifier were
    running concurrently).
(iii) `falsify_normal.log` shipped a one-line stub (same cause).
(iv) `br_certificate.py` and `br_falsifier.py` printed `time.time()` elapsed
    seconds in stdout — a determinism-charter violation (runner
    labels/wall-clock belong in separate receipt files, never in program
    output). `br_mc.py` had the same violation; also fixed.

## 2. Fixes (code)

All wall-clock tokens removed from program stdout in `br_certificate.py`,
`br_falsifier.py`, `br_mc.py`; elapsed seconds now go to sidecar receipts
(`receipt_<mode>.elapsed`, `falsify_<mode>.elapsed`, `rung<r>.elapsed`).
A second violation of the same class was found and fixed during re-review:
the certificate's final line printed the mode-labeled receipt path (a runner
label in program output) — now a fixed mode-independent string. The
certificate's CK6 MC ladder now runs the CANONICAL scale (N = 1500/400) in
BOTH modes (previously normal ran 4000/1500, which made per-mode stdouts
differ in the CK6 lines); production scale lives in the standalone rung
receipts (`rung05.log` N = 4000, `rung025.log` N = 1500).

## 3. Fixes (products, all re-run after the code fixes)

  * certificate, both modes: `cert_normal.stdout`, `cert_O.stdout` —
    ALL CKS PASS, and the pair is BYTE-IDENTICAL (diff-verified). Machine
    receipts `receipt_{normal,O}.json` differ only in the mode label
    (runner labels belong in receipts). Wall-clock in
    `receipt_{normal,O}.elapsed`.
  * falsifier, both modes: `falsify_normal.stdout`, `falsify_O.stdout` —
    BYTE-IDENTICAL (diff-verified); all 4 mutations caught in each mode;
    wall-clock in `falsify_{normal,O}.elapsed`.
  * rung025 production: relaunched and completed (`rung025.log`, N = 1500).
    The bracket pattern `[b]r_mc` was used for every process check
    (pgrep -f self-matches in this environment — the H5 lesson; the earlier
    H5 keepers died silently for exactly this reason). The detached keeper
    script itself is reaped by this environment between runner invocations,
    so the keeper's relaunch watchdog was executed via the monitoring loop;
    the sequence is recorded in `keeper_rung025.receipt` (the log carries
    two "engine" preamble lines: the killed first driver's, then the
    completing driver's).
  * rung05 production: `rung05.log` (N = 4000) — completed pre-errata and
    independently reproduced by the reviewer; the certificate's earlier
    full-scale normal-mode run (pre-unification) matched it exactly
    (counts 3112/888, q = 0.999855). That full-scale stdout was superseded
    by the canonical-scale re-run and is not preserved; `rung05.log` is the
    full-scale receipt of record. FORMAT NOTE: `rung05.log` predates the
    determinism fix and its final line carries the old `[990s]` wall-clock
    token; it is kept as the production receipt of record (its numbers are
    the reviewer's reproduced ones), with the format violation disclosed
    here rather than rewritten.

## 4. §4 receipts paragraph — corrected text

Replaces the §4 block in BRANCH_DIR.md (which remains frozen as hashed):

    ## 4. Receipts and freeze (corrected 2026-09-15)

    Production MC rungs: `rung05.log` (r = 0.05, N = 4000) and
    `rung025.log` (r = 0.025, N = 1500). Certificate program stdout:
    `cert_normal.stdout` ≡ `cert_O.stdout` (BYTE-IDENTICAL pair,
    diff-verified; ALL CKS PASS both modes). Falsifier program stdout:
    `falsify_normal.stdout` ≡ `falsify_O.stdout` (BYTE-IDENTICAL pair,
    diff-verified; all 4 mutations caught in each mode). Machine receipts
    (mode-labeled; may differ): `receipt_{normal,O}.json`. Wall-clock
    sidecars (never in program output): `*.elapsed`. Keeper record:
    `keeper_rung025.receipt`. Freeze: `FREEZE.txt` (code + document hashes at the original freeze)
    and `FREEZE_ERRATA_2026-09-15.txt` (errata-era code + receipts).

## 5. The r³-scaling ladder (completed with rung025)

    r = 0.05:  q = 0.999855   1−q = 1.453e-4 (se 5.1e-05) = 1.162 r³
               counts {success: 3112, A: 888};  W-shares: A = 1.453e-4,
               B1/B2/B4 below MC resolution (N = 4000)
    r = 0.025: q = 0.999989   1−q = 1.075e-5 (se 6.1e-06) = 0.688 r³
               counts {success: 1183, A: 317};  W-shares: A = 1.075e-5,
               B1/B2/B4 below MC resolution (N = 1500)

1−q = O(r³) with rung constants 1.16 and 0.69 — consistent with a common
constant within the importance-sampling standard errors; the failure budget
is carried by class A (the window-saddle carrier lane) at both rungs,
matching C1's point value E_w ≈ 1.6e-4 = 1.28 r³ at r = 0.05 and H5's
certified I_hi with ~500× slack. The certificate's canonical-scale CK6
(N = 1500/400) displays q = 0.999777 / 0.999974 (1−q = 1.78 / 1.65 r³,
se ~1e-4/2e-5) — consistent with the production rungs within the MC
standard errors. The BRANCH_DIR.md mechanism and bound are unchanged.

## 6. Receipt inventory

See `FREEZE_ERRATA_2026-09-15.txt` for sha256 of every file named here.
