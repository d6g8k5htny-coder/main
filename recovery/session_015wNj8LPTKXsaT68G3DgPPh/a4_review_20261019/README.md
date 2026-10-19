# Evidence for C8's review of the A4 elder-sector proof

This folder holds the controls behind `C8-A4-XPROV-REVIEW-20261019`. It is a provider-distinct nonauthor review by Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8): Dylan Roy — delegated AI review. The canonical pickup is [Math-#476/6275238008](https://github.com/d6g8k5htny-coder/Math-/pull/476#issuecomment-6275238008), and the verdict is posted in the same thread. Credit 0. Scientific effect NONE.

## Source reviewed

- `frontiers/gaussian_a4_elder_sector_20261017/PROOF.md` in Math-: blob `86371a58`, 32,065 B, SHA-256 `abed6f94da080a33458b78f1e60ae92ad90083323af41d241de05939d4377ff0`.
- `EVERYWHERE_MARK_EXTENSION.md`: blob `7b60528b`, 3,442 B.
- Both landed in Math-#476 as `4e722922` and are unchanged on Math- main `b6fc0e64`.
- The consumed GAD source is `frontiers/gaussian_short_bar_theory_20261017/smooth_gaussian_manifold_admission/PROOF.md`, blob `d61fd11a`, SHA-256 `8d992e19…faf319`.

## Order of work

C8 read PROOF.md, the extension and the GAD sections that PROOF.md uses, ran the controls below, and wrote `OWN_FINDINGS_before_REVIEW.md`. That file was written at 19:26:38Z on 19 October, with SHA-256 `c83fa7e41d66ed11fae18cd0f17253b81c68ad2bcf75633caeb9bbff48e898f7`. Only then did C8 read the packet's `REVIEW.md` (OpenAI).

## Controls

Each script runs as `python3 -I <script>` and needs `sympy` (1.14.0 here) and `numpy`. Each log is the output of one run.

| Script | Log | What it checks |
|---|---|---|
| `ridge.py` | `ridge_log.txt` | Formula (4): solves the transverse ridge equation as a power series for a general quintic jet. The ridge's third, fourth and fifth derivatives equal `T`, `Q4` and `Q5` exactly. |
| `controls.py` | `controls_log.txt` | (22) and its four heights; the §6 box conditions at 27 points; Jacobian (10) at a general angle; (13); determinant one for the SO(2) action on raw `Sym^3`, `Sym^4` and `Sym^5` coordinates; the §9 exponent ledger. |
| `box_scan.py` | `box_scan_log.txt` | The six §6 conditions on an 11×11×11 grid of the box, for half-widths 0.01 to 0.05. |
| `persist4.py` | `persist4_log.txt` | Superlevel-H0 persistence on a grid: the elder rule computed exactly on an 8-neighbour graph. The field is a planar quintic with the §2 ridge corrections, random `a` and random mixed jets `gamma`, `B` and `eta`, plus a remote decoy bump higher than the birth. The run compares the birth's grid death with the sampled ridge values at the two saddles, for r = 0.1, 0.03, 0.01 and 0.003. The t-spacing is `0.25 r^2.5`, so the sampling error `|a| dt^2/8` stays far below `r^5`. |

## Results

- The symbolic checks agree with the source exactly.
- The box conditions hold at every sampled point for half-width 0.01. The 1/8 gap margin fails at some sampled points from half-width 0.015. At half-width 0.05 the `L0 > 1/2` condition also fails.
  - This is consistent with §6, which only claims some small ε.
  - It shows that U is genuinely small under the source's stated margins.
- **Inside the small-r regime, the birth dies at the farther saddle `s3`, never at the nearer `s1`, despite the decoy.** This held for all three seeds at r = 0.01 and r = 0.003.
  - At r = 0.003, `L_r` = 0.899, 0.851 and 0.874, approaching `L0` = 0.9.
- **Outside that regime, the A4 ordering is absent**, which the theorem's `0 < r < r0` allows.
  - For seeds 2 and 5 at r = 0.1 and r = 0.03, the `s3` saddle lies above `s1`, or no saddle forms near 3r, and the birth dies at `s1`.
  - For those mixed jets, the O(r) constant in `L_r = L0 + O(r)` is about 20.

## Limits

These are finite algebra checks and grid illustrations. They are not a proof, a Gaussian field sample, a Rice integration or an evaluation of `C_A4`. Grid connectivity approximates continuous superlevel connectivity. No author or earlier reviewer program was run.
