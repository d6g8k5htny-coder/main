# Evidence for C8's review of the remaining slices of the non-Gaussian smoothing-admission proof

This folder holds the checks behind `C8-NG-XPROV-REVIEW-HP12-20261019`. It is a provider-distinct nonauthor review by Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8): Dylan Roy — delegated AI review. The canonical pickup is [Math-#476/6277334910](https://github.com/d6g8k5htny-coder/Math-/pull/476#issuecomment-6277334910), and the verdict is posted in the same thread. Credit 0. Scientific effect NONE. The p = 0 slice and its evidence are in `../ng_review_20261019/`.

## Source reviewed

- `frontiers/nongaussian_smoothing_admission_20261017/PROOF.md` in Math-: blob `9df1a2c4`, SHA-256 `322195ff6693f17047d9ec60c52f8178f5f3a215c4f4cba42dd342ff59dbc116`, unchanged on Math- main `4eb86ed3`.
- The HF import: `main` `experiments/universality/higher_homology_fold/PROOF.md`, blob `8e18a790`, 46,174 B, SHA-256 `87f43e6ed94c5c63c385ecf177f68212afa5db54b24062fa008f081f47a78999`, §§1–7. These are the bytes PROOF.md cites.
- **Scope:**
  - (a) the theorem for p = 1, …, d − 1 on oriented X with d ≥ 2, and what differs from p = 0 in §§4, 6, 7 and 10;
  - (b) §12, the bounded-cluster estimate (30) and the iid-copy Poisson law (31)–(32).

## Order of work

1. **21:07Z, 19 October.** Pickup posted.
2. **21:09:14Z.** `OWN_FINDINGS_NG_S12_before_REVIEW.md` written (SHA-256 `f5a3847e7ed70582a568693631a355b1b00d6b33e726fac64eefeaab46dfe561`), before `REVIEW.md` §§11–12 were read. Its numerical line says PENDING; the file is kept unchanged.
3. **21:09–21:12Z.** C8 read PROOF.md §§4, 6, 7 and 10 and HF §§1–7, and wrote `FINDINGS_NG_HP.md`. The Hp findings are not pre-comparison: `REVIEW.md` §§4 and 6 had been read during the p = 0 slice.
4. **About 21:11Z.** `REVIEW.md` §§11–12 read.

## Checks

| File | What it checks |
|---|---|
| `hp_check.py` → `hp_check_log.txt` | The axial profile between exact pins with F_sss = 12k: f(M) − f(S) = kr³, and α_M, α_S = ∓6k. Block identity (17) for d = 3 and 4. The index and degree table j = d − p − 1 for d = 2…6. Needs sympy 1.14.0. |
| `simulate_cluster.py` → `clus_eps{0,1,3}.json` | The same model and persistence code as `../ng_review_20261019/simulate_small.py`, with per-realisation counts N_t recorded. It gives E N_t, P(N_t ≥ 2) and E[N_t 1{N_t ≥ 2}]. |
| `cluster_analysis.py` → `cluster_log.txt` | ρ(t) = E[N_t 1{N_t ≥ 2}] / E N_t, which (30) says tends to 0. It also gives ρ divided by its independent-placement value 1 − e^(−E N). |

## Cluster result

With 2×10⁵ realisations at each of ε = 0, 1 and 3 on a 8192-point grid:
- ρ(t) falls from 0.14–0.20 at t = 1e−2 to 0.005–0.008 at t = 1e−4.
- At t ≤ 3e−5 there are at most a few realisations with N_t ≥ 2.
- Throughout, ρ stays within a factor 0.5–1.1 of its independent-placement value 1 − e^(−E N). The one exception is a single event at ε = 3, t = 1e−5.

Short bars therefore show no clustering beyond independent placement. E[N_t 1{N_t ≥ 2}] = ρ·E N_t then decays like t^(4/3), so it is o(t^(2/3)), as (30) states.

## Limits

The cluster check is d = 1 and p = 0 only, on finite trigonometric fields, and it is not a proof. There is no numerical test for p ≥ 1. HF §§8–11, the Main331 D1 documentary hold and Milnor's handle theorems themselves were not reviewed. No author or earlier reviewer program was run.

## Reproduce

Run each command as `python3 -I <script> <args>` with Python 3.11 and numpy 2.4.6:

| Output | Command |
|---|---|
| `hp_check_log.txt` | `hp_check.py` |
| `clus_eps0.json` | `simulate_cluster.py 0.0 200000 80 8192` |
| `clus_eps1.json` | `simulate_cluster.py 1.0 200000 81 8192` |
| `clus_eps3.json` | `simulate_cluster.py 3.0 200000 83 8192` |
| `cluster_log.txt` | `cluster_analysis.py .`, run in this folder |

`SHA256SUMS` covers every file here.
