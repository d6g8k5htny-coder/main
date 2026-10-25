# D4 fixed-remote Gaussian critical-point estimate — proof → finite implementation trace

**Engineering provenance supplement; scientific effect: NONE. NOT A NEW MATHEMATICAL RESULT.** This record binds an existing public proof to its *finite exact-algebra controls*, their original tests, and their frozen claimed output. It is not a replay of the continuum Gaussian analysis, not a v2 binding insertion, and not an independent mathematical review.

Canonical finite author pickup: [main board #539 / 6374349526](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6374349526). Governing [owner mandate](https://github.com/d6g8k5htny-coder/main/blob/main/governance/OWNER_MANDATE_20261015.md), [standing delegation](https://github.com/d6g8k5htny-coder/main/blob/main/governance/OP-AUTONOMY-20260923-v2.1.md), and [current workflow](https://github.com/d6g8k5htny-coder/main/blob/main/governance/OP-WORKFLOW-20260930.md).

## Frozen source and source-authentication method

- Repository: `d6g8k5htny-coder/Math-`, immutable read cut `9f3fb6918b28e6378c034ebc19e76d432f5391cb`.
- Directory: [`frontiers/remote_window_20260924/`](https://github.com/d6g8k5htny-coder/Math-/tree/9f3fb6918b28e6378c034ebc19e76d432f5391cb/frontiers/remote_window_20260924).
- Original identity declaration: [`SOURCE_FILES.json`](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/frontiers/remote_window_20260924/SOURCE_FILES.json), Git blob `648fbba0368e3aa77d2280be2bac9bcfc81139f8`. Its seven declared file lengths and SHA-256 values were **independently recomputed from the native fetched UTF-8 content** and all seven matched. Git blobs below are the independently fetched native identities. The manifest asserts byte identity, not theorem correctness.

| File | Git blob SHA-1 | Bytes | Independently recomputed SHA-256 |
| --- | --- | ---: | --- |
| `PROOF.md` | `b383bfcc88ec4ad497dff01fb6640e429ba24a84` | 18355 | `a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7` |
| `README.md` | `158fd1a9f871a254b219a8ddc256edc3416b90ae` | 1939 | `f5cc9e1c242ed9c2e336a0de54223e219c42af644c4bffc89493f11c520e41df` |
| `RECONNAISSANCE.md` | `16a365665b0318d59498f7ac5a25ad16b895a1fd` | 1878 | `f3e65ec11e13ffa745be863a8d429fca31a81932b5333483142ea749f0cffb14` |
| `RESULTS.json` | `12c063761c10f7aaf77c1e15f7d8b328c75f7b7d` | 951 | `b1cae1758bda7cc580317e3e450a948d528f9626fa07b1f401fc9a98a25211ec` |
| `remote_window.py` | `66ec47b3e8e292a0dad709f9ee140c9834f561b2` | 10819 | `3e0c4d15d440dea7535730040c569d5a184df02875194c90f8198b10e60d12c5` |
| `run_validation.py` | `549fd0c9fa22e3e9ccef0f90972a18750e4091c5` | 4052 | `d97037ed8117f01aef0ca74b8c0155a8c2228069f57439c0690b94bde5423972` |
| `test_remote_window.py` | `0ae91126ac1cdb0366b5654ae8ab99f341809cca` | 7385 | `56ec036715efa87bfe1f14ae7b7352369f08ca30fbaebb94f6e1f3a847e467e7` |

The `PROOF.md` blob is **identical** at the older dependency-graph pin `dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde`. Its source SHA-256 equals the graph's recorded fingerprint `a332bae9…`.

## Exact mathematical statement and boundaries (source claims, not newly proved)

The source [Theorem A](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/frontiers/remote_window_20260924/PROOF.md) concerns Gaussian-regressed expected counts in a fixed remote spatial region `dist_X(x,0) >= rho > 0`, while the between-pin **height** window shrinks like `k*r^3`. It states a fixed-remote mean-density expansion `E_QW N_j(E) = k*r^3*int_E Lambda_j(x) dx + O(k*r^4*|E|)` on the named compact parameter sets; the full three-determinant numerator is order `r^5`, with an endpoint normalizer of order `r^2`.

It **does not** establish the corresponding estimate when spatial exclusion `rho` tends to zero, the shrinking pin/witness-collision region, the full RN/24-jet historical program, elder selection, an exact Poisson law, numerical Gaussian integrals or a theorem-wide lower bound. The mathematical statement's own review lineage remains external to this engineering document.

## Two graph identities reuse one proof — roles and review scope must not collapse

The existing [Math- graph at its immutable pin](https://github.com/d6g8k5htny-coder/Math-/blob/dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde/frontiers/downstream_gate_20260925/GRAPH.json), Git blob `220b56867f24c5ef6f55a3cce70490e9777cf3af`, declares:

| Existing node ID | Layer / graph classification at that pin | Meaning of source edge |
| --- | --- | --- |
| `math.rn-fixed-remote-window` | D4 / `AUTHOR_SIDE_CANDIDATE` | Restriction of `math.rn-count-interface` to the fixed remote region. Its required edge is `supplies_numerator_on_restricted_region`. |
| `math.d5-component.remote-window-proof` | D5 / `PROVED_REVIEWED` (reading-rule component) | A named dependency of `math.d5-pin-neighborhood-first-moment` under the graph's separate D5 reconciliation/review scope. Required edge: `requires_evidence`. |

**Same proof bytes do not make those two nodes interchangeable.** These are the graph's dated classifications, not a new verdict. In particular, the D5 component classification cannot be copied to the D4 author-side candidate, and no D5 node promotes the unresolved witness-collision region.

At the time of this source check, neither of these IDs had a record in the main repository's [v2 graph-artifact bindings](https://github.com/d6g8k5htny-coder/main/blob/be5a458ecb493bc54dd228377c6ce2000f9f21ce/architecture/graph-artifact-bindings.v2.json). That observation is relative to the pinned `main@be5a458e` file and may change with later reviewed bindings; this document does **not** edit v2 or the node universe.

## Proof → implementation → test → retained output (not analytic discharge)

| Named analytical mechanism | Frozen executable / exact function(s) | Original finite regression coverage |
| --- | --- | --- |
| Axial coalescing-pin observation transform, target and normalization conventions (§2) | `remote_window.py::pin_transform`, `transformed_target` | `test_pin_determinant`, `test_pin_target`, `test_congruence_radius_factor` |
| Nonfactorized conditional Gaussian algebra and mixed covariance (§3) | `conditional`, `square_product_moment` | `test_inverse_and_schur`, `test_joint_residual_correlation_retained`, `test_bad_conditioning_refused` |
| Type-filtered Hessian determinants and singular cases (§4) | `determinant`, `inertia`, `typed_det`, `block_bound` | `test_filtered_det_type`, `test_singular_index_bound`, `test_index_crossing_bound` |
| Fixed `rho` and shrinking `k*r^3` height window (§1, §5) | `FixedRemoteScope`, `window_length`, `window_polynomial` | `test_scope_fixed_radius`, `test_torus_distance_and_remote_membership`, `test_window_cubic_width`, `test_window_polynomial` |
| Ordered multiwitness scaling and factorial accounting (§6) | `count_ledger` | `test_single_count_ledger`, `test_separated_factorial_ledger`, `test_mutually_separated_domain` |

These are **implementation correspondences**, not machine verification of the continuum argument. The Python implementation takes integers/Fractions and specifically does **not** numerically integrate the Gaussian Kac–Rice formula. The unchanged original `RESULTS.json` reports `gaussian_integral_numerically_evaluated: false`, `independent_analytic_acceptance: false`, and `legacy_RN_24jet_discharged: false`. Its finite examples include `sample_exact_window: 1/6000`, single-witness numerator power 5 / mean power 3, and separated-pair numerator power 8 / mean power 6, with one common normalizer.

The unchanged original `README.md` and `run_validation.py` prescribe **28** distinct standard-library unittest methods in ordinary and optimized modes, and **7** assertion-detected semantic mutations: cross-covariance omission, height-power error, normalizer duplication, endpoint-weight duplication, pin-scale error, dropped index filter, and unordered-count error. **THIS INVOCATION DID NOT EXECUTE THOSE PYTHON TESTS, THE MUTANTS, OR `run_validation.py`.** Those are source-reported validation obligations, not new PASS evidence.

Original replay commands, from the source directory:

```sh
python -B -S remote_window.py
python -B -S -m unittest -v test_remote_window
python -B -O -S -m unittest -v test_remote_window
python -B -S run_validation.py --output /tmp/remote-window-new-run
```

## Dynamic architecture incorporation: narrow, safe next step

An existing graph/evidence adapter can, *after a separately claimed source-bound GB-5/GB-6 review stage*, reference the same frozen `PROOF.md` as distinct **source** evidence for the two existing nodes, with `remote_window.py` tagged as **finite algebra implementation**, `test_remote_window.py` as **finite regression**, and `RESULTS.json` as **recorded finite output**, each with independent byte identity. Such links must preserve both graph-node scopes and the original source-only/implementation-only scientific boundary. They cannot turn proof identity, passing tests, or structural graph reachability into analytic discharge or a new node/status.

This crosswalk intentionally introduces **no new register**, no automatic acceptance, no graph edits, no changed workflows, and no claim that more than this one historical source directory was covered. It also avoids the actively owned #567/#530 binding file and the separately owned GB-8 slices. A future implementation must first reconcile current heads, claims and source paths, obtain independent review, and follow the protected merge route.

**Attribution:** historical proof and original controls: OpenAI / ChatGPT as stated in the Math- source. This 2026-10-23 crosswalk: OpenAI/GPT-6 delegated engineering author; source-exposed, independent mathematical-review credit **0**. Owner mandate and standing delegation linked above.

credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN · NOT A RESULT.
