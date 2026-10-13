# Exact samples in shrinking lifetime bins

[Experiment](README.md) · [Deterministic approximation](APPROXIMATION.md) · [Manuscript M4–M5](../../docs/research-translation/20260930/MANUSCRIPT.md#2-the-assembled-planar-statement)

This record gives a reading route to the [complete original proof](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-5976299711)
and its [nonauthor technical review](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-5976527971).
It explains when exact periodic grid samples recover the expected continuum
count in shrinking bins. The original proof and review bodies, identified
below, remain authoritative; this page does not replace either full text.

**Recorded verdict: `PASS_TECHNICAL_SCOPED`, with no required amendment.**
The review checks the implication from the pinned interfaces, rather than
independently re-proving or accepting their complete parent theorem chain.
The actual reviewer is OpenAI/Codex `/root/agent_handoff_review`, on the same
account, with **zero organizational-independence credit**. Source publication
does not change scientific status, formal alignment, or any acceptance flag.

## Exact statement

Let `f` be the smooth periodized Gaussian field used by M4 on the square torus
of fixed side **`L = 24`**. On the periodic `n × n` grid,
`h = L/n`, use the **exact vertex samples of `f`** and the vertex cubical
superlevel filtration defined in the approximation note. Let `N(I)` count
finite ordinary continuum H0 bars with lifetime in `I`, and let `N_h(I)` be
the corresponding cubical count. Counts retain multiplicity and exclude the
essential component.

Fix `0 < λ < μ < ∞` and a deterministic schedule `τ_h ↓ 0`. Put

```text
I_h = [λ τ_h, μ τ_h).
```

Under the inherited M4–M5 density interface, if

```text
h² sqrt(log(1/h)) = o(τ_h),
```

then

```text
E |N_h(I_h) - N(I_h)| / E N(I_h) → 0,
E N_h(I_h) / E N(I_h) → 1.
```

The denominator is nonzero for all sufficiently small `h`. Its inherited
bin-mass asymptotic, with area normalization `L² = 576`, is

```text
L⁻² E N(I_h)
  = (3/2)c₂,₂₄(μ^(2/3) - λ^(2/3)) τ_h^(2/3) + O(τ_h).
```

This is an **ensemble expectation** result for exact samples of the full
field. It supplies neither a realization-by-realization relative error nor
a statistical confidence interval.

## Dependencies and scope

| Interface | What is required or retained |
|---|---|
| Field and law | Fixed side-24 planar torus and the unconditioned smooth periodized Gaussian ensemble used by M4. |
| Sampling and filtration | Exact periodic vertex samples; cells enter at the minimum of their corner values. The approximation note's adaptive PL representative is a proof device for this cubical barcode, not a change to a fixed-diagonal PL observable. |
| Density input | The M4–M5 finite ordinary H0 lifetime-density interface, its positive coefficient, and its existential `C, ℓ_*` retain their recorded hypotheses and source scope. |
| Bin schedule | Fixed `0 < λ < μ < ∞`, deterministic `τ_h ↓ 0`, and `h² sqrt(log(1/h)) = o(τ_h)`. Half-open endpoints remain `[λ τ_h, μ τ_h)`. |
| Gaussian tail input | Hessian-supremum tails and the repaired whole-field marked one-point Kac–Rice interface control the total critical count on the bad event; count/event independence is not assumed. |
| Excluded | spectral truncation; FFT realization error; floating-point sampling; roundoff; persistence-library arithmetic; post hoc bins; confidence intervals; evaluated finite-grid threshold; evaluated M4/Borell/Kac–Rice constants; current-pilot certificate. |

In particular, the retained pilot uses a truncated field, numerical sampling,
and a persistence implementation with separate error obligations. This lemma
does not certify those historical observations or change their inconclusive
coefficient comparison. It does not locate a usable finite grid or evaluate
`C, ℓ_*` or the Gaussian tail constants.

## Proof map

Read the [full proof](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-5976299711)
for the argument and the [full review](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-5976527971)
for the checked endpoint, regression, and rate comparisons.

| Step in the original proof | Role in the conclusion |
|---|---|
| 1. Sharp-bin matching | With lifetime displacement `d = 2ε` and lower endpoint `a > d`, every discordant matched pair is charged injectively to the continuum boundary strips `[a-d,a+d) ∪ [b-d,b+d)`. |
| 2. Exact-sample displacement | The cubical approximation gives `d ≤ h² H(f)/2`, where `H(f)` is the supremum of the Hessian operator norm. On `H(f) ≤ A sqrt(log(1/h))`, use the deterministic displacement `d_h = (A/2)h² sqrt(log(1/h))`. |
| 3. Expected boundary mass | M4 bounds the unrestricted expected strip count by `O(d_h τ_h^(-1/3)) + O(d_h)` per unit area. Restricting to the good event requires no independence. |
| 4. Bad-event integration | Conditional Gaussian regression, Borell tails, and the repaired marked Kac–Rice formula bound the continuum count contribution. Together with at most `n²−1` finite cubical bars, the remaining contribution is `O((h⁻²+1) exp(-c A² log(1/h)))`. |
| 5. Relative comparison | Divide by the M4–M5 bin mass `Θ(τ_h^(2/3))`; the schedule makes the good-event terms vanish. A sufficiently large fixed `A` makes the bad-event term vanish as well. |

The original proof expressly does **not** consume the determinant-tilted Palm
window-count input identified there as Math C6: its law and observable differ
from this unconditioned barcode count. The review distinguishes that objection
from a separate Fourier total-count theorem that could supply an alternate
route. That alternate route is not a dependency of the delivered proof.

## Frozen evidence identities

The following identify the exact decoded GitHub REST **comment body**, not
the surrounding JSON, rendered HTML, or a repository copy with an added newline.
The permanent comment URL identifies the record; the byte count and SHA-256
bind the reviewed version if that record is edited later.

| Record | Canonical public text | UTF-8 bytes | SHA-256 | Terminal newline |
|---|---|---:|---|---|
| Original proof | [main#229 / 5976299711](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-5976299711) | `6104` | `819bc7c7fd637883ca74626b01d777d40233f6b710b6f26b5d370648e09a945d` | None |
| Nonauthor review | [main#229 / 5976527971](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-5976527971) | `9103` | `ad6f20ab98ce288c2ab54f88653df908015a599f30a329f3c1f12eff93e7f16f` | One LF |

These are the consumed source cuts. Contextual local links elsewhere on this
page aid navigation; the pinned records below bind the reviewed interfaces.

| Consumed interface | Immutable source | Git blob | Bytes / SHA-256 |
|---|---|---|---|
| Cubical approximation and endpoint matching | [main / APPROXIMATION.md](https://github.com/d6g8k5htny-coder/main/blob/9093629769f40db76f1311d8ee920d5a82a38b89/experiments/periodic_h0/APPROXIMATION.md) | `137205c149336c98dd66b0aac8d6a0374f4fb8d9` | `17970` / `dae4e01b35602077442796a1150294cce3eb544f2d0c2297f8fa7aa6ab16caef` |
| M4–M5 density and measure definitions | [main / MANUSCRIPT.md](https://github.com/d6g8k5htny-coder/main/blob/9093629769f40db76f1311d8ee920d5a82a38b89/docs/research-translation/20260930/MANUSCRIPT.md) | `0e8c199d2eb48c713682d612f380d8db279c931c` | `20601` / `012d288f7bcd341174fc2670cc10ee1a59fd857f4e5495acc961ad79376ab434` |
| Positive spectrum and whole-field Kac–Rice | [Math- / UNIFORM_MATRIX_CAP_AND_LIFETIME.md](https://github.com/d6g8k5htny-coder/Math-/blob/8404169d33317cc5f01ade17c829c8036d33ea2a/imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md) | `dfed3b8d318a3ab1950957f393307733a4bef3f2` | `40261` / `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7` |
| Recorded Borel repair | [Math- / REPAIR.md](https://github.com/d6g8k5htny-coder/Math-/blob/8404169d33317cc5f01ade17c829c8036d33ea2a/reviews/d1_section9_borel_repair_20260925/REPAIR.md) | `fe9b9ce4999908bb3814b500ee2d0ceb0c6f704a` | `9062` / `845abf9f9c99d672c2a10a887b5a2e7206a3d2de3d876f35f75ff6e2dc13e62f` |

The offline publication test checks these reading routes, frozen source
bindings, and explicit exclusions. It does not retrieve the public bodies,
prove the lemma, certify the pilot, or promote the consumed source statuses.
