# Quantifying the cost of rare clipping events

Every real planar Fourier polynomial with frequencies in the square
[-K,K]^2 has at most **16 K² positive finite H₀ bars**, including
degenerate coefficient choices. The proof uses a conservative Bézout
bound for Morse perturbations and persistence stability.

Under the previously declared independent uniform-word model, this
converts the clipping failure probability into the following absolute
**expected-count correction** for the finite Gaussian polynomial F64,K:

| Cutoff K | Uniform bar cap | Coupling error at most | Clipping probability at most | Expected-count correction at most |
|---|---:|---|---|---|
| 1 | 16 | 0.000000000000000000068095713684 | 0.000000000000011367609938 | 0.000000000000181881759008 |
| 24 | 9216 | 0.000000000000000001465715649537 | 0.000000000003032625717894 | 0.000000027948678616102069 |
| 32 | 16384 | 0.000000000000000001465732013400 | 0.000000000005336461331986 | 0.000000087432582463256044 |

For a positive lifetime bin [a,b), the exact inequalities are
`max(0, E[L] − correction) ≤ E[N_F64,K([a,b))] ≤ min(16 K², E[U] + correction)`.
For polynomial counts, L uses the contracted bin. U uses the expanded
bin only when a > 2 rho; otherwise U = 16 K². For the bounded grid
observables in the proof, the expanded upper count requires
a > 2(epsilon + rho); otherwise U = 16 K². Unresolved grids are kept
with [L,U]=[0,16 K²]. Their outcomes must not be discarded.

**No expectations E[L] or E[U] have been evaluated here.** This delivery
generates no random field or barcode. Sample means need their own
sampling-error argument. The previous deterministic cutoff1 example
does not become a Gaussian observation.

This removes the missing exceptional-count factor for the finite-cutoff
clipping comparison. The infinite smooth field has no such fixed-cutoff
cap. Its exceptional-count moments, the spectral-tail step in expected
counts, and the numerical lifetime remainder remain separate.

[Proof and exact input assumptions](../../FINITE_COUNT_LOSS.md) ·
[Certificate](CERTIFICATE.json) · [Source receipt](RUN.json)

```sh
python -B -S experiments/periodic_h0/finite_count_loss_certificate.py --verify experiments/periodic_h0/results/finite_count_loss1
```

Uniform planar finite-polynomial H0 bar-count bound and finite-cutoff expected-count clipping-loss budgets under the stated IID word-input model. No observed Gaussian ensemble, infinite-field expected count, historical coupling, lifetime-law confirmation or formal proof certified.
