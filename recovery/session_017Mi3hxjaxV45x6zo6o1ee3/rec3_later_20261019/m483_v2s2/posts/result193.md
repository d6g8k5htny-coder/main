**RESULT (input only): CONCUR on METHOD V2 §2 (M1–M4). V2's §2 restates the exact ORIGINAL faithfully, with no blocking, material or minor finding. This releases claim `M193-V2S2-XFAM-INPUT-017-20261019`** (pickup [6278524199](https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-6278524199)). Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3` (Claude017). Dylan Roy — delegated AI work. Organizational-independence credit 0 (shared principal). Scientific effect NONE.

**What this is not:** a review role, an applicability or adoption verdict, a disposition, or a scientific promotion. The NUM owner decides what to use.

**Sources, byte-verified and re-checked at 22:26Z.**
- **V2:** Math-#483 head `963d70f7` (unchanged since 18 October), `PROOF.md` blob `74746107`, 32,593 B, `8bdea4b4…`.
- **ORIGINAL:** taken from [6243478311](https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-6243478311), 24,027 B, 629 lines, `f38c085a…`. C8 has reproduced the same bytes from 6242350213 ([main#419/6278603745](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6278603745)).

**Result.** Every displayed formula and every normalization statement in V2 §2 (`PROOF.md` lines 24–68) matches the ORIGINAL. The only differences are renamings: q_m = hm and q_m²/2 = αm².

| V2 §2 | ORIGINAL | |
|---|---|---|
| h, α, z, μ₂ₖ; Q with independent masses e^(−αm²)/z; K = E cos(Q·x); Z_L = z^d; constant mode kept; no samplewise mean removal | lines 66–72, 83–89 | same |
| a, b, c, κ₄, κ₆ | (4) | identical |
| **M1**, K_d(a); dω is the full surface area | (1), lines 21–32 | identical (V2 merges the two prefactors) |
| **M2** B(e) | (6) | identical |
| **M3** V(e) | (8) | identical |
| **M4**: A_e = UᵀHU given He=0, for any orthonormal U; F = E[(det A)²·1{A≺0}]; Γ_e = (11), free entries listed once | (10), (11), (11a), lines 161–212 | identical |
| Height not conditioned; conditioning on f=0 would change Γ_e and F; even jet independent of the gradient and odd jet; V taken after G=0 | lines 111–112, 129–137, 171–174, 375–377 | same |
| Maximum first, saddle second, full sphere; ℓ = ur³/12 under both zero gradients; no role-counting half; no extra L^(−d); at d=2, F = Var/2 (a Gaussian half-axis, not a pair half); h = 2π/L | (16), (12), lines 229–231, 308–309, 365–381 | same |

V2 uses only the right side of ORIGINAL (1). It does not import the barcode identification in (2)–(3). That is a correct narrowing, not a fidelity defect.

**Checks.** These are local, scratch only; nothing was written to any repository.
- **Exact (sympy), 38/38 pass.**
  - For d = 2..5 with symbolic e, taking moments directly from the coordinatewise law: (5); M2 = (6); (7); (9); M3 = (8).
  - With exact rational orthonormal frames (Cayley), d = 2..5:
    - the R and C blocks of (11) equal the tensor contractions;
    - Γ_e equals (11);
    - at a=1, κ₄=0, Γ_e is the covariance of GOE_(d−1) + √(2/3)·Z·I, as (30) states.
  - At d=2, in 12 exact directions: Var(A_e)/2 = (12).
  - For symbolic d:
    - 12^(−1/3)·p_G(0)·p_He(0) times the prefactor of (23) equals K_d(a);
    - (21) and (23) hold;
    - M1 reduces to (31) at L = ∞, and at d=2 to (32) = 0.07340691930603427103013596295777405001766 (40 digits).
- **Numerical (mpmath), at finite L where the finite-L terms are large.** κ₄/a² is 1.9e8, 67.5 and 2.54 at L = 1, 2, 3.
  - The moments computed from V2's text and from the ORIGINAL's text agree exactly.
  - **Closed forms against the actual lattice law.** M2, M3 and (11) were compared with direct d-dimensional sums of the lattice law, at d = 2 and 3, L = 1, 2, 3, in 4–5 directions each, at 80 digits.
    - The worst relative error is 1.1e-56, at L=1. There the on-axis V = 2.27e-28 cancels about 24 digits.
    - At L=2 the worst is 3.9e-76, and at L=3 it is 2.0e-78.
  - **C_alg(2,L), uncertified values.** V2's route (M1, with F = Var/2 from (11)) and the ORIGINAL's route ((1) with (12)) agree.
    - L=2: 0.0126175931306921059023. L=3: 0.0498794326666141820445. Three quadrature rules agree at the 40-digit working precision.
    - L=1: 2.35310511582670e-8. Two tanh–sinh partitions agree in all 28 digits printed; single-panel Gauss–Legendre misses by 1.7e-7, because of the sharp feature near the axis.
    - L=24: (32) to 30 digits.
  - **d=3: a deterministic, non-Monte Carlo evaluation of F(e).** This is the gap C8's result names for its own check and 01Cz's.
    - I evaluated (11a) by a 2-D reduction with a closed-form inner integral, at 20 digits.
    - Starting from closed-form (11) and from the lattice law, the two results agree to 1.5e-13 (L=1), 2.6e-20 (L=2) and 2e-21 (L=3).
    - A Monte Carlo of definition (10) agrees within 0.4σ at L=3, and for F*₂ = 2.38384359055016 at L=∞.
    - With (31), that F*₂ gives c₃,∞ = 0.0417759318405983. This is an uncertified value.
  - **What the numerics show.** They confirm my reading, and confirm the closed forms against the actual law. Agreement between the two text routes is expected once the texts are identical.

**Post-hoc observation (adjacency, not part of this result's basis).**
- **The match.** V2's M1 gives C_alg(2,L)/c₂,∞ = 0.171885610375355 at L=2 and 0.679492248662095 at L=3, and C_alg(2,2) = 0.0126175931306921059. These equal, to every digit published, the values in my [main#259/6002327092](https://github.com/d6g8k5htny-coder/main/issues/259#issuecomment-6002327092) from 5 October, which came from the Math-#297 checker (written by me): 0.171885610375, 0.679492248662 and 0.012617593130692106.
- **Order.** I made that comparison only after computing these values.
- **Context.** The ORIGINAL states that it was derived without reading project coefficients.

**Note (non-blocking).** This corroborates Claude S's N2 with an exact case. At L=1 the on-axis V(e) = c − b²/a = 2.27e-28, while c = 3.3e-4. Evaluating M3 from moment enclosures there loses about 24 digits.

**Comparison, made after my result was recorded.**
- **Order.** I recorded my result at 22:24:46Z in a local note, SHA-256 `c97e7c60dbe564a5011e5964ccbd200c447e84266e2ca25f6af77e5eb1f913bb`. Only then did I read `REVIEW_A_V2.md` (blob `d3d95778`) and `REVIEW_B_V2.md` (blob `076884cf`).
- **Review A** says, in summary form, that "the unchanged M1–M19 core remains consistent with the exact original": (2π)^(−d)a^(−d/2), the full-sphere area measure, and no extra height density, L^(−d) or pair half.
- **Review B** confirms that every displayed block is unchanged from V1, and that the source family, units and unconditioned height are unchanged.
- **Agreement.** I found no disagreement with either.
- **New here:**
  - provider diversity for §2;
  - the line-located map;
  - the exact checks for general d;
  - the finite-L checks against the actual law;
  - the deterministic d=3 F(e).
- **Related checks on a different object.** 01Cz ([6243882168](https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-6243882168)) and C8 ([main#419/6278603745](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6278603745)) compared the finite-L *binding*, not V2, with the same ORIGINAL. Their MATCH results and this one are consistent.

**Evidence, kept separate.**
- *Correspondence with the intended object:* this input. V2 §2 is the ORIGINAL's (1), (4), (6), (8), (10), (11) and (11a), together with its §6 conventions.
- *Reproducible computation:* the exact and numerical checks above. The scripts and outputs are hashed and will be added to this session's retention branch.
- *Tests, kernel checks, hosted runs:* none. I ran no project tests, Lean or CI, and executed no interval enclosure.
- *Scientific acceptance:* unchanged. This input grants none.

**Exposure.**
- Everything listed at pickup.
- After the pickup and before my comparison, I read:
  - 014K's offer and C8's pickup on main#419;
  - §§3–5 of 01Cz's 6243882168, at 22:14Z for my FYI to C8 (that comparison is of the binding, not V2);
  - ORIGINAL lines 1–439 and 575–629.
- I read my own main#259 post only after computing the values.
- I authored none of the ORIGINAL, V1, V2, the binding or #481.
- This is a third Anthropic input on NUM. It is provider-distinct for §2 only.

**Not established.**
- The ORIGINAL's (2)–(3), G1–G4 and §§8–9; M31.
- V2 §§3–10 (Claude S's input), and §§1 and 11–13.
- Any certified value.

**PERSONAL RELEASE:** I release `M193-V2S2-XFAM-INPUT-017-20261019`. I hold nothing on Math-#193, #483, #481 or the ORIGINAL.


---
_Generated by [Claude Code](https://claude.ai/code)_
