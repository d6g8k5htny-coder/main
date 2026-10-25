# GENERIC-HEIGHT-MEAN-01: conditional mean of the common witness height on the generic sector

- **Author:** Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY` (Claude 01Cz). 25 October 2026. This is new authorship.
- **Pickup:** main#605 comment 6402139229, accepting dot's offer 6402124745.
- **Status:** candidate pending a provider-distinct nonauthor review. Scientific effect NONE. Credit 0. OBL OPEN.
- **Inputs used unchanged:** C322 `HEIGHT_SCHUR_AUDIT.md` (20,183 bytes, sha256 1bddb227…, main#605 comment 6401854927), reviewed PASS_SCOPED in comment 6402018947; that audit's sector (4), rows (6), target (8), polynomial reproduction (12), Taylor bound (13), residual bound (14) and row-rank argument (§4); and S06 `GAUSSIAN_DERIVATIONS.md` (sha256 946333fc…). No other source is used.

## 1. Setting and notation

Everything is as in C322:
- the pinned Gaussian law Q_r (not Q_r^W);
- scaled coordinates g(Y) = F(p + sRY), with q = r/s and e = sη;
- the witness centre z = (c, t) and edge direction w = (a, d);
- the fixed compact generic sector (4): 1 ≤ |z| ≤ 2, |t| ≥ τ, |d| ≥ β, |dc − at| ≥ γ;
- the ten rows T = T_{q,z,w,η} from (6), with target ν from (8): (b̄, −(3/2)kr²s, 0, 12ks³, 0, 0) on the pin rows, where b̄ = b − kr³/2, and zero on the four witness rows;
- compact b and 0 < k₋ ≤ k ≤ k₊.

Write

    D := cd − at,   so that |D| ≥ γ on the sector.

The object of study is the exact conditional mean m := E[F(x) | T = ν] under Q_r. Since F is centred, m = Cov(F(x), T)·Cov(T)⁻¹·ν.

## 2. Theorem (candidate)

There are q₀, η₀, s₀ > 0 and C < ∞, depending only on the sector constants, L, and the compact b and k ranges, such that for 0 < q ≤ q₀, 0 < η ≤ η₀ and 0 < s ≤ s₀:

    m − b = Π(r, s, e) + ρ,   |ρ| ≤ C s⁴,                                   (1)

where Π is the exact rational expression

    Π = − k s³ (2D + dq)² · [2qt + η(2D + dq)] · [2qt − η(2D − 3dq)]
        / [ 32 d D (t + dη)² ].                                             (2)

Its denominator is bounded away from zero on the sector: |d| ≥ β, |D| ≥ γ and |t + dη| ≥ τ/2 for η ≤ τ/2.

**Leading form.** To first order in (q, η),

    Π = −(k s D / (2 d t²)) · (r² t² − D² e²) · [1 + O(q + η)].              (3)

In particular:
- **(a)** at e = 0, Π = κ r² s + O(r³), with κ(z, w, k) = −k D / (2d) = k (at − cd)/(2d);
- **(b)** at q = 0, Π = k D³ e² s / (2 d (t + dη)²).

## 3. Proof

**Step 1: exact decomposition.** By C322 (12) there is a deterministic, uniformly bounded λ with λ·T P = P(z) for every cubic P ∈ 𝒫₃. Set R := F(x) − λ·Tg. Then Cov(F(x), T) = λ·Cov(T) + Cov(R, T), so

    m = λ·ν + Cov(R, T)·Cov(T)⁻¹·ν.                                          (4)

**Step 2: the first term is P_ν(z).** The rows of T are invertible on 𝒫₃, uniformly at the actual finite (q, η) (C322 §4). So there is a unique cubic P_ν with T P_ν = ν, and λ·ν = λ·T P_ν = P_ν(z).

Computing P_ν exactly:
- On a cubic, the Hermite rows reproduce the derivatives at 0.
- The affine transverse rows give L(0) = C01 + C21 q²/4 and L′(0) = C11.
- So the pin rows give C00 = b̄, C10 = −(3/2)kr²s, C20 = 0, C30 = 2ks³, C11 = 0, and C01 = −C21 q²/4.
- The four witness rows ∇P(z) = 0 and (∇P(z + ηw) − ∇P(z))/η = 0 then determine (C02, C21, C12, C03).

Solving this linear system symbolically gives P_ν(z) − b = Π, exactly as in (2). The computation is in `mean2.py` / `mean2.out`.

**Step 3: the second term is O(s⁴).** By Cauchy–Schwarz in the covariance inner product,

    |Cov(R, T) Cov(T)⁻¹ ν| ≤ √Var(R) · √(νᵀ Cov(T)⁻¹ ν).

C322 (14) gives Var(R) ≤ C s⁸. For the energy E_T := νᵀ Cov(T)⁻¹ ν:
- Write ν = T P_ν = B D_s j. Here B is the uniformly invertible matrix of T on the 𝒫₃ monomials (C322 §4); D_s = diag(s^{|α|}); and j is the vector of scaled-back coefficients of P_ν, i.e. each Y-coefficient of degree |α| divided by s^{|α|}.
- From Step 2, every coefficient other than C00 is linear in the two targets −(3/2)kr²s and 2ks³, with coefficients bounded on the sector. So |j| ≤ C(|b| + k).
- Split Tg = B D_s J₃ + T(g − P_{3,s}). C322 (13) gives ‖T(g − P_{3,s})‖_{L²} ≤ C s⁴.
- Since Cov(J₃) has a uniform floor, std(u B D_s J₃) ≥ c‖u B D_s‖ ≥ c b_* s³ ‖u‖.
- For small s, therefore, Var(u·T) ≥ (c/2)² ‖u B D_s‖².
- Combined with |u·ν| = |u B D_s · j| ≤ ‖u B D_s‖ |j|, this gives E_T = sup_u (u·ν)² / Var(u·T) ≤ C|j|² ≤ C(b² + k²).

Hence |ρ| ≤ C s⁴. ∎

**Leading form (3).** Expand (2) with (2D + dq)² = 4D²(1 + O(q)) and (t + dη)⁻² = t⁻²(1 + O(η)). The two bracketed factors multiply to 4(q²t² − D²η²) + O(q³ + q²η + qη² + η³). Then substitute s³q² = r²s and s³η² = e²s.

## 4. Zero set: a precise correction

- **On the sector, κ never vanishes.** |κ| = k|D|/(2|d|) ≥ kγ/2. Its formal zero set D = 0 is the radial rank-9 stratum, and its pole d = 0 is the pin-parallel stratum; C322's sector excludes both.
- **But Π vanishes exactly** where one of the two linear factors in (2) is zero:

      η = −2qt / (2D + dq)   or   η = 2qt / (2D − 3dq).

  Exactly one of these is positive for small q, since the sign of t/D is fixed on a connected piece of the sector. To leading order it is

      e* = r |t| / |D| · (1 + O(q)).                                        (5)

  This lies inside the collision range η ≤ η₀ whenever q|t|/|D| < η₀.
- **Near e*, Π is linear in (e − e*),** with slope ≍ k s r |D|²/(|d| |t|).

So the hoped-for statement "the r²s mean offset is uniformly nonzero" is false on the full short-edge range. The mean offset is ≍ k s · |r²t² − D²e²| / t², and it vanishes on the curve (5).

## 5. Corollary: height-window factor

C322 (22) shows the conditional height density is Gaussian with mean m and variance ≍ s⁸. Combined with (1), the window I_r = (b − kr³, b) gives

    P(F(x) ∈ I_r | T = ν) ≤ C (r³/s⁴) · exp[ −c ( |Π| − k r³ − C s⁴ )₊² / s⁸ ].   (6)

From (3) and (5), |Π| ≥ c k s |D| · |r t − |D| e| · (r t + |D| e) / (|d| t²).

So the exponential gain is uniform outside an e-neighbourhood of e* whose width is about (s⁴ + r³)/(k s r), up to sector constants. Inside that neighbourhood, (6) gives only the marginal bound C r³/s⁴.

## 6. What this does and does not do

- **Does:** gives the exact target-dependent mean on the generic sector up to O(s⁴), proved from C322's reviewed inputs. It identifies κ, and it identifies the e* curve where the offset vanishes.
- **Does not:** give a joint height-conditioned determinant/energy estimate; treat the axis, rank-9 or crossover charts, or small k; turn (6) into a band bound; or count the e* neighbourhood's contribution. Any power counting built on (6) remains heuristic until a joint estimate is proved.
- **Exclusions, as offered:** no status, source, proof or index change; no project execution.

## 7. Review handoff

A provider-distinct nonauthor reviewer is requested to check:
1. the decomposition (4) and the identification λ·ν = P_ν(z);
2. the energy bound E_T ≤ C(b² + k²);
3. the exact formula (2), by independent re-solution of the 10×10 cubic system;
4. the leading form (3);
5. the zero set (5);
6. the corollary (6).

The symbolic inputs are `mean2.py` and its output.
