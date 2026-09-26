# LB-RATE / KIMI-THM-023 landing review

Review date: 2026-09-26.
Reviewer: xAI / Grok 4.6 (source-exposed to the 2026-08-04 addendum packet, the 2026-09-26 Drive landing, GP-LB-REC-001, and AO48-AUD-064). Same-provider technical pass. **Zero organizational-independence credit.**

**Disposition: HOLD — CONDITIONAL ASSEMBLY DRAFT. Do not accept `c = 0.9144` as a proved uniform theorem constant.**

This is not D0–D7 3D lifetime work and does not inherit SIDE24 coefficient status.

---

## 1. Exact objects

Normalized 2026-08-04 addendum carriers (hashes match GP-LB-REC-001):

| Object | Bytes | Whole-file SHA-256 | Body SHA-256 |
|---|---:|---|---|
| KIMI-THM-023.md | 14,936 | `61e14810c55f688561bacd5a62bfa3e0872e0672fa387161c60a0455be65f92a` | `fedcc4b67a792efd5d43496b9a6f26c3412a0e863aac038f3e2aa37654ded3a7` |
| KIMI-AUD-023.md | 5,448 | `df6565acf9231b3ae41e0f5f27f55384d786119013bebc06b79bd87a8e70e874` | `23377bd4257832490cf9e5e863dc7afba7ea1fc4f719b2abfe1e02081cc1fe3d` |
| KIMI-AUD-024.md | 3,369 | `309aefb8bc5c6dab656cf2d6ddae0a570e4ba593a3e327a2008f785c97bdc47c` | `0633402b95a586331848bf12bceed0db519f3425f7e2f8b93ceb7110e514d25f` |
| MANIFEST.sha256 | 1,199 | `73e20364804275cadb84f5452dbec95c3a9f7c171fae79a17670e731274a9a48` | — |

Identity split (do not collapse):

- THM-023 cites AUD-023 as `61f2b702…`. That is AUD-023 **v1.0** body. The landed file above is the normalized carrier of the same campaign. AUD-023 v1.1 body is `a7decb6f4b8b3a896d7e0aadddcd698853e57a9a253ef787173bb3c702136a32`.
- C031 is dual-hashed: `e7998ef0d17d951bc89978f9fe32e510019059dd0650c8f0e0ae33273f40f32e` (THM/AUD-023 cite) vs `e165821b…` (LB-1 / Drive `C031_LBRATE_Integration.md`).

Drive landing of these exact bytes: 2026-09-26T19:30Z, owner Dylan Roy, Drive root. Operator card `README925.txt` is a landing label, not a status upgrade.

---

## 2. Exact estimand

Field: exact normalized periodized Bargmann–Fock on the flat 2-torus $T^2_{24}$, covariance the periodization of $K(u)=e^{-|u|^2/2}$, level $b=6/5$.

Estimand: under the typed maximum–saddle pair-Palm law at separation $r$,

$$
1-q(r,\,6/5).
$$

Claim printed by the addendum: there exist $r_0>0$ and $c>0$ such that

$$
1-q(r,\,6/5)\;\ge\;c\cdot r^3\qquad\text{for all }0<r\le r_0,
$$

with candidate intercept

$$
c \;=\; (1-0.0334)\cdot 0.946 \;=\; 0.9666\times 0.946 \;=\; 0.9144036.
$$

Domain firewall: this object is **not** the 3-dimensional Side-24 first-moment density $\nu_{(3,24)}(\ell)=c_{(3,24)}\ell^{-1/3}(1+o(1))$. Gap convention in the packet: $\ell=r^3/6$.

---

## 3. Arithmetic that holds

- $1-0.0334=0.9666$.
- $0.9666\times 0.946=0.9144036$. Printed $0.9144$ is a 4-decimal truncation, not a round-up.
- WP component sum at $r=0.025$: $1.95\times 10^{-6}+1.37\times 10^{-6}=3.32\times 10^{-6}=0.21248\,r^3$, consistent with printed $0.213\,r^3$ within rounding.
- R2 printed line with denominator $(9-6.25)$ evaluates to $5.18\cdot(\ell/2)$, not the printed $2.1\cdot(\ell/2)$.
- Corrected reading $(9-2.25)$: $0.76\times(18.75/6.75)=2.1\overline{1}\cdot(\ell/2)=0.1759\overline{16}\,r^3$.
- Finite-rung measured $\Lambda$-window: $C^*(0.025)=0.946$.
- C031 derived-on-grid limit (Drive C031 file): $C^*_\infty=0.9091\pm 0.038$ (quad) $\pm 0.009$ (shell). Then $0.9666\times 0.9091=0.87853686$, not $0.9144$. The printed intercept mixes a theorem-grade far term with a finite-rung measured $\Lambda$ constant.

---

## 4. Findings

### F1. Gap: printed R2 denominator

The printed factor $(9-6.25)$ is a ledger defect. The assembly constant $2.1$ stands only under the corrected reading $(9-2.25)$. Falsifier: a C030 source showing collar radius $2.5$ on both zones was intended.

Kind: gap in the printed derivation, not a counterexample to an $O(r^3)$ class.

### F2. Gap + decertification: Lemma WP quantitative upper bound

Two independent defects.

**Intensity (LB-1 §9).** At $r=0.025$, rigidity-zone integral $\approx 1.30\times 10^{-5}$ (hot spot near $(-0.05,-0.575)$ / $(-0.04,-0.58)$, about $1.33\times 10^{-4}$ per unit area). Printed rigidity figure $\sim 3\times 10^{-15}$. The certified subregion is about $3.9$ times the printed *total* WP budget $0.213\,r^3=3.328\times 10^{-6}$, or about $0.830\,r^3$ at that rung.

**Cauchy–Schwarz (AO48-AUD-064, DEF-WP-CS-01; not re-derived this session).** The implemented envelope used a factor $\min(\mathrm{Cantelli},P_W)$ where the displayed CS bound on the indicator requires $\sqrt{\min(\mathrm{Cantelli},P_W)}$. The printed computation is therefore not a proved upper bound. Quantitative WP ub **DECERTIFIED**. Remainder order is not proved $O(r^3)$.

AUD-023 item 4 “VERIFIED at ledger arithmetic” is true of the component sum only. C029-(1) “DISCHARGED at derived-grade-with-measured-constants” is **SUPERSEDED**.

### F3. Scope mismatch: `0.9144` is not a proved uniform constant

Even ignoring F2 and the unquantified outer remainder, the displayed finite-$r$ coefficient is $(0.9666-0.3889\,r^3)\times 0.946$. It equals $\approx 0.91440$ at $r=0.025$ and $\approx 0.91436$ at $r=0.05$, both below $0.9144$. Reaching $0.9144$ requires $r<0.02139\ldots$. LB-2 is certified only at the frozen rungs $0.025$ and $0.05$. Measured $c_\Lambda=0.946$ is not a rigorous $r\to 0$ limit; C031’s own on-grid limit is $\approx 0.909$.

Kind: scope / hypothesis mismatch with the consumer banner “proved floor $0.9144$”.

---

## 5. Inherited statuses (do not upgrade)

From GP-LB-REC-001 (2026-08-04), which intaked these exact normalized bytes:

- KIMI-THM-023: **HOLD — CONDITIONAL ASSEMBLY DRAFT**
- KIMI-AUD-023: **AMEND REQUIRED**
- KIMI-AUD-024: **HOLD — CONDITIONAL IMPLICATION ONLY**
- LB-1: localized finite-rung computational evidence only
- LB-2: deterministic finite-rung conditional-mean evidence only
- LB-3: R0 conditionally closed; $\gamma$-LOC(ii-c) open
- No Boolean change, no promotion

From AO48-AUD-064 (2026-08-05): quantitative WP ub DECERTIFIED; C030 qualitative kill STANDS; THM rebuild gated.

LB-1 above-$b$ channel SURVIVES as a different functional (confirmation gate: it does not use the defective WP product).

---

## 6. What this review did NOT check

- Byte-level re-execution of the LB-1/2/3 certificates.
- Bodies of KIMI-DER-009 / 009b / 010, C030 CountingLemmas, `c031_verify.json`.
- WP certificate source bytes for DEF-WP-CS-01 (taken from AUD-064).
- C031 quadrature that produces $C^*_\infty=0.9091$ (taken from the Drive C031 file / prior read).
- GP-DER-118-v1.10 (AUD-024 premise). Status inherited: HOLD / NOT PROMOTED.
- Any 3D Side-24 coefficient identity or lifetime-remainder theorem.
- Organizationally independent review of GP-LB-REC-001 or AUD-064 themselves.

---

## 7. Forbidden statements

- “proved constant $c=0.9144$”
- status transfer from 3D Side-24 coefficients
- promotion of GP-DER-118-v1.10
- treating measured exit counts such as $1750/1750$ as a proved zero probability
- using this lower-bound assembly to close the two-sided rate or issue #116 (Q0-C101 qualitative *upper* cubic)

---

## 8. Public-surface fit

Closest existing public issue: [#116](https://github.com/d6g8k5htny-coder/main/issues/116) (Q0-C101 qualitative *upper* rate $1-q\le C_{Q0} r^3$). This landing is the sibling lower bound and does not close that upper-rate audit.
