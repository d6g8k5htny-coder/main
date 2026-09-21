# Exact-source crosswalk and review state

## Sources re-fetched in this intake

1. LPW candidate, Drive ID 1p4CsAYJZt30ptsgzANnUSaA-4gU3UcNN: 22614 bytes, SHA cf58f72eb0399c626160c143b50f674ff3bd5c449225211edd6fd378a374e1a5; equals original uploaded package member byte-for-byte.
2. GP-DER-118-v1.10, Drive ID 1qc6ep2S4PIoPMEWDsdDQI1dr0LOwJiK9: 30933 bytes, whole SHA c1d6e559274d7e87a3fa92f44cb2534a5e277a3b5941fd40db380599ddba0564. Its declared marker extraction yields 29293 bytes, body SHA 9b7901e112a4857e3ea59942858684f72fc09a2825b1efa859360dd8fa55f014. Exact unchanged raw bytes are included for the reviewer.
3. AO48-OPR-045, Drive ID 1MfA94SaoYpnnAs7HYLNvowG9EHIR3Opk: whole SHA e48d7c271dff9c120316ee2129a303aed6f78b518c2e0c08337eb357bd60e2e8.
4. GP-LB-STAT-004, Drive ID 1lfH7g57LcshqpLfNbrx-H7gbJckpzPqr: whole SHA 26a9c7ef46b546acfdb0fda11f6ee8dd1e655752202d0add39e0011adf936a4c.
5. LS-CTL-003-v1.3, Drive ID 1W0Ta8KGdUmvNfzbm0S-8gzXON_cVawuQ: whole SHA 081a4e3862f6e7aa0797c6b3aa347530ae4f5491bfa4aafde2309295868ef081.

These hashes verify retrieval identity, not correctness of every statement in the source.

## GP-DER-118 Section 0 versus LPW Section 1

| Attribute | GP-DER-118-v1.10 Section 0 | LPW v1.0 | Comparison |
|---|---|---|---|
| Field | exact normalized periodized BF, side 24 | same with explicit kernel | same underlying declared field |
| Dimension | two coordinates at both sites | d=2 | same |
| Level | b=6/5 | b=6/5 | same |
| Sites | (-r/2,0), (r/2,0) | same | same |
| Values | b, b-r^3/6 | same | same |
| Gradients | zero at both | same | same |
| Weight | absolute pair Hessian determinant, strict maximum/index-one saddle indicators | same | same |
| Normalization | E under six-pin law | same | same |
| Conditional version | conditional law as stated | explicitly continuous Gaussian regression | clarification; reviewer should confirm canonical version compatibility |
| Conclusion event | selected M-ward branch reaches M | elder partner of M is not S | DIFFERENT; no transfer of theorem conclusion |

Source equality in S1 should mean equality of the base law/weight/pins, not equality of these different event probabilities. Supplying these bytes resolves the local source-availability problem; the reviewer's CANNOT VERIFY remains historically accurate until their own supplementary receipt.

## Six reported interface outcomes

R1 and R6: RA_R1_R6.md, prefix 3b7cda65, reported PASS WITHIN SCOPE.
R2: RB_R2.md, prefix 07ceb63c, reported PASS WITHIN SCOPE.
R3: RC_R3.md, prefix 29dcbc23, reported PASS WITHIN SCOPE with A1/A2.
R4 and R5: RD_R4_R5.md, prefix 09aaac1d, reported PASS WITHIN SCOPE.

These underlying Markdown files and scripts are not present in this intake. Their printed prefixes are not fabricated into full hashes. Targeted Drive metadata searches for the named reports and a broader LPW search did not locate them at intake time. The Kimi verdict PDF is a received affirmative report; it is not a substitute for every missing computational artifact.

The lead evidence pack ends with RC review described as in flight, while the final verdict says RC passed. Preserve their chronology as an intermediate lead record plus a later aggregate verdict. Request the final manifest rather than treating the earlier pack as a final run log.

## Independence accounting

The reviewer discloses full candidate exposure before review. Correct label: ordinary hostile review, not blind reconstruction. One Kimi provider-family package reports five working lines. Distinct scripts may demonstrate implementation diversity after inspection/replay; they do not automatically demonstrate five independent provider-family or statistical confirmations. This reconciliation itself contributes no new author-independent credit.

## Two-sided composition check

AO48-OPR-045 is explicitly about the side-24 THREE-torus and the d=3 selection function. LPW concerns the TWO-dimensional six-pin typed elder-pairing probability. A matching exponent and shared side length are insufficient to equate these quantities.

The valid conditional statement is:

    If an accepted d=2 upper theorem gives
    1-q_2(r,6/5)<=C*r^3 for every 0<r<=r_u
    under exactly the LPW law/event conventions,
    then combine with LPW on 0<r<=min(r_u,r_l)
    to obtain c*r^3<=1-q_2(r,6/5)<=C*r^3.

No such upper carrier is supplied by AO48-OPR-045. No two-sided claim is accepted from the attached review.

The lower bound alone DOES logically imply

    liminf_{r->0} (1-q_2(r,6/5))/r^3 >= c > 0.

It does not identify a limit, a sharp leading coefficient, or an evaluated value of c. Interpret the review's 'no liminf coefficient' as withholding those stronger identifications, not denying this elementary liminf consequence.

## K3 status remains exact-object keyed

GP-LB-STAT-004 explicitly names K3-THM-001, 9760 bytes, SHA 715124dba1f089430594a3322b7d32c3508fbcb10c14f6591a432ff582f8854d, and rejects that conditional assembly as written. It separately preserves a valid abstract schema. Kimi's new remark about an H1-H12 assembly does not supply a distinct successor hash or review receipt. Do not rewrite the old disposition as applying only to an unspecified THM-023 v1.1 draft. A corrected successor is reviewable upon delivery.

## W3/W8 and constants

The user's relay reports W8's rung test failing at ratio 3.56 outside [0.4,2.5], and subsequent repair work. No corresponding raw run package was attached. Record this as user-relayed evidence of a failed test in that configuration, not an independently reproduced refutation of every possible Lambda bound. No W3/W8 job was started, stopped, inspected, or monitored by this session. No Kimi constant calculation was executed here.

## Recorded state after this intake

- LPW evidence state: KIMI APPROVE RECEIVED, six interfaces reported PASS, exact target matched, raw-review completeness/replay pending; new additive author clarifications await conversion review.
- Accepted d=2 lower-program control state: unchanged OPEN pending applicable adjudication; receipt of endorsement is separately recorded and must not be concealed by the old candidate-only snapshot.
- Explicit c/r0: NOT ESTABLISHED.
- Two-sided d=2 law from AO48-OPR-045: NOT LICENSED.
- P0.1: HOLD, FOUNDATION/APPLICATION true, INDEPENDENCE/INTEGRITY false under existing overlay.
- RP-C/RP-S d=3: unchanged CLOSED at ratified scope.
- K3 exact rejected carrier: unchanged REFUTED AS WRITTEN/NONCONTROLLING.
- Scientific novelty or human peer-review acceptance: not claimed.
