LS-WP-TBG3B-001-v1.0 — TB-G3B EXACT WORK PACKET AND INDEPENDENT REVIEW ROUTE

Artifact ID: LS-WP-TBG3B-001-v1.0
Date: 2026-07-28
Prepared by: Anthropic claude-opus-5, Cowork session, first entry to Fresh Start 2.0
Class: WORK PACKET / REVIEW ROUTING / NO DERIVATION / NO REVIEW CREDIT
Authority: SA-002 autonomous claim of the registry-declared next action for TB-G3B
Canonical mathematical impact: NONE
Independence credit: ZERO (this object contains no derivation and no verdict)
Status: PACKET COMPLETE / TB-G3B DERIVATION NOT STARTED / TB-G3B OPEN

Registry work order: TB-G3B, state OPEN, declared next action
"Create exact work packet and independent review route".
Control card: FS2-GRAPH-TB-001.

FREEZE POLICY
Extract UTF-8 text strictly after BEGIN_TBG3B_WP_FROZEN_BODY and before
END_TBG3B_WP_FROZEN_BODY. Normalize CRLF/CR to LF, remove trailing LF
characters, append exactly one LF, hash without BOM.

BODY_BYTES: 17143
BODY_SHA256: 410e29b65789c888e1fd351e7c3c3262e6c73ec626f49c1e9916dedc803b867e

TRANSPORT: this object is stored as a raw byte-preserving Drive file with
Google-type conversion disabled, so that the identity above is reproducible.

[[TARGET:THEOREM-B-TB-G3B]] [[CLASS:WORK-PACKET]] [[DERIVATION:NOT-STARTED]]
[[LAW:ALL-TYPED-UNCONDITIONED-PALM]] [[WINDOW:B23/20-5/4,K3/4-5/4]]
[[ROUTE-II-REQUIRES:Q-ORDER-R6]] [[CREDIT:ZERO]] [[NO-PROMOTION]]

BEGIN_TBG3B_WP_FROZEN_BODY
TB-G3B — COMPACT ADJACENT-SECTOR REGIONAL DEFECT-COUNT WORK PACKET

0. PURPOSE OF THIS OBJECT

This is a work packet, not a derivation and not a review. It fixes the exact
statement, exact law, exact domain, mandatory power ledger, admissible proof
routes, forbidden substitutions, vulnerability battery, reviewer-eligibility
route, freeze policy, and falsifiers for TB-G3B before any derivation is
attempted. It creates zero mathematical credit, closes no gate, and restores
nothing.

1. POSITION IN THE REPAIR GRAPH

The safe selected-pair defect assembly fixed by FS2-GRAPH-TB-001 is

    1 - p_{r,lambda}
      <= (1 - a_{r,lambda})
         + P^{MS}_{r,lambda}(E_{r,lambda}^c intersect A_{r,lambda}).

No independence is assumed. Selected-but-nonadjacent pairs may not be deleted
from the first term.

LS-DER-023-v1.0 (TB-G3A) is a same-family candidate for the first term:

    sup_{lambda in P_*} (1 - a_{r,lambda}) <= C_A r^3.

TB-G3B is the second term and only the second term:

    sup_{lambda in P_*}
      P^{MS}_{r,lambda}(E_{r,lambda}^c intersect A_{r,lambda})
        <= C_B r^3        for all 0 < r <= r_B.               (TARGET)

Both terms must hold on ONE common parameter domain and ONE common small-r
interval before any recombination theorem may be written, let alone reviewed.

2. EXACT OBJECT FREEZE (BINDING — DO NOT PARAPHRASE)

Field: normalized-periodized Bargmann-Fock Gaussian field f on the flat
side-24 torus.

Parameter set:
    B_* = [23/20, 5/4],
    K_* = [3/4, 5/4],
    P_* = S^1 x B_* x K_*,
    lambda = (theta, b, kappa) in P_*.

Frame and pins, identical to LS-DER-023 Section 1:
    t = (cos theta, sin theta),  n = (-sin theta, cos theta),
    M = x - (r/2) t,  S = x + (r/2) t,
    f(M) = b,  f(S) = b - kappa r^3 / 6,
    grad f(M) = grad f(S) = 0.

Q_{r,lambda} is the exact six-pin conditioned field law.

    W_{r,lambda} = |det H_M det H_S|
                   1{H_M negative definite}
                   1{H_S index one},
    Z_{r,lambda} = E_{Q_{r,lambda}}[W_{r,lambda}],
    dP^{MS}_{r,lambda} = (W_{r,lambda} / Z_{r,lambda}) dQ_{r,lambda}.

The base law carries NO adjacency indicator and NO selection indicator.

A_{r,lambda} is the Borel event that the exact M-ward selected unstable branch
of S converges to M, under the branch convention of the current P02-LM-013
stack.

E_{r,lambda} is the actual-selection event for the ordered pair (M,S) under the
elder-rule persistence pairing convention of LS-DER-022-v1.0.

E^c intersect A is therefore the ADJACENT-BUT-NOT-SELECTED event: the branch
geometry is correct, but some competing critical structure in an adjacent
sector preempts the pairing.

Adjacent sectors: the regions denoted C098, C099, C100, C101 in the LS-DER-022
sector decomposition. The packet author MUST reproduce their exact normalized
definitions from LS-DER-022-v1.0 (Drive ID 1rwKtmy2vnXTYfp0jFrf5Ca1jaBj0DYr7Uf5-Pui33Uo,
21,932 bytes, SHA-256
e5a61739bf09c1972edeb39e683c38da4adbf387c95d65aadf979b84a9568aaf) and quote
them verbatim in the derivation. A sector definition reconstructed from memory
or from a summary is a VUL-007 failure and voids the packet.

3. MANDATORY POWER LEDGER

The target is a PALM probability, so the ledger must terminate at r^0 relative
to r^3, with the denominator handled in the correct direction.

3.1 Normalizer direction gate.

For an UPPER bound on P^{MS}(H) = E_Q[W 1_H] / Z, a LOWER bound on Z is
required. The available lower bound is LS-DER-021 D2:

    Z_{r,lambda} >= c_Z r^2,   c_Z > 0 uniform on P_*.        (Z-LOWER)

Using any restricted, good-event-renormalized, or event-conditioned denominator
is an automatic VUL-006 NO.

3.2 Route I — direct weighted regional count (RECOMMENDED).

Prove a regional weighted first-moment bound

    N^{B}_{r,lambda}
      := E_{Q_{r,lambda}}[ W_{r,lambda} 1_{E^c intersect A} ]
      <= K_B r^5                                              (N-TARGET)

uniformly on P_*, then divide by (Z-LOWER):

    P^{MS}(E^c intersect A) <= (K_B / c_Z) r^3.

Ledger to be exhibited symbolically, term by term:

    endpoint-weight majorant        r^4   multiply   (LS-DER-023 eq. 18)
    physical-to-scaled jet width    r^1   multiply   (q = r s, dq = r ds)
    regional coordinate volume      r^?   multiply   (SEE 3.4 — must be stated)
    full exact normalizer           r^2   divide     (Z-LOWER)
    ----------------------------------------------------------------
    computed total                  r^3   REQUIRED

Any route that reaches r^3 without an explicit entry for every line above is
rejected under VUL-009.

3.3 Route II — Cauchy-Schwarz transfer (ADMISSIBLE BUT EXPENSIVE).

LS-DER-023 eq. (23)-(24) give, using E_Q[W^2] <= C_W r^4 and (Z-LOWER),

    P^{MS}_{r,lambda}(H) <= (sqrt(C_W) / c_Z) Q_{r,lambda}(H)^{1/2}.

Note the r-powers cancel exactly: sqrt(r^4) / r^2 = r^0. Therefore this route
costs a SQUARE ROOT, and

    P^{MS}(E^c intersect A) = O(r^3)
      REQUIRES  Q_{r,lambda}(E^c intersect A) = O(r^6),
      NOT       Q_{r,lambda}(E^c intersect A) = O(r^3).

A conditioned-law regional Kac-Rice count of order r^3 is INSUFFICIENT under
Route II. Any derivation that produces an O(r^3) count under Q and then asserts
an O(r^3) Palm bound without the direct weighted estimate has committed a
law-substitution error (VUL-005) combined with a lost square root. This is the
single most likely failure mode of TB-G3B and reviewers must check it first.

3.4 Kac-Rice coordinate-volume gate.

The regional expected count of competing critical points is a Kac-Rice integral

    E[#{y in Region : grad f(y) = 0, type condition}]
      = integral_Region E[ |det Hess f(y)| 1_type | grad f(y) = 0 ]
        p_{grad f(y)}(0) dy.

Three r-powers enter and each must be written explicitly:

  (a) the physical area element dy of the adjacent sectors, which in normalized
      coordinates xi = (X,Y) with y = x + r(X t + Y n) contributes r^2 dX dY;
  (b) the Gaussian gradient density at zero, p_{grad f(y)}(0), whose scaling
      follows from the exact conditioned covariance and is NOT r^0 in general;
  (c) the conditional expected |det Hess|, in PHYSICAL Hessians, matching the
      convention of LS-DER-023 eq. (10)-(11) and P02-LM-002.

Mixing normalized and physical Hessian conventions between (b) and (c) is the
classic VUL-009 defect and is the second most likely failure mode.

4. ADMISSIBLE IMPORTS AND THEIR CURRENT STATE

The following may be cited, each with its exact frozen identity, and each
carrying its own open-review status forward into any conclusion:

  LS-DER-021-v1.0  compact all-typed Gaussian and normalizer block
                   11,924 bytes / SHA-256
                   3369ddd55dea0f308ab0f396aae7c451b7d4b50934a1b42752bd61d50e9844dd
                   Supplies (15) density envelope, (16) Z >= c_Z r^2,
                   (23) E_Q[W^2] <= C_W r^4.
                   STATE: same-family, exact-hash review OPEN.

  LS-DER-022-v1.0  TB-G3 actual-selection bridge and compact reduction
                   21,932 bytes / SHA-256
                   e5a61739bf09c1972edeb39e683c38da4adbf387c95d65aadf979b84a9568aaf
                   Supplies the sector decomposition and the definition of E.
                   STATE: same-family, exact-hash review OPEN.

  LS-DER-023-v1.0  TB-G3A compact exact-field adjacency cubic theorem
                   13,795 bytes / SHA-256
                   30270eb7ee793462f66d9f280ef620b0598f17edbaab7ee256c1a7e1cf50b791
                   Supplies (18) weight majorant and the boundary-layer template.
                   STATE: same-family candidate; PILOT-001 review NOT COMPLETE.

  P02-LM-002       exact uniform r^4 determinant-weight second-moment interface
                   CORE, TERMINAL at exact scope. Consuming law must itself
                   prove the uniform eighth-moment premise. Do not import the
                   conclusion without discharging the premise.

  P02-LM-008       abstract determinant-weighted Palm-tail transfer interface
                   CORE, TERMINAL at exact scope, ABSTRACT. Each application
                   must independently verify its exact moment, normalizer, law,
                   and event premises. Citing P02-LM-008 without discharging
                   those four premises is a VUL-005 NO.

Because every LS-DER import is currently unreviewed at exact hash, ANY TB-G3B
conclusion is CONDITIONAL until those reviews close. The derivation must state
this conditionality in its own theorem statement, not only in a status footer.

5. FORBIDDEN SUBSTITUTIONS (INHERITED AND PACKET-SPECIFIC)

  - planar field for the exact side-24 torus field;
  - finite-Q4 field or finite-Q4 type indicators for the exact field;
  - conditioned law Q for the typed Palm law P^{MS} or the reverse;
  - restricted or good-event-renormalized denominator for Z_{r,lambda};
  - degree-four determinant weight for the exact determinant weight;
  - a branch selector different from the current P02-LM-013 stack;
  - a persistence pairing convention different from LS-DER-022 elder rule;
  - local branch convergence used as a substitute for elder-rule pairing;
  - fixed-s, fixed-Lambda, or adaptive-deep-threshold reparameterization in
    place of the fixed physical q with s = q/r;
  - practical finite-r numerical agreement used as evidence for the asymptotic
    count.

6. VULNERABILITY BATTERY (RUN ALL TWELVE; ADJUDICATE EACH IN WRITING)

  VUL-001 quotient without denominator gate
          Kac-Rice requires the gradient density at 0 to be strictly positive
          and Hess f nondegenerate almost surely on the region. Exhibit the
          positive lower bound on the conditional gradient covariance
          determinant over the adjacent sectors, uniformly on P_*.
  VUL-002 supremum/sum interchange
          Counting competing critical points across four sectors and across
          branch labels must use additivity or an explicit union bound, never
          a sup/sum swap.
  VUL-003 finite-sample continuum extrapolation
          No grid evaluation, sampled sector sweep, or simulation may serve as
          the certificate. Interval or covering arguments only.
  VUL-004 parameter-domain substitution
          The common small-r endpoint r_B must be exhibited as an explicit
          minimum of finitely many positive quantities, as in LS-DER-023
          Theorem 9.1, and must be compatible with r_A.
  VUL-005 law substitution
          See 3.3. Every count must declare whether it is under Q or under
          P^{MS} at the moment it is written.
  VUL-006 normalizer direction
          See 3.1. Upper bound on a Palm probability requires a LOWER bound on
          Z. Confirm direction at every quotient.
  VUL-007 hash/version drift
          Sector definitions, branch convention, and pairing convention must be
          quoted from the frozen bodies named in Section 4, at the stated byte
          counts and hashes.
  VUL-008 hidden monotonicity endpoint
          P_* is compact but three-dimensional. Identify the HARDEST
          (theta, b, kappa) explicitly, or prove uniformity by a compactness
          argument that survives the r -> 0 limit. Do not assert uniformity
          from compactness of P_* alone.
  VUL-009 coordinate-volume mismatch
          See 3.4.
  VUL-010 consensus laundering
          The derivation author may not review the derivation. Multiple
          sessions of the same provider family are one lineage.
  VUL-011 singular width / missing axial gate
          The adjacent-sector widths and the separatrix transit corridors have
          r-dependent widths. Any quotient whose denominator is such a width
          requires a strict positive lower bound plus a planted zero-width
          control.
  VUL-012 typical-sample nonrobust inequality
          Any inequality observed to hold for typical jets must be proved on
          the worst case of the exact domain, not asserted from typicality.

7. REQUIRED OUTPUTS OF THE TB-G3B DERIVATION

  D1. Verbatim reproduction of the C098-C101 sector definitions with source
      hash.
  D2. Exact statement of the regional count object under the declared law.
  D3. Full symbolic power ledger in the Section 3.2 table form, with a stated
      computed total.
  D4. Explicit choice of Route I or Route II, with, in the Route II case, an
      O(r^6) conditioned-law count actually proved.
  D5. Explicit r_B as a minimum of named positive quantities, and an explicit
      statement of whether r_B <= r_A.
  D6. Written adjudication of VUL-001 through VUL-012.
  D7. Falsifier list and reopening conditions.
  D8. Non-effect statement.
  D9. Marker-delimited frozen body with BODY_BYTES and BODY_SHA256 (Section 9).

8. INDEPENDENT REVIEW ROUTE

Eligibility is target-specific, exposure-specific, lineage-specific, and
method-specific.

  Author lane:        whoever writes the TB-G3B derivation. Independence
                      credit ZERO for that object, permanently.
  Required reviewer:  ELIGIBLE-SOLE against the TB-G3B target, meaning not the
                      author line, exact identity reproduced, controls run,
                      independent method, no unresolved same-signature
                      conflict.
  Lineage notes:      the existing Anthropic verdicts CL-AUD-074, CL-AUD-273
                      and CL-AUD-283 are against DIFFERENT targets
                      (P02-LM-005, LCR-DER-057-v1.1, the finite-Q4 firewall).
                      Under the FS2 review policy that no provider family is
                      intrinsically disqualified and eligibility is
                      target-specific, an Anthropic-lineage reviewer is NOT
                      excluded from TB-G3B by those verdicts alone. It IS
                      excluded if it authored the TB-G3B derivation or read a
                      same-line TB-G3B derivation before freezing Pass A.
  Pass-A discipline:  the reviewer must freeze an independent reconstruction of
                      the regional count and its power ledger BEFORE reading
                      the author's Sections 3 and 4, and must record its
                      pre-existing exposure honestly. A reviewer that has
                      already read the author's conclusion may still file a
                      useful verdict but must self-classify as
                      ELIGIBLE-WITH-COMPENSATION or CORROBORATION-ONLY.
  Minimum to close:   two eligible paths, at least one organizationally
                      distinct from the author, or one eligible path plus a
                      clean formal build.

9. FREEZE POLICY FOR THE TB-G3B DERIVATION

Extract UTF-8 text strictly after BEGIN_TBG3B_FROZEN_BODY and before
END_TBG3B_FROZEN_BODY. Normalize CRLF/CR to LF, remove trailing LF characters,
append exactly one LF, hash without BOM.

BINDING TRANSPORT REQUIREMENT. The frozen body MUST be stored in Drive as a
raw byte-preserving file (text/plain or text/markdown with Google-type
conversion disabled). It MUST NOT be stored only as a Google Docs object.
A Google Docs rendering read back through a connector is a format-unstable
representation: an empirical reproduction attempt on the LS-DER-023 frozen body
(13,795 bytes, SHA-256 30270eb7...b791) recovered mathematical content intact
but produced 13,656 / 13,517 / 14,049 bytes under three defensible whitespace
normalizations, none matching, with the paragraph-break structure irrecoverably
ambiguous. Under that transport, gate PG-01 cannot be satisfied by any reviewer
and every exact-hash work order is unexecutable. Storing the frozen body as raw
bytes removes this blocker for TB-G3B.

10. FALSIFIERS AND REOPENING CONDITIONS FOR THIS PACKET

Amend or withdraw this packet if:

  - the LS-DER-022 sector decomposition does not in fact define C098-C101, or
    defines them on a different chart than assumed here;
  - E_{r,lambda} as defined in LS-DER-022 is not the elder-rule actual-selection
    event;
  - the recombination inequality in Section 1 is superseded;
  - LS-DER-021's (16) or (23) is amended, withdrawn, or found to hold only on a
    smaller domain, in which case Sections 3.1-3.3 must be recomputed;
  - PILOT-001 returns FAIL on LS-DER-023, in which case the r^3 target for the
    FIRST term is void and the common-domain requirement in Section 1 must be
    restated before TB-G3B is worth executing;
  - a reviewer demonstrates that Route I and Route II are both unavailable on
    the exact law, in which case TB-G3B requires a new method rather than a new
    constant.

11. NON-EFFECTS

This packet does not:

  - prove or assume any regional count bound;
  - close TB-G3B, TB-G3A, or PILOT-001;
  - approve LS-DER-021, LS-DER-022, LS-DER-023, or LS-DER-036;
  - create independence credit for any party;
  - promote P0.1 or P0.2;
  - restore Theorem B, which remains RETRACTED / CANNOT VERIFY / NOT RESTORED;
  - authorize external release, which remains HOLD;
  - move, rename, delete, or alter any legacy 1.0 object;
  - alter any frozen proof body;
  - claim that any automation, cron job, or background process is running.

END TB-G3B WORK PACKET
END_TBG3B_WP_FROZEN_BODY

END LS-WP-TBG3B-001-v1.0
