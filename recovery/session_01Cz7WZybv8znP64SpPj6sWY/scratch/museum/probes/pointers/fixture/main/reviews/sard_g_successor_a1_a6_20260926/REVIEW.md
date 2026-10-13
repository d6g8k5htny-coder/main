# PR122 strict-gradient successor: focused A1/A6 review

Review date: 2026-09-26. Repository: [d6g8k5htny-coder/main, PR122](https://github.com/d6g8k5htny-coder/main/pull/122). Exact reviewed head: `a1fc9581291b6f81c76c903c96a13955b51f8d54`.

**Disposition: A1 AMEND as written; A6's conditional slicing argument is valid, but its application to the stated chart predicate remains AMEND.** The strict gradient-exclusion repair fixes the equality-boundary defect identified in the previous review. A separate geometric boundary remains unspecified: the membership predicate does not require a hit of a finite section to lie in its relative interior. Under the closed-section interpretation explicitly used by the parent manuscript, a chart can contain a transverse endpoint hit and fail to be open. An analytic example below verifies this while all the stated numerical margins have strict slack.

This finding concerns the written chart definition. If “crossing” is intended to include an interior-hit requirement by definition, that convention must be made explicit and applied to every chart member, including the local branch-label sections. I found no such definition in the complete successor, its mapped parent manuscript, or its mapped specialist review. The example is not a counterexample to the desired almost-sure transversality theorem.

| Interface | Disposition on these source bytes | Delimited conclusion |
|---|---|---|
| A1: openness, countability, coverage | **AMEND** | Condition 1 now has the required strict threshold and a sound persistence argument. Finite rational data are countable. The membership rules for sections and first hits need the explicit robust geometric convention described below before openness of every `U_chi` follows. |
| A6: interval slicing | **Conditional argument verified; source application AMEND** | Given genuinely open charts, a C1 mismatch map, and the stated A2–A5 dependencies, the regular-zero and Borel/Tonelli argument works. The present invocation of A1 does not establish that premise for the written predicate. |

No new verdict is issued on A2–A5, finite-jet nondegeneracy, C103, or any downstream result. The full-Gaussian C103 corollary is not discharged by this review and remains on HOLD under the source's own review rule. No author proof, claim register, or scientific status was changed.

## 1. Source identity, authorship, and exposure

The reviewed Git tree is `c20b9638422c6cd594353aaae75a5e1a69fae3a6`. The [complete successor proof](https://github.com/d6g8k5htny-coder/main/blob/a1fc9581291b6f81c76c903c96a13955b51f8d54/research/q0/sard_g/SARD_G_EXECUTION_REPAIR_20260925.md) is `research/q0/sard_g/SARD_G_EXECUTION_REPAIR_20260925.md`.

| Reviewed object | Git blob | Bytes | SHA-256 |
|---|---|---:|---|
| Successor proof | `8f86e0c9ef91fde473e2344a3e461c31f63f8529` | 17,285 | `0fe23db6afc1834610572e307d85f303acfe61300a0c388bd262272857aa7ee4` |
| `SOURCE_MAP.json` | `78a2a4550bbd83071ba7089a2da2c0d6d37dbda3` | 1,354 | `934ae22905f9ac58a3854053202ebe61b79fc53805fdae964749cb9130f49b40` |
| `RECONNAISSANCE.md` | `63f5aa4c0ebfa27ba3b20fad11b5914ba974cd2e` | 1,426 | `90623b128f0de91cc29818bc12392b354a51995e2f4c1bf67a94dd8112db9330` |
| `chart_predicate_negative_control.py` | `7f41ea3a0b0301e0ecdb72de86d00b4d8d9da291` | 10,103 | `cd49c019e5491b15b003eff14f7be406edb6670174633b813db86b7b2bcf158d` |
| Parent Gaussian-transversality working source | `a0f4aa6ab6d0e21e7a24b4a6e1aa4fdebc40efdc` | 19,242 | `4a77f3b7ed390c0ebff3c580baa4d8a107bf3cf2b892c5927e5ffe005ad1c1fe` |
| KIMI-AUD-022 specialist review | `e968385a3bf4e8318a9ab6cb4f57818828f01b18` | 9,527 | `8efcd937502b67ab515d41435eace639000c287ce5e6773105953319f48429b8` |

The parent and specialist paths are pinned in the [source map at the reviewed head](https://github.com/d6g8k5htny-coder/main/blob/a1fc9581291b6f81c76c903c96a13955b51f8d54/research/q0/sard_g/SOURCE_MAP.json). Their full texts are public repository objects. In particular, the [complete parent manuscript](https://github.com/d6g8k5htny-coder/main/blob/a1fc9581291b6f81c76c903c96a13955b51f8d54/drive/mirrors/16_THEMATIC_RESEARCH_TRACKS/T2%20%E2%80%94%20SARD-G%20Transversality/01_TRANSVERSALITY_MANUSCRIPTS/02_GAUSSIAN_TRANSVERSALITY%20%E2%80%94%20WORKING-SOURCE%20%E2%80%94%2019242B%20%E2%80%94%20SHA4a77f3b7%20%E2%80%94%20ID19G4QeIu.md) supplies the closed-section convention discussed below.

The earlier [complete A1/A6 report](https://github.com/d6g8k5htny-coder/main/blob/cc18155abf912d3b8f25ac133ea42bb327287709/reviews/sard_g_a1_a6_20260926/REVIEW.md) reviewed head `6a7f1f6cfb49703b7b758343d65b3b65abe6b11b`, tree `f2deec3727a73ab803b13f314d1b62f9f9f404b4`. Its [old proof](https://github.com/d6g8k5htny-coder/main/blob/6a7f1f6cfb49703b7b758343d65b3b65abe6b11b/research/q0/sard_g/SARD_G_EXECUTION_REPAIR_20260925.md) is blob `a0e9f7d8960f785281fea5aef1bc8b011cbbb055`, 14,672 bytes, SHA-256 `6b04ad32e55c3e0c4f489e45f7687b0f96ec9e1ad25d01b2de5df50a6ab333d0`. The prior report itself is blob `e075929cfcb5e6d10b9f63b2a47c5b2612f85413`, 17,517 bytes, SHA-256 `b56f50ded406d5359c0990cf3aac2669c45917c4b351b4557c3693d9118b2273`; the local copy read for this review matches that public Git object exactly.

Reviewer: OpenAI / ChatGPT, Codex review agent. I did not author the successor proof or the previous report. I read the previous report before completing this review, and reused its elementary trigonometric field to test a different boundary mechanism. This is a source-exposed, nonblind, same-provider technical contribution with **zero organizational-independence credit**. It should not be represented as a fresh provider-independent confirmation. No child agents were used.

I read the complete successor proof, source map, reconnaissance, negative-control script, parent working manuscript, specialist review, and previous A1/A6 report. I checked that successor Sections 4–6 are byte-identical to those at both `6a7f1f6cfb49703b7b758343d65b3b65abe6b11b` and `d49be07074bd0696cc034757379771eae3c852de`: the block from `## 4.` up to `## 7.` is 3,264 bytes with SHA-256 `9300693aa97d54eae580042702d10d1086c82e7966f211259847a60807a04050`. Identity is not a new mathematical verdict. The source map's finite-jet input, blob `1bea9e62ef1ec621f51953dd3f5d5410fba616af`, was not re-proved here.

## 2. The old equality-margin defect is repaired

Successor Section 3 condition 1 now requires

$$m_K(f):=\min_K |\nabla f|>\eta,$$

not equality-permitting `m_K(f) >= eta`. Its estimate

$$|m_K(f)-m_K(g)|\leq\|\nabla f-\nabla g\|_\infty$$

correctly makes this condition open. A perturbation smaller than the actual slack preserves the bound. The same strict inequality appears in the later slicing discussion and in the amended negative-control decision. The previous scaling example at `m_K(f)=eta` is now excluded from that fixed chart, while choosing a smaller rational threshold preserves coverage of its connection.

The rest of the source's condition 1 argument is also sufficient locally. The unique critical point in the closed outer box lies inside its continuation neighborhood because the boundary is in the compact exclusion set. The implicit-function theorem continues its nondegenerate zero inside a small ball. The compact remainder of the outer box outside that ball has a positive gradient minimum, so small C1 perturbations create no other zero. C2 continuity preserves the index and the strict spectral gap. This addresses both the hidden-critical-point failure of an earlier predecessor and the later threshold-equality failure.

The geometric issue below has positive exclusion slack throughout. It does not revive either of those repaired examples.

## 3. What the section membership rules actually say

Successor Section 3 chooses a rational transverse line segment `Sigma` inside a rational tube. Condition 2 requires the declared local branches to meet their local sections uniquely and transversely. Condition 3 requires the branches to remain inside the tube until their unique first crossing of `Sigma`. Condition 4 requires strict speed, angle, and travel-time inequalities. Neither condition 2 nor condition 3 states that the hit lies in the section's relative interior, or assigns positive distance from its endpoints.

This is not resolved by a contrary definition in the mapped sources:

- Parent Section 8.4 explicitly discusses hitting a **fixed closed section** on the unique transverse first-crossing event. Its Section 8.3 lists tube, speed, angle, and time conditions without an endpoint exclusion.
- Parent Section 3 chooses a section through an interior point of the **orbit segment**. That locates a section along the orbit; it does not define every allowed hit to lie in the relative interior of the finite section.
- The specialist review calls this chart bookkeeping an execution-level clarification. It does not supply a global definition of crossing that excludes endpoint hits.
- The successor's coverage paragraph says to choose an “interior transverse section” when constructing a good chart around a connection. That is a choice made for coverage, not a restriction in the preceding definition of every `U_chi`.
- The assertions that all conditions have strict margins and that boundary parameters lie outside a chart interval do not supply the missing membership condition. A transverse endpoint hit can satisfy every numerical inequality that was actually listed.

For the following example, a transverse crossing of a closed line segment means a hit of that segment with nonzero velocity component normal to its supporting line. The crossing angle in the written predicate is then strictly positive, including at an endpoint. This is the closed-section interpretation supported by the parent. If instead a “section” is by definition relatively open, or “crossing” is by definition an interior hit, the example is excluded; that additional convention is precisely what needs to be stated. Merely changing to a relatively open segment also needs care about earlier contacts with its closure, as explained in Section 5 below.

## 4. Endpoint example with strict margins and persistent critical points

Work on the unit torus with ascending gradient flow and set

$$F(x,y)=\sin(2\pi x)\bigl(2-\cos(2\pi y)\bigr),\qquad a=(2\pi)^2.$$

Its first and second derivatives are

$$F_x=2\pi\cos(2\pi x)(2-\cos(2\pi y)),\qquad
F_y=2\pi\sin(2\pi x)\sin(2\pi y),$$

$$F_{xx}=-a\sin(2\pi x)(2-\cos(2\pi y)),\quad
F_{yy}=a\sin(2\pi x)\cos(2\pi y),\quad
F_{xy}=a\cos(2\pi x)\sin(2\pi y).$$

Since `2-cos(2 pi y)` is positive, the complete critical set is

$$x\in\{1/4,3/4\},\qquad y\in\{0,1/2\}.$$

At `p=(3/4,0)` the Hessian is `diag(a,-a)`, and at `q=(1/4,0)` it is `diag(-a,a)`. At `(1/4,1/2)` it is `diag(-3a,-a)`, and at `(3/4,1/2)` it is `diag(3a,a)`. Thus all four points are nondegenerate, and their critical values are respectively `-1,1,3,-3`. They are distinct. This verifies the Morse and distinct-critical-value properties stated for the generic fields in Section 2; it does not make any new probabilistic assertion about the finite-jet input.

The line `y=0` is invariant. In cover coordinates, the arc

$$3/4<x<5/4$$

has velocity `dx/dt=2 pi cos(2 pi x)>0`, and its two asymptotic endpoints are p and the lift `(5/4,0)` of q. It is an actual saddle-to-saddle connection.

Use square endpoint boxes centered at these lifts, with outer, inner, and continuation halfwidths

$$r_+=1/20,\qquad r_-=3/100,\qquad r_N=1/100.$$

Their required compact containments and disjointness hold. Each closed outer box has exactly its designated saddle. If K is the union of the two compact continuation complements, then

$$m=\min_K |\nabla F|>0.$$

One can choose the margins numerically. Write `(u,v)` for coordinates relative to either saddle, so `|u|,|v| <= 1/20` in its outer box. On K either `|u| >= 1/100`, giving `|F_x| >= 2 pi/25`, or `|v| >= 1/100`, giving `|F_y| > pi/25` because `cos(2 pi u) > 1/2`. These bounds use `sin t >= 2t/pi` for `0 <= t <= pi/2`. Hence `m > pi/25 > 3/25`. Choose `eta=1/20` and `delta=1`; in particular `eta<m/2`. These are strict exclusion and spectral margins, with no equality case.

Choose the rational local-section pairs as vertical closed segments with `-1/100 <= y <= 1/100`, at x-coordinates

$$73/100,\ 77/100\quad\hbox{for p},\qquad
123/100,\ 127/100\quad\hbox{for the lift of q}.$$

They lie inside the corresponding inner boxes. Track p's right unstable branch, whose local launch is at `x=77/100`, and q's left stable branch, whose local launch for backward traversal is at `x=123/100`. The local hits are unique, transverse, and in the relative interiors of these sections. Let the open rational polygonal tube be

$$T=(7/10,13/10)\times(-1/50,1/50)$$

in the same coordinate cover. It contains the connection and both tracked compact travel arcs. Define the central **closed** rational section by

$$\Sigma=\{(1,y):0\leq y\leq 1/200\}.$$

The entire section is inside T. Both tracked branches first hit it at `(1,0)`, its lower endpoint. Their x-coordinates are strictly monotone on the travel arcs, so neither branch has any earlier section hit, boundary hit, or tangent contact. At the hit the velocity is horizontal and normal to the vertical supporting line. The crossing-angle sine is exactly 1, so a declared lower bound `1/2` has strict slack.

On the compact horizontal arc `77/100 <= x <= 123/100`, the speed satisfies

$$|\nabla F|\geq 2\pi\sin(\pi/25)\geq 4\pi/25>12/25.$$

Here the middle inequality follows from `sin t >= 2t/pi` on `[0,pi/2]`. Thus a rational speed lower bound `1/4` is strict. Each launch-to-section distance is `23/100`, so each travel time is less than

$$\frac{23/100}{12/25}=\frac{23}{48}<1.$$

The rational upper time bound 1 is strict. As in the source's compact regular-segment construction, speed and travel time refer to finite travel from local sections; they cannot refer to travel starting at the equilibrium itself. The compact travel arcs have positive distance from the tube boundary: at least `1/50` at F. The tube is deliberately open and contains the arcs including their launch and terminal points. No tube-boundary ambiguity is used.

Under the stated closed-section interpretation, all four written membership conditions therefore hold for this fixed rational tuple chi, and `D_chi(F)=0`.

Now translate vertically:

$$F_\varepsilon(x,y)=F(x,y+\varepsilon),\qquad \varepsilon>0.$$

These are smooth torus fields and `F_epsilon -> F` in C2. Their entire critical sets are obtained by shifting those of F down by epsilon. In particular the designated saddles are exactly

$$p_\varepsilon=(3/4,-\varepsilon),\qquad
q_\varepsilon=(5/4,-\varepsilon)$$

in the selected cover. Their Hessians remain exactly `diag(a,-a)` and `diag(-a,a)`. The other two critical points remain a minimum and a maximum, with the same Hessians and the same distinct critical values as before. For `0<epsilon<1/200`, each designated saddle stays strictly inside its continuation square; the explicit critical set shows that no other critical point enters either closed outer box. There is no loss of uniqueness, no spectral degeneration, and no critical-value collision.

On the fixed compact set K, uniform gradient convergence gives

$$\min_K|\nabla F_\varepsilon|\geq m-\|\nabla F_\varepsilon-\nabla F\|_\infty>m/2>\eta$$

for all sufficiently small positive epsilon. Thus condition 1 remains satisfied with the same strict margins.

The translated connection is the line `y=-epsilon`. Its branch-label hits still lie in the interiors of the same local sections. Its finite travel arcs remain strictly inside T; their distance from the horizontal tube boundary is greater than `3/200` for `epsilon<1/200`. Their horizontal dynamics, speed estimates, angles with the vertical supporting lines, and supporting-line travel times are identical to those for F. The central supporting line is crossed once at `(1,-epsilon)`, which lies **below** the closed segment Sigma. No later hit of Sigma can occur, since x is monotone along the connection and tends to its saddle endpoint.

Consequently `F_epsilon` fails the section-hit requirement while preserving the other listed conditions, and

$$F\in U_\chi,\qquad F_\varepsilon\notin U_\chi,\qquad
F_\varepsilon\longrightarrow F\text{ in }C^2.$$

This proves nonopenness for the parent-supported closed-section reading of the written predicate. The failure occurs at a Morse connection with distinct critical values and with strictly positive gradient-exclusion slack. It is different from the old equality-threshold counterexample, which the successor correctly excludes.

## 5. The bounded geometric amendment needed for A1

A sufficient repair is to state the robust geometric membership conditions, rather than infer them from speed, angle, and time bounds. This report does not edit the author source.

For the central section, require the selected hit to lie in the relative interior of its fixed finite segment and require it to be the first hit of the **closed** segment, with no earlier contact of any kind. Equivalently, an open-section convention can be used if it also excludes earlier contact with the closed section's boundary. Apply the corresponding interior-hit and local uniqueness conditions to the branch-label sections. The reference branch must start away from the relevant section and have a positive finite travel time.

This convention supplies the needed first-hit argument. At an interior transverse hit, the implicit-function theorem tracks a nearby hit on the supporting line, and the positive endpoint distance keeps it in the finite segment. Outside a small time neighborhood of that hit, the earlier compact orbit prefix is disjoint from the closed section, hence has positive distance from it. Continuous dependence preserves that separation. Local transversality gives the unique hit in the remaining time neighborhood. A merely unique first **crossing** does not by itself exclude an earlier noncrossing tangent contact or, for a relatively open section, an earlier endpoint contact with its closure.

Likewise, say explicitly that the compact travelled arcs, including launch and terminal points, lie in the open tube. Compact containment then gives positive tube-boundary clearance, so a separate rational clearance parameter is unnecessary. If a closed tube is used instead, strict positive clearance must be imposed. The current numerical bounds alone do not establish this. The endpoint counterexample already uses an open tube with positive clearance, so this observation is a clarification of the general predicate, not an additional mechanism needed for that example.

These strengthened geometric conditions are open using the local branch and flow-dependence results that remain the A2 dependency. Their role is to specify the domain on which those results apply. This report does not issue a new A2 regularity verdict.

The amendment is compatible with connection coverage. At a genuine nondegenerate connection, take a compact regular orbit arc, a sufficiently small transverse flow-box section through an interior point of that arc, and a tube around the compact travelled pieces. The arc is injective; along a nonconstant ascending gradient trajectory, the field value strictly increases. Shrinking the section separates it from compact earlier prefixes, and its hit can be chosen in the relative interior. Choose the endpoint local sections with the same interior and separation properties. All these clearances and the numerical margins can then be preserved by small rational approximations of the finite geometric data. The already repaired endpoint boxes admit a strictly smaller rational eta. Thus the suitably restricted rational charts still cover genuine connections.

Finite rational tuples remain countable. There is no need to select a first admissible chart, or to choose charts measurably as a function of the field. A1 can be closed by this bounded clarification and its openness argument; no new transversality theorem is suggested by this review.

## 6. A6: valid conditional slicing and its exact measurable conclusion

The exact successor says that `I_g,chi` is open because `U_chi` is open. The geometric issue above prevents that implication from being justified for the written predicate. That is why the source application remains AMEND. The following conditional lemma is valid and isolates what remains to be supplied.

Assume that `U_chi` is genuinely open in X, that `D_chi` is C1 on it as supplied by A2, that its derivative is nonzero on H at the relevant connections as supplied by A3–A4, and that A5 supplies the stated countable directions and independent Gaussian decompositions. No new acceptance of these dependencies is made here. For a fixed pair `(chi,j)`, set

$$E_{\chi,j}=\{f\in U_\chi:D_\chi(f)=0,\ dD_\chi(f)[h_j]\ne0\}.$$

Both displayed functions are continuous on the open chart. Consequently E is Borel in X. Write the A5 joint law of `(g_j,xi_j)` as `nu_j x gamma`, where gamma is standard Gaussian measure. The map

$$T_j:X\times\mathbb R\longrightarrow X,\qquad T_j(g,\xi)=g+\xi h_j$$

is continuous, so `T_j^{-1}(E_chi,j)` is a Borel event in the product space. This explicitly supplies the measurability needed to integrate the scalar sections. There is no requirement to enumerate interval components or zeros measurably in g.

For each fixed residual g,

$$I_{g,\chi}=\{\xi:g+\xi h_j\in U_\chi\}$$

is open in the real line and hence is a countable union of disjoint open intervals. On this open set the chain rule gives

$$F_g(\xi)=D_\chi(g+\xi h_j),\qquad
F_g'(\xi)=dD_\chi(g+\xi h_j)[h_j].$$

Every zero counted by E has nonzero derivative and is isolated among the zeros. There are at most countably many such regular zeros: assign each an isolating interval from the countable rational basis, so distinct regular zeros receive distinct isolating basis intervals. Accumulation at a nonregular zero or at an excluded chart boundary does not alter countability. Since gamma has a density, each such scalar section has gamma measure zero. Tonelli then gives

$$\mu(E_{\chi,j})=
\int_X\int_{\mathbb R}{\bf1}_{E_{\chi,j}}(g+\xi h_j)
\,d\gamma(\xi)\,d\nu_j(g)=0.$$

The countable union over chi and j is a Borel null set containing the generic connection event, conditional on the stated coverage and directional nonvanishing inputs. If the connection event itself has not separately been shown Borel, the precise immediate conclusion is outer measure zero and therefore measurability and measure zero in the completed Gaussian probability space. Together with the assumed full-measure generic set, this gives the almost-sure conclusion. No analytic-set projection and no measurable first-chart selection are needed.

The lemma does not require all points of a Gaussian shift line to be Morse. A global degeneracy elsewhere on the torus can be harmless to a selected local chart. Successor Section 8's statement about degenerate or boundary parameters lying outside the interval must concern parameters that actually invalidate that chart. An endpoint hit cannot simply be declared outside the interval after the fact if the membership predicate admits it.

As in the previous report, the conditional A5 premise used here is a countable family with covariance images dense in H, not an assumption that the Banach dual `X*` is norm separable. This delimits the premise of the slicing calculation without issuing a new A5 verdict.

## 7. Validation and limits

I ran the unmodified source control with

```text
python -B -S research/q0/sard_g/chart_predicate_negative_control.py
```

It exited 0 and printed `negative control passed`. The added threshold checks reject binary minimum values 0.5 and 1.0 and accept 1.5 at eta 1.0. Those checks confirm the implemented strict comparison. The other control cases use sampled gradient minima and exercise the hidden-critical-point and isolated-saddle predicates. They do not certify arbitrary gradient minima or test branch hits, finite-section endpoints, tube clearance, first-hit persistence, or Gaussian measurability. Their passing result does not resolve the geometric counterexample.

The analytic example above needs no numerical simulation. Its critical points, Hessians, branch geometry, and margin persistence are derived explicitly. The Borel and slicing calculation is likewise displayed rather than inferred from an engineering check. The finite-jet input, invariant-manifold regularity theorem, endpoint-distribution calculation, RKHS injection, Gaussian decomposition, C103 regional estimates, Theorem B, RN/JETMOD, and all 3D claims were not re-proved or newly accepted.

The source worktree remained clean at the exact head and tree listed above. This reviewer made no remote comment and no source edit for this task. The report is prepared separately for the coordinating agent to review and publish. Its technical findings confer no organizational-independence credit and no scientific promotion.
