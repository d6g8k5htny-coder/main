# Continuation audit — 5 October 2026

Dylan Roy — delegated AI audit. Coordinator and repair implementer: OpenAI/Codex `/root`. Trigger: the owner's explicit request to audit previous work and help with another useful round. Scientific effect: **NONE**.

This is a bounded retrospective source/integration/reader audit, with a separate fresh nonauthor mathematical read. It is **not** a full end-to-end audit of the entire research corpus, a program freeze, or a statement that every roadmap item is finished. No established whole-program predecessor baseline is claimed. [Pickup](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-6003624895), [repair scope](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-6003644816), [expanded engineering scope and UI routing](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-6003725187).

## Cut, ownership and depth

- Public-reader/source cut: main `72a82b438baed6ddaa9c5465210d84027483bf22`, tree `d7a0bef5ffbe19a5abd835d762aac11ecd128abb`.
- Multi-soft mathematical cut: Math- `54ebcedee137e40d22dbc14a25ba51623c073a25`, SC proof blob `16c56821b52fd76b0be791622b9c3809eafde75a`.
- Live Math main advanced first to `c08e83597f94f923b8411dc13cdf28be3f214b37` during the audit and to `f4c33a98a982d50aa490e49ce9327c755682527c` by the 6 October resumption. Its separately owned ledger integration is outside the frozen mathematical slice. [CoS queue](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-6003554874) retains #300/#298/#299/#301 owners. No branch, merge window or hosted rerun was taken from them.
- Other Astra retrospective [6003533605](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-6003533605) covers its own original execution archives and mathematical packets. This report does not duplicate that work or claim its archives were inspected here.
- No source pause or whole-program acknowledgment was requested. These are immutable historical snapshots; other owners continued. Later candidates need their own dependency and gate checks.

Fresh read-only executors were `/root/integration_audit` and `/root/reader_audit` (both inherited coordination context), plus `/root/multisoft_audit` with no inherited conversation and its fresh cubic helper. Root authored the reviewed multi-soft comment and is not its nonauthor reviewer. The asset proposal worker is a contributor, not its own repair reviewer. All are OpenAI/Codex; organizational-independence credit is **zero**. Prior OpenAI/Anthropic/xAI verdicts remain attributed to their actual sources and exposure.

## Prior integration coverage

For the following eight PRs, native merge/head metadata, exposed reviews, relevant premerge gates, landed run outcomes and all changed-path blob identities were checked. All **65** changed paths equal the corresponding landed blobs. This reuses historical execution; it is not a new Lean/browser/Python run or a fresh proof of every imported theorem.

| Object | Actual merge | Preserved changed paths | Observed landed run |
|---|---|---:|---|
| [main#263](https://github.com/d6g8k5htny-coder/main/pull/263) | `5a7a41daac2e04ea513b97d5fae10a7334f422da` | 4 | [SUCCESS](https://github.com/d6g8k5htny-coder/main/actions/runs/37178853466) |
| [main#264](https://github.com/d6g8k5htny-coder/main/pull/264) | `ad7ea9156138911f44b91bcc0e9497a4ba53f181` | 3 | [SUCCESS](https://github.com/d6g8k5htny-coder/main/actions/runs/37180258524) |
| [main#268](https://github.com/d6g8k5htny-coder/main/pull/268) | `5388b602866a9b6cb4a7a6cecb1d96bba538c74d` | 4 | [SUCCESS](https://github.com/d6g8k5htny-coder/main/actions/runs/37350324289) |
| [main#269](https://github.com/d6g8k5htny-coder/main/pull/269) | `72a82b438baed6ddaa9c5465210d84027483bf22` | 27 | [SUCCESS](https://github.com/d6g8k5htny-coder/main/actions/runs/37376179967) |
| [Math-#253](https://github.com/d6g8k5htny-coder/Math-/pull/253) | `2820fc2ed0b1d89b3d535ee0c4fa9ed9fbdcb27f` | 5 | [SUCCESS](https://github.com/d6g8k5htny-coder/Math-/actions/runs/37180422889) |
| [Math-#257](https://github.com/d6g8k5htny-coder/Math-/pull/257) | `3e0ecefbaaf16f104366162b546c55c8b87ec7be` | 5 | [SUCCESS](https://github.com/d6g8k5htny-coder/Math-/actions/runs/37184409283) |
| [Math-#258](https://github.com/d6g8k5htny-coder/Math-/pull/258) | `44314f0ce4ff5e6134c4460f55e4fc5f2ee90633` | 5 | [SUCCESS](https://github.com/d6g8k5htny-coder/Math-/actions/runs/37184961566) |
| [Math-#188](https://github.com/d6g8k5htny-coder/Math-/pull/188) | `f7f083ec9aa7db6306e4f02b0bc9ace53fc84e60` | 12 | [SUCCESS](https://github.com/d6g8k5htny-coder/Math-/actions/runs/37174499250) |

Math-#257/#258 and #188 retain their repair/readback history; an old AMEND is not silently erased. Main#264's later closed-PR intake failure remains recorded and is followed by C133's terminal-merged skip. It does not retroactively invalidate its successful premerge scientific-source custody or prove its theorem.

The additional twenty-one PRs received a lighter native-state/review/selected-thread audit. All are actually merged. Their table is **not** another full blob or execution audit.

| Object | Actual merge | Selected disposition |
|---|---|---|
| [main#243](https://github.com/d6g8k5htny-coder/main/pull/243) | `bf7e51b63596d40d652e3173f5be1fe8006ed5ec` | Successor browser/source/focus repairs recorded. |
| [main#244](https://github.com/d6g8k5htny-coder/main/pull/244) | `daf69ecb3631e37febda70b3002213ed50b5a223` | Locator successor repaired. |
| [main#246](https://github.com/d6g8k5htny-coder/main/pull/246) | `52252bc8ab32df5f7d96620c9965a7e85343447c` | Four asset-release risks remain at audited cut; repair scope below. |
| [main#247](https://github.com/d6g8k5htny-coder/main/pull/247) | `155bdcec6f5042cb134bfc450bc5cced07b605de` | Complete-metadata AMEND repaired/read back. |
| [main#248](https://github.com/d6g8k5htny-coder/main/pull/248) | `af189634392dad8f095c4677db96eec26e5c0f0a` | Two historical cards still need reviewer-lineage clarification. |
| [main#249](https://github.com/d6g8k5htny-coder/main/pull/249) | `80bd5bab6bd6daa6d31c21ae19a5809c3835bf0a` | Pins integration completed/released. |
| [main#254](https://github.com/d6g8k5htny-coder/main/pull/254) | `bbb49404ee301a265a68a7e3443a4e840e163326` | Strict inequality/LaTeX repairs recorded. |
| [main#256](https://github.com/d6g8k5htny-coder/main/pull/256) | `3ed6a07ee745ff66c0321c66382f52f951748db3` | Empty-class/citation/decimal repairs recorded. |
| [main#258](https://github.com/d6g8k5htny-coder/main/pull/258) | `4d8407194a1032db4956cbb987f00bed453b19dc` | Successor harness scoped PASS. |
| [main#260](https://github.com/d6g8k5htny-coder/main/pull/260) | `e48c2f3d18fc9c310739ccdb8244999ad79554a1` | Failed browser harness retained; final review delivered. |
| [main#262](https://github.com/d6g8k5htny-coder/main/pull/262) | `9093629769f40db76f1311d8ee920d5a82a38b89` | Role/alignment wording repaired and reviewed. |
| [main#265](https://github.com/d6g8k5htny-coder/main/pull/265) | `1d0af286b132ce19da6ca047ffca8178ff7079e8` | Terminal-merged intake guard reviewed. |
| [main#266](https://github.com/d6g8k5htny-coder/main/pull/266) | `4a18cf8a6f1cd89ee92a2c781d783b6294312327` | C131/C132 reader integration completed. |
| [Math-#233](https://github.com/d6g8k5htny-coder/Math-/pull/233) | `e9ff8c6165ef0276e1c84d03ee4fb2273bc8278b` | Historical-source verifier repaired/released. |
| [Math-#234](https://github.com/d6g8k5htny-coder/Math-/pull/234) | `c2836c2ae5df1dfdf72549fd9480b83b4621113e` | Scoped integration completed/released. |
| [Math-#249](https://github.com/d6g8k5htny-coder/Math-/pull/249) | `e0bed3ed394f1e208fc5ff28a694e22f25c751f2` | C91–C103 custody completed. |
| [Math-#251](https://github.com/d6g8k5htny-coder/Math-/pull/251) | `e05b8303aa7648cf321d16a4c626d522b9dda3db` | Standing authorization reconciled; optional citation clarification. |
| [Math-#259](https://github.com/d6g8k5htny-coder/Math-/pull/259) | `44ffe91b9772e0a4e9119a7ea3f9d02e917e4701` | K3 source-domain AMEND repaired/read back. |
| [Math-#260](https://github.com/d6g8k5htny-coder/Math-/pull/260) | `0a84192096570a818006f61bf57f8e21c816a1cb` | K4 partial-block/domain AMEND repaired/read back. |
| [Math-#261](https://github.com/d6g8k5htny-coder/Math-/pull/261) | `bbe85e270f2c8b747f2d5d9477c86e86e323fe15` | C124 incorporation/custody completed. |
| [Math-#265](https://github.com/d6g8k5htny-coder/Math-/pull/265) | `c317869a648f05ce74b58493b38a456029f74e5c` | C127 scoped incorporation completed. |

Cross-repository numbers were kept distinct: main#249 is workflow pins while Math-#249 is C91–C103; main#260 is teaching exports while Math-#260 is Lean v2.0; main#265 is intake repair while Math-#265 is C127 incorporation.

## Public-reader and role coverage

At the main cut above:

- Research blob `366a978464d21b47c085c17d4f8df4a0e250a606`: all **30** pinned routes (25 files, five directories) were reachable, and all **10** displayed SHA-256 identities matched. Research/Formal had no missing local target or same-page fragment. No browser/accessibility rerun was performed.
- Four dated reading cuts remain: 2026-10-04T06:20:32Z, 2026-10-04T00:18:14Z, 2026-10-03T20:01:00Z and 2026-10-03T18:00:00Z. Ordinary-bar leading density, designated-pair failure and auxiliary collision work are distinguished.
- C131 exact-sample source blob `54cf568224fa50f958acff385d6c28b5cbd2d0f7`, SHA-256 `3d938d2223d99f7a8dbaa005b9eb3769813a6f8099a7780200ef048049637c82`, remains separate from C132 conditional spectral/nodal composition and actual sampler certification.
- C132 PROOF blob `1a4b544d018d692a265a1ee228360fed03151cd3`, SHA-256 `fa6d443fd5eeeb21c0d311de4d11eb5b406758f3c5ff09a222d02ba54bd0752c`; SOURCES blob `25f958ec24563f250addc94f1ba146ad06829c3d`, SHA-256 `783319599514159bd8e4ca6a4dd02021063d6c547e2c80cb3a8ae1ec1212ccb3`, match their original cut. The unconditioned Fourier total-count interface is not replaced by a Palm-window law.
- C131 [review5976527971](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-5976527971) remains SHA-256 `ad6f20ab98ce288c2ab54f88653df908015a599f30a329f3c1f12eff93e7f16f`; C132 [review5404446570](https://github.com/d6g8k5htny-coder/main/pull/264#pullrequestreview-5404446570) remains `937f05421b4e6b2b72ba18e54c92efcd3b41a1f5665ce976448e7825126b4e56`. The separate [Anthropic C132 review](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-5977511080) also explicitly retains D/C/A and fixed-normalization control3.
- Formal blob `64780e2f1b7780118b10e1e47448e0f89ab7ca5f`: nine coverage rows match the immutable thirteen-declaration SCOPE blob `e5094ac386b72668b7dc42805bf8e6e64a731fac`, 3832 bytes, SHA-256 `350732d7a501fd37d15632b13bd2ac30b8a6d87259b9a3b0931639041a1cfb84`. This remains a historical scalar snapshot, not coverage of the later Lean packages.
- [UI owner](../../docs/agents/UI_UX_OWNER.md) blob `c7b2d407ad28453945e5bcce8ef599915eebeaab` and [Lean auditor](../../docs/agents/LEAN_AUDIT_OWNER.md) blob `49a56b15839e4456e0c9d17e4dbd74bc3b88131d` correctly describe task-local workspace roles. Native acceptances5975108116 and5975146152 were read back. **These roles were not installed as laptop-local agents or background services.** No local installation is asserted by PR262's handoff.

## Findings and disposition

### R1 — omitted normalization premise in the C132 summary

[Original finding4176288112](https://github.com/d6g8k5htny-coder/main/pull/264#discussion_r4176288112) remains valid at README blob `db109becbc4474fc760f2d0b625ca065c2071fb5`. Its displayed accuracy schedule omits `M_n>=m_n`, required by PROOF(8). A fixed denominator has `zeta_M>0`; control3 says its scaling ratio tends to `(1+zeta_M)^(-2/3)`, not one.

The successor README restores the normalization premise, the explicit certificate-failure rate and definitions directly from PROOF(8). It links both actual technical reviews and the immutable original reading cut. The original PROOF/SOURCES and historical review bytes are unchanged. This repairs an overbroad summary, not the already correct theorem.

### R2 — experiment-entry discovery and stale review wording

[Original finding4176288105](https://github.com/d6g8k5htny-coder/main/pull/264#discussion_r4176288105) is already addressed on the public Research page, but the experiment README blob `3282b21b48e3b8d7e9276d6bed0e9fdb1c2c7463` still lacks a C132 route. The successor adds it and replaces the summary's obsolete pending-review paragraph with exact delivered review links. The frozen proof opening remains historical pre-review prose; it is read alongside the linked reviews.

### E1 — four asset-release edge cases

At tool blob `059b2d99fc91f8379876861e710045f0d1736f14` and test blob `90ff5d5c9539c56408c75578505e55f16c609e07`:

1. [4174089213](https://github.com/d6g8k5htny-coder/main/pull/246#discussion_r4174089213): implicit Path ordering is host-dependent; inventory names need explicit POSIX ordering.
2. [4174089219](https://github.com/d6g8k5htny-coder/main/pull/246#discussion_r4174089219): removing the first token from `style.css?site-release=<64hex>&media=screen` creates an intermediate malformed path. The suffix guard can then return the original URL with its stale token; it does not necessarily emit that malformed URL.
3. [4174089224](https://github.com/d6g8k5htny-coder/main/pull/246#discussion_r4174089224): filesystem enumeration includes local untracked/transient files in the release identity.
4. [4174089227](https://github.com/d6g8k5htny-coder/main/pull/246#discussion_r4174089227): regexes inspect inactive comments/string text as imports or resources.

These are current engineering defects, not evidence that historical proofs changed. An isolated standard-library tool/test repair is included in this continuation's precise scope. Its final proposal/review/execution/landing disposition belongs to the linked current PR completion receipt; this historical audit does not assert a successful execution before it occurs.

### U1 — reviewer-lineage details in two historical cards

[PR248 finding4174269866](https://github.com/d6g8k5htny-coder/main/pull/248#discussion_r4174269866) remains applicable to the #233/#251 cards: real review links exist, but local provider/exposure/independence disclosure is missing. Later cards disclose it. **Unresolved; routed through the sole UI owner** in6003725187. No card or historical cut was silently rewritten. That routing is a request, not a delivered review or completed fix.

### H1 — historical mixed-author machine metadata

[PR268 finding4186996536](https://github.com/d6g8k5htny-coder/main/pull/268#discussion_r4186996536) concerns a retained single-author JSON schema spanning mixed historical authors. Markdown/JSON notes disclose the inherited thirteen targets and unknown model provenance; current Math gates support separate proposal authors. The historical record is not a current-package acceptance. **Historical metadata limitation retained, no silent authorship rewrite.**

The separate [ancestry allegation4186996541](https://github.com/d6g8k5htny-coder/main/pull/268#discussion_r4186996541) is stale: native comparison has merge-base `644d06ebe06c748e7f62988942fddaeabddd1147`, behind0/ahead3 to landed `5388b602866a9b6cb4a7a6cecb1d96bba538c74d`. Original evidence remains an ancestor.

### M1 — separate multi-soft review, with precise qualifications

[Proof6003547392](https://github.com/d6g8k5htny-coder/main/issues/259#issuecomment-6003547392) received fresh nonauthor [PASS_SCOPED6003674319](https://github.com/d6g8k5htny-coder/main/issues/259#issuecomment-6003674319). The executor reconstructed the conditioning/full normalizer, eigenvalue Jacobian, h_j^3 factors, internal/cross Vandermonde powers, polynomial-Gaussian integrability and cubic equation(19).

At fixed source parameters,
`M_R^+(E_q(eta)), M_R^j(E_q(eta)) <= C_q eta^(4q+q(q-1)/2)`, j=1,2, for integer1<=q<=d-2. Bare(17) is not finite; the nonempty restriction is. Use source§5's local n_R in the fixed-radius measures; n_R<=n supplies the global majorant. Tensor rotation uses its intrinsic norm or uniform norm equivalence in raw coordinates. The exponent remains a **limiting-kernel** estimate, not a finite-r rate. IBA2-012 overall stays open. Root's author role and zero organizational independence remain explicit.

## Repaired custody gap and exclusions

Root's previous proof/review coordination deliveries had been posted without an explicit retained GET body comparison. This audit read back public proof6003547392, routing6003552362, PR268 review5418397980 and handoff5999789243. New pickups6003624895/6003644816, expansion6003725187 and review6003674319 received exact body readbacks. Earlier R17 rows834–839 were freshly read; the audit/review claim and repair claim/renewal were located by exact UUID in the current tail before dependent publication. No predicted row was accepted as authority.

A targeted Dropbox title search for the selected IBA2 source returned no result; no absent historical carrier was invented and no unrelated private archive was browsed or exported. A Consensus bibliographic lookup did not replace any source proof or imported matching/count premise.

Coverage remains incomplete for private execution/archive payloads, every old issue/review, all remaining mathematical interfaces, live production accessibility, and package-wide Lean alignment. The entire 29-PR metadata set is not an exhaustive repository corpus. PNG/PDF teaching exports, temporal comparisons, package-specific formal coverage and broader metadata-dependent roadmap items remain unfinished; source roles do not install laptop-local executors.

No additional hosted workflow or unchanged million-node experiment was triggered for this retrospective audit. Local runtime was unavailable; planned extra local runs: none. Token/compute cost was not available, so no cost claim is made. The continuation was interrupted after the audit and before repair publication; the prior repair lease expired. A fresh five-path successor claim was located and exactly read back at Work Events846 (UUID9b2f2019-2676-4b15-9b67-981bd786993e), with [native resumption6006746249](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-6006746249). Main72a8 remained unchanged and no overlapping live main PR was found. The asset proposal worker stopped at a usage limit without a completed implementation; root authored the final inventory/query integration and tests, using lexical_design's disclosed lexer contribution. No implementation work or lease continuity is inferred across that gap.

Current repair execution requires its hosted CI and a distinct nonauthor source/diff read. Final expected-head merge, actual landed source/gates and claim releases are recorded only after observed success.

## Release baseline and follow-up

This report freezes the audited facts at the specified cuts. The repair PR's completion receipt supplies its successor head/tree/check identities and disposition of R1/R2/E1. U1 and H1 remain explicitly qualified, and M1 does not close its parent obligation. No uninspected area becomes a full-audit baseline through citation.

The next useful audit milestone is a reviewed integration of the remaining scientific dependency chain, a package-wide alignment report, or a material numerical/sampling-model change. It should follow affected dependencies and retain these coverage gaps rather than repeat timestamp-only replays.

## Asset-tool proposal boundary

The repair takes Git index membership with current worktree bytes, explicit root/POSIX ordering, and URL query-component removal that retains other parameters and fragments. Stage new/deleted inputs first; non-Git inputs, untracked local dependencies and symlinks are refused. Only recognized live HTML/CSS/module URL spans are normalized or rewritten. All pending changes are validated before writes; CRLF bytes are preserved. Comments, inert strings, raw template text and property/computed imports are retained. The standard-library scanner is deliberately bounded: escaped module/CSS URLs, imports in template expressions and module syntax inside ambiguous regex/division contexts are refused. It does not claim full JavaScript grammar support. Fifteen test groups include original behavior plus the earlier four findings, failure/no-write cases and source-preservation cases. Their existence is not an execution receipt: exact hosted results belong to the completion handoff.

## Candidate failures preserved

PR270 initial headfd219850 and tested merge9a304a067f7ce5d223d58f64dfeaa489f4db0bef refused 22 stale URLs in public-shop run37395671105/job112050912136 before unit tests; that canonical change required only the observed cache-key values. [Exact scope addition](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-6006814440), Work Events847. Successorf286058 on tested mergedd89d92d67f7fa5808a4b4df43817873a18ad6b4 passed --check with releasece4cc9c4ecb150312c85c9125729a04d05b118679807deb631951a311dbd86c6, then failed one of14 test groups (public-shop job112051289837): the intended template-expression refusal fixture had been interpolated during JavaScript orchestration into an ordinary `[object Promise]` template. The separate navigation run37395788528 also failed at its normal test-suite step and preserved artifact11383307054; no uninspected artifact failure is assigned a cause here.

Nonauthor integration_audit required the fixture correction and found two bounded-lexer hazards: an ASCII-only identifier scanner could treat a Unicode identifier ending in `import` as a real import, and member calls `.if()`/`?.if()` could be misclassified as control parentheses and swallow actual imports. The successor restores the intended fixture, recognizes complete ordinary Unicode identifiers, refuses unsupported identifier escapes, distinguishes member keywords from control/regex-prefix words, and adds end-to-end preservation/no-write regressions. It also preserves Unicode CSS function identifiers. The original failed candidates and logs remain available; the final exact-head review/current hosted pass belong to the completion receipt, not this proposal record.

A second source-only delta read found that CSS's non-ASCII identifier alphabet is broader than Python Unicode word characters (e.g. symbol/combining prefixes before `url`). The next successor consumes complete CSS non-ASCII identifiers and adds tracked-target controls that retain those custom function names. This required finding is retained rather than hidden behind the earlier passing tests.
