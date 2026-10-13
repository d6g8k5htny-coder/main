

# LENS substitution

## checked
- Fixture authenticity: setup.sh rebuilds the offline inputs. PROOF_INDEX.md@d6628da comes from the bytes embedded in cd32303 tests/test_museum_data.py and matches the pinned identity (12708 bytes, blob e4cb1d94, sha256 f9c4db77). 21 more Math-@d6628da sources come from local git blob objects, each sha256-checked. EC-014 comes from a blob object in the main clone. The main-repo sources come from the cd32303 tree and equal the 71400b94 pins. README.md@d6628da is NOT available offline, so the D2/D4/D6 cards refuse in the harness with a 404 on README. End-to-end renders therefore use the D3, D5, P15 and D1 cards.
- Baseline (p0_baseline.mjs): the shipped cd32303 museum.json passes validateBoundManifest against the real index and status bytes. All 14 shipped proof and review bindings are correct: no review equals its proof, each review matches the review role in its bullet, and every byte-verified review sits at the index commit. Every finding below is a guard or interface defect, not wrong data in the shipped JSON.
- Proof binding (museum.mjs:75): the proof must be the first link in its own quote and share the index repository and commit. A proof at another commit is REFUSED (p2 control), and the existing proof-swap 3<-10 control holds. The builder does not bind the proof commit (p2: it accepts a proof at 33333333), but the frontend catches that, and so does museum_check through the frontend test.
- Class and label promotion: the positional class/source_label checks (museum.mjs:70-72) stop AMEND->ACCEPT and EXACT_COUNTEREXAMPLE->ACCEPT. Existing tests cover this.
- status_quote binding for D2/D3/D4/D6: the frontend requires exactly one matching ACCEPT row (museum.mjs:90-93). The builder takes the first match (museum_data.py:408), but the frontend's uniqueness check covers that.
- Review verdicts: grepped the dispositions of all 7 byte-verified reviews at d6628da (pr28, pr22, TWO_SCALE_REVIEW, replacement REVIEW, pr16, d2_cumulative, p15_full_price). Each records ACCEPT at its stated scope, so the 'ACCEPT' label on those cards has a source. The three AMEND rows render as AMEND/open with 'OPEN / NOT LANDED'.
- Packet: the result path is bound to incoming/<id>/RESULT.md, and its commit must equal status_source.commit. The RESULT.md declarations match the regexes. The card title is 'Packet — not STATUS'. Nothing links the packet to the D3 card. The builder checks the authority fields in IDENTITY.json and output.json. 'Task issue: not stated in the packet' matches RESULT.md. No substitution path found.
- Id collisions: no collision among claim ids, exhibit card ids ('EC-014', 'D4 fixed-remote RN', ...), 'd2-lifetime-fixture', the packet id, the 'd3-side24' alias and the static ids in museum.html. The view parameter is gated by Object.hasOwn(EXHIBITS, ...).
- geometry.mjs:386-391 binds each exhibit kind to a repository and path. That catches a path swap, but only after the card has already been appended. It does not bind the commit or the digest (see Q1-4).
- Current main 513261bb, static read only, not re-probed: the same patterns are still there. museum.mjs:66-68 (sourceContainsPath with the `/${path})` suffix), 99-106 (pointer includes() check and non-pointer path-only check), 118 and 140 (replay check: some line of the source), 121 (only the lifetime exhibit is bound). museum_data.py:431 (bullet substring check) and 473-478 (open_specs zipped by position).
- Probe hygiene: my first run of the Python probes wrote tools/__pycache__/museum_data.cpython-311.pyc into the extracted cd32303 tree. I removed it, and the tree is back to 104 files. builder.py now sets sys.dont_write_bytecode and all Python probes run with -I -B. No git state changes, no network, nothing posted.

## Q1-1 [critical] A claim's own proof (or another link in the same bullet or row) can be shown as its 'Byte-verified review'; builder, museum_check and the frontend guard all accept it
FILES: tools/museum_data.py:401-406; docs/site/museum.mjs:51-53 (sourceContainsPath, including the `/${path})` suffix clause), 80-87 (non-pointer branch; the D5_OPEN_REVIEW check sits only inside the pointer branch at 85), 151 (label); tests/test_museum_data.py:48-52 (only negative control); tests/test_museum_frontend.mjs:127-151
PROBE: <SCRATCH>/museum/probes/substitution/p1_review_role.py, p1_review_role.mjs, p1b_suffix_match.mjs (harness: builder.py, common.mjs, setup.sh)
CMD: cd <SCRATCH>/museum/probes/substitution && sh setup.sh && python3 -I -B p1_review_role.py && /opt/node22/bin/node p1_review_role.mjs && /opt/node22/bin/node p1b_suffix_match.mjs

DESCRIPTION:
The builder binds a review only by checking that the substring "](" + review_path + ")" occurs somewhere in the claim's own PROOF_INDEX bullet. Every reviewed bullet also links the claim's proof. The D2 bullet also links bare main#67, which the bullet says 'remains open for other bundled fronts'. So a CLAIM_SPECS entry whose review is the proof path (or #67) passes project(). museum_check.py regenerates exactly those bytes. The frontend guard accepts them too: for a non-pointer review it only requires sourceContainsPath(quote, review). Nothing checks that the review differs from the proof or fills the review role in the bullet. The frontend check is looser still. Any link in the quote passes, and so does any path that is a suffix of a link, including a mutable branch URL. The D5 AMEND row links .../Math-/blob/main/PROOF_INDEX.md, so the pinned index itself passes as the D5 card's 'Byte-verified review'. Dropping pointer_only also skips the D5_OPEN_REVIEW binding, and the card then shows 'Byte-verified review' next to 'Unmerged review pointer only'. The P15 counterexample card accepts LANDING_CLAIMS.json the same way. The only negative control (test_misdirected_review_pointer_refused) uses a path that is absent from the bullet, so it cannot detect this. The shipped cd32303 museum.json is correct (p0 baseline). The defect is that no layer refuses the substitution.

WHY:
On an ACCEPT-scoped card, the author's own candidate body, or the landing inventory, can appear as the 'Byte-verified review'. That is a review the source does not support. The path is one wrong CLAIM_SPECS entry, a hand-edited table, and builder, CI check and browser guard all pass it. This is exactly the proof/review substitution the museum claims to prevent, and the existing control gives false assurance.

OBSERVED:
REFUSED   control: d5-all-height review := pr16 review (not in bullet): review pointer differs from audited projection
ACCEPTED  A: d5-two-scale review := its own proof
          -> d5-two-scale review=frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md pointer_only=False review_is_proof=True
ACCEPTED  B: d2 review := its own proof ... review_is_proof=True
ACCEPTED  C: d2 review := bare main#67 ('remains open for other bundled fronts') -> ... review=https://github.com/d6g8k5htny-coder/main/issues/67 pointer_only=True
[A: d5-two-scale review := its own proof]
  ACCEPTED by validateBoundManifest
  card says: [...,"Source and review","Byte-verified review"]
  links: [["Open pinned full proof ↗","https://github.com/d6g8k5htny-coder/Math-/blob/d6628da0.../frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md"],["Open review ↗","https://github.com/d6g8k5htny-coder/Math-/blob/d6628da0.../frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md"]]
[B] verifyClaim (review bytes = D2 proof bytes) passed; renderClaim would print "Byte-verified review" -> .../frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md
[d5-pin-neighborhoods-open] review := PROOF_INDEX.md (no pointer_only)
  ACCEPTED by validateBoundManifest
  rendered: ["Byte-verified review","Unmerged review pointer only. Its target is not fetched or adopted by this viewer."]
[p15-demand-one-counterexample] review := claims/LANDING_CLAIMS.json (no pointer_only)
  ACCEPTED by validateBoundManifest
  rendered: ["Byte-verified review"]

VERDICTS (2):
  - repro=True refuted=False sev=important :: BYTES. The extracted files match `git show cd32303:<path>` byte for byte: tools/museum_data.py, tools/museum_check.py, docs/site/museum.mjs, core.mjs, museum.json, museum.html, geometry.mjs, tests/test_museum_data.py, tests/test_museum_frontend.mjs and STATUS.md (sha256 prefixes compared). The PROOF_INDEX fixture is the INDEX_BYTES embedded in the cd32303 test, and its pinned identity is confirmed (12708 bytes, e4cb1d94…, f9c4db77…). Current main 513261bb still has the same logic: museum_data.py:431-433 and museum.mjs:66-68 and 105.

RE-RUN OF THE AUTHOR'S PROBE. Output matches the finding. The control is REFUSED. A, B, C and D are ACCEPTED by project(). validateBoundManifest accepts A, B, C and D. A renders 'Byte-verified review' with the review link equal to the proof link. D5-open with review := index_source and no pointer_only renders 'Byte-verified review' next to 'Unmerged review p
  - repro=True refuted=False sev=critical :: I reproduced the core defect at the cd32303 bytes with my own probes in probes/verify_substitution_1/. I used no network: urlopen and fetch are stubbed. Every fixture source is re-verified against the builder's own SOURCES pins (29 of 30; only README.md@d6628da is missing offline). The cd32303 tree was not modified (0 files newer than the probes).

Code facts at cd32303:
- tools/museum_data.py:403-406 binds a reviewed claim's review only by checking that "](" + review_path + ")" appears somewhere in that claim's own bullet. Every bullet also links the claim's proof, so review := proof passes.
- The builder never binds the review targets of the open rows (open_specs, about lines 421-433) to their STATUS rows.
- tools/museum_check.py:37-43 only regenerates build() and compares bytes (check_export), then runs test_museum_frontend.mjs. It has no review-role logic of its own.
- docs/site/muse

## Q1-2 [important] A byte-verified review's repository and commit are not bound to the index snapshot that names it, so a different version of the review file is shown under the claim
FILES: docs/site/museum.mjs:86 (path-only check) vs 75 (proof is bound to index repository and commit); docs/site/museum.mjs:35 (validateDescriptor allows main|Math-|query-); tools/museum_data.py:405-406 and 25-259 (descriptor(review_path) takes any SOURCES pin; nothing compares it with INDEX['commit'])
PROBE: <SCRATCH>/museum/probes/substitution/p2_review_snapshot.py, p2_review_snapshot.mjs
CMD: cd <SCRATCH>/museum/probes/substitution && sh setup.sh && python3 -I -B p2_review_snapshot.py && /opt/node22/bin/node p2_review_snapshot.mjs

DESCRIPTION:
PROOF_INDEX links reviews by relative path, which means 'this path at the index's own commit d6628da'. The frontend binds the proof to index_source.repository and commit, but for a non-pointer review it checks only that the path appears in the quote. A review descriptor for the same path at any other Math- commit, or in the query- repository, passes. The card then renders 'Byte-verified review' with that other identity. In the probe the alternate bytes are a synthetic stand-in for 'the same file at another commit', with the R7 verdict flipped to AMEND. They are verified against their own digest and shown under the ACCEPT-scoped card. The builder has the same gap: project() never compares SOURCES[review].commit or repository with INDEX. Today every Math- pin happens to sit at d6628da, but nothing enforces that.

WHY:
'Byte-verified' then only proves that the bytes match their own digest. It does not prove they are the review the pinned index cites. A pre-review draft, an amended version, or a same-named file in another repository can be shown as the claim's review. The proof side shows this binding was intended.

OBSERVED:
ACCEPTED (other Math- commit): d5-all-height-annulus review d6g8k5htny-coder/Math-@11111111 vs index d6g8k5htny-coder/Math-@d6628da0
ACCEPTED (query- repository): d5-all-height-annulus review d6g8k5htny-coder/query-@22222222 vs index d6g8k5htny-coder/Math-@d6628da0
synthetic alternate differs from pinned review: true | contains AMEND row: true
[other Math- commit] review d6g8k5htny-coder/Math-@11111111: ACCEPTED by validateBoundManifest
  card renders: ["ACCEPT-scoped","Source label: ACCEPT","Byte-verified review"]
  review identity shown: Repository d6g8k5htny-coder/Math- | Commit 1111111111111111111111111111111111111111
[query- repository] ... ACCEPTED by validateBoundManifest ... Repository d6g8k5htny-coder/query- | Commit 2222222222222222222222222222222222222222
[control: same change applied to the PROOF] REFUSED: Claim proof is not bound by its own pinned source quote

VERDICTS (2):
  - repro=True refuted=False sev=important :: I checked the bytes first. The extracted docs/site/museum.mjs, docs/site/museum.json, tools/museum_data.py, tools/museum_check.py and docs/site/core.mjs have the same sha256 as `git show cd32303:<path>`. The harness imports museum.mjs and museum_data.py from that extracted tree, and global fetch and urlopen are stubbed so the probes make no network calls.

**The defect is real at cd32303.**
- museum.mjs:75 binds a reviewed claim's proof to index_source.repository and index_source.commit.
- museum.mjs:86 checks a non-pointer review only with sourceContainsPath(quote, review). That check is a path or suffix match on the quote, so the review's repository and commit are never compared with the index.
- The builder has the same gap. museum_data.py:405-406 calls descriptor(review_path), which takes whatever SOURCES[path] pins, and project() never compares that pin with INDEX.
- The PROOF_INDEX
  - repro=True refuted=False sev=important :: I reproduced this at cd32303 two ways: by re-running the original p2 probes against the existing fixture, without re-running setup.sh, and with my own independent harness under probes/verify_substitution_1/q1_2/ (b_builder.py, f_frontend.mjs, output.txt). Both import the unmodified cd32303 tools/museum_data.py and docs/site/museum.mjs by path. Network is stubbed to throw. Before any mutation, every input was checked against the pins: PROOF_INDEX (the literal embedded in the cd32303 test), STATUS.md, and the pr28 REVIEW.md bytes all match the builder and manifest pins.

Code facts, checked line by line:
- museum.mjs:75 binds a reviewed claim's proof to index_source.repository and commit.
- museum.mjs:86 checks a non-pointer review only with sourceContainsPath(quote, review), which is a path or URL substring test.
- validateDescriptor (line 35) and core.mjs verifiedBytes accept main, Math-

## Q1-3 [important] The review-pointer binding is an undelimited substring test, so a pointer to a different issue or comment passes; a non-boolean pointer_only also skips the URL and notice checks
FILES: docs/site/museum.mjs:85 (pointerText.includes(review_url)), 40 (regex and review_notice enforced only when pointer_only===true), 82 and 151-154 (truthiness); compare tools/museum_data.py:404 (delimited "]("+url+")")
PROBE: <SCRATCH>/museum/probes/substitution/p3_pointer_substring.mjs
CMD: cd <SCRATCH>/museum/probes/substitution && sh setup.sh && /opt/node22/bin/node p3_pointer_substring.mjs

DESCRIPTION:
The frontend accepts a review pointer when the quote includes its review_url as a raw substring. Any URL that is a prefix of a linked URL also passes the issue-URL regex: main/issues/6 for a bullet linking #65, main/issues/7 for #76, or a truncated comment id. Those pointers are accepted and rendered as 'Review pointer in the verified index' with the 'pinned source binds this review pointer' notice. The builder's check is delimited, so only the browser guard has this gap. A related inconsistency: validateDescriptor enforces the URL shape and review_notice only for pointer_only===true, while validateBoundManifest and renderClaim treat any truthy value as a pointer. With pointer_only:1 and review_url '' (a substring of every quote), the D5 two-scale card replaces its byte-verified review with 'Review pointer in the verified index' linking PROOF_INDEX.md and shows no not-byte-frozen notice.

WHY:
The browser guard exists to refuse a museum.json that is not bound to its sources. Here it certifies a link to an unrelated issue as 'bound by the pinned source', on ACCEPT cards and on the AMEND D1 card. Fix: match "](" + url + ")" as the builder does, and use pointer_only===true everywhere.

OBSERVED:
[d3-side24-coefficient: issue #6 (bullet links #65)]
  quote contains "](https://github.com/d6g8k5htny-coder/main/issues/6)"? false | quote.includes(url)? true
  ACCEPTED by validateBoundManifest
  rendered: ["Review pointer in the verified index","The pinned source binds this review pointer; the linked review is not byte-frozen by this descriptor."]
  link: ["Open review ↗","https://github.com/d6g8k5htny-coder/main/issues/6"]
[d3-side24-coefficient: truncated comment id] ... ACCEPTED by validateBoundManifest
[d4-fixed-remote-rn: issue #7 (bullet links #76)] ... ACCEPTED by validateBoundManifest
[d1-parent-selection-open: issue #6 on the AMEND D1 card] ... ACCEPTED by validateBoundManifest
[d5-two-scale: review := {...index_source, pointer_only: 1, review_url: ""}]
  ACCEPTED by validateBoundManifest
  rendered: ["Review pointer in the verified index","Identity of the pointer source"]
  link: ["Open review ↗","https://github.com/d6g8k5htny-coder/Math-/blob/d6628da0.../PROOF_INDEX.md"]

VERDICTS (2):
  - repro=True refuted=False sev=important :: BYTES. The extracted museum.mjs, core.mjs, museum.json and tools/museum_data.py match the cd32303 git objects (sha256 prefixes 02e9a1f2, 6bbf7fa2, 693a0290, 44b622b9). I checked the PROOF_INDEX.md fixture (f9c4db77…) and STATUS.md (9c3e2114…) against the museum.json pins inside the probe before using them. The cited lines are correct. museum.mjs:85 is `pointerText.includes(claim.review.review_url)`. Line 40 applies the issue-URL regex and the review_notice requirement only when `pointer_only===true`. Lines 82 and 151-154 branch on truthiness. The builder's delimited check is at museum_data.py:403, with the raise on 404.

REPRODUCTION. I re-ran the original probe read-only, without setup.sh, and it gave the observed output exactly. I then wrote an independent harness that does not use common.mjs (probes/verify_substitution_0/q1_3/v_pointer.mjs). The shipped museum.json is ACCEPTED. A non-
  - repro=True refuted=False sev=important :: I confirmed the finding at the cd32303 bytes. The extracted docs/site/museum.mjs, museum.json, tools/museum_data.py, tools/museum_check.py, core.mjs and STATUS.md hash the same as `git show cd32303:<path>` (museum.mjs sha256 02e9a1f2..., museum.json 693a0290...). The PROOF_INDEX.md and STATUS.md I fed in match the index_source and status_source pins in museum.json.

1. I re-ran the finder's probe (p3_pointer_substring.mjs). The output matches the reported output line for line.

2. I wrote my own harness that does not use the finder's common.mjs. It is at <SCRATCH>/museum/probes/verify_substitution_1/q1_3/v_frontend.mjs, with output in v_frontend_output.txt.
   - validateBoundManifest (museum.mjs:85) accepts a review_url that is any prefix of a link in the quote: main/issues/6 on D2, D3, P15 and D1, main/issues/7 on D4, a

## Q1-4 [important] The remote, annulus and p15 exhibits are linked from and titled after the D4/D5/D6 claims but are not bound to those claims' proofs; only the lifetime exhibit is
FILES: docs/site/museum.mjs:100-101 (shape check only; sameIdentity only for lifetime), 6-7 (EXHIBITS titles, VIEW_LINKS from claim cards), 197-203 (quote checked against the exhibit's own bytes); docs/site/geometry.mjs:386-391 (repository and path bound, commit and digest not)
PROBE: <SCRATCH>/museum/probes/substitution/p4_exhibit_binding.mjs
CMD: cd <SCRATCH>/museum/probes/substitution && sh setup.sh && /opt/node22/bin/node p4_exhibit_binding.mjs

DESCRIPTION:
The D4, D5 all-height, D5 height-window and D6 cards link to museum.html?view=remote|annulus|p15. The exhibit card is titled after the claim, e.g. id 'D5 all-height fixed annulus'. validateBoundManifest only runs validateDescriptor on these exhibits. sameIdentity with a claim proof is enforced for exhibits.lifetime alone. The quote check only confirms that exhibit_quotes[kind] occurs in the exhibit's own fetched bytes, and geometry.mjs checks repository and path but not commit. The probe pins exhibits.annulus to another commit. Its bytes are a synthetic stand-in in which the statement is widened to '0<A<B<=infinity (pin neighborhoods and all distances included)' and relabelled 'Theorem (global, all scales)', with a matching excerpt. The exhibit then renders under the D5 title with the geometry mounted, and the widened text appears as the 'Exact source excerpt'. The same change on exhibits.lifetime is refused. Pointing exhibits.remote at a different claim's proof also passes validateBoundManifest; only geometry.mjs refuses it later, after the card has been appended.

WHY:
From an ACCEPT-scoped card, a reader can land on an 'exact source excerpt' labelled with that claim but taken from a different version of the source than the one reviewed, including one with a wider scope than the review accepted. The lifetime binding shows the intended rule; it was not applied to the other three exhibits.

OBSERVED:
validateBoundManifest: ACCEPTED by validateBoundManifest
D5 all-height card links here: museum.html?view=annulus#active-exhibit | D5 proof commit d6628da0 | exhibit commit 44444444
exhibit card rendered: true | title/id: ["D5 all-height fixed annulus","illustration","Fixed annulus"]
identity shown: Commit 4444444444444444444444444444444444444444
geometry mounted for kind: annulus
"Exact source excerpt" shown contains widened scope: true | contains "global, all scales": true
control (same change on exhibits.lifetime): REFUSED: Lifetime source identity differs from D2
exhibits.remote := D5 height-window proof: ACCEPTED by validateBoundManifest

VERDICTS (2):
  - repro=True refuted=False sev=important :: I reproduced the central claim against the cd32303 bytes. I copied the probe into probes/verify_substitution_0/ and repointed HERE there, so it does not overwrite the original fixture. setup.sh re-verified the fixture: PROOF_INDEX sha256 f9c4db77, plus 21 Math- sources checked against their pinned identities. No network was used: fetch is stubbed, and every served byte matches its pinned digest. The p4 output matched the reported output line for line.

The code agrees with the probe:
- museum.mjs:100 only calls validateDescriptor on exhibits ec014, remote, annulus and p15.
- museum.mjs:101 applies sameIdentity to exhibits.lifetime alone.
- museum.mjs:197-199 checks exhibit_quotes[kind] against that exhibit's own fetched bytes.
- geometry.mjs:386-391 binds the repository to Math- and the path to SOURCE_PATHS[kind]. It checks that commit and sha256 are well-formed but does not tie them to 
  - repro=True refuted=False sev=important :: BYTES. I checked that the extracted museum.mjs (02e9a1f2…), geometry.mjs (29b9cfd9…) and museum.json (693a0290…) are byte-identical to `git show cd32303:<path>`. I rebuilt the fixture in my own directory, <SCRATCH>/museum/probes/verify_substitution_1/q1_4/. Its PROOF_INDEX sha256 is f9c4db77…, which equals index_source. The annulus PROOF.md sha256 is d55e2c03…, which equals the pin. All probes import museum.mjs and geometry.mjs straight from the cd32303 tree, and fetch is stubbed out.

REPRODUCED. The original p4 probe gives the same output as the finding. I extended it in v4b. The same commit-only substitution also works for exhibits.remote, which renders under 'D4 fixed-remote RN', and exhibits.p15, which renders under 'D6 P15 full price'. In both cases the widened excerpt is shown as 'Exact source excerpt'. The code m

## Q1-5 [important] A replay command is not bound to its claim: with the shared Math- README as the source, any README command passes for any claim
FILES: tools/museum_data.py:263-341 (commands hard-coded in CLAIM_SPECS), 410-418 (replay_source = README.md for D2/D4/D6; command never checked); docs/site/museum.mjs:96-98 (shape and repository/commit only), 118 (command must be SOME line of the replay source)
PROBE: <SCRATCH>/museum/probes/substitution/p5_replay_binding.py, p5_replay_binding.mjs
CMD: cd <SCRATCH>/museum/probes/substitution && sh setup.sh && python3 -I -B p5_replay_binding.py && /opt/node22/bin/node p5_replay_binding.mjs

DESCRIPTION:
The builder never compares a command with any source. The frontend checks only that the replay source shares the proof's repository and commit, and that the command is some line of the verified replay source. For D2, D4 and D6 that source is the Math- README, which lists replay commands for many packages (three_fronts, remote_window, full_price, price_budget, downstream_gate hard_gate, ...). The probe swaps D2 and D6, gives D4 the D0 hard_gate.py command, and gives the engineering-only P15 demand-one counterexample the D6 full_price.py replay. The builder and validateBoundManifest accept all of these. verifyClaim also passes them against a real Math- README (commit 10e1f191, the locally available stand-in; README@d6628da is not available offline). Its check is a set-membership test, so a swap of two commands that each passed CI for their own card passes again.

WHY:
The card can present one package's test run as the replay of a different claim, including the D0 engineering hard gate or the full-price theorem on the counterexample card. The 'not theorem acceptance' notice limits the status impact. The 'recorded exactly in its verified source' check is real, but it binds nothing claim-specific for the README-sourced claims, and the existing test only rejects an invented command.

OBSERVED:
ACCEPTED d2-lifetime-remainder    command='python -B -S frontiers/full_price_20260924/full_price.py' source=README.md@d6628da0
ACCEPTED d4-fixed-remote-rn       command='python -B -S frontiers/downstream_gate_20260925/hard_gate.py' source=README.md@d6628da0
ACCEPTED d6-p15-full-price        command="python -B -S -m unittest discover -s frontiers/three_fronts_20260924 -p 'test_*.py' -v" source=README.md@d6628da0
swap D2<->D6 replay commands: ACCEPTED by validateBoundManifest
give engineering-only P15 counterexample the D6 full_price.py replay: ACCEPTED by validateBoundManifest | class engineering-only
verifyClaim d4-fixed-remote-rn with command "python -B -S frontiers/full_price_20260924/full_price.py" from README@10e1f191: PASSES
verifyClaim d4-fixed-remote-rn with command "python -B -S frontiers/downstream_gate_20260925/hard_gate.py" from README@10e1f191: PASSES
verifyClaim p15-demand-one-counterexample with command "python -B -S frontiers/full_price_20260924/full_price.py" from README@10e1f191: PASSES

VERDICTS (2):
  - repro=True refuted=False sev=important :: Verdict: reproduced, not refuted, in scope, severity important.

How it was run. I copied the original probe into my own directory (verify_substitution_0/q1_5/) so I would not touch the fixture used by the other lens. setup.sh re-derives PROOF_INDEX.md@d6628da from the INDEX_BYTES embedded in cd32303 tests/test_museum_data.py: 12708 bytes, blob e4cb1d94, sha256 f9c4db77, which equals SOURCES["PROOF_INDEX.md"]. All 21 other Math- sources were byte-verified from local git objects. Re-running p5_replay_binding.py and p5_replay_binding.mjs gave output identical to the finding's.

Code reading at cd32303 agrees with the finding:
- The commands are hard-coded in CLAIM_SPECS (tools/museum_data.py:263-341).
- project() at lines 410-418 sets replay_source = "README.md" for status_id D2, D4 and D6 and never compares the command with any text. build() verifies the README bytes but does not use them
  - repro=True refuted=False sev=important :: I reproduced this against the cd32303 bytes. I copied the author probe into probes/verify_substitution_1/q1_5/, pointed its paths there, and re-ran setup.sh, p5_replay_binding.py and p5_replay_binding.mjs. The output matched the reported output line for line. I also wrote two probes of my own, v5a_builder.py and v5b_frontend.mjs. All output is in q1_5/ALL_OUTPUT.txt.

What is true at cd32303:
(1) The builder never compares a command with any source. museum_data.py:263-341 hard-codes the commands, and at 410-418 the command is copied into the replay with replay_source = README.md for D2, D4 and D6. Even an invented command ('python invented-solver.py') is accepted (v5a #1). The same builder does refuse a misdirected review pointer (v5a #3) and checks proof paths against the bullet. So the hard-coded specs are not trusted for proof and review, but they are trusted for the replay.
(2) museu

## Q1-6 [important] AMEND rows are bound by position: the D5 pin-neighborhood review, proof and 'NO COMPLETE PROOF YET' attach to whatever row sits second in a re-pinned STATUS
FILES: tools/museum_data.py:430-450 (open_specs zipped positionally with no content check, unlike the bullet checks at 401-404); docs/site/museum.mjs:65-66 (id by position), 79 (d5 proof = index_source, row-independent), 84-85 (d5 pointer checked against the PROOF_INDEX 'D5 pin neighborhoods' line, not the row)
PROBE: <SCRATCH>/museum/probes/substitution/p6_open_row_position.py, p6_open_row_position.mjs (writes p6_STATUS_successor.md, p6_manifest.json)
CMD: cd <SCRATCH>/museum/probes/substitution && sh setup.sh && python3 -I -B p6_open_row_position.py && /opt/node22/bin/node p6_open_row_position.mjs

DESCRIPTION:
The three AMEND rows are paired with hard-coded (id, proof, review) specs purely by order. For 'd5-pin-neighborhoods-open', nothing on either side depends on the row's content. The frontend requires title, quote and status_quote to equal whatever row is in position 2. The proof must equal index_source, and the review pointer is checked against the PROOF_INDEX D5 line. The probe simulates the routine maintenance path: a successor STATUS.md, labelled HYPOTHETICAL, replaces the D5 row with another open obligation, and all main SOURCES are re-pinned. The builder emits id d5-pin-neighborhoods-open with the new title, the new reason, availability 'NO COMPLETE PROOF YET', and the PR55 D5 pin-neighborhood AMEND review. The frontend accepts the result and renders it. The D1 and SARD slots are partly protected because their review must appear in the row; the D5 slot is not.

WHY:
STATUS.md is the file that changes most often, and the builder's success line ('3 AMEND rows') stays green while a review and a proof-availability claim from one obligation are shown on another. The status cannot be promoted, since the card stays AMEND/open, but the review attribution is wrong and the source does not support it.

OBSERVED:
builder ACCEPTED successor STATUS:
  id          : d5-pin-neighborhoods-open
  title       : D5 — shrinking multiple-witness collision
  status_quote: HYPOTHETICAL successor row: the shrinking-separation factorial-moment/collision  ...
  proof       : PROOF_INDEX.md | availability: NO COMPLETE PROOF YET
  review      : https://github.com/d6g8k5htny-coder/Math-/blob/4e188e25b1e1ef560f3eeb75c0d354d2ccf0ea22/reviews/d5_pin_neighborhood_20260926/REVIEW.md
validateBoundManifest(successor): ACCEPTED by validateBoundManifest
card id d5-pin-neighborhoods-open rendered: true
  title: D5 — shrinking multiple-witness collision
  shows: ["NO COMPLETE PROOF YET","Review pointer in the verified index","Unmerged review pointer only. Its target is not fetched or adopted by this viewer."]
  review link: ["Open review ↗","https://github.com/d6g8k5htny-coder/Math-/blob/4e188e25.../reviews/d5_pin_neighborhood_20260926/REVIEW.md"]

VERDICTS (2):
  - repro=True refuted=False sev=important :: BYTES. The probe runs against the extracted cd32303 tree. I checked that tree against `git show cd32303:<path>` for tools/museum_data.py, docs/site/museum.mjs, museum.json, core.mjs, STATUS.md, tools/museum_check.py, both museum test files and .github/workflows/public-shop.yml. All match by SHA-256.

CODE MATCHES THE FINDING. tools/museum_data.py:430-450 builds `open_specs` and pairs it with the three AMEND rows using `zip`, with no content check. The reviewed bullets do get checked at 401-404 (`proof pointer differs` and `review pointer differs`). In museum.mjs:
- line 65 checks the id against the slot position;
- lines 66-69 and 88 take quote, title and status_quote from whichever row is in that slot;
- line 79 binds the d5 proof to `index_source`, which does not depend on the row;
- lines 84-85 bind the d5 review pointer to the PROOF_INDEX line beginning `- D5 pin neighborhoods:`, not
  - repro=True refuted=False sev=important :: The finding is reproduced at the cd32303 bytes. I checked the extracted files against `git show cd32303bcf0f...:<path>` for museum_data.py, museum.mjs, museum.json, test_museum_data.py, museum_check.py and STATUS.md, and the SHA-256 of each is identical. The probe pins also check out: PROOF_INDEX and STATUS equal the pinned identities. The finding is in scope. It concerns cross-claim review substitution in tools/museum_data.py and docs/site/museum.mjs, plus the museum tests where those interfaces meet. All the cited lines are correct:
- museum_data.py: the bullet checks are at 401-404, open_specs starts at 430, and the positional zip is at 437. Nothing checks the content of any of the three AMEND rows.
- museum.mjs: line 65 compares the id against a fixed position. Line 79 makes the D5 proof equal to index_source. Lines 84-85 check the D5 review pointer against the PROOF_INDEX '- D5 pin 

## Q1-7 [minor] The D1 AMEND card says 'Review pointer in the verified index' and 'Open review', but the pointer's source is STATUS.md and the target is the 'fail-closed reopening'
FILES: docs/site/museum.mjs:151 (fixed label for every pointer_only review), 83 (D1 pointer source = status_source); tools/museum_data.py:444
PROBE: <SCRATCH>/museum/probes/substitution/p7_labels.mjs
CMD: cd <SCRATCH>/museum/probes/substitution && sh setup.sh && /opt/node22/bin/node p7_labels.mjs

DESCRIPTION:
This is the shipped, unmodified museum.json. For d1-parent-selection-open the pointer descriptor is STATUS.md, which renderClaim itself shows under 'Identity of the pointer source', yet the label says 'in the verified index'. The target, issues/63#issuecomment-5841830743, is what the STATUS row calls the 'fail-closed reopening', and the card offers it as 'Open review ↗'.

WHY:
This misattributes which pinned source binds the pointer, and it labels a reopening as a review. The fix is small: name the pointer source and use the row's own link label.

OBSERVED:
(a) D1 card text: ["Review pointer in the verified index"] | pointer identity Path: STATUS.md | STATUS row calls the target: "fail-closed reopening"

VERDICTS (2):
  - repro=True refuted=False sev=minor :: Confirmed at the cd32303 bytes. I checked the extracted museum.mjs, museum.json, museum_data.py, core.mjs, museum.html, STATUS.md and tests/test_museum_data.py against `git show cd32303bcf0f:<path>`, and all are byte-identical. I wrote an independent probe under probes/verify_substitution_0/ (setup.sh plus v_d1_label.mjs). It does not reuse the original common.mjs harness or fixture, and it does not write into the substitution lens directory. It serves only three local files, and each matches its pinned SHA-256: PROOF_INDEX f9c4db77 (taken from the INDEX_BYTES literal in the cd32303 test, parsed with ast and not executed), STATUS 9c3e2114, and the D1 proof 9350ad6e (from Math- blob dfed3b8d). Global fetch throws, so no network is used. The probe renders the shipped, unmodified museum.json through the real startMuseum.

Results:
(1) The D1 AMEND/open card renders "Review pointer in the ve
  - repro=True refuted=False sev=minor :: I reproduced this on the cd32303 bytes. I re-ran the author's p7_labels.mjs and also wrote my own probe. Both live in my own copy of the harness under probes/verify_substitution_1/. That copy uses a byte-checked fixture: PROOF_INDEX is f9c4db77…, STATUS is 9c3e2114…, and both match the pins in the shipped museum.json. Note that verify_substitution_1/ already held another verifier's files. I added my files beside them without overwriting any, and the only files I rewrote in its fixture/Math- were byte-identical copies whose SHA-256 I checked.

What the probes show:
- With the shipped museum.json left unchanged, the D1 AMEND card renders (it is not refused). Its source column reads "Review pointer in the verified index", then "Open review ↗", which links to issues/63#issuecomment-5841830743. Under "Identity of the pointer source" it shows STATUS.md @71400b94.
- The review descriptor is ide

## Q1-8 [minor] The D5 height-window card's 'source illustration' is an exhibit of a different claim's proof, and the builder's scope_boundary note for it is never rendered
FILES: docs/site/museum.mjs:7 (VIEW_LINKS 'd5-height-window-annulus' -> view=annulus), 6 (EXHIBITS.annulus titled 'D5 all-height fixed annulus'); tools/museum_data.py:509-510 (scope_boundary); museum.mjs has no reference to scope_boundary
PROBE: <SCRATCH>/museum/probes/substitution/p7_labels.mjs
CMD: cd <SCRATCH>/museum/probes/substitution && sh setup.sh && /opt/node22/bin/node p7_labels.mjs

DESCRIPTION:
This is the shipped museum.json. The d5-height-window-annulus card has proof FIXED_ANNULUS_CANDIDATE.md, blob 081abc13. Its 'Open source illustration ↗' leads to the annulus exhibit, whose pinned source is rn_annulus_bridge_20260925/PROOF.md, blob 6f317515, i.e. the d5-all-height-annulus proof. That card is titled for the other claim and shows that claim's 'Statement and exact scope' excerpt. The one scope-separation sentence the builder adds to this exhibit ('The reviewed fixed annulus and the AMEND pin-neighborhood row are distinct scopes.') is never displayed.

WHY:
The height-window (A2) claim and the all-height claim have different statements and different reviews. Sending a reader from one card to the other's excerpt, with no visible scope note, blurs exactly the scope distinction the builder tried to record.

OBSERVED:
(b) d5-height-window proof: frontiers/rn_thin_tube_20260925/FIXED_ANNULUS_CANDIDATE.md 081abc13 | illustration link: museum.html?view=annulus#active-exhibit -> exhibit source frontiers/rn_annulus_bridge_20260925/PROOF.md 6f317515 (= d5-all-height proof: true)
    exhibit card title/id: ["D5 all-height fixed annulus","illustration","Fixed annulus"]
(c) exhibits.annulus.scope_boundary = "The reviewed fixed annulus and the AMEND pin-neighborhood row are distinct scopes."
    referenced in museum.mjs: false | present in rendered exhibit text: false

VERDICTS (2):
  - repro=True refuted=False sev=minor :: I reproduced this at the cd32303 bytes. I copied the probe directory into probes/verify_substitution_0/, repointed HERE so the original fixture was left alone, and re-ran setup.sh, which byte-verified all 21 Math- sources. p7_labels.mjs gives the same output as reported. I also wrote my own probe, v1_q18.mjs, which runs the shipped museum.json through the real museum.mjs and geometry.mjs with the network stubbed.

What the code does at cd32303:
- docs/site/museum.mjs:7 maps 'd5-height-window-annulus' to museum.html?view=annulus#active-exhibit.
- museum.mjs:149 renders that link on the card as 'Open source illustration ↗'.
- museum.json exhibits.annulus is rn_annulus_bridge_20260925/PROOF.md. Its sha256 equals the d5-all-height-annulus proof's sha256, not the height-window proof's (FIXED_ANNULUS_CANDIDATE.md, 081abc13).
- The exhibit renders 'D5 all-height fixed annulus' as its object lab
  - repro=True refuted=False sev=minor :: I reproduced this against the unmodified cd32303 bytes, with no network. I copied the original harness into probes/verify_substitution_1/, pointed it at its own fixture dir, re-ran p7_labels.mjs and got output identical to the finding. I also wrote my own probe, v1_height_window_link.mjs, plus v2_failed_views.mjs.

Confirmed facts:
- The pinned PROOF_INDEX fixture matches index_source (sha256 f9c4db77..., 12708 bytes), and validateBoundManifest accepts the shipped manifest.
- The d5-height-window-annulus card (claims[4]) has proof FIXED_ANNULUS_CANDIDATE.md (sha d55e... no, sha 1fd9fe71...). It renders an "Open source illustration ↗" link to museum.html?view=annulus#active-exhibit.
- That exhibit's pinned source is rn_annulus_bridge_20260925/PROOF.md (sha d55e2c03...). That equals the d5-all-height-annulus proof and is not the height-window proof.
- The rendered exhibit is titled "D5 all

## Q1-9 [minor] claims/LANDING_CLAIMS.json is pinned and byte-verified but never used; for the same proof blobs it records FAIL_CLOSED or HOLD_WITH_DOMAIN where the cards show 'Source label: ACCEPT'
FILES: tools/museum_data.py:122-129 (in SOURCES; no other reference), 423-426 (source_label constant 'ACCEPT' for every reviewed bullet except index 9)
PROBE: <SCRATCH>/museum/probes/substitution/p8_landing_axis.py
CMD: cd <SCRATCH>/museum/probes/substitution && sh setup.sh && python3 -I -B p8_landing_axis.py

DESCRIPTION:
build() fetches and verifies LANDING_CLAIMS.json@d6628da, but nothing reads it. For the exact proof blobs displayed, it records lifetime-remainder (D2, blob 247b3ecf) as FAIL_CLOSED ('theorem remains fail-closed because parent/global elder/Kac-Rice interfaces are still analytically open on #63'), and D3 and D4 as HOLD_WITH_DOMAIN. The museum shows class ACCEPT-scoped and 'Source label: ACCEPT' for all three and does not mention a landing disposition. To be clear: STATUS.md and PROOF_INDEX do support ACCEPT at the stated scopes, and LANDING_CLAIMS is a separate axis ('never theorem acceptance by itself'; the D6 review 'leaves the landing disposition untouched'). The finding is not that ACCEPT is wrong. It is that a source the museum itself pins carries a conflicting machine-readable disposition for the same bytes, and the museum drops it silently.

WHY:
'Source label' reads as a value copied from a source, but it is a constant set by the builder. When a byte-verified source in the same snapshot gives D2 a fail-closed disposition, the card should either show that axis or say it is out of scope. Dropping it silently runs against the repository's rule that status labels are copied from sources, never decided here.

OBSERVED:
museum_data.py lines mentioning LANDING_CLAIMS: [122, 125] (all inside the SOURCES literal)
d2-lifetime-remainder          museum: class=ACCEPT-scoped   label=ACCEPT               | LANDING_CLAIMS lifetime-remainder: FAIL_CLOSED — all named parent source bytes are now in-tree; theorem remains fail-closed because parent/
d3-side24-coefficient          museum: class=ACCEPT-scoped   label=ACCEPT               | LANDING_CLAIMS side24-coefficient: HOLD_WITH_DOMAIN — ...
d4-fixed-remote-rn             museum: class=ACCEPT-scoped   label=ACCEPT               | LANDING_CLAIMS rn-fixed-remote-window: HOLD_WITH_DOMAIN — ...
d6-p15-full-price              museum: class=ACCEPT-scoped   label=ACCEPT               | LANDING_CLAIMS p15-full-price: REVIEWED_SCOPED — ...
LANDING_CLAIMS meaning: machine-readable landing inventory; source/review identity and scope only, never theorem acceptance by itself

VERDICTS (2):
  - repro=True refuted=False sev=minor :: I reproduced the finding against the cd32303 bytes. The original probe p8_landing_axis.py gives the stated output after setup.sh. I also wrote an independent probe in <SCRATCH>/museum/probes/verify_substitution_0/q1_9/ (entry point run.sh, output in output.txt). It runs offline: urlopen and fetch are stubbed, proxies point at a closed port, main is read with read-only `git show 71400b94:<path>`, and Math- comes from the byte-verified fixture.

How the probe establishes each part of the finding:
(1) Static AST check. "LANDING_CLAIMS" appears only on lines 122 and 125, both inside the SOURCES literal (lines 25-258). No string constant outside that literal names it.
(2) build() is run over the offline fixture. Its output matches the shipped docs/site/museum.json except for 9 leaves, all in the README.md replay-source descri
  - repro=True refuted=False sev=minor :: I reproduced Q1-9 against the cd32303 bytes and could not refute it. Its facts hold. Minor is the right severity.

What I checked:
1. I re-ran the finder's probe (p8_landing_axis.py) against its existing fixture, without re-running setup.sh in another lens's directory.
2. I wrote my own probe at probes/verify_substitution_1/v1_landing_unused.py. It does not use the finder's helpers. It takes LANDING_CLAIMS from Math- blob 1e178c2d (sha256 c6fb2fa2..., 16417 bytes, so it matches the SOURCES pin). That is still the blob at the local Math- HEAD 10e1f19. PROOF_INDEX comes from the bytes embedded in cd32303 tests/test_museum_data.py (identity e4cb1d94/f9c4db77). EC-014 comes from main object 7189b4d1. The other Math- sources come from blob objects, each checked by sha256.
3. With those bytes, cd32303 build() reproduces the shipped docs/site/museum.json byte for byte. README.md@d6628da is not 


# LENS pointers

## checked
- Fixture: every one of the 30 SOURCES entries rebuilt from read-only git objects (Math- d6628da from scratchpad/repos/math, main 71400b94 from /home/user/main), each one checked with the cd32303 museum_data.verify_bytes (byte count, git blob, sha256). Baseline: the committed cd32303 museum.json passes validateBoundManifest and renders all 14 cards. Probes make no network requests (global fetch stubbed to throw) (P/build_fixture.py, P/p0_baseline.mjs, out/p0.txt).
- The committed display does treat comment pointers honestly in part. Each pointer review carries the identity of the pinned pointer source (PROOF_INDEX@d6628da, or STATUS@71400b94 for D1) and those bytes are digest-verified. The card says 'Review pointer in the verified index', shows 'Identity of the pointer source' and prints the notice 'the linked review is not byte-frozen by this descriptor'. README line 46 discloses the same thing. The viewer never fetches a github.com issue or comment URL (0 requests, out/p0.txt and out/p4.txt).
- validateDescriptor's regex for pointer review_url (main|Math- issues, optional #issuecomment, or the single D5 blob URL) exists, but it runs only when pointer_only===true. Control: a foreign URL with pointer_only:true is refused (out/p1.txt).
- Removing pointer_only from the D2, D3, D4, P15 and D1 pointer reviews is refused ('Review file differs from the source quote'). Only d5-pin gets through (out/p6.txt, then F4).
- A hand edit of museum.json that the generator does not produce is caught by museum_check.py ('generated museum export differs', exit 1, out/p6b.txt). So the realistic paths for F1 to F3 are (a) a generator edit, which keeps the whole museum CI green (P/ci_sim.py runs the exact public-shop.yml museum commands, with --math-root pointing at the verified fixture), and (b) the browser guard acting as the runtime defence.
- Unmodified cd32303 control under ci_sim: generator OK, 12 unittest OK, museum_check 15/15 node tests pass, geometry 8/8 (out/ci_sim.txt).
- museum.html CSP (script-src 'self' https://cdn.jsdelivr.net, no 'unsafe-inline') should block javascript: hrefs, so F1 is limited to navigation targets. This comes from reading the code only; I did not test it in a browser.
- Python project() binds reviewed-bullet review pointers by the exact markdown target '](url)'. It refuses issues/6 for D2 but accepts any link target in the bullet, including D2's '[main #67]' 'remains open' link (out/p2b.txt). It performs no binding at all for the AMEND-row review targets (open_specs loop, museum_data.py:430-446).
- Code still present on current main 513261bb (static read only, not re-probed against main's manifest): the pointer_only-gated review_url check, pointerText.includes(review_url), the sourceContainsPath '/${path})' suffix clause, anchor 'Open review' using review_url||html_url, and the strip text 'Review wording is quoted at its declared scope'.
- Not checked, out of lens: cross-claim proof substitution beyond the review slot, replay notices, geometry, packet snapshot membership, response size/time limits.

## Q2-F1 [critical] A 'Byte-verified review' card's 'Open review' link can point to any URL; review_url is validated only when pointer_only===true, and the whole museum CI stays green
FILES: docs/site/museum.mjs:40 (review_url check gated on pointer_only===true); museum.mjs:86 (non-pointer review bound only by sourceContainsPath on path); museum.mjs:151 (label 'Byte-verified review' plus anchor 'Open review ↗' -> claim.review.review_url||claim.review.html_url); tools/museum_data.py:406; tests/test_museum_data.py:95-103 (asserts pointer_only absent on claims[3] but not review_url absent)
PROBE: <SCRATCH>/museum/probes/pointers/p1_review_url_on_byte_verified_review.mjs (with lib.mjs, ci_sim.py, render_sim.mjs, p6b_handedit_control.py; outputs in out/p1.txt, out/ci_sim.txt, out/render_sim.txt, out/p6b.txt)
CMD: cd <SCRATCH>/museum/probes/pointers && /opt/node22/bin/node p1_review_url_on_byte_verified_review.mjs && python3 -I ci_sim.py control && python3 -I ci_sim.py F1_review_url_on_byte_verified && /opt/node22/bin/node render_sim.mjs F1_review_url_on_byte_verified d5-all-height-annulus && python3 -I p6b_handedit_control.py

DESCRIPTION:
For a review descriptor without pointer_only, validateDescriptor never inspects review_url, and validateBoundManifest binds only the descriptor's path. renderClaim still prefers review_url over html_url for the 'Open review ↗' link. So a manifest can carry a correct, digest-verified immutable identity (repository, path, commit, sha256, which is what the card shows under 'Review source identity') together with a review_url pointing at a mutable issue comment, a branch-relative blob/main URL or a foreign host. The card still says 'Byte-verified review', the page reports 'Source projections and displayed source bytes verified', and the link a reader actually follows is unverified. This is the lens defect in its strongest form: an immutable identity is displayed, but the click target is a mutable or arbitrary pointer. The check that would catch it (the regex) exists but is skipped exactly when the card claims byte verification. Precondition: no committed cd32303 card shows this today. A generator edit adding review_url to one descriptor passes the regenerate step, the 12 Python tests, museum_check (15/15 frontend tests, including the actual-pinned-museum integration test) and the geometry tests. A hand edit of museum.json alone is caught by museum_check (Q2 control p6b).

WHY:
The card's strongest custody claim ('Byte-verified review', with a pinned sha256 shown) sits next to a link whose target is mutable or arbitrary. A reader who clicks 'Open review' on an ACCEPT-scoped card reads whatever page that link leads to as the byte-verified review. No guard or test in the museum CI refuses this, and the generator-side test that is supposed to keep review custody honest (test_review_pointers_do_not_claim_comment_byte_custody) checks only the pointer_only key. Fix: refuse review_url unless pointer_only===true, and render the non-pointer anchor from html_url only.

OBSERVED:
--- foreign host
validateDescriptor(review): PASS
validateBoundManifest: PASS
card class: ACCEPT-scoped | review block: {"label":"Byte-verified review","open_review_href":"https://example.invalid/forged-review"}
identity shown under "Review source identity" still names: reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md @ d6628da09384
museum-state: Source projections and displayed source bytes verified. These checks establish byte identity, not mathematical acceptance.
(same PASS for review_url=https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-5841570965 and =https://github.com/d6g8k5htny-coder/Math-/blob/main/reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md)
--- control (pointer_only:true, same foreign URL): REFUSED: Invalid review pointer
[F1_review_url_on_byte_verified] exit=0 :: tools/museum_data.py ... :: Museum projection verified: 11 reviewed bullets, 3 AMEND rows, 1 landed packet; scientific effect NONE.
[F1_review_url_on_byte_verified] exit=0 :: -m unittest discover :: Ran 12 tests in 0.054s | OK
[F1_review_url_on_byte_verified] exit=0 :: tools/museum_check.py ... :: # pass 15 | # fail 0 | Museum source export and actual display integration verified. Scientific effect NONE.
[F1_review_url_on_byte_verified] exit=0 :: node --test tests/test_museum_geometry.mjs :: # pass 8 | # fail 0
[F1_review_url_on_byte_verified] d5-all-height-annulus: {"label":"Byte-verified review","open_review_href":"https://example.invalid/forged-review"}
museum_check exit: 1 | ValueError: generated museum export differs: .../sim/handedit_control/docs/site/museum.json

VERDICTS (2):
  - repro=True refuted=False sev=critical :: I reproduced the finding at the cd32303 bytes and could not refute it. All 104 tracked files in the extracted tree are byte-identical to `git show cd32303:<path>`. The tree also holds two __pycache__ .pyc files that are not tracked; they were there before I ran anything.

Code facts I checked by reading the files. Every cited line is accurate:
- docs/site/museum.mjs:40. The review_url regex and the review_notice check run only when `source.pointer_only===true`.
- museum.mjs:82-86. A non-pointer review is bound only by `sourceContainsPath(quote, claim.review)`, which tests the path or html_url. review_url is never read on that branch.
- museum.mjs:151. The card shows the label 'Byte-verified review', then `anchor('Open review ↗', claim.review.review_url||claim.review.html_url)`.
- tools/museum_data.py:405-406. `descriptor(review_path)` accepts **extra.
- tests/test_museum_data.py:95-103. 
  - repro=True refuted=False sev=critical :: BYTES. The extracted files match the cd32303 git objects. I compared docs/site/museum.mjs, museum.json, core.mjs, museum.html, tools/museum_data.py, tools/museum_check.py, tests/test_museum_data.py and tests/test_museum_frontend.mjs against `git show cd32303bc...:<path>` and all are SAME. The fixture's 30 entries were re-verified against the cd32303 museum_data.SOURCES pins (size, git blob, sha256) using museum_data.verify_bytes. The control regeneration of museum.json is byte-identical to the committed file.

CODE CHECKS. Every line reference in the finding is correct at cd32303:
- museum.mjs:40 checks the review_url regex only when `pointer_only===true`.
- museum.mjs:82-86: a review without pointer_only is bound only by `sourceContainsPath(quote, claim.review)`, which checks the path. validateDescriptor, line 39, also binds url and html_url to the identity, but it never looks at review

## Q2-F2 [important] Review-comment pointers are bound by substring, not link equality: any prefix of a quoted URL passes (other issues, truncated comment ids); the generator does not bind AMEND-row pointers at all
FILES: docs/site/museum.mjs:85 (pointerText.includes(claim.review.review_url)); museum.mjs:40 (regex allows /issues/N without a comment anchor); tools/museum_data.py:401-406 ('](' + review_path + ')' in bullet: any link target in the bullet); museum_data.py:430-446 (open_specs review targets never checked against the STATUS row or index line)
PROBE: <SCRATCH>/museum/probes/pointers/p2_pointer_substring_binding.mjs (plus p2b_python_any_link_target.py, ci_sim.py F2_d1_pointer_prefix, render_sim.mjs; outputs out/p2.txt, out/p2b.txt, out/ci_sim.txt, out/render_sim.txt)
CMD: cd <SCRATCH>/museum/probes/pointers && /opt/node22/bin/node p2_pointer_substring_binding.mjs && python3 -I p2b_python_any_link_target.py && python3 -I ci_sim.py F2_d1_pointer_prefix && /opt/node22/bin/node render_sim.mjs F2_d1_pointer_prefix d1-parent-selection-open

DESCRIPTION:
The browser guard accepts a pointer review_url if it is a substring of the pinned quote. Because issue numbers and comment ids are digit strings, any truncation is a different, regex-valid GitHub pointer. Examples: .../issues/6 is a prefix of .../issues/67#issuecomment-..., .../issues/7 is a prefix of .../issues/76#..., and #issuecomment-584 or #issuecomment-58412 are prefixes of real comment ids. On the generator side, project() accepts any markdown link target in a reviewed bullet as the review: D2 can take '[main #67](.../issues/67)', which the bullet says 'remains open for other bundled fronts'. For the three AMEND rows the generator checks nothing, so an edited D1 target of .../issues/6 regenerates and passes the full museum CI. In every case the card shows 'Review pointer in the verified index' with the substituted link, and the details show the genuine pointer-source digest, so the pinned identity appears to vouch for a pointer that the pinned bytes do not contain. The only frontend negative control ('review comment pointers cannot masquerade...') uses a URL absent from the quote, so it cannot detect a prefix.

WHY:
For mutable pointers, the pinned pointer source is the only immutable anchor. Its binding guarantee ('Review pointer is not bound by its pinned source quote') is weaker than its error message: a displayed review link can lead to a different issue or comment than the one the pinned index or STATUS bytes cite, while the card shows the pinned digest as its authority. Fix: parse markdown link targets from the quoted line and require review_url to equal one of them exactly, and, in project(), apply the same exact check to the AMEND-row targets.

OBSERVED:
d2-lifetime-remainder review_url=https://github.com/d6g8k5htny-coder/main/issues/6 (issue #6: prefix of .../issues/67#issuecomment-...) -> PASS | displayed: {"label":"Review pointer in the verified index","open_review_href":"https://github.com/d6g8k5htny-coder/main/issues/6"}
d2-lifetime-remainder review_url=https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-584 (truncated comment id 584 (a different comment anchor)) -> PASS | displayed: {..."open_review_href":"https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-584"}
d4-fixed-remote-rn review_url=https://github.com/d6g8k5htny-coder/main/issues/7 ... -> PASS | displayed: {..."open_review_href":"https://github.com/d6g8k5htny-coder/main/issues/7"}
d3-side24-coefficient review_url=https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-58412 (truncated comment id) -> PASS
d1-parent-selection-open review_url=https://github.com/d6g8k5htny-coder/main/issues/6 (issue #6 on the AMEND/open D1 card) -> PASS
d2-lifetime-remainder review_url=https://github.com/d6g8k5htny-coder/main/issues/99#issuecomment-1 (CONTROL: not a substring of the quote) -> REFUSED: Review pointer is not bound by its pinned source quote
project() D2 review='https://github.com/d6g8k5htny-coder/main/issues/67': ACCEPTED -> review_url='https://github.com/d6g8k5htny-coder/main/issues/67', notice kept='The pinned source binds this review poin'
project() D2 review='https://github.com/d6g8k5htny-coder/main/issues/6': REFUSED (review pointer differs from audited projection)
[F2_d1_pointer_prefix] exit=0 (all four CI steps; unittest OK; museum_check # pass 15 | # fail 0)
[F2_d1_pointer_prefix] d1-parent-selection-open: {"label":"Review pointer in the verified index","open_review_href":"https://github.com/d6g8k5htny-coder/main/issues/6"}

VERDICTS (2):
  - repro=True refuted=False sev=important :: BYTES. I checked that the extracted files are the real cd32303 bytes. For museum.mjs, museum.json, museum_data.py, museum_check.py, both museum test files, core.mjs and museum.html, `git hash-object` of the extracted file equals `git rev-parse cd32303:<path>`. I wrote independent probes under probes/verify_pointers_0/. STATUS.md comes from the local git object 71400b94:STATUS.md (6881 bytes, SHA-256 matches status_source). PROOF_INDEX.md comes from the pointers/ fixture, used read-only and re-hashed against index_source (12708 bytes, SHA-256 matches). startMuseum's own verifiedBytes then re-checks every served source. The network is stubbed.

FRONTEND HALF: CONFIRMED. museum.mjs:85 binds a pointer review_url with `pointerText.includes(claim.review.review_url)`. The other checks on that line only pin the pointer-source identity (sha256/url), and equality is required only for D5. The museu
  - repro=True refuted=False sev=important :: I reproduced this independently against the cd32303 bytes. Before relying on the extracted copies of museum.mjs, museum_data.py and museum.json, I checked that their SHA-256 matches `git show cd32303:<path>`. My own probes are under probes/verify_pointers_1/q2f2/.

v0 builds a separate fixture of all 30 SOURCES entries. Each one is checked with cd32303's own verify_bytes (size, git blob and SHA-256). The pinned PROOF_INDEX bytes are byte-equal to the INDEX_BYTES literal in cd32303's tests/test_museum_data.py.

(1) Browser guard, museum.mjs:85, `pointerText.includes(claim.review.review_url)`. CONFIRMED. v1 lists every string that passes the line-40 pointer regex and is a substring of the exact pinned pointer text, for each of the five pointer_only claims that lack an exact pin (D2, D3, D4, P15 and D1-AMEND). It then runs the real validateBoundManifest on the real museum.json with only rev

## Q2-F3 [important] The only on-card statement that a review is a mutable comment (review_notice) is free text: the guard checks only that it is non-empty, and no test pins it
FILES: docs/site/museum.mjs:40 (!nonempty(source.review_notice) is the only check); museum.mjs:152 (notice rendered verbatim); tools/museum_data.py:371-373 (hard-coded string); tests/test_museum_data.py:95-103 and tests/test_museum_frontend.mjs:81-93 (neither asserts the notice text)
PROBE: <SCRATCH>/museum/probes/pointers/ci_sim.py (patch F3_notice_claims_byte_frozen) + render_sim.mjs; outputs out/ci_sim.txt, out/render_sim.txt
CMD: cd <SCRATCH>/museum/probes/pointers && python3 -I ci_sim.py F3_notice_claims_byte_frozen && /opt/node22/bin/node render_sim.mjs F3_notice_claims_byte_frozen d2-lifetime-remainder

DESCRIPTION:
Display custody for the six comment-pointer cards depends on one sentence: 'The pinned source binds this review pointer; the linked review is not byte-frozen by this descriptor.' The browser guard accepts any non-empty string there. With the notice changed in the generator to 'Byte-frozen review: the linked comment is sealed by the SHA-256 shown below.', the museum regenerates and passes every museum CI step. The D2 ACCEPT-scoped card then tells readers that a mutable issue comment is sealed by the PROOF_INDEX digest printed beneath it. That treats a mutable pointer as an immutable identity, and no check fails.

WHY:
The negative control the repository rules ask for ('fails when the check is weakened') is missing for the one disclosure that separates mutable from immutable review custody. Fix: hard-code the pointer notice in museum.mjs (or require exact equality), and add a test that fails when the text changes.

OBSERVED:
[F3_notice_claims_byte_frozen] exit=0 :: tools/museum_data.py ... :: Museum projection verified: 11 reviewed bullets, 3 AMEND rows, 1 landed packet; scientific effect NONE.
[F3_notice_claims_byte_frozen] exit=0 :: -m unittest discover :: Ran 12 tests in 0.053s | OK
[F3_notice_claims_byte_frozen] exit=0 :: tools/museum_check.py ... :: # pass 15 | # fail 0 | Museum source export and actual display integration verified. Scientific effect NONE.
[F3_notice_claims_byte_frozen] exit=0 :: node --test tests/test_museum_geometry.mjs :: # pass 8 | # fail 0
[F3_notice_claims_byte_frozen] validateBoundManifest: PASS
[F3_notice_claims_byte_frozen] d2-lifetime-remainder: {"label":"Review pointer in the verified index","open_review_href":"https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206"}
[F3_notice_claims_byte_frozen] notice shown: Byte-frozen review: the linked comment is sealed by the SHA-256 shown below.
[F3_notice_claims_byte_frozen] museum-state: Source projections and displayed source bytes verified. These checks establish byte identity, not mathematical acceptance.

VERDICTS (2):
  - repro=True refuted=False sev=important :: I reproduced the finding. It is in scope and correctly rated important, at the lower end of that band.

Byte identity: the extracted files match git cd32303bcf0f exactly (git rev-parse vs git hash-object). That covers docs/site/museum.mjs ba9081f8, museum.json 07df74b5, tools/museum_data.py 31815063, tools/museum_check.py 9e5e53c2, tests/test_museum_data.py 805fef3d, tests/test_museum_frontend.mjs ae966d04, tests/test_museum_geometry.mjs f1f1e3f3, core.mjs, museum.html and .github/workflows/public-shop.yml. I re-verified all 30 fixture files against the cd32303 SOURCES (size, git blob, SHA-256).

Facts confirmed in the code:
- review_notice appears only at museum.mjs:40, where the only check is nonempty, and museum.mjs:152, where it is rendered verbatim.
- museum_data.py:371-373 hard-codes the notice text.
- test_museum_data.py:95-103 never reads review_notice.
- test_museum_frontend.mjs
  - repro=True refuted=False sev=important :: BYTES: I compared the extracted files with `git show cd32303bcf0f:<path>` by sha256: museum.mjs, museum_data.py, museum_check.py, all three museum tests, museum.json, core.mjs and public-shop.yml. All match.

CODE FACTS, all confirmed:
- museum.mjs:40 is the only check on `review_notice`. It is `!nonempty(source.review_notice)` and runs only when `pointer_only===true`.
- museum.mjs:152 renders the notice verbatim.
- museum_data.py:371-373 hard-codes the string.
- test_museum_data.py:95-103 asserts `pointer_only`, the path and `#issuecomment-`, but not the notice.
- test_museum_frontend.mjs:81-93 uses its own notices ('Pointer only.') and never checks the display text, even though its title promises that pointers 'cannot masquerade as byte-frozen comment content'.
- museum_check.py only compares museum.json with the generator's own output (`check_export`). So if the generator changes, the

## Q2-F4 [important] A 'Byte-verified review' is bound by path suffix only: repository and commit are never tied to the pinned pointer source, and a mutable blob/main link satisfies the binding
FILES: docs/site/museum.mjs:51-53 (sourceContainsPath: '](path)', html_url, or any '/${path})' suffix); museum.mjs:86 (the only check for a non-pointer review); compare museum.mjs:79 (proof must equal index_source.repository/commit, which reviews are never required to do); tools/museum_data.py:406 (descriptor(review_path) is taken from the SOURCES table with no commit check against INDEX/STATUS)
PROBE: <SCRATCH>/museum/probes/pointers/p3_byte_verified_review_identity_unbound.mjs (+ p6_controls.mjs; outputs out/p3.txt, out/p6.txt)
CMD: cd <SCRATCH>/museum/probes/pointers && /opt/node22/bin/node p3_byte_verified_review_identity_unbound.mjs && /opt/node22/bin/node p6_controls.mjs

DESCRIPTION:
In the pinned index, a relative review link like '](reviews/pr28.../REVIEW.md)' identifies an immutable object only together with the index's own identity, Math-@d6628da. The guard never enforces that. (a) The d5-pin AMEND card's pointer descriptor is PROOF_INDEX@d6628da. With pointer_only removed, the STATUS row's mutable link 'https://github.com/d6g8k5htny-coder/Math-/blob/main/PROOF_INDEX.md)' matches the '/PROOF_INDEX.md)' clause, so a branch-relative URL is accepted as binding a pinned identity, and the card flips from 'Review pointer' to 'Byte-verified review'. The same removal is refused on the other five pointer cards (control). (b) The same review path at another commit or repository (Math-@bebb77ae, or query-@ffff...) passes validation, and passes fetch-verification wherever bytes at that identity hash to the descriptor's own sha256. (c) A file named just REVIEW.md in any allowed repository, at any commit, is 'contained' in every quote that links some .../REVIEW.md. The probe serves forged bytes with a self-consistent digest at that identity, standing in for a file that someone with write access could publish. (d) Overlaps the substitution lens: a claim's own proof can fill its review slot, so D2's comment pointer becomes a 'Byte-verified review' of LIFETIME_REMAINDER.md. Today's committed data is consistent: every review is at d6628da or 71400b94. The defect is in the guard.

WHY:
'Byte-verified review' is supposed to mean 'the review the pinned source points at, frozen'. The guard proves only 'some bytes whose path ends like a link in the quote'. A branch-relative (mutable) link is accepted as the binding, and nothing ties the review's commit to the snapshot that cites it. Fix: resolve relative links against the quoting source's repository and commit, require review.repository and review.commit to equal them (as is already done for proof), drop the '/${path})' suffix clause, and refuse non-pointer reviews whose identity equals a pointer source.

OBSERVED:
(a) d5-pin: pointer_only removed -> validateBoundManifest: PASS | card d5-pin-neighborhoods-open: {"label":"Byte-verified review","open_review_href":"https://github.com/d6g8k5htny-coder/Math-/blob/d6628da09384728992dcbe6e921cc28ba85aebb0/PROOF_INDEX.md"} | state: Source projections and displayed source bytes verified. Thes
(b) d5-all-height review moved to d6g8k5htny-coder/Math-@bebb77ae (index pins Math-@d6628da) -> validateBoundManifest: PASS | card d5-all-height-annulus: {"label":"Byte-verified review","open_review_href":"https://github.com/d6g8k5htny-coder/Math-/blob/bebb77ae9fe3b22c425a979b780fc99a72c8e761/reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md"}
(b) d5-all-height review moved to d6g8k5htny-coder/query-@ffffffff (index pins Math-@d6628da) -> validateBoundManifest: PASS | card d5-all-height-annulus: {"label":"Byte-verified review","open_review_href":"https://github.com/d6g8k5htny-coder/query-/blob/ffffffffffffffffffffffffffffffffffffffff/reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md"}
(c) d5-all-height review = query-@eeee../REVIEW.md (forged bytes, self-consistent digest) -> validateBoundManifest: PASS | card d5-all-height-annulus: {"label":"Byte-verified review","open_review_href":"https://github.com/d6g8k5htny-coder/query-/blob/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee/REVIEW.md"}
(d) d2 review := d2 proof descriptor -> validateBoundManifest: PASS | card d2-lifetime-remainder: {"label":"Byte-verified review","open_review_href":"https://github.com/d6g8k5htny-coder/Math-/blob/d6628da09384728992dcbe6e921cc28ba85aebb0/frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md"}
(control) d5-all-height review := d5-height-window review (path not in quote) -> validateBoundManifest: REFUSED: Review file differs from the source quote
d2-lifetime-remainder: pointer_only removed -> REFUSED: Review file differs from the source quote
(d3, d4, p15, d1: same REFUSED)
d5-pin-neighborhoods-open: pointer_only removed -> PASS

VERDICTS (2):
  - repro=True refuted=False sev=important :: BYTES. The extracted museum.mjs, core.mjs, museum.json, museum.html, museum_data.py, museum_check.py and both museum test files have the same sha256 as `git show cd32303bc…:<path>`. I built my own fixture: it parses SOURCES out of the cd32303 museum_data.py without importing it, reads each blob with read-only `git cat-file`, and checks size, blob id and sha256 with hashlib. My probes import the cd32303 museum.mjs and museum_data.py read-only and stub out all network access. The original p3/p6 probes give the same output as reported.

THE DEFECT IS IN THE CODE AS DESCRIBED. At museum.mjs:86, the only check on a review that is not a pointer is sourceContainsPath (lines 51-53). That function accepts three things: a link written `](path)`, the review's own html_url, or any `/${path})` suffix. Nothing compares review.repository or review.commit with index_source or status_source. Proofs do ge
  - repro=True refuted=False sev=important :: I reproduced the finding at the cd32303 bytes, wrote my own probes, and also checked whether the bad inputs can get past the generator and CI. They can.

Byte provenance. The extracted museum.mjs, museum.json, core.mjs, museum.html and museum_data.py are sha256-identical to `git show cd32303:<path>`. The probe fixture was checked independently: 30 of 30 entries match the cd32303 SOURCES table on byte count, git blob sha1 and sha256. Built from that fixture, museum_data.build() reproduces the committed museum.json byte for byte.

Code facts confirmed:
- museum.mjs:86 is the only binding for a review that is not a pointer, and it calls `sourceContainsPath(quote, review)`.
- museum.mjs:51-53 accepts any of three matches in the quote: `](path)`, the html_url, or any `/${path})` suffix.
- The review's repository and commit are never compared with the quoting source.
- Proofs are compared, at 

## Q2-F5 [important] Committed display: every comment-pointer card says 'Review wording is quoted at its declared scope' although no review wording is quoted or bound; an edited or retracted comment leaves the page identical
FILES: docs/site/museum.mjs:160 (strip text on every claim card); museum.mjs:114 (verifyClaim's 'review' verification of a pointer re-verifies PROOF_INDEX/STATUS bytes); museum.mjs:171 (final museum-state 'displayed source bytes verified'); docs/site/museum.json claims[0,1,2,9,11,12].review (no comment body, digest, author or updated_at field); pinned PROOF_INDEX.md line 3 ('Review descriptions below report the linked records, checked on 2026-09-26') is not shown anywhere in museum.json, museum.mjs or museum.html
PROBE: <SCRATCH>/museum/probes/pointers/p4_comment_edit_invisible.mjs (output out/p4.txt)
CMD: cd <SCRATCH>/museum/probes/pointers && /opt/node22/bin/node p4_comment_edit_invisible.mjs

DESCRIPTION:
On the actual cd32303 manifest, six cards rest their review on a mutable GitHub pointer. Three are ACCEPT-scoped (D2, D3, D4), one is engineering-only (P15, a whole open issue thread rather than a comment), and two are AMEND (D1 and d5-pin). On every one the green strip reads 'Source byte count and SHA-256 verified. Review wording is quoted at its declared scope.' But the only quotes on those cards are the PROOF_INDEX bullet or STATUS row, plus a STATUS cell. Those are the index author's 'review descriptions', which the pinned index itself dates ('checked on 2026-09-26'). The museum drops that date. The 'review' fetch step for a pointer card verifies the pointer source's bytes again, not the review. I simulated an edit that withdraws the ACCEPT, served at every review URL: the viewer makes 0 requests to github.com, and the rendered cards and page state are byte-identical. The per-card notice ('not byte-frozen') partly mitigates this. But the strip, the page-level 'displayed source bytes verified' and the absence of the dated qualifier together present an undated review description as quoted review wording, which nothing would update if a review comment were edited or deleted.

WHY:
This answers the lens question 'can an edited comment silently change what is shown without any check failing?' The page never shows the comment, so the page does not change. What changes is the review behind three ACCEPT-scoped cards, while those cards keep asserting that review wording is quoted and verified. Fix: on pointer cards, replace the strip with something like 'Review is a mutable GitHub comment; only the pinned index/STATUS pointer is byte-verified', and show the index's own date ('review descriptions checked on 2026-09-26'). If custody is wanted, land the comment bodies as byte-pinned files and point at those.

OBSERVED:
[original] github.com review URLs requested by the viewer: 0
[edited] github.com review URLs requested by the viewer: 0
rendered claim cards identical after the simulated comment edit: true
museum-state: Source projections and displayed source bytes verified. These checks establish byte identity, not mathematical acceptance.
- d2-lifetime-remainder [ACCEPT-scoped] review_url=https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206
    strip: "Source byte count and SHA-256 verified. Review wording is quoted at its declared scope."
    quotes on card come from: scope_quote <- PROOF_INDEX bullet; status_quote <- STATUS cell; review text quoted: none
    review descriptor fields: blob,bytes,commit,html_url,path,pointer_only,repository,review_notice,review_url,sha256,url
    identity shown for the review = d6g8k5htny-coder/Math-:PROOF_INDEX.md@d6628da0 (the pointer source, not the comment)
(identical strip and 'review text quoted: none' for d3-side24-coefficient, d4-fixed-remote-rn, p15-demand-one-counterexample [review_url=.../issues/67], d1-parent-selection-open [identity main:STATUS.md@71400b94], d5-pin-neighborhoods-open)

VERDICTS (2):
  - repro=True refuted=False sev=minor :: BYTES: The extracted docs/site/museum.mjs, museum.json, museum.html, core.mjs and README.md hash the same as `git show cd32303bcf0f:<path>`, so the probe ran against the PR head. I re-ran p4_comment_edit_invisible.mjs and its output is byte-identical to out/p4.txt. My own harness, v1_pointer_strip.mjs, imports museum.mjs read-only, blocks all network access and serves only the sha256-pinned fixture. It confirms each mechanical claim:
(a) The strip sentence 'Review wording is quoted at its declared scope.' appears on all six pointer_only cards (D2, D3, D4, P15, D1, d5-pin). On those cards the only quoted text is the PROOF_INDEX bullet or STATUS row and cell.
(b) For every pointer card, the 'review' step in verifyClaim (line 114) fetches the PROOF_INDEX or STATUS raw URL again. Line 85 forces claim.review.url and sha256 to equal the pointer source, so this step never verifies the review it
  - repro=True refuted=False sev=minor :: BYTES. The extracted docs/site/museum.mjs, museum.json, museum.html, tools/museum_data.py and tools/museum_check.py hash to the same git blobs as cd32303bcf0f (checked with git rev-parse cd32303:<path> against git hash-object). The fixture was built from git objects and checked with museum_data.verify_bytes. My probe checked it again, separately: 37 pins against museum.json's sha256 and byte counts, 0 mismatches. The museum's own verifiedBytes also checks it during the render.

WHAT REPRODUCES. I re-ran the original p4 probe and its output matches out/p4.txt byte for byte. I also wrote my own harness, probes/verify_pointers_1/v_f5.mjs, and it confirms each mechanical claim:
(a) The viewer makes 0 requests to github.com. Serving 'ACCEPT WITHDRAWN' at every review_url leaves all 14 rendered cards and the museum-state identical.
(b) The strip sentence 'Review wording is quoted at its declar

## Q2-F6 [minor] Pointer labels and the single 'Open review' link misdescribe the pointer source on the committed cards
FILES: docs/site/museum.mjs:151 (fixed label 'Review pointer in the verified index' for every pointer); tools/museum_data.py CLAIM_SPECS and open_specs (one hard-coded review URL per claim); museum.mjs:85 (d5-pin pointerText comes from a PROOF_INDEX line that is never displayed)
PROBE: <SCRATCH>/museum/probes/pointers/p5_pointer_labels_and_selection.mjs (output out/p5.txt)
CMD: cd <SCRATCH>/museum/probes/pointers && /opt/node22/bin/node p5_pointer_labels_and_selection.mjs

DESCRIPTION:
(1) The D1 card says 'Review pointer in the verified index', but its pointer source is STATUS.md@71400b94, not the index. (2) Several source lines cite two review comments jointly ('[R1–R4 review] and [R5/R6 delta review] record ACCEPT'; D3 'C1–C6 review' plus 'reconciliation'; D4 'source-bound review' plus 'scoped reconciliation'). The card's single 'Open review ↗' goes to one of them, chosen by a hard-coded spec rather than by the source. For D2 that is the delta (R5/R6) only. (3) D1's 'Open review ↗' goes to the comment the STATUS row labels 'fail-closed reopening', not to a review. (4) The d5-pin 'Open review' URL is bound to a PROOF_INDEX 'Open obligations' line that the card never displays. The displayed STATUS row contains only Math issue #58 and a blob/main index link, so the reader cannot see the text that binds the link they are offered.

WHY:
These are clarity defects, not status changes. The full quote stays visible and the ACCEPT, AMEND and counterexample classes are correct. But a reader following 'Open review' on D2 reaches only the delta review, and on D1 reaches a reopening comment under a label naming the wrong pointer source. Fix: name the actual pointer source in the label, render every review link from the quoted line (not one hard-coded link), and show the d5-pin binding line.

OBSERVED:
d2-lifetime-remainder: label="Review pointer in the verified index" | pointer source actually = Math-:PROOF_INDEX.md
    issue links in the quoted source line: ["R1–R4 review -> main/issues/67#issuecomment-5841270276","R5/R6 delta review -> main/issues/67#issuecomment-5841782206","main #67 -> main/issues/67"]
    "Open review" anchors on card: 1 -> main/issues/67#issuecomment-5841782206
d3-side24-coefficient: ... ["C1–C6 review -> main/issues/65#issuecomment-5841269490","reconciliation -> main/issues/65#issuecomment-5841779222"] ... 1 -> main/issues/65#issuecomment-5841269490
d4-fixed-remote-rn: ... ["source-bound review -> ...5841783172","scoped reconciliation -> ...5841861362"] ... 1 -> main/issues/76#issuecomment-5841783172
p15-demand-one-counterexample: ... ["main #67 -> main/issues/67"] ... 1 -> main/issues/67
d1-parent-selection-open: label="Review pointer in the verified index" | pointer source actually = main:STATUS.md
    issue links in the quoted source line: ["Review issue #63 -> main/issues/63","fail-closed reopening -> main/issues/63#issuecomment-5841830743","source-first erratum -> main/issues/63#issuecomment-5841947606"]
    "Open review" anchors on card: 1 -> main/issues/63#issuecomment-5841830743
d5-pin-neighborhoods-open: label="Review pointer in the verified index" | pointer source actually = Math-:PROOF_INDEX.md
    issue links in the quoted source line: ["Math issue #58 -> Math-/issues/58"]
    "Open review" anchors on card: 1 -> Math-/blob/4e188e25b1e1ef560f3eeb75c0d354d2ccf0ea22/reviews/d5_pin_neighborhood_20260926/REVIEW.md

VERDICTS (2):
  - repro=True refuted=False sev=minor :: Bytes. The extracted docs/site/museum.mjs (sha256 02e9a1f2…), docs/site/museum.json (693a0290…) and tools/museum_data.py (44b622b9…) match `git show cd32303:<path>` byte for byte. The fixture served to the renderer is checked by build_fixture.py with verify_bytes (size, git blob, sha256). The cd32303 core.mjs verifiedBytes also accepted every served source: all 14 claim cards rendered, none were refused, and no fetch went unanswered.

Re-run. The author's p5 probe gives output identical to out/p5.txt. My independent probe (verify_pointers_0/q2f6/v_q2f6_cards.mjs) uses a fresh DOM stub and serves files by the descriptor URLs in museum.json. It dumps the 'Source and review' column of the D2, D1 and d5-pin cards.

Code check:
- museum.mjs:151 always prints 'Review pointer in the verified index' when pointer_only is set.
- museum.mjs:83 makes the D1 pointer source status_source (main STATUS.
  - repro=True refuted=False sev=minor :: I re-ran the finder's probe p5 against the cd32303 bytes. Its output is byte-identical to out/p5.txt.

I then checked independently with my own probe, which does not use the finder's lib.mjs. It is at <SCRATCH>/museum/probes/verify_pointers_1/v_q2f6.mjs, with output in v_q2f6.out.txt. It does four things:
- re-hashes all 26 fixture sources against the descriptors pinned in museum.json@cd32303 (bytes, git blob and sha256 all match);
- confirms that docs/site/museum.mjs, museum.json, core.mjs, museum.html and tools/museum_data.py in the extract equal `git show cd32303:<path>`;
- renders the page through startMuseum with all network stubbed (no request left the fixture; the page state reports all views verified);
- inspects the six pointer_only cards.

All four sub-claims reproduce.

(1) The label at museum.mjs:151 is fixed


# LENS membership

## checked
- INCIDENT, needs operator action. An early version of probe p4 ran the PR's full tests/test_museum_data.py with GIT_DIR=/home/user/main/.git exported, so the probe could read pinned objects. The PR test test_historical_snapshot_ignores_unrelated_working_tree_append (tests/test_museum_data.py:112-133) then ran git init, git add and git commit against /home/user/main. That created commit 9f54729b8cd261513130646470e38928f76d1fd8 ('Pin historical status fixture', author 'Museum Test Fixture <museum-fixture@example.invalid>', parent b1335def2615888463e2d134294ef451895b8d0d) on the checked-out branch claude/drive-audit-github-migration-rrglpp. The commit adds a 58-line STATUS.md (the cd32303 STATUS.md plus one appended probe line). The index now contains STATUS.md and the working tree does not, so `git status` shows ' D STATUS.md'. The re-init also rewrote .git/config; I inspected it and found no core.worktree and no visible change. The commit also wrote 4 loose objects and COMMIT_EDITMSG. Nothing was pushed. I did not undo it, because I was told not to run state-changing git commands. Exact undo for the operator: git -C /home/user/main update-ref -m 'undo accidental museum-probe commit' refs/heads/claude/drive-audit-github-migration-rrglpp b1335def2615888463e2d134294ef451895b8d0d 9f54729b8cd261513130646470e38928f76d1fd8 && git -C /home/user/main reset -q -- STATUS.md. After that, git status should be clean. Since then, every probe reads objects through a bare scratch store, probes/membership/objstore.git, whose objects/info/alternates points at /home/user/main/.git/objects, so no probe can write to /home/user/main. After the final re-run of run_all.sh, /home/user/main HEAD is still 9f54729b (no further writes). Finding F5 is the PR-side defect behind this.
- Baseline at cd32303, no current error (P0). The declared main snapshot is 71400b94 (the per-source commit of STATUS.md and the packet files). `git ls-tree -d 71400b94 incoming/` lists exactly incoming/side24-identity-replay-20260926, and museum.json lists exactly that packet. The packet blobs are the same at 71400b94 and at cd32303. museum_data.build() reproduces cd32303 docs/site/museum.json byte for byte, both from git objects (git show 71400b94:<path>) and from verified plain fixtures. So at the declared snapshot no packet is falsely shown and none is omitted. The merge d77a85a7 has the same bytes as cd32303 for museum.json, museum_data.py, museum_check.py and museum.mjs.
- Browser path (museum.mjs:102-108, 189-192, renderPacket). It checks packet.id against one hard-coded id, requires result.path == incoming/<id>/RESULT.md and result.commit == status_source.commit, and fetches raw.githubusercontent.com/<repo>/<commit>/<path> with byte and sha256 checks plus a regex check of the RESULT.md declarations. I emulated raw-at-commit strictly (git show <commit>:<path>, 404 otherwise). A packet whose path is absent from the declared commit gets a refusal card, 'Source unavailable (404). No result inferred.' So the live display fails closed for non-members; the defect in F1 is in the generator and CI gate.
- Calibration (P2c): if the generator emits no packet at all, museum_check exits 1, because its node 'actual pinned museum' test requires /packet — not STATUS/. Only partial omission gets through (F2).
- packet.identity and packet.output descriptors in museum.json are not validated by museum.mjs at cd32303, but they are not displayed or fetched either. The Python build checks the IDENTITY.json and output.json authority fields (scientific_effect NONE, review_status REVIEW_REQUIRED, scientific_acceptance false) from pinned bytes. Not a finding.
- No branch alias (HEAD, main, latest) is used to name a source anywhere in the three files. The only routes by which a moving ref replaces the declared commit are the read_source working-tree fallback (F1) and the unit tests' read of working-tree STATUS.md (F4).
- CI (.github/workflows/public-shop.yml) checks out with fetch-depth: 0, so in CI the real 71400b94 pins are read from git objects. The fallback fires only when a (commit, path) pair is not in the object store: a path absent at that commit, an unknown commit, or a shallow or sparse checkout.
- museum_check.py builds MUSEUM_FIXTURE keyed by descriptor URL (which embeds the declared commit), with bytes from read_source. The node test then serves those bytes for any matching URL (tests/test_museum_frontend.mjs:131,135), so the harness cannot tell whether a path exists at its declared commit. The non-fixture git show branch of that test is strict, but museum_check always sets MUSEUM_FIXTURE.
- Context only, not reviewed in depth: current main 513261bb still has the read_source fallback (tools/museum_data.py:510-521) and a hard-coded packet list, now two packets with NEW_PACKET_COMMIT a12c178c. Its museum.mjs now validates the identity and output descriptors against PACKET_PINS. Its unit tests read tests/fixtures/status_f2e432e.md instead of live STATUS.md, which fixes F4 there.
- Network: no probe used the network. Math- sources at d6628da come from blob objects in local clones, plus the PROOF_INDEX.md bytes embedded in cd32303 tests/test_museum_data.py, plus a README.md copy recovered by the pointers lens; each was accepted only after verify_bytes. That GitHub raw serves commits that are not on the default branch (F3) is an assumption I did not probe.
- Every probe output is consolidated in <SCRATCH>/museum/probes/membership/ALL_OUTPUT.txt, produced by run_all.sh in the same directory (exit 0). Each modified tree is a copy (wt_*), and none of the extracted cd32303 bytes were edited.
- Scientific effect of this review: NONE. Same-provider reviewer (Anthropic/Claude), zero organizational-independence credit; any gate that needs independence stays open. These findings concern display and guard interfaces only. They do not test, support or weaken any mathematical claim, and a fixed or unfixed probe changes no claim's status.

## Q3-F1 [important] The generator and CI gate never check that a source is in its declared commit: read_source silently falls back to the working tree, and the harness serves those bytes under the pinned-commit URL
FILES: tools/museum_data.py:463-474 (read_source: on any `git show <commit>:<path>` failure it reads Path(root)/path); tools/museum_check.py:27,36-37 (bytes from read_source(ROOT) are stored under descriptor(...)['url'], which embeds the declared commit); tests/test_museum_frontend.mjs:131,135 (the fixture fetcher serves any mapped URL); tests/test_museum_data.py:105-110 (the test named for reading the landed packet from the 'pinned sparse checkout' passes through the fallback), 136-141; docs/site/README.md:48 ('binds ... main source/packet membership to 71400b94...')
PROBE: <SCRATCH>/museum/probes/membership/ (p1_fallback_unit.py, p1_ci_false_green.sh, mkcopy.sh, render.mjs, p1c_env_dependent_source.sh; fixture from build_fixture.py; objstore.git = bare read-only alternates store)
CMD: cd <SCRATCH>/museum/probes/membership && python3 -I build_fixture.py ../../cd32303 /home/user/Math- /home/user/math- /home/user/main && python3 -I p1_fallback_unit.py && ./p1_ci_false_green.sh && for m in harness strict; do /opt/node22/bin/node render.mjs wt_p1/docs/site/museum.json $m; done && ./p1c_env_dependent_source.sh

DESCRIPTION:
read_source() is the only way museum_data.build() and museum_check.py get bytes. It runs `git show <commit>:<path>` and, if that fails for any reason (the path is absent at that commit, the commit is unknown, a shallow checkout, or not a repository at all), it reads the file from the root's working tree. Only size, git blob id and sha256 are then checked. Nothing checks that <path> is in the tree of <commit>, which is exactly what 'packet membership at the declared snapshot' means.

I moved the main pin to bd9fc9d6, a real main commit (the parent of the packet's landing commit d10c2fb). Its STATUS.md, SARD review and notebook are byte-identical to the pins, but it has no incoming/ directory. With only that pin changed, the PR's generator emits museum.json with the packet 'at' bd9fc9d6, and the PR's CI gate tools/museum_check.py passes 15 of 15 node tests and exits 0. That includes the test 'actual pinned museum renders all cards', which renders the packet card with Commit bd9fc9d6.... The unit test test_landed_packet_can_be_read_from_pinned_sparse_checkout also passes.

Under strict raw-at-commit semantics the same manifest renders 'Packet source Unavailable: Source unavailable (404)'. So the deployed page fails closed, but the guard that claims to bind membership to the snapshot passes on a non-member.

The Git-versus-plain-file choice also depends on the environment. It depends only on whether git show succeeds in whatever repository git discovers from cwd. With a discoverable repository that holds 71400b94 (GIT_DIR, or a TMPDIR inside a full clone), the PR's own control test_plain_fixture_directory_does_not_bypass_byte_verification fails with 'ValueError not raised'. No test checks that a commit lacking the path is refused.

WHY:
The docs say main 'source/packet membership' is bound to 71400b94, and museum_check.py is the gate meant to enforce that. In fact the gate checks bytes against whatever the checkout's working tree (HEAD) holds, so membership is computed from a moving ref rather than the declared commit. A snapshot refresh that pins a packet to a commit where it does not exist goes green in CI and only appears after deploy, as a refusal card. The live page fails closed, so no status is misrepresented, but the guard passes on bad input. The fix is to refuse when a git root lacks <commit>:<path>, keeping plain-fixture reads to an explicit fixture mode, and to add a negative control for a path absent at a present commit.

OBSERVED:
71400b94 (declared snapshot, control)
   path in commit tree: True | museum_data.read_source: ACCEPTED (bytes returned) | strict commit:path read: accepted
bd9fc9d6 (main, parent of the packet's landing commit d10c2fb)
   path in commit tree: False | museum_data.read_source: ACCEPTED (bytes returned) | strict commit:path read: REFUSED
4aa316f (merged side-branch commit of #150, packet never in its tree)
   path in commit tree: False | museum_data.read_source: ACCEPTED (bytes returned) | strict commit:path read: REFUSED
ffff... (no such commit)
   path in commit tree: False | museum_data.read_source: ACCEPTED (bytes returned) | strict commit:path read: REFUSED
--- git ls-tree bd9fc9d603fd70946de364afcc3c902f09a5bf29 incoming/ (empty = no packet at the declared snapshot):
[end ls-tree]
--- generated museum.json packets:
side24-identity-replay-20260926 result.commit= bd9fc9d603fd70946de364afcc3c902f09a5bf29 status_source.commit= bd9fc9d603fd70946de364afcc3c902f09a5bf29
# tests 15
# pass 15
# fail 0
Museum source export and actual display integration verified. Scientific effect NONE.
museum_check exit=0
--- PR unit test that claims to read the landed packet from the pinned checkout:
Ran 1 test ... OK
[\harness] museum-state: Source projections and displayed source bytes verified. These checks establish byte identity, not mathematical acceptance.
[\harness] packet-cards: side24-identity-replay-20260926engineering-only Packet — not STATUS ... Commit bd9fc9d603fd70946de364afcc3c902f09a5bf29 ...
[strict] museum-state: Pinned source projections verified. 1 source view(s) unavailable; no result inferred for those views.
[strict] packet-cards: Packet source Unavailable: Source unavailable (404). No result inferred.
--- no discoverable repository:
OK
--- GIT_DIR=.../objstore.git (read-only):
AssertionError: ValueError not raised
FAILED (failures=1)

VERDICTS (2):
  - repro=True refuted=False sev=important :: I reproduced the finding at the cd32303 bytes and it holds. One claim in the write-up needs a wording correction.

BYTES. Every one of the 104 files in the extracted tree matches `git rev-parse cd32303bcf0f:<path>` (`git hash-object`, 104/104 identical). The only extras are `__pycache__` directories left by other probes. `read_source` is at tools/museum_data.py:463-479. When `git show <commit>:<path>` exits nonzero, it reads `Path(root)/path` (line 474). After that, only size, blob id and sha256 are checked.

INDEPENDENT PROBES. I wrote my own probes under probes/verify_membership_0 and ran them against a copy of the cd32303 tree. The copy has a hand-written non-bare `.git` that git discovers from cwd. Its only object source is a read-only alternates file pointing at /home/user/main/.git/objects, and GIT_DIR is not set. This is closer to the CI checkout than the original probe's GIT_DIR 
  - repro=True refuted=False sev=important :: CONFIRMED at the cd32303 bytes, with one correction to how the CI claim is worded. Severity stays important.

Bytes and setup checked. tools/museum_data.py, tools/museum_check.py, both test files, museum.mjs, museum.json, README.md and public-shop.yml each hash to the cd32303 blob (git hash-object compared with git rev-parse cd32303:<path>). The extracted tree equals cd32303's 104 files; the only additions are two __pycache__ files. Every cited line is accurate:
- museum_data.py:463-474: on a non-zero exit from `git show <commit>:<path>`, read_source reads Path(root)/path instead.
- museum_check.py:27 and 36-37: those bytes are stored under descriptor()['url'], which embeds the declared commit.
- test_museum_frontend.mjs:131 and 135: the fixture fetcher serves any mapped URL.
- test_museum_data.py:105-110 and 136-141.
- README.md:48: "binds ... main source/packet membership to 71400b94".

## Q3-F2 [minor] The packet list is hard-coded and never compared with incoming/ at the declared commit, so a packet present at the snapshot can be omitted with every guard green; MAIN_COMMIT is unused and the summary line prints a fixed count
FILES: tools/museum_data.py:20 (MAIN_COMMIT defined, used 0 times), 454-460 (packet_descriptors returns a literal one-element list; its docstring says 'Only the packet already present on the pinned main tree is selected'), 553 (prints the literal '1 landed packet'); docs/site/museum.mjs:102 (allows 0 or 1 packets), 191 ('No packet appears in this pinned projection.'); README.md:96 ('shows submissions present in its pinned default-branch snapshot'); docs/site/README.md:56
PROBE: <SCRATCH>/museum/probes/membership/ (p0_baseline.py, p2_omission.sh, p2c_empty_list_ci.sh, render.mjs, mkcopy.sh)
CMD: cd <SCRATCH>/museum/probes/membership && python3 -I p0_baseline.py && ./p2_omission.sh && ./p2c_empty_list_ci.sh

DESCRIPTION:
Nothing in museum_data.py, museum_check.py or museum.mjs lists incoming/ at the declared commit. Membership is the constant list in packet_descriptors() plus the constant id in museum.mjs. At cd32303 that list happens to be exactly right (P0: one packet at 71400b94).

I simulated a snapshot refresh, the documented path ('excluded until a reviewed engineering refresh'), by moving the main pin to a12c178c. That is the real main merge of #150. Its tree holds two packets (side24-chart-claude-20260926 and side24-identity-replay-20260926), and its STATUS.md, SARD review and notebook are byte-identical to the pins. With that change, the generator, museum_check (exit 0, 15 of 15 node tests) and a strict raw-at-commit render all report 'displayed source bytes verified' while showing one packet. Current main later had to add the chart packet by hand (NEW_PACKET_COMMIT a12c178c), which shows the path is realistic.

The browser validator also accepts packets: [] for the real 71400b94 manifest and prints 'No packet appears in this pinned projection.' over a snapshot that has a packet. CI does catch the fully empty case through the harness regex (P2c), so only partial omission gets through.

The generator's success line is a literal: it printed '1 landed packet' even when it had emitted packets=[] (P2c).

WHY:
The README says the list shows the submissions present in the pinned default-branch snapshot, a completeness claim that no guard enforces. Omission only hides a packet that carries no status effect, so this is minor. But the 'landed packet index' is not derived from the snapshot it names, and a refresh can silently drop members. The fix is to compare packet_descriptors() with `git ls-tree -d <commit> incoming/` in museum_check, and to compute the summary line.

OBSERVED:
MAIN_COMMIT constant: 71400b94f6cb354a8cf7aba73ffede2138a64efa | referenced elsewhere in museum_data.py: 0 times
packets in tree at snapshot: ['incoming/side24-identity-replay-20260926']
packets in museum.json     : ['incoming/side24-identity-replay-20260926']
--- (a) git ls-tree -d a12c178c0f857a130cf434e9efd44233a038195b incoming/:
incoming/side24-chart-claude-20260926
incoming/side24-identity-replay-20260926
# pass 15
# fail 0
Museum source export and actual display integration verified. Scientific effect NONE.
museum_check exit=0
[strict] manifest status_source.commit=a12c178c packets=["side24-identity-replay-20260926@a12c178c"]
[strict] museum-state: Source projections and displayed source bytes verified. These checks establish byte identity, not mathematical acceptance.
--- (b) real cd32303 manifest with packets:[] (git ls-tree -d 71400b94 incoming/ -> incoming/side24-identity-replay-20260926)
[strict] museum-state: Source projections and displayed source bytes verified. ...
[strict] packet-cards: No packet appears in this pinned projection.
##### P2c
Museum projection verified: 11 reviewed bullets, 3 AMEND rows, 1 landed packet; scientific effect NONE.
packets in regenerated museum.json: []
museum_check exit=1
not ok 15 - actual pinned museum renders all cards ...
    The input did not match the regular expression /packet — not STATUS/.

VERDICTS (2):
  - repro=True refuted=False sev=minor :: VERDICT: the finding is CONFIRMED at the cd32303 bytes. It is not refuted, it is in scope, and minor is the right severity. Two details in its wording need tightening (below).

WHAT I DID. I re-ran the finding's own scripts (p0_baseline.py, p2_omission.sh, p2c_empty_list_ci.sh). I ran copies under probes/verify_membership_0/q3f2_orig/ with only the cd32303 path repointed, so the other lens's outputs were not overwritten. Every line of their observed output reproduced. I also wrote independent probes under <SCRATCH>/museum/probes/verify_membership_0/q3f2/:
- run_all.sh runs everything; the output is in ALL_OUTPUT.txt.
- Each worktree is an exact tar copy of cd32303 with a .git whose only object source is a read-only alternate of /home/user/main objects. This mirrors the CI checkout (fetch-depth 0, git discovered from cwd)
  - repro=True refuted=False sev=minor :: Verifier 2 of 2 for Q3-F2. I wrote my own probes in <SCRATCH>/museum/probes/verify_membership_1/q3f2/ (run_all.sh, with output in ALL_OUTPUT.txt). They run against the extracted cd32303 bytes. All 12 relevant files hash-match `git rev-parse cd32303:<path>`, including museum_data.py, museum_check.py, museum.mjs, museum.json, the three test files, README.md, docs/site/README.md and public-shop.yml. Main objects are read through a hand-made bare objstore that only links to /home/user/main via alternates. The Math- fixture was re-verified with the cd32303 verify_bytes (30 of 30). No network was used and no repository state changed.

What I confirmed at cd32303:
(1) MAIN_COMMIT appears only at museum_data.py:20 and is never used. Each main SOURCES entry carries its own literal commit.
(2) None of the seven museum files enumer

## Q3-F3 [minor] Nothing checks that the declared snapshot is on the default branch; a NEVER-MERGE test commit passes every gate while the page says 'pinned landed submissions'
FILES: tools/museum_data.py (SOURCES main pins and read_source: no ancestry or first-parent check); tools/museum_check.py (none); docs/site/museum.mjs:107 (requires only result.commit == status_source.commit); docs/site/museum.html:26 ('These are pinned landed submissions.'); README.md:96 ('pinned default-branch snapshot')
PROBE: <SCRATCH>/museum/probes/membership/p3_unlanded_snapshot.sh (with mkcopy.sh, render.mjs)
CMD: cd <SCRATCH>/museum/probes/membership && ./p3_unlanded_snapshot.sh

DESCRIPTION:
The museum says its packets are 'landed' and come from a 'default-branch snapshot', but the commit pin is only ever checked to exist and to hold matching bytes. I moved the main pin to 3b8c4545 = origin/chatgpt/never-merge-formal-negative-20260927 ('TEST ONLY: inject formal receipt-stage failure; NEVER MERGE'). It is not an ancestor of main 513261bb or of the #151 merge d77a85a7, but its STATUS.md, SARD review, notebook and packet are byte-identical to the pins. The generator and museum_check (exit 0) pass, and a strict commit:path render reports 'displayed source bytes verified'. Grepping the three files for merge-base, is-ancestor, rev-list, first-parent, refs/heads or origin/main finds nothing. The real 71400b94 pin is on main (verified), so cd32303 itself is correct. The hard-coded packet id limits the effect to mislabelling the snapshot rather than showing a different packet. Not probed (no network): that raw.githubusercontent.com serves files for such a commit, which I expect because the branch exists on origin.

WHY:
Being 'landed' is the property the packet index advertises, and no guard checks it. IMPLEMENTATION.md W9 ('Reject draft-PR inclusion') is enforced only by the hard-coded id, not by membership logic. The fix is a first-parent ancestry check of the main pin against the default branch in museum_check.

OBSERVED:
--- 3b8c454 TEST ONLY: inject formal receipt-stage failure; NEVER MERGE
    contained in:   origin/chatgpt/never-merge-formal-negative-20260927
    ancestor of main 513261bb: NO
    ancestor of #151 merge d77a85a7: NO
# pass 15
# fail 0
Museum source export and actual display integration verified. Scientific effect NONE.
museum_check exit=0
[strict] manifest status_source.commit=3b8c4545 packets=["side24-identity-replay-20260926@3b8c4545"]
[strict] museum-state: Source projections and displayed source bytes verified. These checks establish byte identity, not mathematical acceptance.
These are pinned landed submissions. A packet's presence, output or passing check does not promote a claim. Review remains a separate process.
--- ancestry / default-branch checks in the three files:
../../cd32303/tools/museum_data.py:0
../../cd32303/tools/museum_check.py:0
../../cd32303/docs/site/museum.mjs:0

VERDICTS (0):


## Q3-F4 [minor] The unit tests take STATUS.md from the live working tree, not the snapshot; one unrelated STATUS edit breaks 6 tests, including the one named for working-tree immunity, and makes a negative control fail for the wrong reason
FILES: tests/test_museum_data.py:20-21 (setUp: self.status = (ROOT / 'STATUS.md').read_bytes()), 48-52 (test_misdirected_review_pointer_refused), 112-133 (test_historical_snapshot_ignores_unrelated_working_tree_append)
PROBE: <SCRATCH>/museum/probes/membership/p4_status_moving_ref.sh
CMD: cd <SCRATCH>/museum/probes/membership && ./p4_status_moving_ref.sh

DESCRIPTION:
The museum binds STATUS.md, and with it the packet snapshot through museum.mjs:107, to 71400b94. The PR's unit tests instead read STATUS.md from HEAD's working tree and require it to match the 71400b94 pin. I appended one unrelated line to working-tree STATUS.md, leaving the snapshot unchanged. Six tests then fail with 'source identity mismatch: STATUS.md'. One of them is test_historical_snapshot_ignores_unrelated_working_tree_append. The negative control test_misdirected_review_pointer_refused fails for the wrong reason, with "review pointer" not matching "source identity mismatch: STATUS.md", so it no longer tests what it names. The CI gate on the same tree still passes, because it reads git show 71400b94:STATUS.md. This fails closed. STATUS.md changed 4 times on main after cd32303, and current main moved the read to tests/fixtures/status_f2e432e.md.

WHY:
The tests compute the projection from a moving ref while the product binds it to a commit. Normal STATUS maintenance therefore breaks CI, and negative controls fail for the wrong reason, which hides whether they still catch what they name. This fails closed, so it is robustness only. The fix is to read git show <pin>:STATUS.md or a pinned fixture, as main later did.

OBSERVED:
--- control, unmodified STATUS.md:
OK
--- after appending one unrelated line to working-tree STATUS.md:
unit exit=1
      6 ValueError: source identity mismatch: STATUS.md
      1 FAILED (failures=1, errors=5)
      1 FAIL: test_misdirected_review_pointer_refused
      1 ERROR: test_review_pointers_do_not_claim_comment_byte_custody
      1 ERROR: test_replay_commands_bind_source_and_checkout
      1 ERROR: test_historical_snapshot_ignores_unrelated_working_tree_append
      1 ERROR: test_exact_fourteen_projections_and_distinct_d5_scopes
      1 ERROR: test_check_rejects_quote_identity_and_class_tampering
      1 AssertionError: "review pointer" does not match "source identity mismatch: STATUS.md"
--- CI gate on the same tree (full-clone objects via GIT_DIR):
museum_check exit=0
Museum source export and actual display integration verified. Scientific effect NONE.

VERDICTS (0):


## Q3-F5 [minor] test_historical_snapshot_ignores_unrelated_working_tree_append is not isolated from an inherited GIT_DIR: it re-inits and commits into whatever repository GIT_DIR names, then passes (this happened to /home/user/main during this review)
FILES: tests/test_museum_data.py:112-133 (git init / git add / git commit are run with the inherited environment and no env= override)
PROBE: <SCRATCH>/museum/probes/membership/p5_test_writes_inherited_gitdir.sh
CMD: cd <SCRATCH>/museum/probes/membership && ./p5_test_writes_inherited_gitdir.sh

DESCRIPTION:
This sits at the edge of the Q3 lens: it is the PR's snapshot-membership test, and it falls under the fail-open sweep of the test files. The test creates a temporary 'checkout' and runs `git init`, `git add STATUS.md` and `git commit` with the caller's environment. If GIT_DIR is exported (git exports it to hooks, and this review's first object-reading harness exported it), git init re-initializes the GIT_DIR repository, git add stages the temp file into that repository's index, and git commit records 'Pin historical status fixture' on that repository's checked-out branch. The test then passes, because read_source finds the commit through the same GIT_DIR. I reproduced this only against a hand-made scratch host repository. During this review it also happened, unintentionally, to /home/user/main (commit 9f54729b; see checked[0] for the undo).

WHY:
When run under an exported GIT_DIR, the test silently changes the operator's repository: a new commit on the current branch and a staged file. In this program, undisclosed repository edits are themselves a governance problem. The fix is to pass env with GIT_DIR, GIT_WORK_TREE, GIT_INDEX_FILE and GIT_OBJECT_DIRECTORY removed to every git subprocess in the test, and in read_source as well.

OBSERVED:
--- host repo commits before: 0
OK
--- host repo after:
    11f9f4e Museum Test Fixture <museum-fixture@example.invalid> | Pin historical status fixture
     STATUS.md | 56 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     1 file changed, 56 insertions(+)
(and on /home/user/main, read-only inspection: '9f54729b8cd261513130646470e38928f76d1fd8 b1335def2615888463e2d134294ef451895b8d0d | Museum Test Fixture <museum-fixture@example.invalid> | 2026-10-12 17:01:51 +0000 | Pin historical status fixture'; reflog '9f54729 HEAD@{2026-10-12 17:01:51 +0000}: commit: Pin historical status fixture'; git status ' D STATUS.md')

VERDICTS (0):



# LENS limits

## checked
- boundedFetch (museum.mjs:13-32) uses one AbortController and one deadline for both headers and body. Probe p0 C1: headers never arrive and the fetcher ignores the signal -> 'rejected "Source request timed out" after 151ms', signal.aborted=true. The timer is cleared in finally. Promise.race handles the losing promise's rejection, so no unhandled rejection.
- A declared Content-Length above the cap is refused before the body is read (p0 C2). The streaming cap still holds when Content-Length is false ('10') or non-numeric ('abc') (p0 C3, C4). Invalid bound arguments are refused (p0 C5). The streaming cap counts decoded bytes, so Content-Encoding cannot hide a large body.
- The cap is enforced before concatenation and before any hashing or parsing. Order: museum.mjs:27 byte check, then :28 concatenation, then core.mjs:24-26 length check and crypto.subtle.digest, then TextDecoder/JSON.parse. Peak use per request is the cap plus up to two pulled chunks: p2 contrast pulled 4 MiB for the 2 MiB cap with 1 MiB chunks.
- Production sizes against caps: museum.json is 56,448 B (compact 50,173 B) against the 256 KiB cap. museum.json has 28 unique pinned source URLs, the largest 61,961 B against the 2 MiB cap. validateDescriptor's bytes<=2 MiB matches the boundedFetch default. Per-URL promise dedupe (cachedFetch) bounds total memory at about 28 x 2 MiB.
- Failure display (p7 control). While proof.md stalls: placeholder 'Verifying Object 0…', state 'Pinned source projections verified: 11 reviewed index entries and 3 separate AMEND rows. Verifying each linked source before display…'. At t=20.1s: card 'Unavailable: Source request timed out. No result inferred.' and state 'Pinned source projections verified. 13 source view(s) unavailable; no result inferred for those views.' Over-cap source: 'Unavailable: Response exceeds byte limit. No result inferred.' The 'Source byte count and SHA-256 verified' strip is rendered only after verifyClaim resolves. On a global failure every container is cleared.
- A cached rejection (timeout or cap) is shared by every consumer of that URL. This fails closed: each dependent card shows a refusal.
- An OK response with a null body yields empty bytes, which fails the byte-count check or JSON.parse (fails closed).
- A large single chunk is checked only after it arrives: the cap is per chunk, so memory already allocated by the stream producer is not prevented. Browser network chunk sizes make this unrealistic, so it is not a finding. A fetcher that ignores the abort signal and resolves after the deadline would keep reading in the background. Browser fetch honours AbortSignal, so this is not a finding.
- geometry.mjs loadThree has a 15 s timeout and SRI. Timeout or error falls back to the 2D view with an honest status line. mountGeometry's other paths are synchronous.
- museum_data.py verify_bytes checks length, git blob SHA-1 and SHA-256 before any decode or parse. The network read is size-capped at bytes+1 (museum_data.py:478). urllib follows redirects where the browser uses redirect:'error'. That is harmless for this lens because bytes are hash-verified afterwards. It is left to the pointer-identity lens.
- museum_check.py: ThreadPool(8) reads, then base64 fixture, then a 'node --test' subprocess with check=True and no timeout. CI bounds it with public-shop.yml timeout-minutes: 5. The CI checkout uses fetch-depth: 0, so main sources resolve via git show. Math- sources are fetched over the network because no --math-root is passed.
- Current main 513261bb, for context only: boundedFetch's early return on !ok is unchanged (museum.mjs:37). verifyClaim(claim,fetcher=fetch) is unchanged (museum.mjs:133) and core.mjs:33 still calls arrayBuffer() with no bound. museum_data.py:516-524 has the same git show / read_bytes / urlopen(timeout=30) pattern. museum.json is now pinned via config.json with pin.bytes<=256 KiB; finding 4 was not re-probed on main.
- Housekeeping, not caused by this lens. All my git commands were read-only (cat-file, remote). Probes ran with python -I -B and wrote only under probes/limits/. I observed tools/__pycache__ (16:57) and tests/__pycache__ (17:02) inside the extracted cd32303 tree, and 'git -C /home/user/main status' showing ' D STATUS.md' with .git/index modified 17:03:39. These timestamps line up with the membership lane's STATUS.changed/STATUS.control (17:02:18).

## Q4-1 [minor] boundedFetch leaves the body open on a non-OK response: it neither aborts the controller nor cancels the body
FILES: docs/site/museum.mjs:22 (cd32303); unchanged on main 513261bb docs/site/museum.mjs:37
PROBE: <SCRATCH>/museum/probes/limits/p1_nonok_body_not_released.mjs
CMD: /opt/node22/bin/node <SCRATCH>/museum/probes/limits/p1_nonok_body_not_released.mjs <SCRATCH>/museum/cd32303

DESCRIPTION:
Every other exit from boundedFetch calls cancel(): the byte cap, Content-Length over the cap, and the deadline. The early return `if(!response.ok)return {ok:false,status:response.status,bytes:new Uint8Array()};` does not. The 404/500/503 response body stays unlocked and uncancelled and the AbortSignal is never aborted, so the connection (or HTTP/2 stream) and any buffered error-page bytes stay held until garbage collection. The display is honest: verifiedBytes then throws 'Source unavailable (status)' and a refusal card is shown.

WHY:
This is a resource leak on a realistic path (raw.githubusercontent.com does return 404/5xx), and it contradicts the comment above boundedFetch that requests are bounded. It affects robustness only and has no effect on scientific status. Fix: call cancel() (or response.body?.cancel()) before the early return. No test covers it.

OBSERVED:
status=404 result.ok=false result.status=404 | after deadline: signal.aborted=false body.cancelled=false body.locked=false bodyUsed=false
status=500 result.ok=false result.status=500 | after deadline: signal.aborted=false body.cancelled=false body.locked=false bodyUsed=false
status=503 result.ok=false result.status=503 | after deadline: signal.aborted=false body.cancelled=false body.locked=false bodyUsed=false
contrast (200, over cap): rejected "Response exceeds byte limit" signal.aborted=true body.cancelled=true

VERDICTS (0):


## Q4-2 [minor] The size and time bound belongs to startMuseum's private cachedFetch; the exported verifyClaim with its default fetcher (global fetch) is unbounded
FILES: docs/site/museum.mjs:111-119 (verifyClaim(claim,fetcher=fetch)); docs/site/core.mjs:22-24 (fetcher(target,{credentials,redirect}) then await response.arrayBuffer(), with no signal, cap or deadline); tests/test_museum_frontend.mjs:150 calls verifyClaim with a raw fetcher
PROBE: <SCRATCH>/museum/probes/limits/p2_verifyclaim_default_unbounded.mjs
CMD: /opt/node22/bin/node <SCRATCH>/museum/probes/limits/p2_verifyclaim_default_unbounded.mjs <SCRATCH>/museum/cd32303

DESCRIPTION:
verifyClaim is exported and defaults to global fetch. It reaches core.mjs verifiedBytes, which buffers the whole body with arrayBuffer() and passes no AbortSignal or deadline. Only after the whole body is buffered is the byte count compared. In production at cd32303 every call goes through cachedFetch -> boundedFetch, so the shipped page is bounded. The verification API itself is not, and the module's own integration test calls it with a raw fetcher. The same unbounded primitive serves app.js (the shop page, outside this review's scope).

WHY:
Any other caller of the exported verifyClaim (or of verifiedBytes) gets no memory or time protection. That includes a future detail page or a test harness. The pinned byte count is known before the fetch and could serve as the cap. Fix: make verifiedBytes itself bounded, capped at pin.bytes with a deadline, or make verifyClaim default to boundedFetch. Production display at cd32303 is unaffected.

OBSERVED:
A default fetcher: rejected "Source byte count mismatch" only after buffering 64 MiB (pin.bytes=13); init.signal: no signal passed; 186ms
A boundedFetch contrast: rejected "Response exceeds byte limit" after 4 MiB pulled
B default fetcher, stalled body: STILL PENDING after 25002ms (boundedFetch default deadline is 20000ms)

VERDICTS (0):


## Q4-3 [minor] No negative control pins startMuseum's use of boundedFetch or its production bounds; removing or loosening every bound leaves the frontend suite green
FILES: docs/site/museum.mjs:180 (manifest boundedFetch at 256 KiB), :184 (cachedFetch -> boundedFetch defaults 2 MiB / 20 s), :15 (defaults); tests/test_museum_frontend.mjs:69-80 (boundedFetch tested directly, with explicit bounds of 12 B / 15 ms only), :113-125 and :127-151 (startMuseum served complete, immediate in-memory Responses)
PROBE: <SCRATCH>/museum/probes/limits/p6_bound_wiring_mutants.mjs
CMD: /opt/node22/bin/node <SCRATCH>/museum/probes/limits/p6_bound_wiring_mutants.mjs <SCRATCH>/museum/cd32303

DESCRIPTION:
Five versions were run against the unmodified cd32303 test file, each a scratch copy of museum.mjs (the extracted tree was not modified): the unmodified copy as baseline, then four mutants. One makes the manifest fetch unbounded. One makes cachedFetch bypass boundedFetch. One does both. One raises the production defaults to 1 GiB / 1 day. All four mutants pass 14/14. The pinned-museum test (skipped offline, run in CI under MUSEUM_FIXTURE) was not executed here: the Math- README blob abe16871 is not available locally. Its fetcher at lines 133-139 returns complete, immediate Responses of at most 61,961 B, which an unbounded mutant accepts identically, so by inspection it cannot detect these mutations either.

WHY:
The bounded-fetch fix that IMPLEMENTATION.md:45 cites as resolving earlier findings can regress silently. Fix: add a startMuseum-level control. Serve an over-cap museum.json and an over-cap or stalled source through startMuseum, and assert the refusal text, signal.aborted and a bounded number of pulled chunks.

OBSERVED:
M0 unmodified copy (baseline): exit=0 pass=14 fail=0 skipped=1
M1 manifest fetch unbounded (no 256 KiB cap, no deadline): exit=0 pass=14 fail=0 skipped=1
M2 source fetches unbounded (cachedFetch bypasses boundedFetch): exit=0 pass=14 fail=0 skipped=1
M3 M1+M2 together: exit=0 pass=14 fail=0 skipped=1
M4 production defaults raised to 1 GiB / 1 day: exit=0 pass=14 fail=0 skipped=1

VERDICTS (0):


## Q4-4 [minor] CI checks the 256 KiB museum.json cap against re-serialized compact JSON, not the bytes the page serves; nothing bounds the export's size
FILES: tests/test_museum_frontend.mjs:134 (`if(url==='museum.json')return new Response(JSON.stringify(manifest));`); docs/site/museum.mjs:180 (256*1024 cap); tools/museum_data.py:531-533 (dump: indent=2, sort_keys) and :535-537 (check_export compares bytes only, no size bound); tools/museum_check.py:37-43
PROBE: <SCRATCH>/museum/probes/limits/p3_manifest_cap_tested_on_reserialized_bytes.mjs
CMD: /opt/node22/bin/node <SCRATCH>/museum/probes/limits/p3_manifest_cap_tested_on_reserialized_bytes.mjs <SCRATCH>/museum/cd32303

DESCRIPTION:
The page fetches the pretty-printed bytes that museum_data.dump() writes. The only CI path that exercises the cap serves JSON.stringify(manifest), which is about 11% smaller. Neither museum_data.py nor museum_check.py enforces the 256 KiB limit at export time. The probe pads the unvalidated 'meaning' field until compact output is exactly the cap. The CI path then passes the cap and moves on to source fetches. The production bytes (268,419 B from museum_data.dump) are refused. Consequence: fails closed (the whole page shows Unavailable). Likelihood is low: the current export has 4.64x headroom.

WHY:
A green CI run could ship an export that the live page always refuses, so the cap is not tested where it applies. Fix: serve the real museum.json bytes in the integration test, or have museum_data/museum_check assert len(dump(data)) <= 256 KiB. Current main pins museum.json through config.json; this gap was not re-probed there.

OBSERVED:
actual museum.json: served(pretty)=56448 compact=50173 cap=262144
padded export: compact=262144 museum_data.dump()=268419 cap=262144
CI test path (JSON.stringify): museum-state = "Unavailable: Source unavailable (404). No result inferred."
production path (dump bytes): museum-state = "Unavailable: Response exceeds byte limit. No result inferred."

VERDICTS (0):


## Q4-5 [minor] museum_data.read_source network path: timeout=30 is a per-socket-operation timeout, not a deadline, so a trickling response is never timed out
FILES: tools/museum_data.py:477-478 (`with urlopen(..., timeout=30) as response: raw = response.read(identity["bytes"] + 1)`); unchanged on main 513261bb tools/museum_data.py:524; CI bound .github/workflows/public-shop.yml:12 (timeout-minutes: 5)
PROBE: <SCRATCH>/museum/probes/limits/p4_urlopen_timeout_is_per_recv.py
CMD: python3 -I -B <SCRATCH>/museum/probes/limits/p4_urlopen_timeout_is_per_recv.py <SCRATCH>/museum/cd32303

DESCRIPTION:
The size cap (bytes+1) is correct. The timeout, however, applies to each recv, so a peer that sends one byte every <30 s is never timed out. The probe used the real stdlib path that urlopen returns: http.client.HTTPResponse.read(amt) over a socket with read_source's own timeout value (30 s), served over a local socketpair with no network. The response delivered the last two bytes 20 s apart. read_source accepted after 40.0 s without error. In CI the 5-minute job timeout turns this into a failure, not a hang (fails closed). Local runs of museum_data.py or museum_check.py without --math-root have no overall bound.

WHY:
The exporter and checker can stall for an unbounded time on a degraded network or CDN. That is robustness only: bytes are still hash-verified, so nothing false is accepted. Fix: enforce a total deadline, for example by reading in chunks against time.monotonic().

OBSERVED:
url=https://raw.githubusercontent.com/d6g8k5htny-coder/main/71400b94f6cb354a8cf7aba73ffede2138a64efa/incoming/side24-identity-replay-20260926/output.json
timeout passed by read_source=30s; byte gap=20.0s
read_source: ACCEPTED 362 verified bytes after 40.0s (> the 30s timeout, no error)
worst-case per source at one byte per 29.9 s: largest pin 61961 B -> 21.4 days

VERDICTS (0):


## Q4-6 [minor] museum_data.read_source local paths read the whole git object or working-tree file before any size check; a FIFO blocks forever and /dev/zero exhausts memory
FILES: tools/museum_data.py:466-475 (subprocess.run(['git','show',...], capture_output=True) with no timeout or size limit; fallback `(Path(root) / identity['path']).read_bytes()`); unchanged on main 513261bb tools/museum_data.py:516-521
PROBE: <SCRATCH>/museum/probes/limits/p5_local_fallback_uncapped.py
CMD: python3 -I -B <SCRATCH>/museum/probes/limits/p5_local_fallback_uncapped.py <SCRATCH>/museum/cd32303

DESCRIPTION:
The pinned byte count is known before the read, and the network path uses it (read(bytes+1)). The local paths do not: git show output is captured whole, and when git show fails the working-tree file is read whole (following symlinks). Only then does verify_bytes compare. The probe used a plain-folder --math-root, with each case in a child process limited to RLIMIT_AS 1 GiB and 10 s wall clock. A FIFO at a pinned path blocks indefinitely. A /dev/zero symlink raises MemoryError. A 300 MiB file is read fully before the mismatch error. Reachability: --math-root given a plain fixture folder or a checkout without the pinned commit. This path is not reached in CI at cd32303: fetch-depth 0 means main sources come from git show, and Math- sources come from the network.

WHY:
The local and network read paths behave differently: only the network path caps its read, and the local hang or OOM happens before verification. This affects robustness only (fails closed; no bad bytes are accepted). Fix: refuse anything that is not a regular file, read at most bytes+1, and give git show a timeout (or use git cat-file --batch with a size check).

OBSERVED:
fifo     at math_root/frontiers/full_price_20260924/PROOF.md: STILL BLOCKED after 10.0s wall clock (killed by the probe)
devzero  at math_root/frontiers/full_price_20260924/PROOF.md: exit=0 child: MemoryError: '' after 0.8s; pinned bytes=11352 
bigfile  at math_root/frontiers/full_price_20260924/PROOF.md: exit=0 child: ValueError: 'source identity mismatch: frontiers/full_price_20260924/PROOF.md' after 1.0s; pinned bytes=11352 

VERDICTS (0):


## Q4-7 [minor] The geometry module import and mount is the only unbounded await in startMuseum; if it never settles, the page-level state never leaves its interim text
FILES: docs/site/museum.mjs:176 (geometryLoader=()=>import('./geometry.mjs')), :202 (await geometryLoader(); await module.mountGeometry(...) with no deadline), :206 (await Promise.all(jobs))
PROBE: <SCRATCH>/museum/probes/limits/p8_geometry_loader_no_deadline.mjs
CMD: /opt/node22/bin/node <SCRATCH>/museum/probes/limits/p8_geometry_loader_no_deadline.mjs <SCRATCH>/museum/cd32303

DESCRIPTION:
Every source fetch has a 20 s deadline and loadThree has 15 s. The dynamic import of geometry.mjs and the mountGeometry promise have no deadline. If the loader stalls (for example a stalled same-origin module response), Promise.all(jobs) never resolves. museum-state then stays at the interim text indefinitely, even though all 14 cards have rendered. The exhibit card stays without its 'Exact source excerpt'. The interim text is honest: it does not claim completion or acceptance, so this is not a status misrepresentation. Control: a loader that rejects produces an honest refusal and a final state.

WHY:
The page's final summary depends on an await with no time limit, which breaks the otherwise consistent 'every wait is bounded' design. This is robustness and clarity only. Fix: race the loader and mount against a deadline and route expiry to the existing refusal path.

OBSERVED:
control (loader rejects): state="Pinned source projections verified. 1 source view(s) unavailable; no result inferred for those views."
control exhibit="Source illustrationUnavailable: module load failed. No result inferred."
stalled loader, t=25s: startMuseum settled=false; claim cards rendered=14/14
stalled loader, t=25s: state="Pinned source projections verified: 11 reviewed index entries and 3 separate AMEND rows. Verifying each linked source before display…"
stalled loader, t=25s: exhibit="D4 fixed-remote RNillustrationFixed-remote regionRepositoryd6g8k5htny-coder/Math-Pathproof.mdCommitaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaSHA-25693feb43892283b"

VERDICTS (0):



# LENS failopen

## checked
- Probe dir: <SCRATCH>/museum/probes/failopen/. run_all.sh re-runs every probe offline and writes ALL_OUTPUT.txt (re-run once; the mutant table came out identical). The extracted cd32303 bytes were only read. Every mutant ran on a fresh scratch copy that was then deleted.
- Fixture: ../pointers/fixture/verified-inputs.json was checked by verify_fixture.py against all 30 cd32303 SOURCES pins (byte count, git blob SHA-1, SHA-256) before use: 'fixture verified: 30 of 30 SOURCES pins'. The fetcher is stubbed and only serves museum.json plus those 30 pinned raw URLs. Anything else throws. No network calls.
- Baseline on a scratch copy, CI steps run offline: test_museum_data 12/12 OK; museum_check --math-root fixture passes 15/15 frontend tests with the pinned integration test executed (0 skipped); test_museum_geometry 8/8. p0_control: the committed museum.json passes validateBoundManifest and renders fully verified with correct labels. The shipped bytes are not wrong; the findings are about what would not be refused.
- Every catch in museum.mjs was traced: the claim, lifetime, packet and exhibit catches each render a refusal and return false. The outer catch clears all containers. The reader.cancel().catch only swallows cancel errors. None of the five continues to a verified display in the shipped code. In geometry.mjs, mountEC014's catch falls back to the source-defined 2D diagram, which is illustration only and claims no verification.
- boundedFetch: a non-ok response returns ok:false and verifiedBytes refuses it. A NaN content-length skips only the header pre-check; the streaming cap still applies. A body-less 200 gives 0 bytes, which validateDescriptor's bytes>=1 refuses. The response cache keeps rejected promises, so a failure stays a failure. Size and time limits are left to the limits lens.
- Unexpected types: validateDescriptor uses Number.isSafeInteger, so NaN, string and bool byte counts are refused. claim.review of {}, false, 0 or undefined is refused. A falsy replay.source with a non-empty command is refused. manifest null leads to the outer refusal. verify_bytes rejects bool byte counts, and a None commit raises TypeError (still fail-closed). Exceptions inside ThreadPoolExecutor.map propagate in build() and museum_check.
- Duplicate JSON keys: the pinned IDENTITY.json, output.json and museum.json have none (checked with object_pairs_hook). museum.json is byte-compared by check_export, and the browser parse is last-wins, the same object for validation and rendering. No divergence.
- geometry.mjs: NaN/Infinity/negative inputs to ec014Model, remoteModel, annulusModel and p15Model are refused or return false. The mountGeometry identity guard is killed by the tests (mutant G01). Surviving mutants G02 (p15 duplicate-selection guard) and G06 (Three.js revision check) have no reachable display path: the UI uses a Set, and SRI pins the CDN bytes. Not reported as findings. remoteModel/annulusModel are never called by the drawing code.
- core.mjs (outside scope but relied on): deleting the SHA-256 comparison is caught by test_public_shop_frontend.mjs in the same workflow. In test_museum_frontend.mjs the 'Changed proof.' control has a different length, so it only exercises the byte-count check.
- Mutants that the tests killed (controls showing the harness detects failures): M11 pointer-source identity, M17 replay-source verification, M24 proof-byte verification, M28 status authority, G01 geometry identity, P02 review-pointer binding, P11 check_export.
- Informational, current main 513261bb (validation only; STATUS blob 3aeb0412 checked against its pinned sha256): FO-1 and FO-2 validator behaviour persists (p9: all three tampers PASS). The test-gap findings were not re-checked on main, whose test files differ.
- Overlap with other lenses: the review_url-on-byte-verified-review link, review_notice wording, suffix/substring pointer binding, replay-command-as-any-README-line, empty packets list, read_source working-tree fallback and limits are covered by the pointers, substitution, membership and limits probe directories. They are not repeated here except where a fail-open mechanism differs.
- What this sweep does not establish: it says nothing about the mathematical correctness or review status of any claim. A green or red probe is a record of a computation, not evidence about a theorem. It gives no independence credit, and any gate requiring organizational independence remains open. No status label is changed.

## FO-1 [critical] Status-bearing card text (proof.availability, replay.notice, replay.url without replay.source) is never bound to a source: missing or arbitrary values pass validateBoundManifest and render inside a 'verified' card
FILES: docs/site/museum.mjs:79 (sameIdentity compares 8 keys only, not availability), :96-98 (replay.notice only non-empty; replay.url only shape-checked; no verification when replay.source is absent), :146-148 (the proof link label and the 'No landed proof is supplied' line both switch on claim.proof.availability), :156-158 (replay.notice and replay.url rendered as given); tools/museum_data.py:440-449 (availability and the AMEND replay notice are generator literals that no test pins)
PROBE: <SCRATCH>/museum/probes/failopen/p1_availability_unbound.mjs ; p5_replay_unbound.mjs ; p6_generator_path_ci_green.py (same dir)
CMD: cd <SCRATCH>/museum/probes/failopen && PATH=/opt/node22/bin:$PATH node p1_availability_unbound.mjs && PATH=/opt/node22/bin:$PATH node p5_replay_unbound.mjs && python3 -I p6_generator_path_ci_green.py

DESCRIPTION:
validateBoundManifest binds scope_quote, title, class, source_label, status_quote, and the proof/review/replay identities. It never looks at proof.availability or at the content of replay.notice. replay.url gets only a URL-shape regex, and verifyClaim fetches nothing when replay.source is absent. renderClaim uses these fields to decide what an open obligation looks like:
- availability absent: the link reads 'Open pinned full proof' instead of 'Open pinned source contract', and both 'NO COMPLETE PROOF YET' and 'No landed proof is supplied for this open obligation.' disappear.
- availability set to arbitrary text: the text is printed verbatim.
- replay.notice: any non-empty text is printed in the Replay column.
- replay.url without a source: shown as 'Open pinned replay source' with no byte check.
The banner and strip still say 'Source byte count and SHA-256 verified'. A realistic path also gets past museum_check. That checker compares museum.json with the output of the same generator, so a generator edit to these literals regenerates a museum.json that every museum CI step accepts (p6). Mitigation: class AMEND/open is enforced, so the line 'OPEN / NOT LANDED — the claimed closure remains open.' stays on the card, and the card contradicts itself rather than flipping class.

WHY:
PROOF_INDEX says 'NO COMPLETE PROOF YET' for the D5 pin-neighbourhood obligation, yet the page can label it 'Open pinned full proof' or print 'Replay reproduces the closed result' on an AMEND card. Both validateBoundManifest and the CI gate pass, and the banner reports verification. The PR's offer, in the scope it asked to have reviewed, says catalog presence 'must not imply ... admissible or scientifically accepted'. Fix: derive these strings from class and claim id in museum.mjs (fixed literals per class), or require exact equality with an allow-listed value per claim id, and add negative controls.

OBSERVED:
p1:
== availability deleted on claims 12,13
   validateBoundManifest: PASS
   museum-state: Source projections and displayed source bytes verified. These checks establish byte identity, not mathematical acceptance.
   d5-pin-neighborhoods-open: proof link = "Open pinned full proof ↗ -> https://github.com/d6g8k5htny-coder/Math-/blob/d6628da09384728992dcbe6e921cc28ba85aebb0/PROOF_INDEX.md"; "NO COMPLETE PROOF YET"=false; "No landed proof is supplied"=false; ...
== availability = arbitrary text on claim 12
   validateBoundManifest: PASS ... d5-pin-neighborhoods-open: ... "COMPLETE PROOF LANDED"=true
(control: 'Open pinned source contract ↗', "NO COMPLETE PROOF YET"=true)

p5:
== AMEND d1 card: unverified replay url + arbitrary notice
   validateBoundManifest: PASS
   museum-state: Source projections and displayed source bytes verified. ...
   card class=AMEND/open | Replay column: "ReplayReplay reproduces the closed parent theorem. ACCEPT.Open pinned replay source ↗" | links=["Open pinned replay source ↗ -> https://github.com/d6g8k5htny-coder/Math-/blob/eeee…eeee/frontiers/never_landed/replay_closure.py"]
   requests for the replay url or its raw form: 0 | strip shown: true

p6 (generator edit: drop availability literal, rewrite AMEND replay notice; regenerate museum.json):
regenerate museum.json: 0 Museum projection verified: 11 reviewed bullets, 3 AMEND rows, 1 landed packet; scientific effect NONE.
S1 test_museum_data: exit 0 OK
S2 museum_check: exit 0 Museum source export and actual display integration verified. Scientific effect NONE.
S3 test_museum_geometry: exit 0
museum-state: Source projections and displayed source bytes verified. ...
d5-pin-neighborhoods-open | class AMEND/open | "Open pinned full proof ↗ -> https://github.com/d6g8k5htny-coder/Math-/blob/d6628da0…/PROOF_INDEX.md" | NO COMPLETE PROOF YET: false | Replay: "ReplayReplay reproduces the closed result."

p9 (current main 513261bb, validation only): delete availability ...: PASS ; AMEND claims replay.notice = "Replay reproduces the closed result.": PASS

VERDICTS (0):


## FO-2 [important] pointer_only type confusion: the review-pointer guard runs only for === true, but the binding check and the renderer branch on truthiness, and includes('') is vacuously true
FILES: docs/site/museum.mjs:40 (validateDescriptor: `source.pointer_only===true&&(...URL shape...||!nonempty(review_notice))`), :82-85 (validateBoundManifest: `if(claim.review.pointer_only){ ... pointerText.includes(claim.review.review_url) ...`), :151-154 (renderClaim truthy branches; link = review_url||html_url)
PROBE: <SCRATCH>/museum/probes/failopen/p2_pointer_only_truthy.mjs
CMD: cd <SCRATCH>/museum/probes/failopen && PATH=/opt/node22/bin:$PATH node p2_pointer_only_truthy.mjs

DESCRIPTION:
With pointer_only:1 or 'yes', validateDescriptor skips the issue/comment URL-shape check and the non-empty notice check, but validateBoundManifest still takes the pointer branch. That branch binds the pointer only by pointerText.includes(review_url), so review_url '' is accepted (String.prototype.includes('') === true), as is any substring of the quote, such as 'D2'. Effects:
- The card shows 'Review pointer in the verified index'.
- 'Open review' points at PROOF_INDEX.md itself, or at a relative link like 'D2'.
- The 'linked review is not byte-frozen' notice is dropped.
- A byte-frozen review file (d5-two-scale, or the STATUS-pinned sard-g review) can be replaced by a pseudo-pointer to the Math- index, and the banner still says verified.
The same input with pointer_only:true is refused, which shows that only the type check fails open. Mutant M13 (delete the line-40 guard) also survives every CI step: no test covers a pointer with a missing or empty notice. Reaching this needs a museum.json that differs from the generator output (the generator emits JSON true), so museum_check catches it in CI. The browser validator, which is meant to refuse exactly such manifests, does not.

WHY:
The interface's own distinction between a mutable review-comment pointer and an immutable source identity, which is part of the offered scope, can be bypassed with an unexpected type. The disclosure that the linked review is not byte-frozen is then removed. Fix: require typeof pointer_only === 'boolean' (or reject any value other than true/absent) everywhere, branch on === true in validateBoundManifest and renderClaim, and require review_url non-empty and equal to one exact link target of the quote.

OBSERVED:
== pointer_only:true, review_url:"", notice removed (guard applies)
   validateBoundManifest: REFUSED: Invalid review pointer
== pointer_only:1, review_url:"", notice removed
   validateBoundManifest: PASS
   museum-state: Source projections and displayed source bytes verified. These checks establish byte identity, not mathematical acceptance.
   d2-lifetime-remainder: review label="Review pointer in the verified index"; "Open review ↗ -> https://github.com/d6g8k5htny-coder/Math-/blob/d6628da09384728992dcbe6e921cc28ba85aebb0/PROOF_INDEX.md"; "not byte-frozen" notice shown=false
== pointer_only:"yes", review_url:"D2", notice removed
   validateBoundManifest: PASS ... "Open review ↗ -> D2"; "not byte-frozen" notice shown=false
== claim 13 sard-g (byte-frozen review file) replaced by pointer_only:1
   validateBoundManifest: PASS ...
   sard-g-a1-a6-open: review label="Review pointer in the verified index"; "Open review ↗ -> https://github.com/d6g8k5htny-coder/Math-/blob/d6628da0…/PROOF_INDEX.md"; "not byte-frozen" notice shown=false
(mutants: M13 SURVIVED S1=ok S2=ok S3=ok S4=ok | delete review-pointer URL shape + notice guard)
(p9, current main 513261bb: claim 0 review -> pointer_only:1, review_url:"", no notice: PASS)

VERDICTS (0):


## FO-3 [important] The live page's verification wiring has no negative control: CI stays green if startMuseum stops calling validateBoundManifest/verifyClaim, or if a refusal is swallowed into a verified card
FILES: docs/site/museum.mjs:183-207 (startMuseum: index/status verification, validateBoundManifest call at :186, verifyClaim and catch at :189, lifetime/packet jobs, banner at :207); tests/test_museum_frontend.mjs:127-150 (the only startMuseum assertions on pinned data: banner text at :142 and `claim-cards.children.length === 14` at :143)
PROBE: <SCRATCH>/museum/probes/failopen/mutants.py ; p3_mutant_impact.py + p3_render_tampered.mjs ; p7_banner_and_count.mjs (same dir)
CMD: cd <SCRATCH>/museum/probes/failopen && python3 -I mutants.py M31 M32 M33 M21 M20 M22 M23 M16 && python3 -I p3_mutant_impact.py && PATH=/opt/node22/bin:$PATH node p7_banner_and_count.mjs

DESCRIPTION:
Every substitution and promotion control calls validateBoundManifest or verifyClaim directly. No test drives startMuseum with a bad manifest or bad bytes and asserts a refusal. Each of the following mutants of museum.mjs survived all four CI steps of public-shop.yml:
- M31: startMuseum never calls validateBoundManifest.
- M32: index/status bytes used without digest verification.
- M33: claim cards rendered without verifyClaim.
- M21: the claim catch renders the card as verified.
- M20: the banner always says verified.
- M22: lifetime card without byte verification.
- M23: packet card without fetching RESULT.md.
- M16: 'Byte-verified review' without verifying the review bytes.
The one per-card assertion in the pinned integration test, children.length === 14, counts the placeholder hosts, which exist whether a card rendered or was refused (p7b). Under the mutants, p3 shows the shipped page rendering a manifest with an AMEND row promoted to ACCEPT-scoped, and a proof with a wrong digest, as fully verified.

WHY:
The refusal behaviour of the deployed page is the property the museum exists to provide. A refactor that disconnects it would ship a page that displays promotions and unverified bytes as verified, with every check green. The project's own convention says negative controls are the deliverable. Fix: add startMuseum-level controls that feed a tampered manifest (promoted class, swapped review, wrong digest, bad packet bytes) and assert the refusal card or banner and the absence of the data-object-id card. Replace the children.length assertion with a count of rendered versus refused cards.

OBSERVED:
M16 SURVIVED S1=ok S2=ok S3=ok S4=ok | docs/site/museum.mjs: verifyClaim: skip review byte verification
M20 SURVIVED ... banner always says 'displayed source bytes verified' even with refused views
M21 SURVIVED ... claim catch swallows verification failure and renders the card as verified
M22 SURVIVED ... lifetime card rendered without byte verification
M23 SURVIVED ... packet card rendered without fetching/verifying RESULT.md
M31 SURVIVED ... startMuseum never calls validateBoundManifest
M32 SURVIVED ... startMuseum uses index/status bytes without digest verification
M33 SURVIVED ... startMuseum renders claim cards without verifyClaim
### tamper=promote_amend  mutant=M31
[original cd32303] museum-state: Unavailable: Claim class promotion or source label mismatch. No result inferred.
[mutant impact_M31] museum-state: Source projections and displayed source bytes verified. ...
  d1-parent-selection-open: data-object-class=ACCEPT-scoped | ... | strip="Source byte count and SHA-256 verified…"
### tamper=bad_proof_digest  mutant=M21
[original cd32303] museum-state: Pinned source projections verified. 1 source view(s) unavailable; ... refusal cards: 1
[mutant impact_M21] museum-state: Source projections and displayed source bytes verified. ... rendered claim cards: 14 | refusal cards: 0
p7 (b) 3 proofs with wrong digests  museum-state: Pinned source projections verified. 3 source view(s) unavailable; ...
    claim-cards.children.length = 14 (test asserts 14) | refusal cards = 3

VERDICTS (0):


## FO-4 [important] Most individual validateBoundManifest guards, including the cross-claim review-substitution guard, can be deleted with every CI step green
FILES: docs/site/museum.mjs:68 (id/order), :70 (row counts), :74 (title), :79 (open-claim identity), :76-78 (proof repo/commit; D1 parent), :86 (review-file binding `sourceContainsPath`), :85 (D5 open-review pin), :88 (AMEND reason), :96-98 (replay declaration/URL/checkout), :101 (packet cap), :170 (renderPacket declarations), :199 (exhibit quote), :145/:148/:152 (AMEND line, availability line, pointer notice); tests/test_museum_frontend.mjs (no control for these)
PROBE: <SCRATCH>/museum/probes/failopen/mutants.py ; p3_mutant_impact.py (same dir)
CMD: cd <SCRATCH>/museum/probes/failopen && python3 -I mutants.py M01 M02 M03 M04 M05 M06 M07 M08 M09 M10 M12 M13 M14 M15 M18 M19 M25 M26 M27 && python3 -I p3_mutant_impact.py

DESCRIPTION:
Guard-deletion mutants that survived S1–S4:
- M01 title binding
- M02 byte-frozen review-file binding (the cross-claim review-substitution guard named in the offer)
- M03 AMEND reason
- M04, M05, M06 replay declaration, URL and checkout bindings
- M07 packet-count cap
- M08 proof repository/commit = index snapshot
- M09 D1 parent binding
- M10 open-claim identity
- M12 D5 open-review pin
- M13 pointer URL/notice guard
- M14 id order
- M15 pinned row counts
- M18 RESULT.md 'Scientific effect: **NONE**' / 'Review status: **REVIEW_REQUIRED**' check
- M19 exhibit quote-in-source check
- M25, M26, M27 removing the 'source contract' label, the 'OPEN / NOT LANDED' line and the 'not byte-frozen' notice from the rendered card
The integration test substitutes only the proof (claims[3] by claims[10]), status_quote and the replay command, never the review. Under M02, p3 shows D6's P15 review rendered as the 'Byte-verified review' of the D5 all-height annulus, with the banner verified. With the original code the same input is refused ('Review file differs from the source quote').

WHY:
These guards currently hold, but nothing keeps them from regressing. Cross-claim review substitution and the AMEND/open disclosures are exactly the properties the offer asked to have reviewed. Fix: one negative control per guard, each mutating one field of the real or fixture manifest and asserting the specific error. Include review swaps (claims[3].review = claims[10].review) and rendered-text assertions for the AMEND line, the availability label and the pointer notice.

OBSERVED:
M01 SURVIVED S1=ok S2=ok S3=ok S4=ok | docs/site/museum.mjs: delete claim-title binding
M02 SURVIVED S1=ok S2=ok S3=ok S4=ok | docs/site/museum.mjs: delete byte-frozen review-file binding (cross-claim review substitution guard)
M03 SURVIVED ... delete AMEND reason binding
M04 SURVIVED ... M05 SURVIVED ... M06 SURVIVED ... M07 SURVIVED ... M08 SURVIVED ... M09 SURVIVED ... M10 SURVIVED ...
M12 SURVIVED ... delete D5 open-review pin in validateBoundManifest
M13 SURVIVED ... M14 SURVIVED ... M15 SURVIVED ...
M18 SURVIVED ... renderPacket: delete source-declaration check
M19 SURVIVED ... exhibit view: delete quote-in-source check
M25 SURVIVED ... open obligations labelled 'Open pinned full proof'
M26 SURVIVED ... drop OPEN / NOT LANDED line from AMEND cards
M27 SURVIVED ... drop 'not byte-frozen' pointer notice
### tamper=swap_review  mutant=M02
[original cd32303] museum-state: Unavailable: Review file differs from the source quote. No result inferred.
[mutant impact_M02] museum-state: Source projections and displayed source bytes verified. ...
  d5-all-height-annulus: data-object-class=ACCEPT-scoped | "Open review ↗ -> https://github.com/d6g8k5htny-coder/Math-/blob/d6628da09384728992dcbe6e921cc28ba85aebb0/reviews/p15_full_price_nonauthor_20260926/REVIEW.md" | strip="Source byte count and SHA-256 verified…"
(killed controls in the same run: M11, M17, M24, M28)

VERDICTS (0):


## FO-5 [minor] museum_data.project() silently defaults a missing STATUS ACCEPT row to status_quote None and silently takes the first of duplicate rows, then prints 'Museum projection verified'
FILES: tools/museum_data.py:408-409 (`status_quote = next((cells[1] for _, cells in accepted_rows if status_id and cells[0].startswith("**" + status_id + " —")), None)`); contrast docs/site/museum.mjs:90-94 (requires exactly one row)
PROBE: <SCRATCH>/museum/probes/failopen/p4_status_quote_default.py
CMD: cd <SCRATCH>/museum/probes/failopen && python3 -I p4_status_quote_default.py ../../cd32303 ./fixture

DESCRIPTION:
For D2/D3/D4/D6 the generator looks up the STATUS 'ACCEPT — scoped' row with next(..., None). If a successor STATUS withdraws the D4 row, project() still emits d4-fixed-remote-rn as class ACCEPT-scoped with status_quote null and no error. The class comes from PROOF_INDEX bullet membership, not from STATUS. If the D4 row is duplicated, the first row is quoted silently. The browser validator refuses both cases, so the page refuses the card, and museum_check's integration test would then fail on the banner. The Python step on its own fails open, and only the downstream browser check catches it. This is reachable only when a maintainer repins STATUS.md in SOURCES; the probe updates the pin exactly that way in a scratch import.

WHY:
The generator and the browser enforce different rules for the same projection. The generator's success message ('Museum projection verified') is then false for a STATUS that has withdrawn an accepted row. Fix: in project(), collect the matching rows and raise unless exactly one matches for each status_id, mirroring museum.mjs, and add a test that patches STATUS with a removed and a duplicated row.

OBSERVED:
== control (pinned STATUS)
   python : project() OK -> d4-fixed-remote-rn class=ACCEPT-scoped status_quote="Fixed positive spatial separation `rho`, fixed witness separ"
   browser validateBoundManifest: PASS
== D4 ACCEPT row withdrawn
   python : project() OK -> d4-fixed-remote-rn class=ACCEPT-scoped status_quote=null
   browser validateBoundManifest: REFUSED: STATUS scope quote does not match this claim
== D4 row duplicated, narrower first
   python : project() OK -> d4-fixed-remote-rn class=ACCEPT-scoped status_quote="WITHDRAWN PENDING RE-REVIEW; fixed positive spatial separati"
   browser validateBoundManifest: REFUSED: STATUS scope quote does not match this claim

VERDICTS (0):


## FO-6 [minor] museum_data.py guards with no negative control (including the landed-packet authority check)
FILES: tools/museum_data.py:176-178 (verify_bytes repository allow-list and path guard; the SHA-256 comparison at :183 is also never isolated), :208 (section heading count), :401-402 (proof-pointer binding), :396-397 (eleven bullets), :436-437 (three AMEND rows), :483-484 (excerpt boundary), :517-520 (IDENTITY/output scientific_effect / review_status / scientific_acceptance)
PROBE: <SCRATCH>/museum/probes/failopen/mutants.py
CMD: cd <SCRATCH>/museum/probes/failopen && python3 -I mutants.py P01 P02 P03 P04 P05 P06 P07 P08 P09 P10 P11

DESCRIPTION:
Surviving mutants, with all S1–S4 green:
- P01 proof-pointer binding
- P03 eleven-bullet guard
- P04 three-AMEND-row guard
- P05 landed-packet authority check
- P06 excerpt boundary
- P07 repository allow-list
- P08 path-traversal guard
- P09 SHA-256 comparison (the git blob SHA-1 is still compared, and the browser re-checks SHA-256, so P09 has no display impact)
- P10 missing/duplicate section guard
Because the inputs are pinned by digest, these guards only become live when someone repins SOURCES. That is exactly when a deleted guard would be missed. The review-pointer binding (P02) and check_export (P11) are covered.

WHY:
P05 is the generator-side refusal of a packet whose pinned IDENTITY/output no longer says scientific_effect NONE and scientific_acceptance false. Its deletion is invisible to CI, and the browser-side RESULT.md declaration check (M18, FO-4) is also untested. Fix: unit tests that patch INDEX/STATUS/IDENTITY bytes together with their pins and assert each specific ValueError.

OBSERVED:
P01 SURVIVED S1=ok S2=ok S3=ok S4=ok | tools/museum_data.py: project: delete proof-pointer binding
P02 killed   S1=FAIL S2=ok S3=ok S4=ok | tools/museum_data.py: project: delete review-pointer binding (control)
P03 SURVIVED ... delete eleven-bullet guard
P04 SURVIVED ... delete three-AMEND-row guard
P05 SURVIVED ... build: delete landed-packet authority check
P06 SURVIVED ... excerpt: delete boundary check
P07 SURVIVED ... verify_bytes: delete repository allow-list
P08 SURVIVED ... verify_bytes: delete path traversal guard
P09 SURVIVED ... verify_bytes: delete SHA-256 comparison (git blob SHA-1 still compared)
P10 SURVIVED ... section: delete missing/duplicate heading guard
P11 killed   S1=FAIL S2=ok S3=ok S4=ok | tools/museum_data.py: check_export: never differs (control)

VERDICTS (0):


## FO-7 [minor] museum_check.py has no negative control, and it reports 'actual display integration verified' when the pinned integration test was skipped
FILES: tools/museum_check.py:38 (check_export call), :41 (MUSEUM_FIXTURE env), :42-43 (`node --test tests/test_museum_frontend.mjs ... check=True`), :44 (unconditional success print); tests/test_museum_frontend.mjs:127 (skip predicate on MUSEUM_MATH_ROOT/MUSEUM_FIXTURE)
PROBE: <SCRATCH>/museum/probes/failopen/p8_museum_check_skip_is_pass.sh ; mutants.py (C01 C02 C03)
CMD: cd <SCRATCH>/museum/probes/failopen && sh p8_museum_check_skip_is_pass.sh && python3 -I mutants.py C01 C02 C03

DESCRIPTION:
No test invokes museum_check.py. Mutants that drop the export comparison (C01) or ignore node test failures (C02) survive every CI step. The checker treats a node exit code of 0 as proof that 'actual display integration' ran. The integration test is guarded by a skip predicate on an environment variable, and a skipped test exits 0. In p8 the only change is the name of the variable museum_check exports (simulating drift between checker and test). The pinned integration test is then SKIPPED and museum_check still prints its success line and exits 0.

WHY:
museum_check is the only CI step that exercises the real pinned museum.json in the display code. If its comparison or its integration run were lost, the gate would stay green, and with FO-1 in place a hand-edited museum.json would ship. Fix: have the integration test fail rather than skip when museum_check invokes it (or pass a flag the test must see), parse the node summary and require skipped == 0, and add a test that runs museum_check against a tampered museum.json and expects a nonzero exit.

OBSERVED:
ok 15 - actual pinned museum renders all cards and rejects cross-claim source, scope and replay substitutions # SKIP
# tests 15
# pass 14
# fail 0
# skipped 1
Museum source export and actual display integration verified. Scientific effect NONE.
museum_check exit=0
C01 SURVIVED S1=ok S2=ok S3=ok S4=ok | tools/museum_check.py: museum_check: drop the export comparison
C02 SURVIVED S1=ok S2=ok S3=ok S4=ok | tools/museum_check.py: museum_check: ignore node test failures
C03 SURVIVED S1=ok S2=ok [# skipped 1] S3=ok S4=ok | tools/museum_check.py: museum_check: env var drift silently skips the pinned integration test

VERDICTS (0):


## FO-8 [minor] A refusal for an undeclared ?view= is not counted, so the banner still says all displayed source bytes were verified
FILES: docs/site/museum.mjs:195 (unknown-view refusal appended without pushing a job), :206-207 (banner computed only from jobs)
PROBE: <SCRATCH>/museum/probes/failopen/p7_banner_and_count.mjs
CMD: cd <SCRATCH>/museum/probes/failopen && PATH=/opt/node22/bin:$PATH node p7_banner_and_count.mjs

DESCRIPTION:
For ?view= values outside the four declared exhibits (for example ?view=lifetime), startMuseum appends a 'Unknown exhibit … Unavailable' refusal card but does not push a false outcome. The banner then reads 'Source projections and displayed source bytes verified.' while a refusal is on screen. Nothing unverified is displayed, so this is a consistency defect in the refusal accounting, not a status misstatement.

WHY:
The banner is the page-level summary of what was verified and refused. Every refusal path should feed it, so the summary cannot read as complete success while a refusal is shown. Fix: push a resolved false (or a distinct 'not requested' outcome) for the unknown-view branch.

OBSERVED:
(a) ?view=lifetime  museum-state: Source projections and displayed source bytes verified. These checks establish byte identity, not mathematical acceptance.
    active-exhibit: Unknown exhibitUnavailable: Select one of the four declared source illustrations. No result inferred.

VERDICTS (0):
