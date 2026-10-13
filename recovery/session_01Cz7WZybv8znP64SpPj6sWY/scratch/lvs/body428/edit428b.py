import re
S='<SCRATCH>'
t=open(f'{S}/lvs/body428/stage1.md',encoding='utf-8').read()
P='https://github.com/d6g8k5htny-coder/Math-/pull/'
def rep(a,b):
    global t
    n=t.count(a); assert n==1,(n,a[:90]); t=t.replace(a,b)
C='companions/torus_filtration_20261011/'
# table
start=t.index('| Path | Blob | SHA-256 | Bytes |')
end=t.index('Read, never edited: `formal/lean-toolchain`')
rows=[
 ('TorusTopology.lean','ed678ea322fe5315b3aaff44e4e2f06b018399d8','ddb145b3f791c181d4f2a51e54e711937297bd6476ba55bcd5252b7ddb57cfe1','12257',None),
 ('ClosedSuperlevel.lean','a4323cb50fb2d5861c4e6240a18684bfb566fb80','90d1481b75d9c6d14e462c3fbf0bf7c7cdd0efb9eaad68d6732400658753b1c2','8088',None),
 ('SuperlevelBar.lean` (C263 successor, new','b047384e20306ee030154885536c531633b88bd8','346eea948e18d5904422904006827930e27d3141ccbc19ee8ac2c24bcf65a4a9','15309','new'),
 ('OrdinaryH0.lean` (C264 successor, new','7bc5f80028271e0e33d857e60496ca4cadaa9364','3d6b0d6b53413c484b1d010bccf0d53ae7f75becb2d747868f22dbd673fe0a88','19803','new'),
 ('TorusFiltration.lean` (root: imports the four modules in dependency order','29e41ef12ae1a24eb36de48b9ee0738fc939d77b','33b88648df5c0144a492544f3da83e6ebbe72aadc6fccc1ece930335ad2a3af5','84',('d8e52f866943d54959cd38d5ec496739f739bc17','5498eb85d94965073f8cee18fda4bd8a6f4ceef36ed85efd7fb78784ab4f0e58','45')),
 ('check.py','999eb4bf06273e572f19e9e34125898657832a8c','c4416dc9e890895156bec553971d4dbcb0f4a6feeddaa9fcdbe055e405f4cf0a','49482',('afc6492bb39f9830f99b1a28ee16ab04537fd45d','d8fb2add2ca430b0f935f1578adecc90da798a7fb33f693ada156a09af99077d','40225')),
 ('replay.sh','ff47d2393e341232273186d5e9ff44559eac1e3d','34b739182179806a802633eff78b2006c8e5baa49cd04652f951a121a3bc8f2d','5236',('05602a3f7f5fd2cada98625b27c5268a407cc8f1','cb6b54e1aabda382c48eca7316d5e892845f68e2a708082a30b5e2acf81fe95c','4382')),
 ('test_torus_filtration.py` (28 tests','ef12443fb8b2d6e0cbb61706d45d023815bad261','007365ea76c68f59ff85f3190dff6fe0a3676ade44915fa2f260542f02be8a84','108149',('0fd3c887cdec71cd7b232933cdd08b4f34918fee','ab95294403739a4f20df7fafc6bca7bf0e25748595581cdaaba02fc975a38d2f','84884')),
 ('SOURCE_FILES.json','839dafcca7b2dc88f47e07e67020735f7b392fd3','79376ae0d8a39ee45dd81935fb3d95e36953ab0b1d58256ca686fe256e3ff773','13308',('09cb314ab33ba710819af668a01f8c7133522cf1','44fbf9f2c7c24e86936bf9414b1f487c153bdeae7df505567a6a872dcf14dcc1','5896')),
 ('README.md','671b4cb7c295461efb3e4922537f476139177dfb','f1e0a8c57012a024f744da74229885bb3062e0db842ad36ba489dbb7e06f74fe','56162',('2f99b89767fb37648bc0742ed4e8328c245b74b1','8084b313486bbf72fa32ab57fc604fb88bccb658b21569c225f5b6b68a783f49','30582')),
 ('SCOPE.md','5d0b673aa604173c2161d839f700d4c397ea7bfb','b2239cf0548f87b588925eb9d86aa17b74dff7774d2653925d4d888389977072','22627',('6f78f7300475eec8b97455133714310fbdf35a29','cb7d9cb9929a05fd3b0d07f91dfbc9b2b0f796cbd59da5f1e34bc2ab08a55d7b','11141')),
]
out=['| Path | Blob | SHA-256 | Bytes |','|---|---|---|---|']
for name,b,s,n,old in rows:
    if '`' in name:
        base,_,note=name.partition('` (')
        p='`'+C+base+'` ('+note+')'
    else:
        p='`'+C+name+'`'
    if old and old!='new':
        out.append(f'| {p} | `{b}` (at `4c33d8cd`: `{old[0]}`) | `{s}` (at `4c33d8cd`: `{old[1]}`) | {n} (at `4c33d8cd`: {old[2]}) |')
    else:
        out.append(f'| {p} | `{b}` | `{s}` | {n} |')
out.append('| `.github/workflows/torus-filtration.yml` | `becc58ff7e8362160c305bcdc8b386221966afad` | `3b6b06856bad22f7e9d28f1caf0e1710db3284cd73d65bf6f7330bfbd45dc033` | 2084 |')
table='\n'.join(out)+'\n\n'
table+=("Identities are at the current head `940dffc9`, as the hosted receipt's `sources` records them. The seven rows with a `4c33d8cd` parenthesis moved in the registration commit; the two successor modules are new in it; the three rows without either (`TorusTopology.lean`, `ClosedSuperlevel.lean`, the workflow) are byte-identical at `4c33d8cd`, `a898bb44` and `0dd0cb17`. The identities at `a898bb44` and `0dd0cb17` are in GitHub's edit history of this body and in the C262 readback. ")
t=t[:start]+table+t[end:]

# provenance table: add rows after Handoff bytes row
anchor="| This lane | local elaboration report"
rep(anchor, "| C263 handoff (`SuperlevelBar.lean`) | Math- #428 comments [6113033147]("+P+"428#issuecomment-6113033147) (bytes, sha256 `d74dc8b2c25eff5b90ce7133e51a0888c7885cae24ca6d5e648bc514089637a8`, 14,904 B) and [6113033545]("+P+"428#issuecomment-6113033545) (owner task); allocation [6112439759]("+P+"419#issuecomment-6112439759); compile report [6113229784]("+P+"428#issuecomment-6113229784) |\n| C264 handoff (`OrdinaryH0.lean`) | Math- #428 comments [6113276830]("+P+"428#issuecomment-6113276830), [6113277173]("+P+"428#issuecomment-6113277173), [6113277485]("+P+"428#issuecomment-6113277485) (sha256 `261dd0526d42890d8a4ac1225fcf47228eb3c9799af4106cb643a528bcbf968e`, 18,469 B); failure report [6113300903]("+P+"428#issuecomment-6113300903); repair handover [6113372506]("+P+"428#issuecomment-6113372506); full repair diff [6113438262]("+P+"428#issuecomment-6113438262) |\n"+anchor)
rep("| Namespace mapping | `ResearchFormalCoreR1.<n>` (handoff) -> `TorusFiltration.<n>` (companion); no other name changes |",
    "| Namespace mapping | C254: `ResearchFormalCoreR1.<n>` (handoff) -> `TorusFiltration.<n>` (companion), no other name changes. C263 and C264 were authored directly in namespace `TorusFiltration` |")

# compile fixes
rep("## Compile fixes at the pin (the whole diff from the handoff bytes)\n\nEvery theorem and lemma statement",
    "## Compile fixes at the pin (the whole diff from the handoff bytes)\n\nC254 modules. Every theorem and lemma statement")
rep("- Module docstring: three lines recording the above; `namespace ResearchFormalCoreR1` -> `namespace TorusFiltration`.\n",
    "- Module docstring: three lines recording the above; `namespace ResearchFormalCoreR1` -> `namespace TorusFiltration`.\n\n"
    "`SuperlevelBar.lean` (C263; handoff 14,904 B -> successor 15,309 B): the handoff bytes produce exactly one diagnostic at the pin with warnings as errors, the deprecated `if_pos` at line 123. `theorem superlevelBar_death_eq_sSup_of_nonempty`, proof: `simp only [superlevelBar_deathHeight, if_pos hne]` -> `simp only [superlevelBar_deathHeight, ite_eq_left hne]` (Lean's own named replacement, the same statement). Six module-docstring lines (a blank separator and five text lines) record the handoff hash and the fix. No other byte changed; the 20 theorem names and the 1 + 3 definition names keep their order.\n\n"
    "`OrdinaryH0.lean` (C264; handoff 18,469 B -> successor 19,803 B; owner repair handed to this lane in [6113372506]("+P+"428#issuecomment-6113372506), full diff in [6113438262]("+P+"428#issuecomment-6113438262)):\n"
    "- `variable`: `[Preadditive C] [CategoryWithHomology C]` enter the scope only after the three declarations that do not use them, so `def ordinaryH0_componentFiltration` and `theorem ordinaryH0_componentFiltration_map_iota` move up (text unchanged; no `omit`, no linter disabled).\n"
    "- `theorem ordinaryH0_iso_naturality`, proof: the handoff's `simp only [...]` is replaced by explicit rewrites (`rw [← Category.assoc, ordinaryH0_pointClass_map, ordinaryH0_pointClass_iso]` on the left, `erw [← Category.assoc, ordinaryH0_pointClass_iso, ordinaryH0_componentCoproductMap_iota]` on the right, then `rfl`); no naturality or compatibility field is assumed.\n"
    "- `isoWhiskerLeft` -> `Functor.isoWhiskerLeft`.\n"
    "- The three declaration names containing `ι` are spelled `ordinaryH0_componentCoproductMap_iota`, `ordinaryH0_componentIota_isoInv`, `ordinaryH0_componentFiltration_map_iota` (the companion's ASCII identifier rule), all references updated.\n"
    "- Fourteen docstring lines (a blank separator and thirteen text lines) record the above. No statement changed; the 20 theorem and 7 definition names are the handoff's except the three ASCII respellings. Whether every statement still says what the handoff intended is part of the pending exact-source review.\n")

# gate checks section
rep("the 28 protocol tests in both Python modes (`PYTHONOPTIMIZE` unset; each `check.py` child observed in the outer mode); `check.py emit` (writes `Audit.lean`, `Types.lean`, `Positive.lean`, `Census.lean` and six control files); inside `lake env` from `formal/`: `lean -DwarningAsError=true --root=<companion>` on the three modules,",
    "the 28 protocol tests in both Python modes (`PYTHONOPTIMIZE` unset; each `check.py` child observed in the outer mode); `check.py emit` (writes `Audit.lean`, `Types.lean`, `Positive.lean`, `Census.lean` and eight control files); inside `lake env` from `formal/`, with `LEAN_NUM_THREADS` defaulting to 2: `lean -DwarningAsError=true --root=<companion>` on the five modules in dependency order (`TorusTopology`, `ClosedSuperlevel`, `SuperlevelBar`, `OrdinaryH0`, the root),")
rep("the axiom audit (`#print axioms` of all 58 declarations,", "the axiom audit (`#print axioms` of all 109 declarations,")
rep("`leanchecker TorusTopology ClosedSuperlevel TorusFiltration` (modules mode:", "`leanchecker TorusTopology ClosedSuperlevel SuperlevelBar OrdinaryH0 TorusFiltration` (modules mode:")
rep("three false-statement controls Lean itself must refuse with exactly `⊢ False`;", "five false-statement controls Lean itself must refuse with exactly `⊢ False` (two new at `940dffc9`: `RejectEssentialDeathBound`, `RejectReducedPointClass`);")
rep("and writes `receipt.json` with `log_sha256` over the evidence directory.",
    "and writes `receipt.json` once (`O_EXCL`, `O_NOFOLLOW`) with `log_sha256` over exactly the 54 evidence files. The scanner admits two prelude forms added for `OrdinaryH0.lean`, an import of an earlier companion module and `open scoped <Namespace>`, both with negative controls; this widening is disclosed for the readback to ratify.")

# not established
rep("That is the pending independent alignment review, which this same-account lane cannot supply.",
    "That is the pending independent alignment review, which this same-account lane cannot supply.\n\n"
    "For the C263 and C264 successors, in addition: no review of the compiled successors exists (this lane compiled, repaired and registered them and has not reviewed them), so FORMAL HOLD stands; the concrete ordinary-H0 controls of the C264 owner task (empty, point, two-component, merge/non-merge and zero-coefficient fixtures) are not provided; none of the modules' own listed exclusions is lifted; no Field or Lifetime inhabitant, Field adapter or coupling to the `Specifications` package of #419; no elder relation, bar pairing, finite-bar count, interval decomposition or stochastic law; no organizational-independence credit.")

# coordination
rep("Receipted heads are never amended; successors stack on them.",
    "C263/C264 registration and readback request: [6115569460]("+P+"428#issuecomment-6115569460). Receipted heads are never amended; successors stack on them.")

open(f'{S}/lvs/body428/body428_after.md','w',encoding='utf-8').write(t)
print('ok',len(t))
