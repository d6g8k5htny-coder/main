#!/bin/sh
# Aside 1: the builder's AMEND-row loop (museum_data.py:437-445) never checks that review_target occurs
#          in the row: an unrelated ACCEPT review is accepted for SARD-G by project(); only the frontend
#          refuses it (non-pointer branch requires the path in the quote).
# Aside 2: the frontend pointer branch (museum.mjs:85) uses plain includes(), so a truncated pointer
#          (issues/6, a different issue) is accepted for the D2 card. (Pointer lens; noted only.)
P=<SCRATCH>/museum/probes/verify_substitution_0
T=<SCRATCH>/museum/cd32303
SUB=<SCRATCH>/museum/probes/substitution
python3 -I -B - "$T" "$SUB" <<'EOF'
import importlib.util, pathlib, sys
T, SUB = map(pathlib.Path, sys.argv[1:3]); sys.dont_write_bytecode = True
src = (T / "tools/museum_data.py").read_text()
old = '("sard-g-a1-a6-open", "STATUS.md", "reviews/sard_g_successor_a1_a6_20260926/REVIEW.md")'
assert src.count(old) == 1
src = src.replace(old, '("sard-g-a1-a6-open", "STATUS.md", "reviews/p15_full_price_nonauthor_20260926/REVIEW.md")')
ns = {"__name__": "museum_variant"}; exec(compile(src, "museum_data_variant.py", "exec"), ns)
ns["urlopen"] = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("NETWORK BLOCKED"))
claims = ns["project"]((SUB / "PROOF_INDEX.md").read_bytes(), (T / "STATUS.md").read_bytes())
c = claims[13]; print("builder ACCEPTED", c["id"], "review :=", c["review"]["path"], "| in row:", c["review"]["path"] in c["scope_quote"])
EOF
/opt/node22/bin/node --input-type=module -e "
import fs from 'node:fs';
globalThis.fetch=async u=>{throw Error('NETWORK BLOCKED')};
const m=await import('$T/docs/site/museum.mjs');
const idx=fs.readFileSync('$SUB/fixture/Math-/PROOF_INDEX.md','utf8'), st=fs.readFileSync('$T/STATUS.md','utf8');
const base=()=>JSON.parse(fs.readFileSync('$T/docs/site/museum.json','utf8'));
const t=(name,f)=>{const x=base();f(x);try{m.validateBoundManifest(x,idx,st);console.log('frontend ACCEPTED',name)}catch(e){console.log('frontend REFUSED ',name,'->',e.message)}};
t('sard-g review := p15 ACCEPT review (builder-accepted above)',x=>{x.claims[13].review={...x.claims[10].review};});
t('d2 pointer review_url := .../main/issues/6 (prefix of issues/67)',x=>{x.claims[0].review.review_url='https://github.com/d6g8k5htny-coder/main/issues/6';});
"
