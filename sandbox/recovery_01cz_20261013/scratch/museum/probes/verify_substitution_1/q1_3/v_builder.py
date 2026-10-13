# Verifier 2 for Q1-3, builder side. Loads cd32303 tools/museum_data.py source text, edits ONE
# review URL in an in-memory copy (never written to the cd32303 tree), runs project() on the
# pinned PROOF_INDEX/STATUS bytes, then hands the resulting claims to the cd32303 museum.mjs guard.
import ast, hashlib, json, subprocess, sys, types, pathlib
sys.dont_write_bytecode = True
S = pathlib.Path("<SCRATCH>/museum")
TREE = S / "cd32303"; HERE = S / "probes/verify_substitution_1/q1_3"
SRC = (TREE / "tools/museum_data.py").read_text()
print("museum_data.py sha256", hashlib.sha256(SRC.encode()).hexdigest()[:16])
tsrc = (TREE / "tests/test_museum_data.py").read_text()
INDEX = next(n.value.func.value.value for n in ast.parse(tsrc).body
             if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "INDEX_BYTES").encode()
STATUS = (TREE / "STATUS.md").read_bytes()
def load(text, name):
    mod = types.ModuleType(name); exec(compile(text, name, "exec"), mod.__dict__)
    mod.urlopen = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("NETWORK BLOCKED")); return mod
base = load(SRC, "museum_data_base")
base.verify_bytes(INDEX, base.INDEX); base.verify_bytes(STATUS, base.STATUS)
shipped = json.loads((TREE / "docs/site/museum.json").read_text())
assert base.project(INDEX, STATUS) == shipped["claims"], "builder claims differ from shipped"
print("unmodified builder project() == shipped museum.json claims: True")
def run(label, old, new):
    assert SRC.count(old) == 1, old
    mod = load(SRC.replace(old, new), "museum_data_edit")
    try:
        claims = mod.project(INDEX, STATUS)
    except Exception as e:
        print(f"[{label}] builder REFUSES: {e}"); return
    m = dict(shipped, claims=claims)
    (HERE / "edited_manifest.json").write_text(json.dumps(m))
    js = ("const S='%s';globalThis.fetch=async()=>{throw Error('blocked')};const fs=await import('node:fs');"
          "const mu=await import(S+'/cd32303/docs/site/museum.mjs');const m=JSON.parse(fs.readFileSync('%s','utf8'));"
          "try{mu.validateBoundManifest(m,fs.readFileSync(S+'/probes/substitution/fixture/Math-/PROOF_INDEX.md','utf8'),fs.readFileSync(S+'/cd32303/STATUS.md','utf8'));console.log('ACCEPT')}catch(e){console.log('REFUSE: '+e.message)}"
          % (S, HERE / "edited_manifest.json"))
    out = subprocess.run(["/opt/node22/bin/node", "--input-type=module", "-e", js], capture_output=True, text=True).stdout.strip()
    print(f"[{label}] builder ACCEPTS (review_url={[c['review']['review_url'] for c in claims if c['id']==label.split()[0]][0]}); museum.mjs guard: {out}")
run("d3-side24-coefficient CLAIM_SPECS -> issues/6",
    '"https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5841269490"',
    '"https://github.com/d6g8k5htny-coder/main/issues/6"')
run("d1-parent-selection-open open_specs -> issues/6",
    '"https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-5841830743"),',
    '"https://github.com/d6g8k5htny-coder/main/issues/6"),')
run("d1-parent-selection-open open_specs -> truncated comment id (typo)",
    '"https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-5841830743"),',
    '"https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-584183074"),')
run("d1-parent-selection-open open_specs -> erratum comment (delimited in row)",
    '"https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-5841830743"),',
    '"https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-5841947606"),')
