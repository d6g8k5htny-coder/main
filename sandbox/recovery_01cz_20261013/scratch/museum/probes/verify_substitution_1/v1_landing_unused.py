# Verifier 2 probe for Q1-9 (NON-CERTIFYING engineering probe; no network; read-only on cd32303 bytes).
# (1) static: every reference to LANDING_CLAIMS in cd32303 museum files;
# (2) dynamic: build() output is byte-identical whether LANDING_CLAIMS bytes are the pinned ones or a
#     mutated copy (every disposition rewritten), i.e. the pinned source has no data flow into museum.json;
# (3) dynamic: build() still hard-fails if LANDING_CLAIMS cannot be read (fetched-but-unused dependency);
# (4) the pinned LANDING_CLAIMS dispositions for the exact proof blobs displayed, vs. the shipped card fields.
import ast, hashlib, importlib.util, json, pathlib, sys
sys.dont_write_bytecode = True
S = pathlib.Path("<SCRATCH>/museum")
T = S / "cd32303"; V = S / "probes/verify_substitution_1"; FX = V / "fixture/Math-"
spec = importlib.util.spec_from_file_location("md_cd32303", T / "tools/museum_data.py")
md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
def _blocked(*a, **k): raise RuntimeError("NETWORK BLOCKED")
md.urlopen = _blocked
KEY = "claims/LANDING_CLAIMS.json"
# (1)
for rel in ["tools/museum_data.py", "tools/museum_check.py", "docs/site/museum.mjs", "docs/site/geometry.mjs",
            "tests/test_museum_data.py", "tests/test_museum_frontend.mjs", "tests/test_museum_geometry.mjs"]:
    lines = [n for n, l in enumerate((T / rel).read_text().splitlines(), 1) if "LANDING" in l]
    print(f"(1) {rel}: lines mentioning LANDING -> {lines}")
src = (T / "tools/museum_data.py").read_text()
start = next(n for n, l in enumerate(src.splitlines(), 1) if l.startswith("SOURCES = json.loads"))
end = next(n for n, l in enumerate(src.splitlines(), 1) if l.startswith("}''')"))
print(f"(1) SOURCES literal spans lines {start}-{end}; label constant at line",
      [n for n, l in enumerate(src.splitlines(), 1) if 'source_label="EXACT_COUNTEREXAMPLE" if counterexample else "ACCEPT"' in l])
# (2)
def real_bytes(identity):
    if identity["repository"].endswith("/main"):
        return (T / identity["path"]).read_bytes()
    if identity["path"] == "README.md":
        return None  # README.md@d6628da not available offline; build() never parses it
    return (FX / identity["path"]).read_bytes()
landing_real = real_bytes(md.SOURCES[KEY])
md.verify_bytes(landing_real, md.SOURCES[KEY])
mut = json.loads(landing_real)
for c in mut["claims"]:
    c["disposition"] = "FAIL_CLOSED"; c["disposition_reason"] = "MUTATED BY PROBE"
mut["meaning"] = "MUTATED"
landing_mut = json.dumps(mut).encode()
def make_reader(landing):
    def rs(identity, main_root, math_root):
        raw = real_bytes(identity)
        if identity["path"] == "README.md": return b"README placeholder (unverified, unparsed)\n"
        if identity["path"] == KEY:
            if landing is None: raise FileNotFoundError("LANDING_CLAIMS unavailable (simulated)")
            return landing  # verification deliberately bypassed for this key only
        return md.verify_bytes(raw, identity)
    return rs
md.read_source = make_reader(landing_real); a = md.dump(md.build(T, FX))
shipped = (T / "docs/site/museum.json").read_bytes()
print("(2) build(pinned LANDING) == shipped cd32303 museum.json:", a == shipped, hashlib.sha256(a).hexdigest()[:16])
md.read_source = make_reader(landing_mut); b = md.dump(md.build(T, FX))
print("(2) build(mutated LANDING, all dispositions FAIL_CLOSED) == build(pinned):", a == b)
# (3)
md.read_source = make_reader(None)
try: md.build(T, FX); print("(3) build without LANDING: succeeded")
except Exception as e: print("(3) build without LANDING: FAILS ->", type(e).__name__, e)
# (4)
landing = json.loads(landing_real); by_blob = {c["source"]["blob"]: c for c in landing["claims"]}
for c in json.loads(shipped)["claims"]:
    l = by_blob.get(c["proof"]["blob"])
    if l:
        mentions = "landing" in c["scope_quote"].lower() or "landing" in (c["status_quote"] or "").lower()
        print(f"(4) {c['id']:28} blob={c['proof']['blob'][:8]} card class={c['class']:15} label={c['source_label']:21}"
              f"| LANDING {l['claim_id']}: {l['disposition']}; card text mentions 'landing': {mentions}")
print("(4) LANDING meaning:", landing["meaning"]); print("(4) LANDING audited_default_commit:", landing["audited_default_commit"])
print("(4) lifetime-remainder reason:", by_blob["247b3ecf80bfbe896948d5d489b2d5842a81c481"]["disposition_reason"])
