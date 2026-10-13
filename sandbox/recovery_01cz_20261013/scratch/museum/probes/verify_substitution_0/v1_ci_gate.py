# Verifier probe: does the CI path (museum_check.py -> museum_data.build + check_export) let the
# substituted manifest land? No network: urlopen is replaced; README.md@d6628da (unavailable offline,
# and used only as a replay-source descriptor, never as exhibit text) is served as placeholder bytes
# AFTER the identity check is bypassed for that single path only.
import sys, json, copy, tempfile, pathlib, hashlib
T = "<SCRATCH>/museum/cd32303"
V = "<SCRATCH>/museum/probes/verify_substitution_0"
sys.path.insert(0, T + "/tools")
import museum_data as md
md.urlopen = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("NETWORK BLOCKED"))
orig = md.read_source
def rs(identity, main_root, math_root):
    if identity["path"] == "README.md": return b"placeholder (README.md@d6628da not available offline)"
    return orig(identity, main_root, math_root)
md.read_source = rs
expected = md.build(pathlib.Path(T), pathlib.Path(V + "/fixture/Math-"))
shipped_path = pathlib.Path(T + "/docs/site/museum.json")
try: md.check_export(shipped_path, expected); print("check_export(shipped museum.json): PASS")
except ValueError as e: print("check_export(shipped):", e)
# Structural binding inside the builder: one SOURCES identity per path.
keys = ['repository','path','commit','blob','bytes','sha256','url','html_url']
for k, ci in (("remote", 2), ("annulus", 3), ("p15", 10), ("lifetime", 0)):
    same = all(expected["exhibits"][k][f] == expected["claims"][ci]["proof"][f] for f in keys)
    print(f"builder: exhibits.{k} identity == claims[{ci}].proof ({expected['claims'][ci]['id']}): {same}")
print("builder: D5 height-window card links to view=annulus whose source is",
      expected["exhibits"]["annulus"]["path"], "; height-window proof is", expected["claims"][4]["proof"]["path"])
# The probe's substitution, applied to the shipped manifest, then put through the CI comparison.
m = json.loads(shipped_path.read_bytes())
m["exhibits"]["annulus"]["commit"] = "4" * 40
with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / "museum.json"; p.write_bytes(md.dump(m))
    try: md.check_export(p, expected); print("check_export(substituted annulus commit): PASS")
    except ValueError as e: print("check_export(substituted annulus commit): REFUSED:", e)
