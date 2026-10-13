# Verifier probe for Q1-6. Patch a COPY of cd32303 tools/museum_data.py (never the original) to
# simulate a routine STATUS re-pin. Each substitution asserts its anchor count, so no silent no-op.
#   argv: copy_root variant readme_path status_path
#   variant REPIN_ONLY : STATUS bytes unchanged, every main source re-pinned to commit 555..5
#   variant SUCCESSOR  : STATUS replaced by the successor (D5 row -> different open obligation), re-pinned
import hashlib, pathlib, sys

root = pathlib.Path(sys.argv[1]); variant = sys.argv[2]
readme = pathlib.Path(sys.argv[3]).read_bytes(); status = pathlib.Path(sys.argv[4]).read_bytes()
f = root / "tools/museum_data.py"; src = f.read_text()

def ident(raw):
    return (hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest(), len(raw), hashlib.sha256(raw).hexdigest())

def sub(old, new, count=1):
    global src
    assert src.count(old) == count, ("anchor count", src.count(old), old[:80])
    src = src.replace(old, new)

# Fixture artifact shared by both variants: README.md@d6628da is not available offline.
b, n, s = ident(readme)
sub('"blob":"abe16871da7740fa28ae4dbb28fb31853a5868bd","bytes":6023,"sha256":"89d4c618ad4560373f8fba3c787522cea8f0191fd8e176f3e0741c6230390730"',
    f'"blob":"{b}","bytes":{n},"sha256":"{s}"')
# Re-pin main snapshot (constant + six main SOURCES entries).
sub("71400b94f6cb354a8cf7aba73ffede2138a64efa", "5" * 40, count=7)
if variant == "SUCCESSOR":
    old = (root / "STATUS.md").read_bytes()   # the copy still holds the cd32303 STATUS bytes here
    ob, on, os_ = ident(old); nb, nn, ns = ident(status)
    for a, z in ((ob, nb), (os_, ns)):
        sub(f'"{a}"', f'"{z}"')
    sub(f'"bytes": {on},\n    "sha256": "{ns}"', f'"bytes": {nn},\n    "sha256": "{ns}"')
    (root / "STATUS.md").write_bytes(status)
elif variant != "REPIN_ONLY":
    raise SystemExit("unknown variant")
f.write_text(src)
print("patched", variant)
