# Verifier probe Q1-5: patch a COPY of the cd32303 tree's tools/museum_data.py.
#  1. Re-pin SOURCES["README.md"] to the stand-in README bytes (Math-@10e1f191; README@d6628da is not
#     available offline). repository/commit/path stay d6628da so every other check is unchanged.
#  2. Apply the variant's CLAIM_SPECS edit (replay commands only).
# Usage: patch_q15.py <copy-root> <variant> <stand-in-README>
import hashlib, json, pathlib, re, sys
root, variant, readme = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]).read_bytes()
p = root / "tools/museum_data.py"; src = p.read_text()
ident = {"repository": "d6g8k5htny-coder/Math-", "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
         "path": "README.md", "blob": hashlib.sha1(b"blob %d\0" % len(readme) + readme).hexdigest(),
         "bytes": len(readme), "sha256": hashlib.sha256(readme).hexdigest()}
src, n = re.subn(r'^SOURCES\["README.md"\] = \{.*\}$', 'SOURCES["README.md"] = ' + json.dumps(ident), src, flags=re.M)
assert n == 1, "README pin line not found"
D2, D4, D6 = 0, 2, 10
FULL = "python -B -S frontiers/full_price_20260924/full_price.py"
GATE = "python -B -S frontiers/downstream_gate_20260925/hard_gate.py"
edits = {
    "baseline": [],
    "SWAP_D2_D6__D4_gets_D0_hard_gate": ["CLAIM_SPECS[0][4], CLAIM_SPECS[10][4] = CLAIM_SPECS[10][4], CLAIM_SPECS[0][4]",
                                          f"CLAIM_SPECS[2][4] = {GATE!r}"],
    "D2_D4_D6_all_get_full_price": [f"CLAIM_SPECS[{i}][4] = {FULL!r}" for i in (D2, D4, D6)],
    "CONTROL_D2_invented_command": ["CLAIM_SPECS[0][4] = 'python invented-solver.py'"],
}[variant]
anchor = "\n]''')\n\n\ndef verify_bytes"
assert src.count(anchor) == 1, "CLAIM_SPECS end anchor not found"
src = src.replace(anchor, "\n]''')\n" + "".join(e + "\n" for e in edits) + "\n\ndef verify_bytes")
p.write_text(src)
print(variant, "patched:", edits or "(no CLAIM_SPECS edit)")
