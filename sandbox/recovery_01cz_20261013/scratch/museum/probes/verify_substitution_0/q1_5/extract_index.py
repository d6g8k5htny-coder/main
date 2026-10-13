# Extract the PROOF_INDEX fixture bytes embedded in cd32303 tests/test_museum_data.py
# without importing/executing the test module; verify against the pinned identity.
import ast, hashlib, sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text()
tree = ast.parse(src)
for node in tree.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "INDEX_BYTES":
        call = node.value  # "<str>".encode("utf-8")
        raw = call.func.value.value.encode("utf-8")
        break
blob = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
print(len(raw), blob, hashlib.sha256(raw).hexdigest())
pathlib.Path(sys.argv[2]).write_bytes(raw)
