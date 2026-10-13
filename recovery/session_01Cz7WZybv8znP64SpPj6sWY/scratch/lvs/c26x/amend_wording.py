from pathlib import Path
p = Path('<SCRATCH>/repos/math_c263reg/companions/torus_filtration_20261011/README.md')
s = p.read_text(encoding='utf-8')
old = """organizational-independence credit; any hosted receipt for the registered set (the first hosted
run of this commit will produce one, and it will be a record of a computation). The compile fix"""
new = """organizational-independence credit; any hosted receipt for the registered set (a green hosted
run of this commit would produce one, and a receipt is a record of a computation). The compile fix"""
assert s.count(old) == 1
p.write_text(s.replace(old, new), encoding='utf-8')
print('wording amended')
