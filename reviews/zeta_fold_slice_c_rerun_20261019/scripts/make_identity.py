#!/usr/bin/env python3
"""Write IDENTITY.json for an incoming/ package (CONTRIBUTING schema 1): every artifact except the
identity file with bytes and SHA-256, plus the public source files the result is bound to."""
import hashlib
import json
import sys
from pathlib import Path

pkg = Path(sys.argv[1]).resolve()
MAIN = "0a7e5145b55f020e02e7782fed318266d13c2447"
sources = [
    ("experiments/zeta_fold_exploration/README.md", "d9416dc35e198de3a1c2799b6c5fd23f5e93a16d054b35c403604a3a2728913c"),
    ("experiments/zeta_fold_exploration/MANIFEST.json", "99441cc43dae0ace1ea5a080880e93e2495c61665ca74bfdd42dfd452812700e"),
    ("experiments/zeta_fold_exploration/requirements.txt", "f5eed4fa0ff9d50171b9857728e41177edf2e0f8195a51f285746c54ade5ecda"),
    ("experiments/zeta_fold_exploration/CLAIMS.md", "3f255628933e766795d8fd90ea41b8e68a69e26db8b1b678a0bfa4eafe1fdc80"),
]
arts = []
for p in sorted(pkg.rglob("*")):
    if p.is_file() and p.name != "IDENTITY.json":
        b = p.read_bytes()
        arts.append(dict(path=p.relative_to(pkg).as_posix(), bytes=len(b), sha256=hashlib.sha256(b).hexdigest()))
ident = dict(
    schema=1,
    scientific_effect="NONE",
    review_status="REVIEW_REQUIRED",
    sources=[dict(repository="d6g8k5htny-coder/main", path=p, commit=MAIN, sha256=h) for p, h in sources],
    artifacts=arts,
)
(pkg / "IDENTITY.json").write_text(json.dumps(ident, indent=1) + "\n")
print(json.dumps(ident, indent=1))
