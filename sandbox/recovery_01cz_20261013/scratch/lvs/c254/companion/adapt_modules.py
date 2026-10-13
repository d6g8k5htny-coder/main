"""Copy the compile-fixed C254 modules from the worktree's formal/ overlay into the companion directory,
changing only the namespace ResearchFormalCoreR1 -> TorusFiltration (declaration and `end` lines) and
qualified self-references, and verify that the theorem/def inventories are unchanged.
Usage: adapt_modules.py <worktree> ; writes companions/torus_filtration_20261011/{TorusTopology,ClosedSuperlevel}.lean
"""
import hashlib, re, sys
from pathlib import Path

WT = Path(sys.argv[1])
SRC = WT / 'formal/ResearchFormalCoreR1'
DST = WT / 'companions/torus_filtration_20261011'
ORIG = Path('<SCRATCH>/lvs/c254/overlay/ResearchFormalCoreR1')
DECL = re.compile(r'^(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_]*)', re.M)
DEFS = re.compile(r'^(?:noncomputable\s+)?(?:def|abbrev|instance)\s+([A-Za-z_][A-Za-z0-9_]*)', re.M)
report = {}
for name in ('TorusTopology', 'ClosedSuperlevel'):
    fixed = (SRC / f'{name}.lean').read_text(encoding='utf-8')
    orig = (ORIG / f'{name}.lean').read_text(encoding='utf-8')
    assert DECL.findall(fixed) == DECL.findall(orig), (name, 'theorem inventory changed by the fix')
    assert DEFS.findall(fixed) == DEFS.findall(orig), (name, 'definition inventory changed by the fix')
    n_ns = len(re.findall(r'^namespace ResearchFormalCoreR1\s*$', fixed, re.M))
    n_end = len(re.findall(r'^end ResearchFormalCoreR1\s*$', fixed, re.M))
    assert n_ns == 1 and n_end == 1, (name, n_ns, n_end)
    out = re.sub(r'^namespace ResearchFormalCoreR1\s*$', 'namespace TorusFiltration', fixed, flags=re.M)
    out = re.sub(r'^end ResearchFormalCoreR1\s*$', 'end TorusFiltration', out, flags=re.M)
    out = out.replace('ResearchFormalCoreR1.', 'TorusFiltration.')   # qualified self-references, if any
    leftovers = [l for l in out.split('\n') if 'ResearchFormalCoreR1' in l and not l.lstrip().startswith(('--', '/-', '*', 'Source', 'The'))]
    DST.mkdir(parents=True, exist_ok=True)
    (DST / f'{name}.lean').write_text(out, encoding='utf-8')
    report[name] = {'fixed_sha256': hashlib.sha256(fixed.encode()).hexdigest(), 'companion_sha256': hashlib.sha256(out.encode()).hexdigest(),
                    'original_sha256': hashlib.sha256(orig.encode()).hexdigest(), 'theorems': len(DECL.findall(out)), 'defs': DEFS.findall(out),
                    'remaining_core_mentions': leftovers[:5]}
import json; print(json.dumps(report, indent=1))
