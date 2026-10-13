"""Build mutant copies of the HEAD tree with one G3 gate weakened each; the caller runs FinishFixture against them."""
import shutil, sys
from pathlib import Path

V = Path('<SCRATCH>/lvs/c262/verify_fable_g3')
TREE = V / 'tree'
REL = 'companions/torus_filtration_20261011/check.py'
MUTANTS = {
    'M1_no_receipt_gate': (
        "    require(not receipt_path.exists() and not receipt_path.is_symlink(),\n"
        "            'receipt.json already present: finish never replaces a receipt; a new attempt needs a fresh evidence directory')\n",
        "    pass\n"),
    'M2_subset_membership': (
        "    require(not missing and not unexpected, 'evidence membership differs: missing %s, unexpected %s' % (missing, unexpected))\n",
        "    require(not missing, 'evidence membership differs: missing %s, unexpected %s' % (missing, unexpected))\n"),
    'M3_no_regular_file_check': (
        "        require(member.is_file() and not member.is_symlink(), 'evidence member is not a regular file: ' + name)\n",
        "        pass\n"),
    'M4_forty_members_if_raise': (
        "    | {'leanchecker.cmd', 'packages.log', 'toolchain.log', 'source.json', 'tests.log', 'tests-optimized.log'}\n",
        "    | {'leanchecker.cmd', 'packages.log', 'toolchain.log', 'source.json', 'tests-optimized.log'}\n"),
    'M5_forty_members_assert': (
        "    | {'leanchecker.cmd', 'packages.log', 'toolchain.log', 'source.json', 'tests.log', 'tests-optimized.log'}\n"
        "if len(EVIDENCE_FILES) != 41:   # an explicit check, not an `assert`: the gate must hold under -O too\n"
        "    raise RuntimeError('EVIDENCE_FILES must enumerate exactly 41 members')\n",
        "    | {'leanchecker.cmd', 'packages.log', 'toolchain.log', 'source.json', 'tests-optimized.log'}\n"
        "assert len(EVIDENCE_FILES) == 41\n"),
}
for name, (old, new) in MUTANTS.items():
    dst = V / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(TREE, dst, symlinks=True)
    p = dst / REL
    text = p.read_text()
    if text.count(old) != 1:
        print('MUTATION TARGET NOT UNIQUE/ABSENT:', name, text.count(old)); sys.exit(2)
    p.write_text(text.replace(old, new))
    print('built', name)
