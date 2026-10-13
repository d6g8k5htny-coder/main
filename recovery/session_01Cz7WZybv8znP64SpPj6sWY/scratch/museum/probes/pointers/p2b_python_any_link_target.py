"""F2 (generator side): project() checks only '](' + review_path + ')' in bullet, so the review slot accepts ANY
link target in the same bullet. For D2 that includes the '[main #67](.../issues/67)' link which the bullet says
'remains open for other bundled fronts' - not a review. For the AMEND rows, project() checks nothing at all."""
import copy, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; CD = (HERE / "../../cd32303").resolve()
sys.path.insert(0, str(CD / "tools")); import museum_data as md
idx = (HERE / "fixture/Math-/PROOF_INDEX.md").read_bytes(); st = (HERE / "fixture/main/STATUS.md").read_bytes()
for target in ["https://github.com/d6g8k5htny-coder/main/issues/67", "https://github.com/d6g8k5htny-coder/main/issues/6"]:
    specs = copy.deepcopy(md.CLAIM_SPECS); specs[0][2] = target
    md.CLAIM_SPECS = specs
    try:
        c = md.project(idx, st)[0]; print(f"project() D2 review={target!r}: ACCEPTED -> review_url={c['review']['review_url']!r}, notice kept={c['review']['review_notice'][:40]!r}")
    except ValueError as e:
        print(f"project() D2 review={target!r}: REFUSED ({e})")
