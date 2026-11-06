import UniversalLaw.Side24.Ledger
import UniversalLaw.Side24.ConeMoments
import UniversalLaw.Side24.Endpoints
import UniversalLaw.P15.PriceBoundary
import UniversalLaw.P15.RealizedCovers
import UniversalLaw.P15.FullPrice

/-!
# Universal Law — main-side formal package (SIDE24 and P15 arithmetic skeletons)

Root module. `tools/formal_gate_check.py` requires this file to import exactly
the modules registered in `formal/manifest.json`, derives the target inventory
from their `theorem` declarations, and in `--run-lean` mode builds the package,
runs `leanchecker`, audits `#print axioms` for every target (only `propext`,
`Classical.choice`, `Quot.sound` are allowed) and executes negative controls.
-/
