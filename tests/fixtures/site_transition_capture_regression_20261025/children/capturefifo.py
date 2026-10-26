import os
from pathlib import Path
p = Path('runner/current-site-transition-candidate/tap.stdout.tap')
p.unlink()
os.mkfifo(p, 0o600)
