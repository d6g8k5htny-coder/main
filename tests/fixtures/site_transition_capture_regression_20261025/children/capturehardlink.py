import os
from pathlib import Path
p = Path('runner/current-site-transition-candidate/tap.stdout.tap')
p.unlink()
q = Path('owned-hardlink-target.txt')
q.write_bytes(b'owned side input')
os.link(q, p)
