import os
from pathlib import Path
p = Path('inputs/member-00.txt')
p.unlink()
os.mkfifo(p, 0o600)
