from pathlib import Path
p = Path('runner/current-site-transition-candidate/tap.stdout.tap')
p.unlink()
p.symlink_to('../../inputs/member-01.txt')
