set -euo pipefail
python -B -S -m unittest discover -s tests -v
python -B -O -S -m unittest discover -s tests -v
