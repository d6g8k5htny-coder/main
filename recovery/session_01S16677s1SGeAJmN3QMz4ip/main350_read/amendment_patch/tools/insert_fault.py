import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text()
anchor = "                old_criticals=[], incidence=[], events=[])\n"
insert = anchor + ("            if fault == 'height_disagree':\n"
                   "                # Critical box (0, 0) keeps its field value 4/9 while the\n"
                   "                # supplied algebraic height moves to 1: only the root-box\n"
                   "                # field-versus-algebra overlap check can object.\n"
                   "                selection['critical_heights'][0]['centered_interval'] = (Q(1), Q(1))\n")
assert text.count(anchor) == 1, text.count(anchor)
assert "height_disagree" not in text
path.write_text(text.replace(anchor, insert))
print("inserted fault branch")
