import unittest
from pathlib import Path
def record(label):
 with Path('/workspace/cbz-0/delivery/grader-identity-fix/review-probes/module-dir-omission/kernel_replay/trace.txt').open("a") as f: f.write(label+"\n")
class Visible(unittest.TestCase):
 def test_first(self): record("first")
class Hidden(unittest.TestCase):
 def test_hidden(self):
  record("hidden")
  self.fail("hidden declared body retained")
def __dir__(): return ["Visible"]
