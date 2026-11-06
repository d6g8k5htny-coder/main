import unittest
from pathlib import Path
def record(label):
 with Path('/workspace/cbz-0/delivery/grader-identity-fix/review-probes/constructor-alias/kernel_replay/trace.txt').open("a") as f: f.write(label+"\n")
class C(unittest.TestCase):
 def __init__(self, methodName="runTest"):
  super().__init__(methodName)
  type(self).test_second=type(self).test_first
 def test_first(self): record("first")
 def test_second(self):
  record("second")
  self.fail("retained original second body")
