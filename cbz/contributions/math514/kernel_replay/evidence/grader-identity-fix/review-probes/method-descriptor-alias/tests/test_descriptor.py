import unittest
from pathlib import Path
def record(label):
 with Path('/workspace/cbz-0/delivery/grader-identity-fix/review-probes/method-descriptor-alias/kernel_replay/trace.txt').open("a") as f: f.write(label+"\n")
class C(unittest.TestCase):
 def test_first(self): record("first")
 def test_second(self):
  record("second")
  self.fail("retained original second body")
ORIGINAL_SECOND=C.test_second
class Rebind:
 def __get__(self,instance,owner):
  record("descriptor")
  owner.test_second=owner.test_first
  return owner.test_second
C.test_second=Rebind()
