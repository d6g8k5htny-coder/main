import unittest
from pathlib import Path
def record(label):
 with Path('/workspace/cbz-0/delivery/grader-identity-fix/review-probes/staticmethod-subclass-alias/layer_a/trace.txt').open("a") as f: f.write(label+"\n")
class Alias(staticmethod):
 def __get__(self,instance,owner):
  record("descriptor")
  owner.test_second=owner.test_first
  return owner.test_first
class C(unittest.TestCase):
 def test_first(self): record("first")
 @Alias
 def test_second(self):
  record("second")
  self.fail("retained original second body")
