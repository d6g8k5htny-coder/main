# Emulates "python3 -B -m unittest -v test_moment_kernel.py" from WORK under -I.
import sys, os
work = sys.argv[1]
os.chdir(work)
sys.path.insert(0, work)
sys.argv = ["python -m unittest", "-v", "test_moment_kernel.py"]
import unittest
unittest.main(module=None)
