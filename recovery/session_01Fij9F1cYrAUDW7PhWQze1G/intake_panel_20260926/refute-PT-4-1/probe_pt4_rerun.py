"""Re-run of the finder's F4 probe verbatim against the fresh checker copy."""
import unittest
from harness import IntakeTests

class F4_BaseShaNotBranchTip(IntakeTests):
    def test_branch_tip_never_consulted(self):
        result = self.run_check()
        self.assertEqual(result['lane'], 'incoming')
        self.assertFalse(any('/git/trees/main' in c or '/git/ref' in c or '/branches/' in c for c in self.api.calls))
        print('\nAPI calls:', *self.api.calls, sep='\n  ')

if __name__ == '__main__':
    unittest.main()
