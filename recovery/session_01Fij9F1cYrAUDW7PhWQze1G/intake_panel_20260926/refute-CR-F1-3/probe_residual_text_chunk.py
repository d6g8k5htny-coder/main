import importlib.util, io, os, struct, sys, unittest, zipfile, zlib
from pathlib import Path
TREE = Path(__file__).resolve().parent / 'head'
spec = importlib.util.spec_from_file_location('house_tests', TREE / 'tests/test_public_intake.py')
house = importlib.util.module_from_spec(spec); spec.loader.exec_module(house)
class Residual(house.IntakeTests):
    def test_zip_inside_tEXt_chunk_still_accepted_INFO(self):
        # Finder's own residual note: bytes inside an ancillary chunk's data field are not inspected.
        self.raw['plot.png'] = house.png(after=[(b'tEXt', b'Comment\0' + house.zip_archive())])
        self.save()
        r = self.run_check(); self.assertEqual(r['lane'], 'incoming'); print('INFO: accepted, artifact_files=', r['artifact_files'], file=sys.stderr)
suite = unittest.TestLoader().loadTestsFromName('probe_residual_text_chunk.Residual.test_zip_inside_tEXt_chunk_still_accepted_INFO')
sys.exit(not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful())
