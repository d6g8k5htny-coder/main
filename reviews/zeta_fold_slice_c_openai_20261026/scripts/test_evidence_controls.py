"""Standard-library negative controls for evidence bookkeeping, not math proofs."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("build_evidence", HERE / "build_evidence.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class EvidenceControls(unittest.TestCase):
    def test_timing_mask_does_not_hide_residual_or_value_change(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ.update(ZF_PKT=d, ZF_LOG=d)
            s = importlib.util.spec_from_file_location("frozen_compare", HERE/"frozen_compare.py")
            c = importlib.util.module_from_spec(s)
            s.loader.exec_module(c)
            self.assertEqual(build.classify(b"time 1.23\n",b"time 5.67\n",c.mask),"TIMING-ONLY")
            self.assertEqual(build.classify(b"t* 1.23\ntime 1.23\n",b"t* 1.24\ntime 5.67\n",c.mask),"DIFFERENT")
            self.assertEqual(build.classify(b"4 a b 4.128e-50\n",b"4 a b 8.3279e-51\n",c.mask),"DIFFERENT")

    def fixture(self, d):
        rows = []
        for i in range(59):
            name = "job"+str(i)
            b = (name+"\n").encode()
            (d/(name+".stdout")).write_bytes(b)
            (d/(name+".stderr")).write_bytes(b"")
            rows.append(dict(name=name,argv=["python","-B",name+".py"],rc=0,cwd=str(d/build.REL),
                stdout_sha256=build.sha(b),stderr_bytes=0))
        return rows

    def test_complete_multiset_and_actual_streams(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            expected = self.fixture(p)
            build.validate_jobs(list(reversed(expected)), expected, p, "python", p)
            for label, changed in [("missing",expected[:-1]),("duplicated",expected[:-1]+[expected[0]])]:
                with self.subTest(label=label),self.assertRaises(ValueError):
                    build.validate_jobs(changed,expected,p,"python",p)
            changed = [dict(r) for r in expected]
            changed[0]["rc"] = 1
            with self.assertRaisesRegex(ValueError,"nonzero exit"):
                build.validate_jobs(changed,expected,p,"python",p)
            changed = [dict(r) for r in expected]
            changed[0]["argv"] = ["python","-B","wrong.py"]
            with self.assertRaisesRegex(ValueError,"multiset"):
                build.validate_jobs(changed,expected,p,"python",p)
            for field,value,reason in [("argv",["wrong-python","-B","job0.py"],"interpreter"),
                                       ("cwd",str(p/"wrong"),"working directory")]:
                changed = [dict(r) for r in expected]
                changed[0][field] = value
                with self.subTest(field=field), self.assertRaisesRegex(ValueError,reason):
                    build.validate_jobs(changed,expected,p,"python",p)
            (p/"job0.stdout").write_bytes(b"forged\n")
            with self.assertRaisesRegex(ValueError,"stdout hash"):
                build.validate_jobs(expected,expected,p,"python",p)

    def test_wrong_recorded_environment_refused(self):
        good = dict(python="3.13.16 build",numpy="2.5.3",mpmath="1.4.1",mpmath_backend="python",
                    thread_environment=dict(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1"))
        build.validate_environment(good)
        for field,value in [("python","3.12.14"),("numpy","2.3.5"),("mpmath","1.3.0"),("mpmath_backend","gmpy")]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                build.validate_environment(dict(good,**{field:value}))
        with self.assertRaisesRegex(ValueError,"thread pin"):
            build.validate_environment(dict(good,thread_environment=dict(OPENBLAS_NUM_THREADS="8")))

    def test_explicit_hash_recovery_preserves_and_reports_inconsistent_capture(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            expected = self.fixture(p)
            original = (p/"job0.stdout").read_bytes()
            (p/"job0.stdout").write_bytes(original[:2])
            with self.assertRaisesRegex(ValueError,"stdout hash"):
                build.validate_jobs(expected,expected,p,"python",p)
            recovered = p/"recovered0.txt"
            recovered.write_bytes(original)
            recovery = {"job0":dict(path=str(recovered),retained_sha256=build.sha(original[:2]),retained_bytes=2)}
            report = build.validate_jobs(expected,expected,p,"python",p,recovery)
            self.assertEqual(len(report),1)
            self.assertEqual(report[0]["name"],"job0")
            self.assertEqual((p/"job0.stdout").read_bytes(),original[:2])
            recovered.write_bytes(b"wrong recovery")
            with self.assertRaisesRegex(ValueError,"recovery hash"):
                build.validate_jobs(expected,expected,p,"python",p,recovery)
            recovered.write_bytes(original)
            (p/"job0.stdout").write_bytes(b"new damaged capture")
            with self.assertRaisesRegex(ValueError,"damaged capture changed"):
                build.validate_jobs(expected,expected,p,"python",p,recovery)
            (p/"job0.stdout").write_bytes(original)
            with self.assertRaisesRegex(ValueError,"unused or unknown"):
                build.validate_jobs(expected,expected,p,"python",p,recovery)


if __name__ == "__main__":
    unittest.main(verbosity=2)
