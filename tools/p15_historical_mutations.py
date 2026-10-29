#!/usr/bin/env python3
"""Seven reason-bound source mutations for historical P15 byte verification."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

def detected_failure(code, stderr):
    return (code==1 and "Ran 1 test" in stderr and "FAILED (failures=1)" in stderr
            and "AssertionError: ValueError not raised" in stderr and "ERROR:" not in stderr)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    source=root/"tools/p15_historical_artifact.py"
    tests=root/"tests/test_p15_historical_artifact.py"
    cases=[
     ("archive","if digest(raw)!=expected_zip:  # MUT_ARCHIVE","if False:  # MUT_ARCHIVE","test_archive_hash_refuses_changed_zip"),
     ("manifest","if digest(source)!=expected_manifest:  # MUT_MANIFEST","if False:  # MUT_MANIFEST","test_manifest_hash_refuses_changed_source"),
     ("receipt","if digest(receipt_bytes)!=expected_receipt:  # MUT_RECEIPT","if False:  # MUT_RECEIPT","test_receipt_hash_refuses_changed_receipt"),
     ("targets","if len(p15)!=expected_p15:  # MUT_TARGETS","if False:  # MUT_TARGETS","test_p15_target_count_refuses_other_count"),
     ("axioms","if any(axioms[name]!=[] for name in targets):  # MUT_AXIOMS","if False:  # MUT_AXIOMS","test_nonempty_axiom_list_is_refused"),
     ("logs","if got!=want:  # MUT_LOG","if False:  # MUT_LOG","test_wrong_log_bytes_are_refused"),
     ("source","if digest(source_data)!=want:  # MUT_SOURCE_FILE","if False:  # MUT_SOURCE_FILE","test_source_blob_mismatch_is_refused")]
    report={"status":"NOT_PASS","executions":[],"threshold":"seven named mutants per mode"}
    try:
        original=source.read_text()
        for mode in ("normal","optimized"):
            flag=["-O"] if mode=="optimized" else []
            command=[sys.executable,"-B","-S",*flag,"-m","unittest","discover","-s","tests","-p","test_p15_historical_artifact.py","-v"]
            base=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=60)
            (args.output/(mode+"-baseline.log")).write_text(base.stdout+base.stderr)
            report["executions"].append({"mode":mode,"name":"baseline","exit":base.returncode})
            if base.returncode or "Ran 15 tests" not in base.stderr or "OK" not in base.stderr or "skipped=" in base.stderr:
                raise RuntimeError("baseline incomplete")
            for name,old,new,method in cases:
                if original.count(old)!=1:raise ValueError("mutation anchor not unique: "+name)
                with tempfile.TemporaryDirectory() as temporary:
                    d=Path(temporary);(d/"tools").mkdir();(d/"tests").mkdir()
                    (d/"tools/p15_historical_artifact.py").write_text(original.replace(old,new,1))
                    shutil.copyfile(tests,d/"tests/test_p15_historical_artifact.py")
                    command=[sys.executable,"-B","-S",*flag,"-m","unittest","discover","-s","tests","-p","test_p15_historical_artifact.py","-k",method,"-v"]
                    child=subprocess.run(command,cwd=d,capture_output=True,text=True,timeout=60)
                    (args.output/(mode+"-"+name+".log")).write_text(child.stdout+child.stderr)
                    intended=detected_failure(child.returncode,child.stderr)
                    report["executions"].append({"mode":mode,"name":name,"exit":child.returncode,"intended_failure":intended})
                    if not intended:raise RuntimeError("mutant not killed for intended reason: "+name)
            harness=Path(__file__).read_text()
            marker='and "'+'AssertionError: ValueError not raised'+'" in stderr'
            if harness.count(marker)!=1:raise ValueError("harness mutation anchor not unique")
            with tempfile.TemporaryDirectory() as temporary:
                d=Path(temporary);(d/"tools").mkdir();(d/"tests").mkdir()
                shutil.copyfile(source,d/"tools/p15_historical_artifact.py")
                (d/"tools/p15_historical_mutations.py").write_text(harness.replace(marker,"and True",1))
                shutil.copyfile(tests,d/"tests/test_p15_historical_artifact.py")
                command=[sys.executable,"-B","-S",*flag,"-m","unittest","discover","-s","tests","-p","test_p15_historical_artifact.py","-k","test_false_kill_rejection_controls","-v"]
                child=subprocess.run(command,cwd=d,capture_output=True,text=True,timeout=60)
                (args.output/(mode+"-harness-mutant.log")).write_text(child.stdout+child.stderr)
                intended=(child.returncode==1 and "Ran 1 test" in child.stderr and "FAILED (failures=1)" in child.stderr and
                          "AssertionError: True is not false" in child.stderr and "ERROR:" not in child.stderr)
                report["executions"].append({"mode":mode,"name":"harness","exit":child.returncode,"intended_failure":intended})
                if not intended:raise RuntimeError("harness mutant was not caught for intended reason")
        report["status"]="ALL_SEVEN_SOURCE_AND_ONE_HARNESS_MUTANTS_KILLED_IN_BOTH_MODES"
    finally:
        (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report["status"])


if __name__=="__main__":
    main()
