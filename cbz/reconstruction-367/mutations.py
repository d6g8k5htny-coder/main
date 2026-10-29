"""Exercise exact guard removals; never count setup or import failures as kills."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

root=Path(__file__).resolve().parent
source=(root/"regenerate.py").read_text()
cases=[
 ("axioms","set(axioms) <= APPROVED","True","FamilyValidation.test_unapproved_axiom_refused"),
 ("source_commit",'row["source_commit"] == commit',"True","FamilyValidation.test_wrong_checked_commit_refused"),
 ("target_path",'row["path"] == target_paths[name]',"True","FamilyValidation.test_altered_target_path_refused"),
 ("receipt_source",'receipt["source_sha256"] == manifest["files"]',"True","FamilyValidation.test_receipt_source_table_drift_refused"),
 ("git_digest","expected is None or checksum == expected","True","GitBinding.test_wrong_digest_rejected")]
out=[]
for mode in ([],["-O"]):
    base=[sys.executable,"-B","-S",*mode]
    normal=subprocess.run(base+[str(root/"test_regenerate.py"),"-v"],capture_output=True,text=True)
    if (normal.returncode != 0 or
        re.search(r"^Ran 15 tests in ", normal.stderr, re.M) is None or
        re.search(r"^OK$", normal.stderr, re.M) is None):
        raise SystemExit("baseline failed: "+normal.stdout+normal.stderr)
    for name,old,new,method in cases:
        if source.count(old)!=1:
            raise SystemExit("non-unique mutation "+name)
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            (directory/"regenerate.py").write_text(source.replace(old,new))
            shutil.copy(root/"test_regenerate.py",directory/"test_regenerate.py")
            run=subprocess.run(base+[str(directory/"test_regenerate.py"),method,"-v"],
                               capture_output=True,text=True)
            text=run.stdout+run.stderr
            intended=(run.returncode==1 and "ValueError not raised" in text
                      and re.search(r"^Ran 1 test in ",text,re.M) is not None
                      and "FAILED (failures=1)" in text and "ERROR:" not in text
                      and ("FAIL: "+method.rsplit(".",1)[-1]+" (") in text)
            if not intended:
                raise SystemExit("mutation did not fail at intended assertion: "+name+"\n"+text)
            out.append({"mode":"optimized" if mode else "normal","mutant":name,"exit":run.returncode})
test_source=(root/"test_regenerate.py").read_text()
anchor='    exec(compile(data, str(path), "exec", dont_inherit=True), module.__dict__)'
if test_source.count(anchor)!=1:
    raise SystemExit("source-loader mutation anchor is not unique")
with tempfile.TemporaryDirectory() as temp:
    directory=Path(temp)
    shutil.copy(root/"regenerate.py",directory/"regenerate.py")
    mutant=test_source.replace(anchor,'    spec.loader.exec_module(module)',1)
    (directory/"test_regenerate.py").write_text(mutant)
    for mode in ([],["-O"]):
        run=subprocess.run([sys.executable,"-B","-S",*mode,str(directory/"test_regenerate.py"),
                            "SourceLoader.test_poisoned_bytecode_does_not_replace_source","-v"],
                           capture_output=True,text=True)
        text=run.stdout+run.stderr
        if not (run.returncode==1 and "AssertionError: 'POISON' != 'SOURCE'" in text
                and re.search(r"^Ran 1 test in ",text,re.M) is not None
                and "FAILED (failures=1)" in text
                and "FAIL: test_poisoned_bytecode_does_not_replace_source (" in text):
            raise SystemExit("poisoned-loader harness mutant not detected: "+text)
        out.append({"mode":"optimized" if mode else "normal","mutant":"poisoned-loader","exit":run.returncode})
print(json.dumps({"status":"ALL_DECLARED_MUTANTS_DETECTED","results":out},sort_keys=True))
