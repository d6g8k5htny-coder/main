import pathlib, shutil, subprocess, sys, tempfile
repo = pathlib.Path(sys.argv[1]); src = (repo/'tools/release_snapshot.py').read_text()
M = {
 'M1 identity check skipped in read()': ("                if paginated:\n                    check_identity(value, identity)\n", "                if False:\n                    check_identity(value, identity)\n"),
 'M2 duplicates silently skipped (dedup)': ("            raise ValueError(message)\n", "            continue\n"),
 'M3 filename identity replaced by blob SHA': ("            key = row['filename']\n", "            key = row.get('sha')\n"),
 'M4 malformed PR rows counted, not deferred': ("                continue  # Malformed PR rows keep the per-row error path in collect().\n", "                raise\n"),
 'M5 missing policy accepted': ("        raise ValueError('paginated collection has no explicit identity policy')\n", "        return\n"),
 'M6 rulesets route unwired': ("read('/rulesets', 'rulesets.json', paginated=True, identity=ID_IDENTITY)", "read('/rulesets', 'rulesets.json', paginated=True)"),
 'M7 files route given id policy': ("                     identity=FILENAME_IDENTITY)", "                     identity=ID_IDENTITY)"),
 'M8 bool accepted as id': ("type(row.get('id')) is not int", "not isinstance(row.get('id'), int)"),
 'M9 duplicate-PR consequence not stated': ("                message += '; open pull requests not processed'\n", "                pass\n"),
 'M10 first-wins: second duplicate dropped from raw rows': ("                write_json(dest/filename, value)\n", "                write_json(dest/filename, list({(r.get('id') or r.get('filename') or id(r)) if isinstance(r, dict) else id(r): r for r in value}.values()) if paginated and isinstance(value, list) else value)\n"),
}
killed = 0
for name, (old, new) in M.items():
    assert src.count(old) == 1, (name, src.count(old))
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d); (root/'tools').mkdir(); (root/'tests').mkdir()
        (root/'tools/release_snapshot.py').write_text(src.replace(old, new))
        shutil.copy(repo/'tests/test_release_snapshot.py', root/'tests/')
        r = subprocess.run([sys.executable, '-B', '-S', '-m', 'unittest', 'tests.test_release_snapshot'],
                           cwd=root, capture_output=True, text=True)
        tail = [l for l in r.stderr.splitlines() if l.startswith(('Ran', 'FAILED', 'OK'))]
        caught = r.returncode != 0; killed += caught
        print(f"{'KILLED' if caught else 'SURVIVED'}  {name}  ::  {' '.join(tail)}")
print(f'{killed}/{len(M)} mutants killed')
