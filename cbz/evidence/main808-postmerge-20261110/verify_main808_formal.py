import hashlib
import json
import pathlib
import zipfile

root = pathlib.Path('/workspace/scratch/7abea556a9be/cbz_v51_build/private')
archive = root / 'main808-formal.zip'
manifest = pathlib.Path('/workspace/scratch/7abea556a9be/cbz_v49_build/native/main/cbz/evidence/formal-main811-20261110/main811-formal-manifest.json').read_bytes()
expected_commit = '02973f3975a1c4cc91b7c51d372f65717c9b3c5c'
expected_tree = '5d3160d982b4513f17848de45e0f4569233acd4e'


def sha(data):
    return hashlib.sha256(data).hexdigest()


raw = archive.read_bytes()
if (len(raw) != 81224 or sha(raw) != '38f8f2b00f51ceace6e746552699c5709757be7fa059d27e7255c4d1012dcff1'
        or sha(manifest) != '022cf67419a08402ace3b689b1603c61283a633128fb7fcb12f212c0468d4b73'):
    raise ValueError('archive or exact native manifest identity')
source = json.loads(manifest)
with zipfile.ZipFile(archive) as z:
    members = z.namelist()
    if len(members) != len(set(members)) or len(members) != 70 or z.testzip() is not None:
        raise ValueError('archive population or CRC')
    receipt_raw = z.read('receipt/receipt.json')
    receipt = json.loads(receipt_raw)
    binding = json.loads(z.read('receipt/required-check-binding.json'))
    if (z.read('commit.txt').decode().strip() != expected_commit
            or receipt['checked_commit'] != expected_commit
            or receipt['repository'] != 'd6g8k5htny-coder/main'
            or receipt['workflow_run_id'] != '42781606883'
            or receipt['workflow_run_attempt'] != '1'
            or receipt['manifest_sha256'] != sha(manifest)
            or receipt['formalization_status'] != 'kernel-checked'
            or receipt['alignment_status'] != 'PENDING_INDEPENDENT_REVIEW'
            or receipt['scientific_effect'] != 'NONE'
            or receipt['scientific_status_authority'] is not False):
        raise ValueError('receipt identity or scope')
    if (binding['checked_commit'] != expected_commit or binding['run_id'] != '42781606883'
            or binding['run_attempt'] != '1' or binding['receipt_sha256'] != sha(receipt_raw)):
        raise ValueError('required-check binding')
    targets = {item['name'] for item in source['targets']}
    if (len(targets) != 59 or set(receipt['axioms']) != targets
            or any(axioms != [] for axioms in receipt['axioms'].values())
            or set(receipt['compiled_inventory']['public_theorems']) != targets
            or receipt['compiled_inventory']['axioms'] != []):
        raise ValueError('target or axiom boundary')
    if len(receipt['negative_controls']) != 18 or any(
            not value.startswith('REJECTED_BY_') for value in receipt['negative_controls'].values()):
        raise ValueError('negative controls')
    for name, expected in receipt['logs'].items():
        if sha(z.read('receipt/' + name)) != expected:
            raise ValueError('receipt log hash: ' + name)
    logs = []
    for mode in ('normal', 'optimized'):
        text = z.read('tests-' + mode + '.log').decode()
        if ('Ran 147 tests in ' not in text or not text.rstrip().endswith('OK')
                or ' ... skipped ' in text or 'skipped=' in text
                or 'test_real_package_executes_with_receipt' not in text
                or 'test_actual_clean_and_omitted_constants' not in text):
            raise ValueError('test population: ' + mode)
        logs.append({'mode': mode, 'tests': 147, 'skips': 0})
    leanlog = z.read('run-lean.log').decode()
    if 'leanchecker: exit 0' not in leanlog or '4.34.1' not in receipt['lean_version']:
        raise ValueError('Lean or leanchecker outcome')
out = {
    'run': 42781606883, 'job': 129712719992, 'artifact': 13746426465,
    'sha256': sha(raw), 'bytes': len(raw), 'entries': 70, 'crc': 'valid',
    'tested_commit': expected_commit, 'tested_tree': expected_tree,
    'manifest_git_blob': '7bd51fe201ade413fcd18314854c40ee4aae3df6',
    'manifest_sha256': sha(manifest), 'receipt_sha256': sha(receipt_raw),
    'targets': len(targets), 'target_axioms': [], 'negative_controls': 18,
    'receipt_log_hashes': len(receipt['logs']), 'tests': logs,
    'alignment': 'PENDING_INDEPENDENT_REVIEW', 'scientific_effect': 'NONE',
    'scope': 'exact main-side arithmetic package only; not whole-federation G3/G4/G5',
}
archive.with_name('main808-formal-verification.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out))
