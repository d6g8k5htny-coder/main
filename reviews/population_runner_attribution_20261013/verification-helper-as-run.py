"""Record actual local commands; no campaign, hosted or scientific admission."""
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
CHECKOUT = ROOT / 'checkout'
POPULATION = 'experiments/universality/two_parameter_cusp_population'
stage = sys.argv[1]
directory = ROOT / 'evidence' / stage
directory.mkdir(parents=True, exist_ok=False)

def identities():
    return {name: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            for name in (POPULATION + '/run_population_certificate.py',
                         POPULATION + '/test_population_certificate.py')
            for raw in [(CHECKOUT / name).read_bytes()]}

groups = [('regression', ['-m', 'unittest', '-v',
          'test_population_certificate.RunnerAttributionTests.test_omitted_disclosures_do_not_claim_a_provider',
          'test_population_certificate.RunnerAttributionTests.test_cli_disclosures_and_defaults_reach_a_source_refusal_receipt'],
          CHECKOUT / POPULATION)] if stage == 'red' else [
    ('population', ['-m', 'unittest', 'discover', '-s', POPULATION,
                    '-p', 'test_population_certificate.py', '-v'], CHECKOUT),
    ('universality', ['-m', 'unittest', 'discover', '-s', 'experiments/universality',
                      '-p', 'test_*.py', '-v'], CHECKOUT),
    ('repository', ['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'], CHECKOUT)]
environment = os.environ.copy()
if stage in ('qualified', 'qualified-outside-git'):
    groups = [groups[-1]]
    temporary = (ROOT / 'resolved-runtime-tmp' if stage == 'qualified' else
                 Path('<outside-git-resolved-temp>'))
    temporary.mkdir(exist_ok=True)
    if stage == 'qualified-outside-git':
        probe = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=temporary,
                               capture_output=True)
        if probe.returncode == 0:
            raise RuntimeError('Qualification temporary directory is inside Git')
    environment['TMPDIR'] = str(temporary.resolve())
    environment['PATH'] = str(Path(sys.executable).parent) + os.pathsep + environment['PATH']
record = {'stage': stage, 'actual_executor': 'OpenAI/Codex root01a11cbf-068d-7102-861e-75814e715c98',
          'provider': 'OpenAI', 'source_exposed': True, 'organizational_independence_credit': 0,
          'scientific_effect': 'NONE', 'python': sys.version, 'platform': platform.platform(),
          'machine': platform.machine(), 'source_before': identities(), 'commands': [],
          'test_environment': {'TMPDIR': environment.get('TMPDIR'),
                               'python_on_PATH': shutil.which('python', path=environment['PATH'])}}
for name, arguments, cwd in groups:
    for mode in ('normal', 'optimized'):
        command = [sys.executable, '-B', '-E', '-S'] + (['-O'] if mode == 'optimized' else []) + arguments
        started = datetime.now(timezone.utc).isoformat()
        try:
            result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, timeout=180)
            code, stdout, stderr = result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired as error:
            code, stdout, stderr = 'TIMEOUT', error.stdout or b'', error.stderr or b''
        item = {'group': name, 'mode': mode, 'argv': command, 'cwd': str(cwd), 'exit_code': code,
                'started_utc': started, 'ended_utc': datetime.now(timezone.utc).isoformat()}
        for stream, raw in (('stdout', stdout), ('stderr', stderr)):
            path = directory / (name + '-' + mode + '.' + stream + '.log')
            path.write_bytes(raw)
            item[stream] = {'path': path.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        record['commands'].append(item)
        (directory / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'group': name, 'mode': mode, 'exit_code': code,
                          'tail': stderr.decode(errors='replace')[-600:]}), flush=True)
record['source_after'] = identities()
record['source_drift'] = record['source_before'] != record['source_after']
(directory / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
