"""Check the retained post-merge Workspace pin-job transcript's bounded claims."""
import hashlib
import json
from pathlib import Path
import re

path = Path(__file__).with_name('pin-job-129740862312.log')
raw = path.read_bytes()
log = raw.decode('utf-8-sig')
sha = '5ee815d60d5e11cb590b2a3b274fd13104bd915d'
if not re.search(r'(?m)^\S+ ' + sha + r'$', log):
    raise ValueError('checkout did not report the merged commit')
expected = {'main', 'Math-', 'meta-framework', 'query-', 'google-drive',
            'trial', 'governance-', 'sandbox'}
actual = re.findall(r'(?m)^\S+ info: repos/([^:]+): pin equals remote default main tip$', log)
if len(actual) != 8 or set(actual) != expected:
    raise ValueError('current pin observations differ')
if 'verify_pins: OK (8 pin(s) agree with tree and .gitmodules; remotes checked; current tips required)' not in log:
    raise ValueError('current pin summary missing')
if 'verify_canonical_inputs: OK (4 canonical inputs; default-tip currency is separate)' not in log:
    raise ValueError('canonical input summary missing')
if not re.search(r'(?m)^\S+ Ran 258 tests in [0-9.]+s$', log):
    raise ValueError('test count missing')
if not re.search(r'(?m)^\S+ OK$', log):
    raise ValueError('test result missing')
if re.search(r'(?m)^\S+ .* \.\.\. (?:skipped|FAIL|ERROR)\b|^\S+ FAILED\s*\(|^\S+ OK \(skipped=', log):
    raise ValueError('a skipped or failed test was observed')
result = {'job': 129740862312, 'checkout': sha, 'sha256': hashlib.sha256(raw).hexdigest(),
          'bytes': len(raw), 'tests': 258, 'skips': 0, 'current_pins': sorted(actual),
          'canonical_inputs': 4, 'scope': 'this post-merge job; not final federation G7 or G5'}
Path(__file__).with_name('pin-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
