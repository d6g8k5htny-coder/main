# Targeted composition check for the one consumer that changed since C219's tested base:
# architecture/evidence_adapter.py on main b7deb198. Start from the adapter test's accepted
# synthetic main-formal packet, then add the candidate's new native log "preflight-version.log"
# to BOTH the captured logs and the receipt's log inventory (what the candidate checker produces);
# the adapter must still accept. Negative: the log captured but absent from the receipt inventory,
# and the inverse, must each be refused.
import sys; sys.path.insert(0, '.')
import tests.test_architecture_evidence_adapter as A
from architecture.evidence_adapter import adapt_packet, parse_packet
t = A.ArchitectureEvidenceAdapterContract('setUp'); t.setUp()
def run(packet):
    return adapt_packet(parse_packet(A.encoded(packet)), t.expected)
base = t.with_formal(); r0 = run(base); print('baseline accepted; formal records:', len(r0.get('formal_records', r0.get('formal', []))) if isinstance(r0, dict) else type(r0))
PRE = b"Lean (version 4.34.1, TEST fixture only)\n"
p1 = t.with_formal(); f = p1['formal_records'][0]
cap = A.run_capture(PRE, 'preflight-version.log'); f['logs'].append(cap)
t.update_receipt(f, lambda r: r['logs'].update({'preflight-version.log': A.sha(PRE)}))
r1 = run(p1); print('candidate-shaped packet (10 logs incl. preflight-version.log) accepted:', isinstance(r1, dict))
import json; s0=json.dumps(r0, sort_keys=True); s1=json.dumps(r1, sort_keys=True); print('report identical to baseline except log inventory:', s0==s1, '| lengths', len(s0), len(s1))
p2 = t.with_formal(); f2 = p2['formal_records'][0]; f2['logs'].append(A.run_capture(PRE, 'preflight-version.log'))
try: run(p2); print('UNEXPECTED: unbound captured log accepted')
except ValueError as e: print('refused (log captured, not in receipt):', e)
p3 = t.with_formal(); f3 = p3['formal_records'][0]
t.update_receipt(f3, lambda r: r['logs'].update({'preflight-version.log': A.sha(PRE)}))
try: run(p3); print('UNEXPECTED: receipt entry without capture accepted')
except ValueError as e: print('refused (receipt entry, no capture):', e)
