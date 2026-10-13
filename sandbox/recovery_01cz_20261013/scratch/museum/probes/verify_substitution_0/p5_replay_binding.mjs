// Q1/P5 (frontend half): validateBoundManifest (museum.mjs:96-98) checks the replay only for shape and
// for source repo/commit == proof repo/commit; verifyClaim (118) checks that the command is SOME line
// of the replay source. With the shared README as source, any README command passes for any claim.
import fs from 'node:fs';
import {shipped, tryValidate, museum, makeDescriptor, makeFetcher, HERE} from './common.mjs';
const m = shipped(); const [d2, d4, p15x, d6] = ['d2-lifetime-remainder', 'd4-fixed-remote-rn', 'p15-demand-one-counterexample', 'd6-p15-full-price'].map(id => m.claims.find(c => c.id === id));
const d2cmd = d2.replay.command; d2.replay.command = d6.replay.command; d6.replay.command = d2cmd; // swap
console.log('swap D2<->D6 replay commands:', tryValidate(m));
const n = shipped(); const x = n.claims.find(c => c.id === 'p15-demand-one-counterexample'); x.replay = structuredClone(n.claims[10].replay);
console.log('give engineering-only P15 counterexample the D6 full_price.py replay:', tryValidate(n), '| class', x.class);
// verifyClaim's command test with a REAL Math- README (commit 10e1f191, the locally available
// stand-in; README@d6628da is not available offline). The test is a set-membership test, so it is
// identical for the swapped commands whenever each original command was a README line.
const readme = fs.readFileSync(`${HERE}/readme_10e1f191.md`);
const src = makeDescriptor('d6g8k5htny-coder/Math-', '10e1f191c7d9f2755ca971ddbba87874d2475619', 'README.md', readme);
for (const [claim, cmd] of [[d4, 'python -B -S frontiers/full_price_20260924/full_price.py'], [d4, 'python -B -S frontiers/downstream_gate_20260925/hard_gate.py'], [x, 'python -B -S frontiers/full_price_20260924/full_price.py']]) {
  const c = structuredClone(claim); c.replay = {command: cmd, source: src, url: null, notice: 'x', checkout: c.proof.commit};
  const ok = await museum.verifyClaim(c, makeFetcher(m, new Map([[src.url, readme]]))).then(() => 'PASSES', e => `REFUSED: ${e.message}`);
  console.log(`verifyClaim ${c.id} with command ${JSON.stringify(cmd)} from README@10e1f191: ${ok}`);
}
