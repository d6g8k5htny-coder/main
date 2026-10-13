// Verifier 2 / Q1-5 (frontend half). Network stubbed (common.mjs). Read-only against cd32303 bytes.
// README.md@d6628da (sha 89d4c618...) is not available offline. Part B uses a LABELLED STAND-IN:
// the real Math- README@10e1f191 bytes, served at the d6628da README URL under a descriptor that
// carries the stand-in's own sha256/bytes/blob. This exercises the whole startMuseum pipeline; it
// does not claim the stand-in equals README@d6628da. Part C is the deduction for the real bytes.
import fs from 'node:fs';
import {shipped, tryValidate, render, findCard, texts, makeDescriptor, HERE, MATH_PIN} from './common.mjs';

const ids = ['d2-lifetime-remainder', 'd4-fixed-remote-rn', 'p15-demand-one-counterexample', 'd6-p15-full-price'];
const at = (m, id) => m.claims.find(c => c.id === id);

console.log('A1 shipped manifest:', tryValidate(shipped()));
{ const m = shipped(); const a = at(m, ids[0]), b = at(m, ids[3]); [a.replay.command, b.replay.command] = [b.replay.command, a.replay.command];
  console.log('A2 swap D2<->D6 commands:', tryValidate(m)); }
{ const m = shipped(); at(m, ids[1]).replay.command = 'python invented-solver.py';
  console.log('A3 D4 invented command (validateBoundManifest alone):', tryValidate(m)); }
{ const m = shipped(); const x = at(m, ids[2]); x.replay = structuredClone(at(m, ids[3]).replay);
  console.log('A4 P15 counterexample given D6 replay:', tryValidate(m), '| class', x.class, '| label', x.source_label); }
{ const m = shipped(); at(m, ids[0]).proof = structuredClone(at(m, ids[3]).proof);
  console.log('A5 control: D2 given D6 proof:', tryValidate(m)); }

// Part B: full pipeline with the stand-in README for every README-sourced replay.
const readme = fs.readFileSync(`${HERE}/readme_10e1f191.md`);
const stand = makeDescriptor('d6g8k5htny-coder/Math-', MATH_PIN, 'README.md', readme);
const overrides = new Map([[stand.url, readme]]);
function withStandIn(m) { for (const c of m.claims) if (c.replay?.source?.path === 'README.md') c.replay.source = structuredClone(stand); return m; }
async function show(label, m) {
  const {document} = await render(m, {overrides});
  const state = document.getElementById('museum-state').textContent;
  const root = document.getElementById('claim-cards');
  const refused = root.children.map(h => h.children[0]).filter(c => c.className === 'museum-card refused').map(c => c.textContent);
  console.log(`${label}: state="${state}" | refused=${JSON.stringify(refused)}`);
  for (const id of [ids[0], ids[1], ids[3]]) {
    const card = findCard(root, id); const t = card ? texts(card) : [];
    console.log(`   card ${id.padEnd(22)} rendered=${!!card} class=${card?.attributes['data-object-class']} replay=${JSON.stringify(t.find(s => s.startsWith('python')))}`);
  }
}
await show('B1 baseline (shipped commands, stand-in README)', withStandIn(shipped()));
{ const m = withStandIn(shipped()); const a = at(m, ids[0]), b = at(m, ids[3]); [a.replay.command, b.replay.command] = [b.replay.command, a.replay.command];
  at(m, ids[1]).replay.command = 'python -B -S frontiers/downstream_gate_20260925/hard_gate.py';
  await show('B2 D2<->D6 swapped, D4 = hard_gate.py', m); }
{ const m = withStandIn(shipped()); at(m, ids[1]).replay.command = 'python invented-solver.py';
  await show('B3 control: D4 invented command', m); }

// Part C: the real README@d6628da. D2/D4/D6 carry one identical replay.source descriptor, so the
// verifyClaim test "command is a trimmed line of replay.source" is invariant under any permutation
// of the three commands; the shipped CI test (museum_check -> test_museum_frontend 'actual pinned
// museum renders all cards') requires each original command to pass it against the real bytes.
const m = shipped(); const srcs = [ids[0], ids[1], ids[3]].map(id => JSON.stringify(at(m, id).replay.source));
console.log('C1 D2/D4/D6 replay.source descriptors identical:', new Set(srcs).size === 1, '| sha256', at(m, ids[0]).replay.source.sha256.slice(0, 16), '| bytes', at(m, ids[0]).replay.source.bytes);
console.log('C2 stand-in README sha256', stand.sha256.slice(0, 16), 'bytes', stand.bytes, '(differs from the pin: stand-in only)');
