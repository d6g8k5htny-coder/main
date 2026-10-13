// Q4 limits lens -- CONTROL probe (expected honest): what the page shows while a source stalls,
// after the 20 s production deadline fires, and when a source exceeds the 2 MiB cap.
// Uses the same synthetic manifest shape as tests/test_museum_frontend.mjs fixture().
// No network: in-process stub fetcher.
// Usage: node p7_failure_display_control.mjs <path-to-cd32303-tree>
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
const root = process.argv[2];
const museum = await import(pathToFileURL(path.join(root, 'docs/site/museum.mjs')));
globalThis.fetch = async () => { throw Error('network forbidden in probe'); };

const digest = v => createHash('sha256').update(v).digest('hex');
const pin = (p, v) => ({repository: 'd6g8k5htny-coder/Math-', path: p, commit: 'a'.repeat(40), blob: 'b'.repeat(40), bytes: Buffer.byteLength(v), sha256: digest(v), url: `https://raw.githubusercontent.com/d6g8k5htny-coder/Math-/${'a'.repeat(40)}/${p}`, html_url: `https://github.com/d6g8k5htny-coder/Math-/blob/${'a'.repeat(40)}/${p}`});
const ids = ['d2-lifetime-remainder', 'd3-side24-coefficient', 'd4-fixed-remote-rn', 'd5-all-height-annulus', 'd5-height-window-annulus', 'd5-two-scale', 'd5-inner-belt-density', 'd5-fixed-transverse', 'cumulative-transfer-correction', 'p15-demand-one-counterexample', 'd6-p15-full-price'];
function fixture() {
  const proof = pin('proof.md', 'Pinned proof.');
  const claims = ids.map((id, i) => ({id, title: `Object ${i}`, scope_quote: `- Object ${i}: [proof](proof.md). Scope ${i}.`, source_label: i === 9 ? 'EXACT_COUNTEREXAMPLE' : 'ACCEPT', class: i === 9 ? 'engineering-only' : 'ACCEPT-scoped', proof, review: null, replay: {command: null, url: null, notice: 'n'}, status_quote: null}));
  for (const i of [0, 1, 2, 10]) claims[i].status_quote = `Accepted scope ${i}.`;
  const open = ['d1-parent-selection-open', 'd5-pin-neighborhoods-open', 'sard-g-a1-a6-open'].map((id, i) => ({id, title: `Open ${i}`, scope_quote: `| **Open ${i}** | Still open ${i}. | [Proof](proof.md) |`, source_label: 'AMEND / open', class: 'AMEND/open', proof, review: null, replay: {command: null, url: null, notice: 'n'}, status_quote: `Still open ${i}.`}));
  const index = `# Index\n\n## Reviewed scoped results\n\n${claims.map(c => c.scope_quote).join('\n')}\n\n## Open or conditional results with complete proof text in GitHub\n- D1 parent lifetime theorem: full proof [proof](proof.md).\n`;
  const acceptedRows = [[0, 'D2'], [1, 'D3'], [2, 'D4'], [10, 'D6']].map(([i, id]) => `| **${id} — scope** | Accepted scope ${i}. | Review | Limits |`).join('\n');
  const status = `# Status\n\n## ACCEPT — scoped\n\n| Object | Accepted scope | Source and review | Explicit limits |\n|---|---|---|---|\n${acceptedRows}\n\n## AMEND / open\n\n| Object | Current reason | Source |\n|---|---|---|\n${open.map(c => c.scope_quote).join('\n')}\n\n## Engineering only\n`;
  const indexSource = pin('PROOF_INDEX.md', index), statusSource = pin('STATUS.md', status); open[1].proof = indexSource; open[2].proof = statusSource;
  return {index, status, manifest: {schema_version: 1, scientific_status_authority: false, index_source: indexSource, status_source: statusSource, claims: [...claims, ...open], exhibits: {ec014: proof, remote: proof, annulus: proof, p15: proof, lifetime: proof}, packets: []}};
}
class Element {
  constructor(tag) { this.tagName = tag.toUpperCase(); this.children = []; this.attributes = {}; this._text = ''; this.className = ''; this.id = ''; }
  set textContent(v) { this._text = String(v); this.children = []; } get textContent() { return this._text + this.children.map(c => c.textContent ?? String(c)).join(''); }
  append(...c) { this.children.push(...c); } replaceChildren(...c) { this._text = ''; this.children = c; }
  setAttribute(k, v) { this.attributes[k] = String(v); }
}
const makeDoc = () => { const nodes = new Map(['museum-state', 'claim-cards', 'lifetime-fixture', 'packet-cards', 'active-exhibit'].map(id => [id, new Element('div')])); return {createElement: t => new Element(t), getElementById: id => nodes.get(id) ?? null}; };

const f = fixture();
const serve = proofResponse => async url => {
  if (url === 'museum.json') return new Response(JSON.stringify(f.manifest));
  if (url === f.manifest.index_source.url) return new Response(f.index);
  if (url === f.manifest.status_source.url) return new Response(f.status);
  if (url === f.manifest.claims[0].proof.url) return proofResponse();
  throw Error('unexpected ' + url);
};
const card0 = d => d.getElementById('claim-cards').children[0].textContent.slice(0, 120);

// A. proof.md stalls (headers arrive, body never completes) -> 20 s production deadline
{
  const d = makeDoc(); const t = Date.now();
  const run = museum.startMuseum({document: d, search: '', fetcher: serve(() => new Response(new ReadableStream({pull() { return new Promise(() => {}); }})))});
  await new Promise(r => setTimeout(r, 1000));
  console.log(`A t=1s  state="${d.getElementById('museum-state').textContent}"`);
  console.log(`A t=1s  card0="${card0(d)}"`);
  await run;
  console.log(`A t=${((Date.now() - t) / 1000).toFixed(1)}s state="${d.getElementById('museum-state').textContent}"`);
  console.log(`A final card0="${card0(d)}"`);
  console.log(`A final lifetime="${d.getElementById('lifetime-fixture').textContent.slice(0, 100)}"`);
}
// B. proof.md streams past the 2 MiB production cap
{
  const d = makeDoc();
  await museum.startMuseum({document: d, search: '', fetcher: serve(() => new Response(new ReadableStream({pull(c) { c.enqueue(new Uint8Array(1 << 16)); }})))});
  console.log(`B state="${d.getElementById('museum-state').textContent}"`);
  console.log(`B card0="${card0(d)}"`);
}
