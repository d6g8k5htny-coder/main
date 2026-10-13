// Q4 limits lens -- FINDING probe (minor, fail-closed): the 256 KiB museum.json cap
// (museum.mjs:180) is exercised in CI only against JSON.stringify(manifest) (compact), never against
// the bytes museum_data.dump() writes and the page actually serves (indent=2, sort_keys).
// tests/test_museum_frontend.mjs:134 `if(url==='museum.json')return new Response(JSON.stringify(manifest));`
// Neither museum_data.py nor museum_check.py bounds the export size.
// Construct an export in the gap: compact == cap (passes the CI path), pretty > cap (served bytes refused).
// No network: in-process stub fetcher; sources answer 404 so we only observe how far startMuseum gets.
// Usage: node p3_manifest_cap_tested_on_reserialized_bytes.mjs <path-to-cd32303-tree>
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
const root = process.argv[2];
const museum = await import(pathToFileURL(path.join(root, 'docs/site/museum.mjs')));
globalThis.fetch = async () => { throw Error('network forbidden in probe'); };
const CAP = 256 * 1024;

const actual = fs.readFileSync(path.join(root, 'docs/site/museum.json'));
const manifest = JSON.parse(actual);
console.log(`actual museum.json: served(pretty)=${actual.length} compact=${Buffer.byteLength(JSON.stringify(manifest))} cap=${CAP}`);

// Pad the free-text 'meaning' field (not validated by validateBoundManifest) so compact == CAP exactly.
const base = Buffer.byteLength(JSON.stringify({...manifest, meaning: ''}));
const padded = {...manifest, meaning: 'x'.repeat(CAP - base)};
const compact = JSON.stringify(padded);
// Serialize exactly as tools/museum_data.py dump() does (python -I -B: no bytecode written into the tree).
const py = `import sys,json; sys.path.insert(0, sys.argv[1]); import museum_data; sys.stdout.buffer.write(museum_data.dump(json.load(sys.stdin)))`;
const pretty = execFileSync('python3', ['-I', '-B', '-c', py, path.join(root, 'tools')], {input: compact, maxBuffer: 8 << 20});
console.log(`padded export: compact=${Buffer.byteLength(compact)} museum_data.dump()=${pretty.length} cap=${CAP}`);

class Element {
  constructor(tag) { this.tagName = tag.toUpperCase(); this.children = []; this.attributes = {}; this._text = ''; this.className = ''; }
  set textContent(v) { this._text = String(v); this.children = []; } get textContent() { return this._text + this.children.map(c => c.textContent ?? String(c)).join(''); }
  append(...c) { this.children.push(...c); } replaceChildren(...c) { this._text = ''; this.children = c; }
  setAttribute(k, v) { this.attributes[k] = String(v); }
}
const doc = () => { const nodes = new Map(['museum-state', 'claim-cards', 'lifetime-fixture', 'packet-cards', 'active-exhibit'].map(id => [id, new Element('div')])); return {createElement: t => new Element(t), getElementById: id => nodes.get(id) ?? null}; };

for (const [label, bytes] of [['CI test path (JSON.stringify)', compact], ['production path (dump bytes)', pretty]]) {
  const document = doc();
  await museum.startMuseum({document, search: '', fetcher: async url => url === 'museum.json' ? new Response(bytes) : new Response('', {status: 404})});
  console.log(`${label}: museum-state = "${document.getElementById('museum-state').textContent}"`);
}
