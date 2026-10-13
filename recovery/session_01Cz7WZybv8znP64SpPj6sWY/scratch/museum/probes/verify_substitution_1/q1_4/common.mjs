// Shared harness for the Q1 substitution probes. Read-only against the cd32303 bytes.
// No network: global fetch is replaced by a stub that throws; every source is served
// from local bytes whose SHA-256 is checked against the pinned identity before use.
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';

export const SCRATCH = '<SCRATCH>/museum';
export const TREE = `${SCRATCH}/cd32303`;
export const HERE = `${SCRATCH}/probes/verify_substitution_1/q1_4`;
export const MATH_PIN = 'd6628da09384728992dcbe6e921cc28ba85aebb0';
export const MAIN_PIN = '71400b94f6cb354a8cf7aba73ffede2138a64efa';

globalThis.fetch = async url => { throw Error(`NETWORK BLOCKED in probe: ${url}`); };
export const museum = await import(`${TREE}/docs/site/museum.mjs`);
export const sha256 = b => createHash('sha256').update(b).digest('hex');
export const gitBlob = b => createHash('sha1').update(Buffer.concat([Buffer.from(`blob ${b.length}\0`), b])).digest('hex');
export const shipped = () => JSON.parse(fs.readFileSync(`${TREE}/docs/site/museum.json`, 'utf8'));

// Local bytes for the two pinned snapshots (Math- from the verified fixture directory,
// main from the cd32303 tree, whose STATUS/review/packet/notebook bytes equal the 71400b94 pins).
export function pinnedBytes(repository, commit, p) {
  if (repository === 'd6g8k5htny-coder/Math-' && commit === MATH_PIN) {
    const f = `${HERE}/fixture/Math-/${p}`; return fs.existsSync(f) ? fs.readFileSync(f) : null;
  }
  if (repository === 'd6g8k5htny-coder/main' && commit === MAIN_PIN) {
    const f = `${TREE}/${p}`; return fs.existsSync(f) ? fs.readFileSync(f) : null;
  }
  return null;
}
export function makeDescriptor(repository, commit, p, bytes) {
  const enc = p.split('/').map(encodeURIComponent).join('/');
  return {repository, commit, path: p, blob: gitBlob(bytes), bytes: bytes.length, sha256: sha256(bytes),
    url: `https://raw.githubusercontent.com/${repository}/${commit}/${enc}`,
    html_url: `https://github.com/${repository}/blob/${commit}/${enc}`};
}

// fetcher: overrides (url -> Buffer) first, then the pinned local bytes. Logs misses.
export function makeFetcher(manifest, overrides = new Map(), log = []) {
  return async url => {
    const u = String(url);
    if (u === 'museum.json') return new Response(JSON.stringify(manifest));
    if (overrides.has(u)) return new Response(overrides.get(u));
    const m = u.match(/^https:\/\/raw\.githubusercontent\.com\/(d6g8k5htny-coder\/[^/]+)\/([0-9a-f]{40})\/(.+)$/);
    const b = m && pinnedBytes(m[1], m[2], decodeURIComponent(m[3]));
    if (!b) { log.push(`404 (no local bytes): ${u}`); return new Response('missing', {status: 404}); }
    return new Response(b);
  };
}

class Element {
  constructor(tag){this.tagName=tag.toUpperCase();this.children=[];this.attributes={};this.listeners={};this.hidden=false;this._text='';this.style={};}
  set textContent(v){this._text=String(v);this.children=[];} get textContent(){return this._text+this.children.map(c=>c.textContent??String(c)).join('');}
  append(...c){this.children.push(...c);} replaceChildren(...c){this._text='';this.children=c;}
  setAttribute(k,v){this.attributes[k]=String(v);} getAttribute(k){return this.attributes[k]??null;} addEventListener(k,f){this.listeners[k]=f;}
  querySelector(){return null;} remove(){} get firstChild(){return this.children[0]??null;}
}
export function makeDocument() {
  const html = fs.readFileSync(`${TREE}/docs/site/museum.html`, 'utf8');
  const nodes = new Map([...html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"/gi)].map(m => [m[2], new Element(m[1])]));
  const doc = {nodes, head: new Element('head'), createElement: t => new Element(t), createElementNS: (_ns, t) => new Element(t),
    createTextNode: v => ({textContent: v}), getElementById: id => nodes.get(id) ?? null};
  return doc;
}
export function walk(node, f) { if (!node || typeof node !== 'object') return; f(node); for (const c of node.children ?? []) walk(c, f); }
export function findCard(root, id) { let hit = null; walk(root, n => { if (!hit && n.attributes?.['data-object-id'] === id) hit = n; }); return hit; }
export function anchors(node) { const out = []; walk(node, n => { if (n.tagName === 'A') out.push([n.textContent, n.href]); }); return out; }
export function texts(node) { const out = []; walk(node, n => { if (n._text) out.push(n._text); }); return out; }

export async function render(manifest, {search = '', overrides = new Map(), withGeometry = false} = {}) {
  const document = makeDocument(); const log = [];
  if (withGeometry) globalThis.document = document; // geometry.mjs draws through the global document
  const geometryLoader = withGeometry ? () => import(`${TREE}/docs/site/geometry.mjs`) : () => { throw Error('geometry not loaded in this probe'); };
  await museum.startMuseum({document, fetcher: makeFetcher(manifest, overrides, log), search, geometryLoader});
  return {document, log};
}
export function textOf(p) { return fs.readFileSync(p, 'utf8'); }
export function indexText() { return textOf(`${HERE}/fixture/Math-/PROOF_INDEX.md`); }
export function statusText() { return textOf(`${TREE}/STATUS.md`); }
export function tryValidate(manifest, index = indexText(), status = statusText()) {
  try { museum.validateBoundManifest(manifest, index, status); return 'ACCEPTED by validateBoundManifest'; }
  catch (e) { return `REFUSED: ${e.message}`; }
}
