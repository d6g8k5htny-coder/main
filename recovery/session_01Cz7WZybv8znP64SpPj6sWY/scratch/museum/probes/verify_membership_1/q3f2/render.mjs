// render.mjs <tree-root> <museum.json>: run <tree-root>/docs/site/museum.mjs startMuseum() with the network stubbed
// as raw.githubusercontent.com would answer: main URLs ONLY from `git show <commit>:<path>` in ./objstore.git
// (404 if the path is not in that commit's tree); Math- URLs (d6628da is not in the local object store) from the
// verify_bytes-checked ./fixture. Minimal DOM stub; geometry loader disabled (no ?view= selected).
import fs from 'node:fs'; import path from 'node:path'; import {execFileSync} from 'node:child_process';
const H = path.dirname(new URL(import.meta.url).pathname);
const [root, manifestPath] = process.argv.slice(2);
const museum = await import(path.resolve(root, 'docs/site/museum.mjs'));
const manifestText = fs.readFileSync(manifestPath, 'utf8'), manifest = JSON.parse(manifestText);
const gitShow = (c, p) => { try { return execFileSync('git', [`--git-dir=${H}/objstore.git`, 'show', `${c}:${p}`], {stdio: ['ignore', 'pipe', 'ignore'], env: {PATH: process.env.PATH, GIT_OPTIONAL_LOCKS: '0'}}); } catch { return null; } };
const log = [];
const fetcher = async url => {
  url = String(url); if (url === 'museum.json') return new Response(manifestText);
  const m = url.match(/^https:\/\/raw\.githubusercontent\.com\/d6g8k5htny-coder\/(main|Math-)\/([0-9a-f]{40})\/(.+)$/);
  if (!m) { log.push('404 ' + url); return new Response('', {status: 404}); }
  const p = decodeURIComponent(m[3]); let bytes = null;
  if (m[1] === 'main') bytes = gitShow(m[2], p);
  else { const f = `${H}/fixture/Math-/${p}`; if (fs.existsSync(f)) bytes = fs.readFileSync(f); }
  log.push(`${bytes ? 200 : 404} ${m[1]}@${m[2].slice(0, 8)}:${p}`);
  return bytes ? new Response(bytes) : new Response('', {status: 404});
};
class El { constructor(t){this.tagName=t.toUpperCase();this.children=[];this.attributes={};this._t='';this.id='';this.className='';}
  set textContent(v){this._t=String(v);this.children=[];} get textContent(){return [this._t,...this.children.map(c=>c.textContent??String(c))].filter(Boolean).join(' ');}
  append(...c){this.children.push(...c);} replaceChildren(...c){this._t='';this.children=c;} setAttribute(k,v){this.attributes[k]=String(v);} addEventListener(){} }
const html = fs.readFileSync(path.resolve(root, 'docs/site/museum.html'), 'utf8');
const nodes = new Map([...html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"/gi)].map(m => [m[2], new El(m[1])]));
const document = {createElement: t => new El(t), createTextNode: v => ({textContent: v}), getElementById: id => nodes.get(id) ?? null};
await museum.startMuseum({document, fetcher, search: '', geometryLoader: () => { throw Error('geometry not loaded'); }});
const cards = nodes.get('claim-cards').children.map(h => h.children[0]?.className || '?');
console.log(`  manifest status_source.commit=${manifest.status_source.commit.slice(0, 8)}; packets in manifest=${JSON.stringify(manifest.packets.map(p => p.id + '@' + p.result.commit.slice(0, 8)))}`);
console.log(`  claim cards rendered=${cards.filter(c => !c.includes('refused')).length} refused=${cards.filter(c => c.includes('refused')).length}`);
console.log(`  museum-state: ${nodes.get('museum-state').textContent}`);
console.log(`  packet-cards: ${nodes.get('packet-cards').textContent.replace(/\s+/g, ' ').slice(0, 260)}`);
console.log(`  packet requests: ${log.filter(l => l.includes('incoming/')).join(' | ') || '(none)'}`);
console.log(`  non-200 requests: ${log.filter(l => !l.startsWith('200')).length}`);
