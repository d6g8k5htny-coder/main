// v3_render.mjs <worktree> <mode>: run cd32303 docs/site/museum.mjs startMuseum() against <worktree>/docs/site/museum.json.
// Network is stubbed. mode "harness": every URL served from bytes chosen by museum_data.read_source (what museum_check.py
// passes via MUSEUM_FIXTURE). mode "strict": raw.githubusercontent.com semantics for main = `git show <commit>:<path>`
// in the worktree's object store, else 404; Math- (d6628da not in local objects) served from the verified fixture.
import fs from 'node:fs'; import {execFileSync} from 'node:child_process';
const H = new URL('.', import.meta.url).pathname; const CD = new URL('../../cd32303/', import.meta.url);
const museum = await import(new URL('docs/site/museum.mjs', CD));
const [wt, mode] = process.argv.slice(2);
const manifestText = fs.readFileSync(`${wt}/docs/site/museum.json`, 'utf8');
const harness = mode === 'harness' ? JSON.parse(execFileSync('python3', ['-I', '-B', `${H}v3_harness_map.py`, wt, `${H}fixture/Math-`], {maxBuffer: 1 << 26})) : null;
const log = [];
const fetcher = async url => {
  url = String(url); if (url === 'museum.json') return new Response(manifestText);
  const m = url.match(/^https:\/\/raw\.githubusercontent\.com\/d6g8k5htny-coder\/(main|Math-)\/([0-9a-f]{40})\/(.+)$/);
  if (!m) return new Response('', {status: 404});
  let bytes = null; const path = decodeURIComponent(m[3]);
  if (harness) bytes = Object.hasOwn(harness, url) ? Buffer.from(harness[url], 'base64') : null;
  else if (m[1] === 'main') { try { bytes = execFileSync('git', ['-C', wt, 'show', `${m[2]}:${path}`], {stdio: ['ignore', 'pipe', 'ignore'], env: {...process.env, GIT_OPTIONAL_LOCKS: '0'}}); } catch { bytes = null; } }
  else { const f = `${H}fixture/Math-/${path}`; bytes = fs.existsSync(f) ? fs.readFileSync(f) : null; }
  if (path.startsWith('incoming/')) log.push(`${bytes ? 200 : 404} ${m[2].slice(0, 8)}:${path}`);
  return bytes ? new Response(bytes) : new Response('', {status: 404});
};
class El { constructor(t){this.tagName=t.toUpperCase();this.children=[];this._t='';this.attributes={};this.className='';}
  set textContent(v){this._t=String(v);this.children=[];} get textContent(){return [this._t,...this.children.map(c=>c.textContent??String(c))].filter(Boolean).join(' ');}
  append(...c){this.children.push(...c);} replaceChildren(...c){this._t='';this.children=c;} setAttribute(k,v){this.attributes[k]=String(v);} addEventListener(){} }
const html = fs.readFileSync(new URL('docs/site/museum.html', CD), 'utf8');
const nodes = new Map([...html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"/gi)].map(x => [x[2], new El(x[1])]));
const document = {createElement: t => new El(t), createTextNode: v => ({textContent: v}), getElementById: id => nodes.get(id) ?? null};
await museum.startMuseum({document, fetcher, search: '', geometryLoader: () => { throw Error('no geometry in probe'); }});
console.log(`[${mode}] museum-state: ${nodes.get('museum-state').textContent}`);
console.log(`[${mode}] packet-cards: ${nodes.get('packet-cards').textContent.replace(/\s+/g, ' ').slice(0, 300)}`);
console.log(`[${mode}] packet fetches: ${log.join(' | ')}`);
