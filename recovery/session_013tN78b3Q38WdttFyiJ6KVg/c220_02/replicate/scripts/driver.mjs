// Independent C220 replication driver (CDP only).
// usage: node driver.mjs SCENARIO OUTDIR PORT THREE_JS_PATH RAW_CACHE_DIR
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import crypto from 'node:crypto';

const [scenario, outDir, port, threePath, rawCache] = process.argv.slice(2);
const BASE = 'http://127.0.0.1:8765/museum.html';
const THREE_URL = 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.js';
const logFile = fs.createWriteStream(path.join(outDir, 'driver.log'), { flags: 'a' });
const ts = (ms = Date.now()) => new Date(ms).toISOString().slice(11, 23);
const log = (...a) => { const line = `${ts()} ${a.join(' ')}`; logFile.write(line + '\n'); console.log(line); };
const sleep = ms => new Promise(r => setTimeout(r, ms));

const SCEN = {
  S1: { url: `${BASE}?view=ec014`, three: 'block', steps: [
    ['settle', 'first load settled', 110], ['drain', 'final drain', 15]] },
  S2: { url: `${BASE}?view=ec014`, three: 'serve', steps: [
    ['settle', 'first load settled', 110], ['drain', 'final drain', 15]] },
  S3: { url: `${BASE}?view=ec014`, three: 'serve', steps: [
    ['settle', 'first load settled', 110],
    ['focus', 'Show 2D diagram', 12], ['enter', 'Show 2D diagram', 20],
    ['focus', 'Show 3D view', 12], ['enter', 'Show 3D view', 25],
    ['drain', 'final drain', 15]] },
  S4: { url: `${BASE}?view=p15`, three: 'serve', steps: [
    ['settle', 'first load settled', 110],
    ['focus', 'Toggle x1', 12], ['enter', 'Toggle x1', 20],
    ['focus', 'Toggle x1', 12], ['enter', 'Toggle x1', 20],
    ['focus', 'Swap palette colors', 12], ['enter', 'Swap palette colors', 20],
    ['drain', 'final drain', 15]] },
  S5: { url: `${BASE}?view=ec014`, three: 'serve', steps: [
    ['settle', 'first load settled', 110], ['lose', 'loseContext', 30], ['drain', 'final drain', 15]] },
};
SCEN.S3F = { ...SCEN.S3, url: `${BASE}?view=ec014#active-exhibit` };
SCEN.S4F = { ...SCEN.S4, url: `${BASE}?view=p15#active-exhibit` };
SCEN.T = { url: `${BASE}?view=p15#active-exhibit`, three: 'serve', steps: [['settle', 'first load settled', 30], ['focus', 'Toggle x1', 5], ['enter', 'Toggle x1', 8], ['drain', 'final drain', 5]] };
const sc = SCEN[scenario] || SCEN[scenario.replace(/r\d+$/, '')];
if (!sc) throw new Error('unknown scenario ' + scenario);

async function pageTarget() {
  for (let i = 0; i < 80; i++) {
    try { const j = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); const p = j.find(t => t.type === 'page'); if (p) return p; } catch {}
    await sleep(250);
  }
  throw new Error('no page target');
}
const target = await pageTarget();
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
let nextId = 0; const pending = new Map();
function send(method, params = {}) {
  const id = ++nextId; ws.send(JSON.stringify({ id, method, params }));
  return new Promise((res, rej) => pending.set(id, { res, rej, method }));
}
const threeBody = fs.readFileSync(threePath).toString('base64');
fs.mkdirSync(rawCache, { recursive: true });
function rawBytes(url) {
  const f = path.join(rawCache, crypto.createHash('sha256').update(url).digest('hex'));
  if (!fs.existsSync(f)) execFileSync('curl', ['-sS', '--fail', '-o', f, url], { timeout: 60000 });
  return fs.readFileSync(f);
}
ws.onmessage = async (m) => {
  const d = JSON.parse(m.data);
  if (d.id && pending.has(d.id)) { const p = pending.get(d.id); pending.delete(d.id); d.error ? p.rej(new Error(p.method + ': ' + JSON.stringify(d.error))) : p.res(d.result); return; }
  if (d.method === 'Fetch.requestPaused') {
    const { requestId, request } = d.params; const u = request.url;
    try {
      if (u === THREE_URL && sc.three === 'serve') {
        log('fetch: FULFILL three.js');
        await send('Fetch.fulfillRequest', { requestId, responseCode: 200, responseHeaders: [
          { name: 'Content-Type', value: 'application/javascript; charset=utf-8' }, { name: 'Access-Control-Allow-Origin', value: '*' }], body: threeBody });
      } else if (u.startsWith('https://raw.githubusercontent.com/')) {
        const b = rawBytes(u);
        log('fetch: FULFILL raw', b.length, u);
        await send('Fetch.fulfillRequest', { requestId, responseCode: 200, responseHeaders: [
          { name: 'Content-Type', value: 'text/plain; charset=utf-8' }, { name: 'Access-Control-Allow-Origin', value: '*' }], body: b.toString('base64') });
      } else {
        log('fetch: FAIL BlockedByClient', u);
        await send('Fetch.failRequest', { requestId, errorReason: 'BlockedByClient' });
      }
    } catch (e) { log('fetch: ERROR', u, e.message); try { await send('Fetch.failRequest', { requestId, errorReason: 'Failed' }); } catch {} }
  } else if (d.method === 'Runtime.bindingCalled' && d.params.name === '__c220rep') {
    const r = JSON.parse(d.params.payload);
    log(`dom(${ts(r.t)}): [${r.kind}] ${JSON.stringify(r.text)}${r.note ? ' ' + r.note : ''}`);
  } else if (d.method === 'Runtime.exceptionThrown') {
    log('page exception:', JSON.stringify(d.params.exceptionDetails.exception?.description || d.params.exceptionDetails.text).slice(0, 300));
  } else if (d.method === 'Runtime.consoleAPICalled') {
    log('console.' + d.params.type + ':', JSON.stringify(d.params.args.map(a => a.value ?? a.description)).slice(0, 300));
  } else if (d.method === 'Page.frameNavigated' && !d.params.frame.parentId) {
    log('frameNavigated', d.params.frame.url);
  } else if (d.method === 'Page.loadEventFired') {
    log('loadEventFired');
  }
};

const OBSERVER = `(() => {
  const last = new WeakMap();
  const rep = (kind, text, note) => { try { globalThis.__c220rep(JSON.stringify({ t: Date.now(), kind, text, note })); } catch {} };
  const label = el => el.closest('.geometry-host') ? 'WIDGET status' : ('outer status ' + (el.id || el.className || el.tagName));
  new MutationObserver(records => {
    const touched = new Set();
    for (const r of records) {
      const n = r.target.nodeType === 1 ? r.target : r.target.parentElement;
      const s = n && n.closest && n.closest('[role=status]');
      if (s) touched.add(s);
      for (const a of r.addedNodes) if (a.nodeType === 1) { if (a.matches('[role=status]')) touched.add(a); a.querySelectorAll && a.querySelectorAll('[role=status]').forEach(x => touched.add(x)); }
    }
    for (const s of touched) { const t = s.textContent; rep(label(s), t, last.get(s) === t ? '(same text rewritten)' : ''); last.set(s, t); }
  }).observe(document, { subtree: true, childList: true, characterData: true });
})();`;

await send('Page.enable');
await send('Runtime.enable');
await send('Runtime.addBinding', { name: '__c220rep' });
await send('Page.addScriptToEvaluateOnNewDocument', { source: OBSERVER });
await send('Fetch.enable', { patterns: [{ urlPattern: 'https://*', requestStage: 'Request' }] });
log(`scenario ${scenario} url=${sc.url} three=${sc.three}`);

async function evalJS(expr) {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
  return r.result?.value ?? r.exceptionDetails?.text;
}
const STATE = `JSON.stringify({active: document.activeElement && (document.activeElement.tagName + ':' + (document.activeElement.textContent||'').trim().slice(0,40)),
  buttons: [...document.querySelectorAll('.geometry-host button')].map(b => b.textContent + (b.hasAttribute('aria-pressed') ? '[' + b.getAttribute('aria-pressed') + ']' : '')),
  status: [...document.querySelectorAll('.geometry-host [role=status]')].map(s => s.textContent),
  canvas: !!document.querySelector('.geometry-host canvas')})`;

let stepNo = 0;
const mark = (label) => log(`STEP ${stepNo}: ${label}`);
mark('before navigation (start page)');
await sleep(9000);
for (const [kind, arg, secs] of sc.steps) {
  stepNo++;
  if (kind === 'settle') {
    mark(`navigate + ${arg}`);
    await send('Page.navigate', { url: sc.url });
  } else if (kind === 'focus') {
    const r = await evalJS(`(() => { const b = [...document.querySelectorAll('.geometry-host button')].find(b => b.textContent === ${JSON.stringify(arg)}); if (!b) return 'not found'; const was = document.activeElement === b; b.focus(); return was ? 'already focused' : 'focused'; })()`);
    mark(`focus "${arg}" via Runtime.evaluate -> ${r}`);
  } else if (kind === 'enter') {
    const pre = await evalJS(`document.activeElement && document.activeElement.textContent`);
    mark(`key Enter via Input.dispatchKeyEvent (active element "${pre}")`);
    await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13, nativeVirtualKeyCode: 13, text: '\r', unmodifiedText: '\r' });
    await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13, nativeVirtualKeyCode: 13 });
  } else if (kind === 'lose') {
    const r = await evalJS(`(() => { const c = document.querySelector('.geometry-host canvas'); if (!c) return 'no canvas'; const gl = c.getContext('webgl2'); const g = gl || c.getContext('webgl'); const ext = g && g.getExtension('WEBGL_lose_context'); if (!ext) return 'no extension'; ext.loseContext(); return 'loseContext() called on ' + (gl ? 'webgl2' : 'webgl'); })()`);
    mark(`WEBGL_lose_context.loseContext() via Runtime.evaluate -> ${r}`);
  } else if (kind === 'drain') {
    mark('final drain');
  }
  await sleep(secs * 1000);
  log(`state after step ${stepNo}: ${await evalJS(STATE)}`);
}
log('END');
logFile.end();
setTimeout(() => process.exit(0), 300);
