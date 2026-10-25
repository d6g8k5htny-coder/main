// C220-02 CDP driver. Launches nothing: connects to an already running Chrome
// (--remote-debugging-port), intercepts cdn.jsdelivr.net (fulfill three.js or fail it),
// navigates, performs keyboard-style steps and records the widget status after each step.
// usage: node cdp_driver.mjs SCENARIO_ID URL THREE_MODE(fulfill|fail) OUT_DIR [PORT=9333]
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';

const [scenario, url, threeMode, outDir, portArg] = process.argv.slice(2);
const port = Number(portArg || 9333);
const THREE_URL = 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.js';
const THREE_PATH = '/tmp/claude-0/-home-user-main/d5c985ca-97e5-58d8-a9d4-1e0a19065a22/scratchpad/three/package/build/three.js';
const AUDIO_RAW = path.join(outDir, 'audio.raw');
const threeBody = fs.readFileSync(THREE_PATH).toString('base64');
// Chromium here does not trust the agent proxy's TLS CA, so pinned raw.githubusercontent.com
// sources are relayed: curl (which trusts the proxy CA) fetches the bytes once into a shared cache
// and CDP Fetch.fulfillRequest serves them. The page still verifies every byte count and SHA-256.
const RAW_CACHE = path.join(path.dirname(outDir), 'raw_cache');
fs.mkdirSync(RAW_CACHE, { recursive: true });
function rawRelay(u) {
  const key = crypto.createHash('sha256').update(u).digest('hex');
  const body = path.join(RAW_CACHE, key + '.body'), meta = path.join(RAW_CACHE, key + '.json');
  if (!fs.existsSync(meta)) {
    const code = execFileSync('curl', ['-sS', '-o', body, '-w', '%{http_code}', u], { encoding: 'utf8' }).trim();
    const bytes = fs.readFileSync(body);
    fs.writeFileSync(meta, JSON.stringify({ url: u, status: Number(code), bytes: bytes.length, sha256: crypto.createHash('sha256').update(bytes).digest('hex'), fetched: new Date().toISOString() }));
  }
  const m = JSON.parse(fs.readFileSync(meta, 'utf8'));
  return { ...m, body: fs.readFileSync(body).toString('base64') };
}
const events = [];   // CDP-side event log (network, console, fetch interception)
const steps = [];    // driver steps with DOM snapshots
const iso = ms => new Date(ms).toISOString();
const log = (kind, data) => { const t = Date.now(); events.push({ t, iso: iso(t), kind, ...data }); };
const sleep = ms => new Promise(r => setTimeout(r, ms));

// Step plans. Each key step = focus (Runtime.evaluate .focus()) then a key (Input.dispatchKeyEvent).
const PLANS = {
  S1: { kind: 'ec014', keys: [] },
  S2: { kind: 'ec014', keys: [] },
  S3: { kind: 'ec014', keys: [
    { label: 'Show 2D diagram', key: 'Enter', expect: 'button' },
    { label: 'Show 3D view', key: 'Enter', expect: 'button' } ] },
  S4: { kind: 'p15', keys: [
    { label: 'Toggle x1', key: 'Enter' },
    { label: 'Toggle x1', key: 'Enter' },
    { label: 'Swap palette colors', key: 'Enter' } ] },
  S5: { kind: 'ec014', keys: [], loseContext: true }
};
const plan = PLANS[scenario.replace(/[FX]$/, '')];
// C220_KEYMODE=xtest (variant runs only): send the key as a real X event via AT-SPI/XTest instead of CDP.
const KEYMODE = process.env.C220_KEYMODE === 'xtest' ? 'xtest' : 'cdp';
if (!plan) throw new Error(`unknown scenario ${scenario}`);

async function pageTarget() {
  for (let i = 0; i < 80; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
      const page = list.find(t => t.type === 'page');
      if (page) return page;
    } catch {}
    await sleep(250);
  }
  throw new Error('no page target on DevTools port');
}

const target = await pageTarget();
const ws = new WebSocket(target.webSocketDebuggerUrl);
let nextId = 0; const pending = new Map(); const listeners = [];
function send(method, params = {}) {
  const id = ++nextId;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject, method }));
}
function once(method, timeoutMs) {
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(null), timeoutMs);
    listeners.push({ method, fn: p => { clearTimeout(timer); resolve(p); } });
  });
}
ws.onmessage = m => {
  const d = JSON.parse(m.data);
  if (d.id && pending.has(d.id)) {
    const p = pending.get(d.id); pending.delete(d.id);
    d.error ? p.reject(new Error(`${p.method}: ${d.error.message}`)) : p.resolve(d.result);
    return;
  }
  if (!d.method) return;
  for (let i = listeners.length - 1; i >= 0; i--) if (listeners[i].method === d.method) { listeners[i].fn(d.params); listeners.splice(i, 1); }
  const p = d.params || {};
  if (d.method === 'Fetch.requestPaused') {
    const u = p.request.url;
    if (threeMode === 'fulfill' && u === THREE_URL) {
      log('fetch.fulfill', { url: u });
      send('Fetch.fulfillRequest', { requestId: p.requestId, responseCode: 200, responseHeaders: [
        { name: 'Content-Type', value: 'application/javascript' }, { name: 'Access-Control-Allow-Origin', value: '*' } ], body: threeBody }).catch(e => log('error', { msg: String(e) }));
    } else if (u.startsWith('https://raw.githubusercontent.com/')) {
      let r; try { r = rawRelay(u); } catch (e) { log('fetch.relay-error', { url: u, msg: String(e) }); send('Fetch.failRequest', { requestId: p.requestId, errorReason: 'Failed' }); return; }
      log('fetch.relay', { url: u, status: r.status, bytes: r.bytes, sha256: r.sha256 });
      send('Fetch.fulfillRequest', { requestId: p.requestId, responseCode: r.status, responseHeaders: [
        { name: 'Content-Type', value: 'text/plain; charset=utf-8' }, { name: 'Access-Control-Allow-Origin', value: '*' } ], body: r.body }).catch(e => log('error', { msg: String(e) }));
    } else {
      log('fetch.fail', { url: u });
      send('Fetch.failRequest', { requestId: p.requestId, errorReason: 'BlockedByClient' }).catch(e => log('error', { msg: String(e) }));
    }
  } else if (d.method === 'Network.responseReceived') {
    const r = p.response; if (!/^data:/.test(r.url)) log('net.response', { url: r.url, status: r.status });
  } else if (d.method === 'Network.loadingFailed') {
    log('net.failed', { requestId: p.requestId, errorText: p.errorText, blockedReason: p.blockedReason, corsErrorStatus: p.corsErrorStatus });
  } else if (d.method === 'Network.requestWillBeSent') {
    if (!/^data:/.test(p.request.url)) log('net.request', { requestId: p.requestId, url: p.request.url });
  } else if (d.method === 'Runtime.consoleAPICalled') {
    log('console', { type: p.type, args: p.args.map(a => a.value ?? a.description) });
  } else if (d.method === 'Runtime.exceptionThrown') {
    log('exception', { text: p.exceptionDetails.text, desc: p.exceptionDetails.exception?.description });
  } else if (d.method === 'Log.entryAdded') {
    log('log', { level: p.entry.level, text: p.entry.text, url: p.entry.url });
  } else if (d.method === 'Page.loadEventFired' || d.method === 'Page.domContentEventFired') {
    log(d.method, {});
  }
};
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

// Passive in-page recorder: logs every textContent change of any live/status element.
const RECORDER = `(() => {
  const rec = window.__c220 = [];
  const last = new Map();
  const name = el => {
    if (el.id) return '#' + el.id;
    const host = el.closest('.geometry-host');
    if (host) return 'widget:' + (host.getAttribute('data-illustration') || '?');
    return (el.className ? '.' + String(el.className).split(' ')[0] : el.tagName.toLowerCase());
  };
  const scan = () => {
    for (const el of document.querySelectorAll('[role=status],[aria-live],[role=alert],[role=log]')) {
      const text = el.textContent;
      if (last.get(el) !== text) { rec.push({ t: Date.now(), el: name(el), text, prev: last.has(el) ? last.get(el) : null }); last.set(el, text); }
    }
  };
  new MutationObserver(scan).observe(document, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ['hidden'] });
})();`;

await send('Network.enable');
await send('Page.enable');
await send('Runtime.enable');
await send('Log.enable');
await send('Fetch.enable', { patterns: [{ urlPattern: 'https://cdn.jsdelivr.net/*', requestStage: 'Request' }, { urlPattern: 'https://raw.githubusercontent.com/*', requestStage: 'Request' }] });
await send('Page.addScriptToEvaluateOnNewDocument', { source: RECORDER });

async function evaluate(expr) {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result.value;
}
const SNAP = `(() => {
  const host = document.querySelector('.geometry-host');
  const st = host?.querySelector('[role=status]');
  const ae = document.activeElement;
  const btns = host ? [...host.querySelectorAll('button')].map(b => ({ text: b.textContent, pressed: b.getAttribute('aria-pressed'), hiddenAncestor: !!b.closest('[hidden]') })) : [];
  const stage = host?.querySelector('.geometry-3d-stage');
  const fig = host?.querySelector('figure');
  return {
    t: Date.now(),
    widget_status: st ? st.textContent : null,
    widget_status_attrs: st ? { role: st.getAttribute('role'), live: st.getAttribute('aria-live') } : null,
    stage_hidden: stage ? stage.hidden : null, figure_hidden: fig ? fig.hidden : null,
    canvas: !!host?.querySelector('canvas'),
    buttons: btns,
    active: ae ? (ae.tagName.toLowerCase() + (ae.id ? '#' + ae.id : '') + (ae.textContent && ae.tagName === 'BUTTON' ? ' "' + ae.textContent + '"' : '')) : null,
    hasFocus: document.hasFocus(),
    outer: { museum_state: document.getElementById('museum-state')?.textContent ?? null,
             conditional_route_status: document.getElementById('conditional-route-status')?.textContent ?? null },
    webgl_globals: { WebGLRenderingContext: typeof globalThis.WebGLRenderingContext, WebGL2RenderingContext: typeof globalThis.WebGL2RenderingContext },
    three_revision: globalThis.THREE?.REVISION ?? null
  };
})()`;
async function record(action, extra = {}) {
  const snap = await evaluate(SNAP);
  const s = { step: steps.length, action, t: snap.t, iso: iso(snap.t), ...extra, snapshot: snap };
  steps.push(s);
  console.log(`[${s.iso}] step ${s.step} ${action} :: status=${JSON.stringify(snap.widget_status)} active=${snap.active}`);
  return snap;
}
function audioSize() { try { return fs.statSync(AUDIO_RAW).size; } catch { return -1; } }
// Wait at least minMs, then until the paced audio sink has been idle for idleMs (max maxMs).
async function settle(minMs, idleMs = 1500, maxMs = 20000) {
  const start = Date.now(); await sleep(minMs);
  let size = audioSize(), stableSince = Date.now();
  while (Date.now() - start < maxMs) {
    await sleep(200);
    const now = audioSize();
    if (now !== size) { size = now; stableSince = Date.now(); }
    else if (Date.now() - stableSince >= idleMs) break;
  }
  return { waited_ms: Date.now() - start, audio_idle: Date.now() - stableSince >= idleMs };
}

// ---- run ----
const w0 = await settle(2000, 1000, 10000);
await record('before navigation (neutral start page)', { wait: w0 });
const tNav = Date.now();
log('navigate', { url });
const loadP = once('Page.loadEventFired', 30000);
await send('Page.navigate', { url });
const loaded = await loadP;
log('load-wait', { loaded: !!loaded });
// Wait for the widget to reach its first-load end state.
const deadline = Date.now() + 30000; let ready = null;
while (Date.now() < deadline) {
  const s = await evaluate(SNAP).catch(() => null);
  if (s && s.widget_status !== null) {
    if (plan.kind === 'p15' && s.widget_status) { ready = s; break; }
    if (plan.kind === 'ec014' && !/Checking optional 3D view/.test(s.widget_status)) { ready = s; break; }
  }
  await sleep(100);
}
log('widget-ready', { ready: !!ready, ms_after_navigate: ready ? ready.t - tNav : null });
// Orca's default sayAllOnLoad plus queued live-region messages keep speech busy; allow the first-load
// backlog to drain (or 100 s) before interacting.
const firstWait = await settle(20000, 2500, 100000);
await record('first load settled', { navigate_t: tNav, navigate_iso: iso(tNav), widget_ready_t: ready?.t ?? null, widget_ready_iso: ready ? iso(ready.t) : null, wait: firstWait });

for (const k of plan.keys) {
  const found = await evaluate(`(() => { const b = [...document.querySelectorAll('.geometry-host button')].find(b => b.textContent === ${JSON.stringify(k.label)}); if (!b) return false; b.focus(); return document.activeElement === b; })()`);
  const tf = Date.now();
  const w1 = await settle(3000, 1200, 12000);
  await record(`focus "${k.label}" via Runtime.evaluate`, { focused: found, action_t: tf, action_iso: iso(tf), wait: w1 });
  const keyDef = k.key === 'Enter'
    ? { key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13, nativeVirtualKeyCode: 13, text: '\r', unmodifiedText: '\r' }
    : { key: ' ', code: 'Space', windowsVirtualKeyCode: 32, nativeVirtualKeyCode: 32, text: ' ', unmodifiedText: ' ' };
  const tk = Date.now(); let how = 'Input.dispatchKeyEvent';
  if (KEYMODE === 'xtest') {
    how = 'AT-SPI generate_keyboard_event (XTest): ' + execFileSync('/usr/bin/python3.12', ['-I', path.join(path.dirname(new URL(import.meta.url).pathname), 'xtest_key.py'), k.key === 'Enter' ? 'Return' : 'space'], { encoding: 'utf8' }).trim();
  } else {
    await send('Input.dispatchKeyEvent', { type: 'keyDown', ...keyDef });
    await send('Input.dispatchKeyEvent', { type: 'keyUp', key: keyDef.key, code: keyDef.code, windowsVirtualKeyCode: keyDef.windowsVirtualKeyCode, nativeVirtualKeyCode: keyDef.nativeVirtualKeyCode });
  }
  const w2 = await settle(4000, 1500, 20000);
  await record(`key ${k.key} on "${k.label}" via ${KEYMODE === 'xtest' ? 'XTest' : 'Input.dispatchKeyEvent'}`, { action_t: tk, action_iso: iso(tk), how, wait: w2 });
}

if (plan.loseContext) {
  const tl = Date.now();
  const res = await evaluate(`(() => { const c = document.querySelector('.geometry-host canvas'); if (!c) return 'no canvas';
    const gl = c.getContext('webgl2') || c.getContext('webgl'); if (!gl) return 'no context';
    const ext = gl.getExtension('WEBGL_lose_context'); if (!ext) return 'no WEBGL_lose_context';
    ext.loseContext(); return 'loseContext() called on ' + (gl instanceof WebGL2RenderingContext ? 'webgl2' : 'webgl'); })()`);
  const w = await settle(4000, 1500, 20000);
  await record('WEBGL_lose_context.loseContext() via Runtime.evaluate', { result: res, action_t: tl, action_iso: iso(tl), wait: w });
}

const wEnd = await settle(4000, 2500, 40000);
await record('final drain', { wait: wEnd });
const domLog = await evaluate('window.__c220 || []');
fs.writeFileSync(path.join(outDir, 'driver.json'), JSON.stringify({ scenario, url, threeMode, port, steps, dom_status_changes: domLog, cdp_events: events }, null, 2));
console.log('driver done');
ws.close();
process.exit(0);
