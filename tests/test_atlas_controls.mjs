import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {runInNewContext} from 'node:vm';

// Execute the production helper with an explicit clock and browser lifecycle
// boundary. Actual native option behavior and speech have separate replays.
const source = (await readFile(new URL('../docs/site/atlas-controls.mjs', import.meta.url), 'utf8')).replace(/^export /gm, '');
function fixture() {
  const document = new EventTarget(), window = new EventTarget();
  document.hidden = false;
  let clock = 0, next = 0;
  const timers = new Map(), output = {textContent: ''};
  const request = runInNewContext(source + '\ncreateEquationAnnouncement(output);', {
    document, window, output,
    setTimeout(fn, ms) { const id = ++next; timers.set(id, {at: clock + ms, fn}); return id; },
    clearTimeout(id) { timers.delete(id); }
  });
  return {document, window, output, request, tick(ms) {
    clock += ms;
    for (const [id, timer] of timers) if (timer.at <= clock) { timers.delete(id); timer.fn(); }
  }};
}
test('only the current query is announced after 500ms of quiet, even if its count repeats', () => {
  const f = fixture();
  f.request('108 excerpts'); f.tick(450); f.request('23 excerpts'); f.tick(450);
  assert.equal(f.output.textContent, '');
  f.request('23 excerpts'); f.tick(499); assert.equal(f.output.textContent, '');
  f.tick(1); assert.equal(f.output.textContent, '23 excerpts');
  f.request('0 excerpts'); f.tick(100); f.request('108 excerpts');
  f.tick(500); assert.equal(f.output.textContent, '108 excerpts');
});
test('hidden documents and page-cache transitions cancel pending speech and resume current results', () => {
  const f = fixture();
  f.request('23 excerpts'); f.tick(100);
  f.document.hidden = true; f.document.dispatchEvent(new Event('visibilitychange'));
  f.tick(1000); assert.equal(f.output.textContent, '');
  f.request('0 excerpts'); f.document.hidden = false;
  f.document.dispatchEvent(new Event('visibilitychange')); f.tick(500);
  assert.equal(f.output.textContent, '0 excerpts');
  f.request('108 excerpts'); f.window.dispatchEvent(new Event('pagehide'));
  f.tick(1000); assert.equal(f.output.textContent, '');
  f.window.dispatchEvent(new Event('pageshow')); f.tick(500);
  assert.equal(f.output.textContent, '108 excerpts');
});
