import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { runInNewContext } from "node:vm";
import { compareSnapshots, selectionDescription } from "../docs/site/compare-model.mjs";
import { sourceURL } from "../docs/site/current-model.mjs";

// Run the real controller and comparison model. Only publication acquisition and
// the DOM boundary are doubled; browser layout and native AT need browser checks.
class Element extends EventTarget {
  constructor(tag, document) {
    super();
    this.tagName = tag.toUpperCase();
    this.ownerDocument = document;
    this.children = [];
    this.attributes = new Map();
    this.dataset = {};
    this.hidden = false;
    this.disabled = false;
    this._text = "";
    this._value = "";
  }
  set textContent(value) { this.replaceChildren(); this._text = String(value); }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
  set id(value) { this.setAttribute("id", value); }
  get id() { return this.getAttribute("id") || ""; }
  set href(value) { this.setAttribute("href", value); }
  get href() { return this.getAttribute("href") || ""; }
  setAttribute(key, value) { this.attributes.set(key, String(value)); }
  getAttribute(key) { return this.attributes.get(key) ?? null; }
  removeAttribute(key) { this.attributes.delete(key); }
  append(...children) {
    for (const child of children) {
      child.parentElement = this;
      this.children.push(child);
    }
  }
  replaceChildren(...children) {
    for (const child of this.children) child.parentElement = null;
    this.children = [];
    this._text = "";
    this.append(...children);
  }
  get options() { return this.children; }
  get value() {
    if (this.tagName !== "SELECT") return this._value;
    return (this.children.find(child => child.value === this._value) || this.children[0])?.value || "";
  }
  set value(value) { this._value = String(value); }
  focus() { this.ownerDocument.activeElement = this; }
}
const descendants = node => node.children.flatMap(child => [child, ...descendants(child)]);
const repository = "d6g8k5htny-coder/main";
function fixtures() {
  const source = index => ({ repository, path: `proofs/${String(index).padStart(2, "0")}.md`, commit: "a".repeat(40), blob: "b".repeat(40), bytes: 100 });
  const before = {
    schema_version: 1,
    graph: { data: { nodes: { claim: { label: "Before" } }, edges: [{ from: "a", to: "b" }, { from: "a", to: "b" }] } },
    sources: Array.from({ length: 23 }, (_, index) => source(index)),
    equations: [{ ...source(0), id: "excerpt", text: "before excerpt" }],
    references: [{ ...source(0), label: "before reference" }],
  };
  const after = structuredClone(before);
  after.graph.data.nodes.claim.label = "After";
  after.graph.data.edges.pop();
  for (const record of after.sources) record.commit = "c".repeat(40);
  after.sources[0].blob = "d".repeat(40);
  after.sources[1].bytes = 101;
  after.equations[0].text = "after excerpt";
  after.references[0].label = "after reference";
  return { before, after };
}
async function controller() {
  const document = { activeElement: null, createElement: tag => new Element(tag, document) };
  const root = document.createElement("main");
  const html = await readFile(new URL("../docs/site/compare.html", import.meta.url), "utf8");
  for (const match of html.matchAll(/<([a-z][\w-]*)\b[^>]*\bid="([^"]+)"[^>]*>/g)) {
    const node = document.createElement(match[1]);
    node.id = match[2];
    root.append(node);
  }
  const get = id => descendants(root).find(node => node.id === id) || null;
  document.getElementById = get;
  document.querySelector = () => ({ dataset: { currentDigest: "e".repeat(64), currentBytes: "1" } });
  const choices = ["1", "2"].map((digit, index) => ({
    key: `archive:${digit.repeat(64)}`, id: digit.repeat(64), current: false, version: 1,
    observed_at: `2026-10-${index + 10}T00:00:00Z`,
    descriptor: { path: `snapshots/${digit.repeat(64)}/current.json`, sha256: digit.repeat(64), bytes: 1 },
  }));
  const { before, after } = fixtures();
  const original = structuredClone({ before, after });
  const location = new URL("https://example.test/site/compare.html?reader=keep#results-heading");
  const window = new EventTarget();
  const writes = [];
  const history = { pushState(state, title, url) { location.href = new URL(url, location.href).href; writes.push(location.href); } };
  let failure = false, blocked = null;
  const source = await readFile(new URL("../docs/site/compare.mjs", import.meta.url), "utf8");
  // Strip module declarations only; execute the unchanged production lifecycle.
  const program = source.replace(/^import\s+[\s\S]*?from\s+"[^"]+";\n/gm, "");
  await runInNewContext(program, {
    document, window, location, history, URL, compareSnapshots, selectionDescription, sourceURL,
    loadPublication: async () => ({ choices }),
    loadSelection: async choice => {
      if (blocked) await blocked;
      if (failure) throw Error("Fixture payload digest mismatch.");
      return choice === choices[0] ? before : after;
    },
  });
  assert.equal(get("compare-state").dataset.state, "completed", get("compare-state").textContent);
  const settle = async () => { await new Promise(resolve => setImmediate(resolve)); };
  const event = async (id, type) => {
    const node = get(id);
    assert.ok(node, `Missing control ${id}`);
    node.dispatchEvent(new Event(type, { cancelable: true }));
    await settle();
  };
  const choose = async key => {
    const select = get("compare-category");
    assert.ok(select, "Verified comparisons need a category selector");
    select.focus();
    select.value = key;
    await event("compare-category", "change");
    return select;
  };
  return {
    get, choose, event, window, location, writes, document, before, after, original,
    rows: () => get("comparison-results").children.filter(node => node.tagName === "ARTICLE"),
    fail() { failure = true; },
    block() { let release; blocked = new Promise(resolve => { release = resolve; }); return release; },
    settle,
  };
}
const rowIDs = fixture => fixture.rows().map(row => row.children[1].textContent);

test("verified category counts expose content and references directly without changing the source selection", async () => {
  const f = await controller();
  const select = f.get("compare-category");
  assert.ok(select, "Verified comparisons need a category selector");
  const label = descendants(select.parentElement).find(node => node.tagName === "LABEL");
  assert.ok(label?.getAttribute("for") === select.id || select.parentElement.tagName === "LABEL", "Native category control has an associated label");
  assert.equal(select.value, "all");
  assert.deepEqual(select.options.map(option => option.value), ["all", "nodes", "edges", "pins", "content", "sourceRecords", "addedSources", "removedSources", "excerpts", "references"]);
  assert.deepEqual(select.options.map(option => Number(option.textContent.match(/\((\d+)\)$/)?.[1])), [52, 1, 1, 23, 2, 23, 0, 0, 1, 1]);
  const selection = [f.get("from").value, f.get("to").value, f.get("share").href, f.location.href];
  await f.choose("content");
  assert.deepEqual(rowIDs(f), [`${repository} · proofs/00.md`, `${repository} · proofs/01.md`]);
  assert.equal(f.document.activeElement, select);
  assert.equal(f.rows()[0].children[2].children[1].textContent, JSON.stringify(f.before.sources[0], null, 2));
  assert.equal(f.rows()[0].children[3].children[1].textContent, JSON.stringify(f.after.sources[0], null, 2));
  assert.equal(f.rows()[0].children[2].children[2].href, `https://github.com/${repository}/blob/${"a".repeat(40)}/proofs/00.md`);
  await f.choose("references");
  assert.deepEqual(rowIDs(f), [`${repository} · proofs/00.md`]);
  assert.match(f.rows()[0].textContent, /after reference/);
  assert.deepEqual([f.get("from").value, f.get("to").value, f.get("share").href, f.location.href], selection);
  assert.deepEqual(f.writes, []);
  assert.equal(f.get("compare-state").dataset.state, "completed");
  assert.equal(f.get("compare-category-status").getAttribute("role"), "status");
  assert.match(f.get("compare-category-status").textContent, /1[–-]1 of 1/);
  assert.deepEqual({ before: f.before, after: f.after }, f.original);
});

test("changing categories resets both pagers and retains native selector focus", async () => {
  const f = await controller();
  await f.choose("pins");
  await f.event("next-top", "click");
  assert.deepEqual(rowIDs(f), [`${repository} · proofs/20.md`, `${repository} · proofs/21.md`, `${repository} · proofs/22.md`]);
  assert.match(f.get("page-status").textContent, /21[–-]23 of 23/);
  const select = await f.choose("content");
  for (const suffix of ["", "-top"]) {
    assert.match(f.get("page-status" + suffix).textContent, /1[–-]2 of 2/);
    assert.equal(f.get("previous" + suffix).disabled, true);
    assert.equal(f.get("next" + suffix).disabled, true);
  }
  assert.equal(f.document.activeElement, select);
});

test("All preserves ordered overlapping categories and complete edge multiplicity", async () => {
  const f = await controller();
  await f.choose("edges");
  const edge = f.rows()[0];
  assert.equal(edge.children[2].children[1].textContent, "2");
  assert.equal(edge.children[3].children[1].textContent, "1");
  await f.choose("all");
  const all = [];
  while (true) {
    all.push(...f.rows().map(row => ({ kind: row.children[0].textContent, id: row.children[1].textContent })));
    if (f.get("next").disabled) break;
    await f.event("next", "click");
  }
  assert.equal(all.length, 52);
  assert.deepEqual(all.map(row => row.kind).filter((kind, index, kinds) => kind !== kinds[index - 1]), ["Node records", "Complete edge records (multiplicity)", "Source commit pins", "Source blob / byte identities", "Other / all shared source record fields", "Included excerpt records", "Reference identity records"]);
  assert.equal(all.filter(row => row.id === `${repository} · proofs/00.md`).length, 4, "The same identity remains present in pin, content, shared-record and reference categories");
  assert.deepEqual(all.slice(2, 5).map(row => row.id), [`${repository} · proofs/00.md`, `${repository} · proofs/01.md`, `${repository} · proofs/02.md`]);
});

test("an empty category reports zero results with both pagers hidden", async () => {
  const f = await controller();
  await f.choose("addedSources");
  assert.equal(f.rows().length, 0);
  assert.match(f.get("comparison-results").textContent, /No record differences/);
  assert.match(f.get("compare-category-status").textContent, /0[–-]0 of 0/);
  for (const suffix of ["", "-top"]) {
    assert.equal(f.get("pagination" + suffix).hidden, true);
    assert.equal(f.get("previous" + suffix).disabled, true);
    assert.equal(f.get("next" + suffix).disabled, true);
  }
});

test("selection changes clear category state and ignore an older in-flight comparison", async () => {
  const f = await controller();
  await f.choose("references");
  const release = f.block();
  await f.event("compare-form", "submit");
  assert.equal(f.get("compare-category"), null);
  f.get("from").value = f.get("to").value;
  await f.event("from", "change");
  release();
  await f.settle();
  assert.equal(f.get("compare-category"), null);
  assert.equal(f.get("compare-category-status"), null);
  assert.equal(f.rows().length, 0);
  assert.equal(f.get("share").getAttribute("href"), null);
  assert.equal(f.get("compare-state").dataset.state, "awaiting");
  await f.event("compare-form", "submit");
  assert.equal(f.get("compare-category").value, "all");
  assert.match(f.get("compare-category-status").textContent, /0[–-]0 of 0/);
  const share = new URL(f.get("share").href);
  assert.equal(share.searchParams.get("reader"), "keep");
  assert.equal(share.hash, "#results-heading");
  assert.equal(share.searchParams.get("from"), f.get("from").value);
  assert.equal(share.searchParams.get("to"), f.get("to").value);
});

test("payload refusal and malformed history discard category results and announcements", async () => {
  const f = await controller();
  await f.choose("content");
  f.fail();
  await f.event("compare-form", "submit");
  assert.equal(f.get("compare-state").dataset.state, "refused");
  assert.match(f.get("compare-state").textContent, /digest mismatch/);
  assert.equal(f.get("compare-category"), null);
  assert.equal(f.get("compare-category-status"), null);
  assert.equal(f.rows().length, 0);
  assert.equal(f.get("share").getAttribute("href"), null);
  const g = await controller();
  await g.choose("references");
  g.location.search = "?from=bad&from=bad&to=bad";
  g.window.dispatchEvent(new Event("popstate"));
  await g.settle();
  assert.equal(g.get("compare-state").dataset.state, "refused");
  assert.equal(g.get("compare-category"), null);
  assert.equal(g.rows().length, 0);
});
