import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { webcrypto } from "node:crypto";
const moduleURL = new URL("../docs/site/compare-model.mjs", import.meta.url);
const load = () => import(moduleURL);
const site = new URL("../docs/site/", import.meta.url);
const registry = JSON.parse(
  await readFile(new URL("snapshots/index.json", site)),
);
const archives = registry.archives.toSorted((a, b) =>
  a.observed_at.localeCompare(b.observed_at),
);
const raw = await Promise.all(
  archives.map((a) =>
    readFile(new URL(`snapshots/${a.id}/current.json`, site)),
  ),
);
const data = raw.map((x) => JSON.parse(x));
test("comparison model is implemented", async () =>
  assert.equal(typeof (await load()).compareSnapshots, "function"));
test("literal fields preserve array order and number versus boolean", async () => {
  const { canonicalJSON } = await load();
  assert.notEqual(canonicalJSON([1, true]), canonicalJSON([true, 1]));
  assert.equal(canonicalJSON({ b: 2, a: 1 }), canonicalJSON({ a: 1, b: 2 }));
});
test("real retained archives distinguish pin moves from blob/byte moves", async () => {
  const m = await load();
  const admitted = await Promise.all(
    raw.map((r, i) =>
      m.parsePayload(
        r,
        archives[i].files.find((f) => f.path === "current.json"),
        1,
        webcrypto,
      ),
    ),
  );
  const d = m.compareSnapshots(...admitted);
  const shared = data[0].sources.filter((a) =>
    data[1].sources.some(
      (b) => a.repository === b.repository && a.path === b.path,
    ),
  );
  assert.equal(
    d.pins.length,
    shared.filter(
      (a) =>
        data[1].sources.find(
          (b) => a.repository === b.repository && a.path === b.path,
        ).commit !== a.commit,
    ).length,
  );
  assert.equal(
    d.content.length,
    shared.filter((a) => {
      const b = data[1].sources.find(
        (b) => a.repository === b.repository && a.path === b.path,
      );
      return a.blob !== b.blob || a.bytes !== b.bytes;
    }).length,
  );
  assert.equal(d.nodes.length, 0);
  assert.equal(d.edges.length, 0);
});
test("payload integrity refuses byte and digest mismatch", async () => {
  const m = await load();
  const d = archives[0].files.find((f) => f.path === "current.json");
  await assert.rejects(
    m.parsePayload(raw[0], { ...d, bytes: d.bytes + 1 }, 1, webcrypto),
    /byte/i,
  );
  await assert.rejects(
    m.parsePayload(raw[0], { ...d, sha256: "0".repeat(64) }, 1, webcrypto),
    /digest/i,
  );
});
test("unsafe public paths and unsupported versions refuse", async () => {
  const m = await load();
  for (const p of [
    "../x",
    "/x",
    "https://evil/x",
    "x%2fy",
    "x?y",
    "x#y",
    "x\\y",
  ])
    assert.throws(() => m.publicPath(p));
  assert.throws(() => m.validatePayload({ ...data[0], schema_version: 9 }, 9));
  assert.throws(() =>
    m.validatePayload(
      { ...data[0], coverage: "mathematical-material-only" },
      1,
    ),
  );
});
test("duplicate inventory identities refuse; complete edge multiplicity survives", async () => {
  const m = await load();
  const a = structuredClone(data[0]);
  a.sources.push(a.sources[0]);
  assert.throws(() => m.validatePayload(a, 1), /duplicate/i);
  const b = structuredClone(data[0]);
  b.graph.data.edges.push(b.graph.data.edges[0]);
  const d = m.compareSnapshots(data[0], b);
  assert.equal(d.edges.length, 1);
  assert.equal(d.edges[0].after, 2);
});
test("schema2 validates projection and refuses to describe omitted inventory as deleted", async () => {
  const m = await load();
  const a = structuredClone(data[0]);
  a.schema_version = 2;
  a.coverage = "mathematical-material-only";
  const wanted = new Set([
    `${a.graph.identity.repository}\0${a.graph.identity.path}`,
    ...a.equations.map((e) => `${e.repository}\0${e.path}`),
  ]);
  for (const n of Object.values(a.graph.data.nodes))
    for (const k of ["source", "review_source"])
      if (typeof n[k] === "string")
        wanted.add(`d6g8k5htny-coder/Math-\0${n[k]}`);
  a.sources = a.sources.filter((s) => wanted.has(`${s.repository}\0${s.path}`));
  // Match mathematical_projection's explicit repository/path ordering.
  assert.throws(() => m.validatePayload(a, 2), /Unsorted/);
  a.sources.sort((left, right) => {
    const leftKey = `${left.repository}\0${left.path}`;
    const rightKey = `${right.repository}\0${right.path}`;
    return leftKey < rightKey ? -1 : leftKey > rightKey ? 1 : 0;
  });
  m.validatePayload(a, 2);
  assert.equal(m.compareSnapshots(data[0], a).inventoryComparable, false);
  assert.deepEqual(m.compareSnapshots(data[0], a).removedSources, []);
  a.sources.push(
    data[0].sources.find((s) => !wanted.has(`${s.repository}\0${s.path}`)),
  );
  assert.throws(() => m.validatePayload(a, 2));
});
test("registry binds selected own descriptors and metadata publication", async () => {
  const m = await load();
  const meta = JSON.parse(await readFile(new URL("current-meta.json", site)));
  const choices = m.validateRegistry(registry, meta);
  assert.equal(choices.length, archives.length + 1);
  assert.throws(() =>
    m.validateRegistry({ ...registry, current: "0".repeat(64) }, meta),
  );
  const dup = structuredClone(registry);
  dup.archives.push(dup.archives[0]);
  assert.throws(() => m.validateRegistry(dup, meta), /duplicate/i);
});
test("bounded acquisition refuses oversized responses and follows no redirects", async () => {
  const m = await load();
  await assert.rejects(
    m.readBounded(async () => new Response("abcd"), "current.json", 3),
    /byte/i,
  );
  await assert.rejects(
    m.readBounded(async () => new Response("ab"), "current.json", 10, 3),
    /byte/i,
  );
});
test("publication acquisition verifies registry hash before admitting choices", async () => {
  const m = await load();
  const fetcher = async (path) => {
    const body = await readFile(new URL(path, site));
    return new Response(body);
  };
  const p = await m.loadPublication(fetcher, webcrypto);
  assert.equal(
    p.currentPayload.sources.length,
    JSON.parse(await readFile(new URL("current.json", site))).sources.length,
  );
  await assert.rejects(
    m.loadPublication(async (path) => {
      const response = await fetcher(path);
      return path === "snapshots/index.json"
        ? new Response((await response.text()) + " ")
        : response;
    }, webcrypto),
    /digest/i,
  );
});
test("page advertised identity refuses a different publication", async () => {
  const m = await load();
  await assert.rejects(
    m.loadPublication(
      async (path) => new Response(await readFile(new URL(path, site))),
      webcrypto,
      { sha256: "0".repeat(64), bytes: 1 },
    ),
    /page.*publication/i,
  );
});
test("decoded byte identity survives HTTP content encoding", async () => {
  const m = await load();
  const raw = await m.readBounded(
    async () =>
      new Response("abc", {
        headers: { "content-length": "21", "content-encoding": "gzip" },
      }),
    "current.json",
    10,
    3,
  );
  assert.equal(raw.length, 3);
});

const digestOf = async (bytes) =>
  Buffer.from(await webcrypto.subtle.digest("SHA-256", bytes)).toString("hex");
test("raw duplicate node IDs refuse even when exact byte digest matches", async () => {
  const m = await load();
  const id = Object.keys(data[0].graph.data.nodes)[0];
  const duplicate = JSON.stringify({
    ...data[0].graph.data.nodes[id],
    classification: "duplicate-control",
  });
  const text = JSON.stringify(data[0]).replace(
    '"nodes":{',
    '"nodes":{' + JSON.stringify(id) + ":" + duplicate + ",",
  );
  const bytes = new TextEncoder().encode(text);
  await assert.rejects(
    m.parsePayload(
      bytes,
      {
        path: "current.json",
        bytes: bytes.length,
        sha256: await digestOf(bytes),
      },
      1,
      webcrypto,
    ),
    /duplicate JSON key/i,
  );
});
test("strict raw JSON refuses escaped equivalent keys and accepts scoped nested keys", async () => {
  const m = await load();
  for (const text of [
    '{"a":1,"\\u0061":2}',
    '{"nodes":{"n":{},"\\u006e":{}}}',
    '{"x":[{"same":1,"same":2}]}',
    '{"__proto__":1,"__proto__":2}',
  ])
    assert.throws(() => m.parseStrictJSON(text), /duplicate JSON key/i);
  const safe =
    '{"a":{"same":1},"b":[{"same":2},{"same":3}],"text":"braces {} [] \\\" and : escaped \\\\"}';
  assert.deepEqual(m.parseStrictJSON(safe), JSON.parse(safe));
  assert.throws(
    () => m.parseStrictJSON("[".repeat(130) + "0" + "]".repeat(130)),
    /depth/i,
  );
});
test("raw duplicate metadata and hash-matching registry keys refuse", async () => {
  const m = await load();
  const meta = JSON.parse(await readFile(new URL("current-meta.json", site)));
  const duplicateMeta =
    '{"sha256":"' + "0".repeat(64) + '",' + JSON.stringify(meta).slice(1);
  const fetcher = async (path) =>
    new Response(
      path === "current-meta.json"
        ? duplicateMeta
        : await readFile(new URL(path, site)),
    );
  await assert.rejects(
    m.loadPublication(fetcher, webcrypto),
    /duplicate JSON key/i,
  );
  const duplicateRegistry =
    '{"current":"' + "0".repeat(64) + '",' + JSON.stringify(registry).slice(1);
  const modifiedMeta = {
    ...meta,
    history_manifest_sha256: await digestOf(
      new TextEncoder().encode(duplicateRegistry),
    ),
  };
  await assert.rejects(
    m.loadPublication(
      async (path) =>
        new Response(
          path === "current-meta.json"
            ? JSON.stringify(modifiedMeta)
            : path === "snapshots/index.json"
              ? duplicateRegistry
              : await readFile(new URL(path, site)),
        ),
      webcrypto,
    ),
    /duplicate JSON key/i,
  );
});
test("producer-admitted identical current/archive bytes retain distinct typed selections", async () => {
  const m = await load();
  const identical = structuredClone(registry);
  const a = identical.archives.find((a) => a.id === identical.current);
  const d = a.files.find((f) => f.path === "current.json");
  identical.total_bytes +=
    d.bytes - identical.current_observation.payload.bytes;
  identical.capacity.bytes = identical.total_bytes;
  identical.current_observation.payload = { ...d, path: "current.json" };
  const registryBytes = new TextEncoder().encode(JSON.stringify(identical));
  const meta = {
    archive: a.id,
    sha256: d.sha256,
    bytes: d.bytes,
    observed_at: a.observed_at,
    history_manifest_sha256: await digestOf(registryBytes),
  };
  const fetcher = async (path) =>
    new Response(
      path === "current-meta.json"
        ? JSON.stringify(meta)
        : path === "snapshots/index.json"
          ? registryBytes
          : path === "current.json"
            ? await readFile(new URL(`snapshots/${a.id}/current.json`, site))
            : await readFile(new URL(path, site)),
    );
  const publication = await m.loadPublication(fetcher, webcrypto);
  const archive = publication.choices.find((c) => !c.current && c.id === a.id),
    current = publication.choices.find((c) => c.current);
  assert.equal(archive.key, `archive:${a.id}`);
  assert.equal(current.key, `current:${a.id}`);
  assert.notEqual(archive.key, current.key);
  const result = m.compareSnapshots(
    await m.loadSelection(archive, publication, fetcher, webcrypto),
    await m.loadSelection(current, publication, fetcher, webcrypto),
  );
  for (const [key, value] of Object.entries(result))
    if (Array.isArray(value)) assert.equal(value.length, 0, key);
});
test("material selection labels distinguish archive semantic ID from exact payload digest and bytes", async () => {
  const m = await load();
  const choice = {
    id: "a".repeat(64),
    key: "archive:" + "a".repeat(64),
    current: false,
    version: 2,
    observed_at: "2026-10-25T00:00:00Z",
    descriptor: {
      sha256: "b".repeat(64),
      bytes: 123,
      path: "snapshots/a/current.json",
    },
  };
  const label = m.selectionDescription(choice);
  assert.match(label, /material archive ID/);
  assert.ok(label.includes(choice.id));
  assert.ok(label.includes("payload SHA-256 " + choice.descriptor.sha256));
  assert.match(label, /123 bytes/);
});
