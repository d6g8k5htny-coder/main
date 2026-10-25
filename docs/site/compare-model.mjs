import { validateSnapshot, parseVerifiedSnapshot } from "./current-model.mjs?site-release=a30d68465e3acda48eedce64854afebc7af5c84903a4e6b0c649e08007b08999";
const HEX = /^[0-9a-f]{64}$/;
const ORIGIN = "https://d6g8k5htny-coder.github.io/main/site/";
export const MAX_BYTES = 32 * 1024 * 1024;
const fail = (message) => {
  throw Error(message);
};
export function publicPath(path) {
  if (
    typeof path !== "string" ||
    !path ||
    /[\\%:?#\s\u0000-\u001f\u007f]/.test(path) ||
    path.split("/").some((p) => !p || p === "." || p === "..")
  )
    fail("Unsafe public path.");
  return path;
}
function descriptor(d) {
  if (
    !d ||
    !HEX.test(d.sha256) ||
    !Number.isSafeInteger(d.bytes) ||
    d.bytes < 1 ||
    d.bytes > MAX_BYTES
  )
    fail("Invalid descriptor byte bounds or digest.");
  publicPath(d.path);
  return d;
}
export function canonicalJSON(value) {
  if (Array.isArray(value))
    return "[" + value.map(canonicalJSON).join(",") + "]";
  if (value && typeof value === "object")
    return (
      "{" +
      Object.keys(value)
        .sort()
        .map((k) => JSON.stringify(k) + ":" + canonicalJSON(value[k]))
        .join(",") +
      "}"
    );
  return JSON.stringify(value);
}
const sourceKey = (s) => s.repository + "\0" + s.path;
export function validatePayload(data, version) {
  if (![1, 2].includes(version) || data?.schema_version !== version)
    fail("Unsupported payload version.");
  if (version === 1 && Object.hasOwn(data, "coverage"))
    fail("Full snapshot coverage mismatch.");
  if (version === 2 && data.coverage !== "mathematical-material-only")
    fail("Projection coverage mismatch.");
  validateSnapshot(version === 1 ? data : { ...data, schema_version: 1 });
  for (const rows of [data.sources, data.references]) {
    const seen = new Set();
    for (const s of rows) {
      const key = sourceKey(s);
      if (seen.has(key)) fail("Duplicate source identity.");
      seen.add(key);
    }
  }
  const equations = new Set();
  for (const e of data.equations) {
    const key = sourceKey(e) + "\0" + e.id;
    if (equations.has(key)) fail("Duplicate excerpt identity.");
    equations.add(key);
  }
  if (version === 2) {
    const wanted = new Set([
      sourceKey(data.graph.identity),
      ...data.equations.map(sourceKey),
    ]);
    for (const n of Object.values(data.graph.data.nodes))
      for (const k of ["source", "review_source"])
        if (typeof n[k] === "string")
          wanted.add("d6g8k5htny-coder/Math-\0" + n[k]);
    if (data.sources.some((s) => !wanted.has(sourceKey(s))))
      fail("Irrelevant inventory in material projection.");
    if (
      canonicalJSON(data.sources) !==
      canonicalJSON(
        [...data.sources].sort((a, b) =>
          a.repository < b.repository
            ? -1
            : a.repository > b.repository
              ? 1
              : a.path < b.path
                ? -1
                : a.path > b.path
                  ? 1
                  : 0,
        ),
      )
    )
      fail("Unsorted material projection inventory.");
  }
  return data;
}
export async function verifyBytes(raw, d, cryptoAPI = globalThis.crypto) {
  descriptor({ ...d, path: d.path || "payload.json" });
  const bytes = raw instanceof Uint8Array ? raw : new Uint8Array(raw);
  if (bytes.byteLength !== d.bytes) fail("Payload byte count mismatch.");
  if (!cryptoAPI?.subtle) fail("Digest verification unavailable.");
  const digest = Array.from(
    new Uint8Array(await cryptoAPI.subtle.digest("SHA-256", bytes)),
    (b) => b.toString(16).padStart(2, "0"),
  ).join("");
  if (digest !== d.sha256) fail("Payload digest mismatch.");
  return bytes;
}
export async function parsePayload(
  raw,
  d,
  version,
  cryptoAPI = globalThis.crypto,
) {
  const bytes = await verifyBytes(raw, d, cryptoAPI);
  let data;
  if (version === 1) data = await parseVerifiedSnapshot(bytes, d, cryptoAPI);
  else
    data = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  return validatePayload(data, version);
}
export function validateMetadata(meta) {
  descriptor({ ...meta, path: "current.json" });
  if (
    !HEX.test(meta.archive) ||
    !HEX.test(meta.history_manifest_sha256) ||
    typeof meta.observed_at !== "string" ||
    !Number.isFinite(Date.parse(meta.observed_at))
  )
    fail("Invalid publication metadata.");
  return meta;
}
export function validateRegistry(registry, meta) {
  validateMetadata(meta);
  if (
    registry?.schema_version !== 2 ||
    registry.scientific_effect !== "NONE" ||
    registry.origin !== ORIGIN ||
    registry.representation !== "asset-release-normalized-html-v1" ||
    !Array.isArray(registry.archives) ||
    registry.archives.length < 1 ||
    registry.archives.length > 512
  )
    fail("Unsupported history registry.");
  const seen = new Set(),
    choices = [];
  for (const a of registry.archives) {
    const version = a.archive_version ?? 1;
    if (!HEX.test(a.id) || !HEX.test(a.semantic_sha256) || seen.has(a.id))
      fail("Duplicate or invalid archive ID.");
    seen.add(a.id);
    if (
      ![1, 2].includes(version) ||
      a.renderer !== `research-static-v${version}` ||
      (version === 2 && a.semantic_sha256 !== a.id) ||
      !Number.isFinite(Date.parse(a.observed_at))
    )
      fail("Unsupported archive descriptor.");
    if (!Array.isArray(a.files) || a.files.length < 1 || a.files.length > 10003)
      fail("Invalid archive members.");
    const paths = new Set();
    for (const d of a.files) {
      descriptor(d);
      if (paths.has(d.path.toLowerCase())) fail("Duplicate archive member.");
      paths.add(d.path.toLowerCase());
    }
    const d = a.files.find((d) => d.path === "current.json");
    if (!d || (version === 1 && d.sha256 !== a.id))
      fail("Archive payload identity mismatch.");
    choices.push({
      id: a.id,
      observed_at: a.observed_at,
      version,
      descriptor: { ...d, path: `snapshots/${a.id}/${d.path}` },
      current: false,
    });
  }
  if (!seen.has(registry.current) || registry.current !== meta.archive)
    fail("Publication archive mismatch.");
  const current = descriptor(registry.current_observation?.payload);
  if (
    current.path !== "current.json" ||
    current.sha256 !== meta.sha256 ||
    current.bytes !== meta.bytes
  )
    fail("Publication current payload mismatch.");
  choices.sort(
    (a, b) =>
      a.observed_at.localeCompare(b.observed_at) || a.id.localeCompare(b.id),
  );
  if (seen.has(meta.sha256)) fail("Ambiguous current and archive IDs.");
  choices.push({
    id: meta.sha256,
    observed_at: meta.observed_at,
    version: 1,
    descriptor: current,
    current: true,
  });
  return choices;
}
export async function readBounded(fetcher, path, maxBytes, exactBytes) {
  publicPath(path);
  const response = await fetcher(path, {
    cache: "no-store",
    credentials: "omit",
    redirect: "error",
  });
  if (!response.ok) fail("Source unavailable: " + path);
  const encoding = response.headers?.get("content-encoding");
  const advertised =
    encoding && encoding !== "identity"
      ? null
      : response.headers?.get("content-length");
  if (
    advertised !== null &&
    advertised !== undefined &&
    (!/^\d+$/.test(advertised) ||
      Number(advertised) > maxBytes ||
      (exactBytes !== undefined && Number(advertised) !== exactBytes))
  )
    fail("Response byte bounds mismatch.");
  const chunks = [];
  let size = 0;
  if (response.body?.getReader) {
    const reader = response.body.getReader();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        size += value.byteLength;
        if (size > maxBytes || (exactBytes !== undefined && size > exactBytes))
          fail("Response byte bounds exceeded.");
        chunks.push(value);
      }
    } catch (e) {
      await reader.cancel().catch(() => {});
      throw e;
    }
  } else {
    const bytes = new Uint8Array(await response.arrayBuffer());
    size = bytes.byteLength;
    chunks.push(bytes);
  }
  if (size > maxBytes || (exactBytes !== undefined && size !== exactBytes))
    fail("Response byte count mismatch.");
  const raw = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    raw.set(chunk, offset);
    offset += chunk.length;
  }
  return raw;
}
export async function loadPublication(
  fetcher = globalThis.fetch,
  cryptoAPI = globalThis.crypto,
  expectedIdentity,
) {
  const decode = (b) =>
    JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(b));
  const meta = validateMetadata(
    decode(await readBounded(fetcher, "current-meta.json", 16384)),
  );
  if (
    expectedIdentity &&
    (meta.sha256 !== expectedIdentity.sha256 ||
      meta.bytes !== expectedIdentity.bytes)
  )
    fail("Page and publication identity mismatch.");
  const registryRaw = await readBounded(
    fetcher,
    "snapshots/index.json",
    MAX_BYTES,
  );
  await verifyBytes(
    registryRaw,
    {
      path: "snapshots/index.json",
      bytes: registryRaw.length,
      sha256: meta.history_manifest_sha256,
    },
    cryptoAPI,
  );
  const choices = validateRegistry(decode(registryRaw), meta);
  const current = choices.at(-1);
  const payload = await parsePayload(
    await readBounded(
      fetcher,
      current.descriptor.path,
      MAX_BYTES,
      current.descriptor.bytes,
    ),
    current.descriptor,
    1,
    cryptoAPI,
  );
  if (payload.observed_at !== meta.observed_at)
    fail("Publication observation mismatch.");
  return { meta, choices, currentPayload: payload };
}
export async function loadSelection(
  choice,
  publication,
  fetcher = globalThis.fetch,
  cryptoAPI = globalThis.crypto,
) {
  if (!publication.choices.includes(choice)) fail("Unadmitted selection.");
  if (choice.current) return publication.currentPayload;
  const data = await parsePayload(
    await readBounded(
      fetcher,
      choice.descriptor.path,
      MAX_BYTES,
      choice.descriptor.bytes,
    ),
    choice.descriptor,
    choice.version,
    cryptoAPI,
  );
  if (data.observed_at !== choice.observed_at)
    fail("Archive observation mismatch.");
  return data;
}
function recordDiff(before, after, key) {
  const a = new Map(before.map((x) => [key(x), x])),
    b = new Map(after.map((x) => [key(x), x]));
  return [...new Set([...a.keys(), ...b.keys()])]
    .sort()
    .filter((k) => canonicalJSON(a.get(k)) !== canonicalJSON(b.get(k)))
    .map((id) => ({ id, before: a.get(id), after: b.get(id) }));
}
export function compareSnapshots(before, after) {
  const nodes = recordDiff(
    Object.entries(before.graph.data.nodes).map(([id, record]) => ({
      id,
      record,
    })),
    Object.entries(after.graph.data.nodes).map(([id, record]) => ({
      id,
      record,
    })),
    (n) => n.id,
  );
  const multiset = (edges) => {
    const m = new Map();
    for (const e of edges) {
      const key = canonicalJSON(e);
      m.set(key, (m.get(key) || 0) + 1);
    }
    return m;
  };
  const a = multiset(before.graph.data.edges),
    b = multiset(after.graph.data.edges);
  const edges = [...new Set([...a.keys(), ...b.keys()])]
    .sort()
    .filter((k) => a.get(k) !== b.get(k))
    .map((id) => ({ id, before: a.get(id) || 0, after: b.get(id) || 0 }));
  const inventoryComparable =
    before.schema_version === 1 && after.schema_version === 1;
  const changes = recordDiff(before.sources, after.sources, sourceKey),
    shared = changes.filter((c) => c.before && c.after);
  return {
    nodes,
    edges,
    inventoryComparable,
    pins: shared.filter((c) => c.before.commit !== c.after.commit),
    content: shared.filter(
      (c) => c.before.blob !== c.after.blob || c.before.bytes !== c.after.bytes,
    ),
    sourceRecords: shared,
    addedSources: inventoryComparable ? changes.filter((c) => !c.before) : [],
    removedSources: inventoryComparable ? changes.filter((c) => !c.after) : [],
    excerpts: recordDiff(
      before.equations,
      after.equations,
      (e) => sourceKey(e) + "\0" + e.id,
    ),
    references: recordDiff(before.references, after.references, sourceKey),
  };
}
