import {
  loadPublication,
  loadSelection,
  compareSnapshots,
} from "./compare-model.mjs?site-release=6c69a2165f77b03d7f2d07cb3189e49a5f44f8ac2309341cd6076abd583594cc";
import { sourceURL } from "./current-model.mjs?site-release=a30d68465e3acda48eedce64854afebc7af5c84903a4e6b0c649e08007b08999";
const $ = (id) => document.getElementById(id);
let publication,
  generation = 0,
  pages = [],
  page = 0;
const status = (text, state) => {
  $("compare-state").textContent = text;
  $("compare-state").dataset.state = state;
};
function clear() {
  pages = [];
  page = 0;
  $("comparison-summary").replaceChildren();
  $("comparison-results").replaceChildren();
  $("pagination").hidden = true;
  $("share").removeAttribute("href");
}
function element(tag, text) {
  const e = document.createElement(tag);
  e.textContent = text;
  return e;
}
function selectionFromURL() {
  const params = new URL(location.href).searchParams;
  const ids = ["from", "to"].map((k) => params.get(k));
  if (params.getAll("from").length > 1 || params.getAll("to").length > 1)
    throw Error("Duplicate selection parameters.");
  if (ids.every((x) => x === null)) {
    const archives = publication.choices.filter((c) => !c.current);
    return [archives[0], archives.at(-1)];
  }
  if (ids.some((x) => !x || !/^[0-9a-f]{64}$/.test(x)))
    throw Error("Both exact selection IDs are required.");
  return ids.map((id) => {
    const c = publication.choices.find((c) => c.id === id);
    if (!c)
      throw Error("Selection ID is absent from this verified publication.");
    return c;
  });
}
function selected() {
  return ["from", "to"].map((id) =>
    publication.choices.find((c) => c.id === $(id).value),
  );
}
function selectionURL(choices) {
  const url = new URL(location.href);
  url.searchParams.set("from", choices[0].id);
  url.searchParams.set("to", choices[1].id);
  return url;
}
function drawPage() {
  const list = $("comparison-results");
  list.replaceChildren();
  const start = page * 20;
  for (const row of pages.slice(start, start + 20)) {
    const article = element("article", "");
    article.className = "change";
    article.append(
      element("h3", row.kind),
      element("code", row.change.id.replaceAll("\0", " · ")),
    );
    for (const key of ["before", "after"]) {
      const value = row.change[key];
      const detail = document.createElement("details");
      detail.append(
        element("summary", key === "before" ? "Before record" : "After record"),
      );
      detail.append(
        element(
          "pre",
          value === undefined
            ? "Absent in this retained coverage"
            : JSON.stringify(value, null, 2),
        ),
      );
      if (value && typeof value === "object") {
        const url = sourceURL(value);
        if (url) {
          const link = element("a", "Open exact pinned source ↗");
          link.href = url;
          detail.append(link);
        }
      }
      article.append(detail);
    }
    list.append(article);
  }
  if (!pages.length)
    list.append(element("p", "No record differences in these categories."));
  $("page-status").textContent =
    `${pages.length ? start + 1 : 0}–${Math.min(start + 20, pages.length)} of ${pages.length} differences`;
  $("previous").disabled = page === 0;
  $("next").disabled = start + 20 >= pages.length;
  $("pagination").hidden = pages.length === 0;
}
function render(result, choices, before, after) {
  const summary = $("comparison-summary");
  summary.replaceChildren(element("h2", "What changed between these records?"));
  summary.append(
    element(
      "p",
      `${Object.keys(before.graph.data.nodes).length} → ${Object.keys(after.graph.data.nodes).length} nodes · ${before.graph.data.edges.length} → ${after.graph.data.edges.length} edges · ${before.sources.length} → ${after.sources.length} retained source identities.`,
    ),
  );
  const counts = [
    ["Node records", result.nodes],
    ["Complete edge records (multiplicity)", result.edges],
    ["Source commit pins", result.pins],
    ["Source blob / byte identities", result.content],
    ["Other / all shared source record fields", result.sourceRecords],
    ["Added inventory identities", result.addedSources],
    ["Removed inventory identities", result.removedSources],
    ["Included excerpt records", result.excerpts],
    ["Reference identity records", result.references],
  ];
  const dl = document.createElement("dl");
  dl.className = "comparison-counts";
  for (const [label, rows] of counts) {
    dl.append(element("dt", label), element("dd", String(rows.length)));
  }
  summary.append(dl);
  summary.append(
    element(
      "p",
      result.inventoryComparable
        ? "Inventory coverage: complete source inventories are comparable."
        : "Inventory coverage: incomparable. At least one archive retains mathematical material only. Omitted inventory is unretained, never reported as deleted. Shared retained identities remain comparable.",
    ),
  );
  const exact = document.createElement("p");
  exact.className = "exact-selection";
  choices.forEach((c, i) => {
    exact.append(element("strong", i ? " To: " : "From: "));
    const a = element("a", `${c.observed_at} · ${c.id}`);
    a.href = c.descriptor.path;
    exact.append(a);
  });
  summary.append(exact);
  pages = counts.flatMap(([kind, rows]) =>
    rows.map((change) => ({ kind, change })),
  );
  page = 0;
  drawPage();
}
async function compare(push = false) {
  const token = ++generation;
  clear();
  try {
    if (!publication) throw Error("Publication has not been verified.");
    const choices = selected();
    if (choices.some((c) => !c))
      throw Error("Choose two verified observations.");
    if (push) history.pushState(null, "", selectionURL(choices));
    status("Running: verifying the exact selected payloads…", "running");
    const [before, after] = await Promise.all(
      choices.map((c) => loadSelection(c, publication)),
    );
    if (token !== generation) return;
    render(compareSnapshots(before, after), choices, before, after);
    $("share").href = selectionURL(choices).href;
    status(
      "Completed: exact selected bytes verified; record comparison displayed.",
      "completed",
    );
  } catch (error) {
    if (token !== generation) return;
    clear();
    status(
      "Refused: " +
        error.message +
        " Use Retry to verify this publication again.",
      "refused",
    );
  }
}
async function initialize() {
  const token = ++generation;
  clear();
  publication = null;
  for (const id of ["from", "to", "compare"]) $(id).disabled = true;
  status(
    "Running: verifying publication metadata, registry and current bytes…",
    "running",
  );
  try {
    const binding = document.querySelector(
      "[data-current-digest][data-current-bytes]",
    );
    if (!binding) throw Error("Page publication binding missing.");
    const candidate = await loadPublication(
      globalThis.fetch,
      globalThis.crypto,
      {
        sha256: binding.dataset.currentDigest,
        bytes: Number(binding.dataset.currentBytes),
      },
    );
    if (token !== generation) return;
    publication = candidate;
    for (const id of ["from", "to"]) {
      const select = $(id);
      select.replaceChildren();
      for (const c of publication.choices) {
        const option = element(
          "option",
          `${c.observed_at} · ${c.current ? "current observation" : "retained archive"} · ${c.id}`,
        );
        option.value = c.id;
        select.append(option);
      }
      select.disabled = false;
    }
    const choices = selectionFromURL();
    choices.forEach((c, i) => ($(i ? "to" : "from").value = c.id));
    $("compare").disabled = false;
    await compare();
  } catch (error) {
    if (token !== generation) return;
    clear();
    status(
      "Refused: " +
        error.message +
        " Use Retry to verify this publication again.",
      "refused",
    );
  }
}
$("compare-form").addEventListener("submit", (event) => {
  event.preventDefault();
  compare(true);
});
for (const id of ["from", "to"])
  $(id).addEventListener("change", () => {
    ++generation;
    clear();
    status(
      "Awaiting comparison: selection changed. Load the selected observations.",
      "awaiting",
    );
  });
$("retry").addEventListener("click", initialize);
$("previous").addEventListener("click", () => {
  page--;
  drawPage();
  $("results-heading").focus();
});
$("next").addEventListener("click", () => {
  page++;
  drawPage();
  $("results-heading").focus();
});
window.addEventListener("popstate", () => {
  if (!publication) {
    initialize();
    return;
  }
  try {
    const choices = selectionFromURL();
    choices.forEach((c, i) => ($(i ? "to" : "from").value = c.id));
    compare();
  } catch (error) {
    ++generation;
    clear();
    status("Refused: " + error.message, "refused");
  }
});
initialize();
