"""Current-only content associations, independent of scientific status or graph binding."""
import hashlib
import math
import re

REPOSITORIES = {"d6g8k5htny-coder/main", "d6g8k5htny-coder/Math-"}
MATH = "d6g8k5htny-coder/Math-"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
MAX_BYTES = 16 * 1024 * 1024
LABELS = {"metadata_only": "Metadata only", "graph_data_included": "Parsed graph",
          "source_identity_only": "Reference identity", "excerpts_included": "Included excerpts",
          "unavailable": "Unavailable"}
LEGEND = ("Metadata only: discoverable identity without included document content. "
          "Parsed graph: source read and verified during generation; parsed data included. "
          "Reference identity: source read and verified during generation; identity retained, document text omitted. "
          "Included excerpts: exact text spans, not the complete document. Unavailable: no qualified association. "
          "Mathematical review: unknown in this inventory. Applicability: unknown in this inventory. "
          "Acceptance: unknown in this inventory. Unknown does not mean unreviewed, rejected or inapplicable.")
UNAVAILABLE = "Acquisition associations could not be qualified for this observation; no content association inferred."


def _unicode(value):
    return isinstance(value, str) and not any(0xD800 <= ord(c) <= 0xDFFF for c in value)


def _integer(value, low, high):
    return (type(value) in (int, float) and low <= value <= high and math.isfinite(value)
            and value == int(value))


def _path(value):
    return (_unicode(value) and 0 < len(value) <= 4096
            and not any(c in value for c in ("\\", ":", "%"))
            and not any(ord(c) < 32 or ord(c) == 127 for c in value)
            and all(p and p not in (".", "..") for p in value.split("/")))


def project_current_acquisitions(data):
    """Return a deterministic, all-or-nothing annotation projection; never mutate data."""
    reasons = set()
    if not isinstance(data, dict):
        return {"disposition": "unavailable", "reasons": ["invalid_acquisition_input"], "rows": []}
    commits = {}
    repos = data.get("repositories")
    if not isinstance(repos, list) or len(repos) != 2:
        reasons.add("invalid_acquisition_input")
    else:
        for repo in repos:
            if (not isinstance(repo, dict) or not isinstance(repo.get("repository"), str)
                    or repo.get("repository") not in REPOSITORIES
                    or repo.get("repository") in commits
                    or not isinstance(repo.get("commit"), str) or not HEX40.fullmatch(repo["commit"])):
                reasons.add("invalid_acquisition_input")
            else:
                commits[repo["repository"]] = repo["commit"]

    def identity(record, whole=True):
        if (not isinstance(record, dict) or not isinstance(record.get("repository"), str)
                or record.get("repository") not in REPOSITORIES
                or record.get("commit") != commits.get(record.get("repository"))
                or not isinstance(record.get("commit"), str) or not HEX40.fullmatch(record["commit"])
                or not _path(record.get("path"))):
            return None
        key = (record["repository"], record["commit"], record["path"])
        if whole and (not isinstance(record.get("blob"), str) or not HEX40.fullmatch(record["blob"])
                      or not _integer(record.get("bytes"), 0, MAX_BYTES)):
            return None
        return key

    sources = data.get("sources")
    if not isinstance(sources, list):
        reasons.add("invalid_acquisition_input")
        sources = []
        inventory_shape = False
    else:
        inventory_shape = True
    rows, index = [], {}
    for source in sources:
        key = identity(source)
        row = {"key": list(key) if key else None,
               "identity": [*key, source["blob"], int(source["bytes"])] if key else None,
               "categories": [], "whole_files": [], "excerpts": []}
        rows.append(row)
        if key is None:
            reasons.add("invalid_acquisition_identity")
        elif key in index:
            reasons.add("duplicate_inventory_identity")
        else:
            index[key] = (source, row)
    whole_seen = set()

    def whole(record, category, graph=False):
        key = identity(record)
        if (key is None or (graph and record.get("repository") != MATH)
                or not isinstance(record.get("sha256"), str) or not HEX64.fullmatch(record["sha256"])):
            reasons.add("invalid_acquisition_identity")
            return
        if key in whole_seen:
            reasons.add("duplicate_acquisition_identity")
        whole_seen.add(key)
        if not inventory_shape:
            return
        if key not in index:
            reasons.add("missing_acquisition_inventory")
            return
        source, row = index[key]
        if source["blob"] != record["blob"]:
            reasons.add("acquisition_blob_conflict")
        if source["bytes"] != record["bytes"]:
            reasons.add("acquisition_bytes_conflict")
        row["categories"].append(category)
        row["whole_files"].append({"category": category, "sha256": record["sha256"]})

    graph = data.get("graph")
    if not isinstance(graph, dict) or not isinstance(graph.get("identity"), dict):
        reasons.add("invalid_acquisition_input")
    else:
        whole(graph["identity"], "graph_data_included", True)
    references = data.get("references")
    if not isinstance(references, list):
        reasons.add("invalid_acquisition_input")
    else:
        for reference in references:
            whole(reference, "source_identity_only")
    equations = data.get("equations")
    ids, spans = set(), set()
    if not isinstance(equations, list):
        reasons.add("invalid_acquisition_input")
    else:
        for excerpt in equations:
            key = identity(excerpt, False)
            if (key is None or not _unicode(excerpt.get("id")) or not excerpt["id"]
                    or not _unicode(excerpt.get("text"))
                    or not _integer(excerpt.get("line_start"), 1, 9007199254740991)
                    or not _integer(excerpt.get("line_end"), 1, 9007199254740991)
                    or excerpt["line_start"] > excerpt["line_end"]
                    or not isinstance(excerpt.get("sha256"), str) or not HEX64.fullmatch(excerpt["sha256"])
                    or hashlib.sha256(excerpt["text"].encode("utf-8")).hexdigest() != excerpt["sha256"]):
                reasons.add("invalid_excerpt_record")
                continue
            span = (*key, excerpt["line_start"], excerpt["line_end"])
            if excerpt["id"] in ids or span in spans:
                reasons.add("duplicate_excerpt_identity")
            ids.add(excerpt["id"])
            spans.add(span)
            if not inventory_shape:
                continue
            if key not in index:
                reasons.add("missing_excerpt_inventory")
                continue
            row = index[key][1]
            if "excerpts_included" not in row["categories"]:
                row["categories"].append("excerpts_included")
            row["excerpts"].append({field: excerpt[field] for field in ("id", "line_start", "line_end", "sha256")})
    for row in rows:
        if reasons:
            row.update(categories=["unavailable"], whole_files=[], excerpts=[])
        elif not row["categories"]:
            row["categories"] = ["metadata_only"]
    return {"disposition": "unavailable" if reasons else "qualified", "reasons": sorted(reasons), "rows": rows}


# Presentation is current-only. Frozen historical renderer helpers do not import it.
def _source_url(source):
    from urllib.parse import quote
    if (not isinstance(source, dict) or not isinstance(source.get('repository'), str)
            or source.get('repository') not in REPOSITORIES
            or not isinstance(source.get('commit'), str) or not HEX40.fullmatch(source['commit'])
            or not _path(source.get('path'))):
        return None
    return f"https://github.com/{source['repository']}/blob/{source['commit']}/" + quote(source['path'], safe='/')


def inventory_legend(identifier='current-inventory-legend'):
    from html import escape
    return f'<p id="{escape(identifier, quote=True)}" class="inventory-legend">{escape(LEGEND)}</p>'


def acquisition_notice(projection):
    from html import escape
    if projection['disposition'] == 'qualified':
        return ''
    return ('<p class="inventory-unavailable">' + UNAVAILABLE + '</p>'
            '<details class="inventory-diagnostics"><summary>Acquisition association diagnostics</summary><p>'
            + escape(', '.join(projection['reasons'])) + '</p></details>')


def current_source_list(sources, projection):
    from html import escape
    lookup = {tuple(r['key']): r for r in projection['rows'] if r['key'] is not None}
    parts = []
    for source in sources:
        url = _source_url(source)
        label = escape(source['repository'].split('/')[1] + ' · ' + source['path'])
        name = f'<a href="{escape(url, quote=True)}">{label}</a>' if url else label
        row = lookup.get((source['repository'], source['commit'], source['path']))
        categories = row['categories'] if row else ['unavailable']
        labels = [LABELS[c] + (f" ({len(row['excerpts'])})" if c == 'excerpts_included' else '') for c in categories]
        extra = ''
        if row and (row['excerpts'] or row['whole_files']):
            identities = ''.join('<p>Git blob <code>' + escape(source['blob']) + '</code> · ' + str(source['bytes'])
                                 + ' bytes · producer-recorded complete-file SHA-256 <code>' + file['sha256']
                                 + '</code>. The viewer has not fetched or rehashed the complete source document.</p>'
                                 for file in row['whole_files'])
            items = []
            for excerpt in row['excerpts']:
                start, end = excerpt['line_start'], excerpt['line_end']
                fragment = f'#L{start}' if start == end else f'#L{start}-L{end}'
                text = f"{excerpt['id']} · lines {start}–{end}"
                link = f'<a href="{escape(url + fragment, quote=True)}">{escape(text)}</a>' if url else escape(text)
                items.append('<li>' + link + ' · excerpt SHA-256 <code>' + excerpt['sha256'] + '</code></li>')
            extra = '<details><summary>Acquisition identities</summary>' + identities + ('<ul>' + ''.join(items) + '</ul>' if items else '') + '</details>'
        parts.append('<li>' + name + '<small class="acquisition-label">Acquisition evidence: '
                     + escape(' · '.join(labels)) + '</small>' + extra + '</li>')
    return '\n'.join(parts)


def date_explanation(data, *, prefix='', snapshot_prefix=''):
    from html import escape
    source = data['graph']['identity']
    url = _source_url(source)
    graph = ('<a href="' + escape(url, quote=True) + '">Exact graph classification source</a>') if url else 'Graph source unavailable'
    return ('<details id="current-date-explanation" class="current-date-explanation"><summary>Why these dates differ</summary>'
            '<p>The observation time records when this source snapshot was collected. Graph classifications, historical records and curated stories keep their own source cuts; a newer observation or publication does not re-review their mathematics.</p>'
            '<dl><dt>Source observation</dt><dd><time datetime="' + escape(data['observed_at'], quote=True) + '">'
            + escape(data['observed_at']) + '</time>: collection time for the displayed pinned repository identities. The two-hour stale rule concerns this observation only.</dd>'
            '<dt>Graph classification source</dt><dd>' + graph + ' at <code>' + escape(source['commit'])
            + '</code>. Classifications quote those graph bytes, which may remain unchanged while the repository advances; classification date not recorded separately.</dd>'
            '<dt>Historical records</dt><dd><a href="' + escape(snapshot_prefix + 'nodes/index.html', quote=True)
            + '">Preserved dependency snapshot</a>, <a href="' + escape(prefix + 'workspace.html#inventory', quote=True)
            + '">Library catalog</a> and <a href="' + escape(prefix + 'workspace.html#board', quote=True)
            + '">recorded status board</a> retain their separate printed capture dates and source cuts.</dd>'
            '<dt>Curated Research stories</dt><dd>The <a href="' + escape(prefix + 'research.html#latest-work', quote=True)
            + '">dated reading cut</a> identifies the editorial selection and its sources; it does not claim that all later work is included or newly reviewed.</dd></dl></details>')
