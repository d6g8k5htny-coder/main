export const meta = {
  name: 'drive-liveness-listing',
  description: 'List Drive items modified after the 2026-09-17 snapshot, by time window, into local TSVs (metadata only)',
  phases: [{ title: 'List', detail: 'one agent per modifiedTime window; pages search_files; writes a TSV' }],
}
const OUT = '<SCRATCH>/gdrive/parts'
const W = [
  ['w01', '2026-09-17T00:00:00Z', '2026-09-21T00:00:00Z'],
  ['w02', '2026-09-21T00:00:00Z', '2026-09-26T00:00:00Z'],
  ['w03', '2026-09-26T00:00:00Z', '2026-10-01T00:00:00Z'],
  ['w04', '2026-10-01T00:00:00Z', '2026-10-08T00:00:00Z'],
  ['w05', '2026-10-08T00:00:00Z', '2026-10-15T00:00:00Z'],
  ['w06', '2026-10-15T00:00:00Z', '2026-10-19T00:00:00Z'],
  ['w07', '2026-10-19T00:00:00Z', '2026-10-22T00:00:00Z'],
  ['w08', '2026-10-22T00:00:00Z', '2026-10-23T12:00:00Z'],
  ['w09', '2026-10-23T12:00:00Z', '2026-10-24T12:00:00Z'],
  ['w10', '2026-10-24T12:00:00Z', '2026-10-25T00:00:00Z'],
]
const SCHEMA = { type: 'object', properties: { window: { type: 'string' }, files_written: { type: 'number' }, pages: { type: 'number' }, complete: { type: 'boolean' }, notes: { type: 'string' } }, required: ['window', 'files_written', 'pages', 'complete', 'notes'] }
phase('List')
const results = await parallel(W.map(([w, a, b]) => () => agent(`You are listing Google Drive file METADATA for one time window, read-only. Do not open, read, download, export, move, share or edit any file. Do not post anything anywhere. Do not run git. Do not mention or copy any Drive id, URL or email address into your final return value (write them only into the TSV file named below).

Steps:
1. Call ToolSearch with query "select:mcp__Google_Drive__search_files" to load the Drive search tool.
2. Call mcp__Google_Drive__search_files with query exactly:
   modifiedTime > '${a}' and modifiedTime <= '${b}'
   and pageSize 100 and excludeContentSnippets true. Keep calling it with the returned nextPageToken as pageToken (same query, same pageSize, excludeContentSnippets true) until the response has no nextPageToken or returns no files. Count the pages.
3. For every file object in every page, append one line to the file ${OUT}/${w}.tsv using the Bash tool with a Python heredoc or printf (create the file on the first write; tab-separated; no header; one line per file; replace any tab or newline inside a field with a space). Columns in this order: id, title, mimeType, fileSize (empty if absent), createdTime, modifiedTime, parentId (empty if absent), starred (true/false), owner (the owner field as given, or empty). Write the lines faithfully from the tool output; do not invent or summarise fields. It is fine to write each page's lines with one Bash call.
4. When done, run: wc -l ${OUT}/${w}.tsv  and report that count.

Return a JSON object: window "${w}", files_written (the wc -l count), pages (number of search calls made), complete (true only if the last page had no nextPageToken), notes (one or two sentences: any error, any page that failed, whether you had to retry; no ids, URLs or emails).`, { label: `list:${w}`, phase: 'List', schema: SCHEMA, effort: 'low' })))
const ok = results.filter(Boolean)
log(`${ok.length}/${W.length} windows returned; files: ${ok.reduce((s, r) => s + (r.files_written || 0), 0)}; incomplete: ${ok.filter(r => !r.complete).map(r => r.window).join(',') || 'none'}`)
return { windows: ok, missing: W.map(x => x[0]).filter(w => !ok.find(r => r.window === w)) }