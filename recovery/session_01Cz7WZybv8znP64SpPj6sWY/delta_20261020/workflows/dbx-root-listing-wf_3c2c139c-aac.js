export const meta = {
  name: 'dbx-root-listing',
  description: 'Complete the recursive metadata listing of the Dropbox folder /2026 into a local TSV (metadata only, no download)',
  phases: [{ title: 'List', detail: 'one agent pages list_folder from the saved cursor to has_more=false' }],
}
const OUT = '<SCRATCH>/dbxroot/parts'
const SCHEMA = { type: 'object', properties: { lines_written: { type: 'number' }, pages: { type: 'number' }, complete: { type: 'boolean' }, notes: { type: 'string' } }, required: ['lines_written', 'pages', 'complete', 'notes'] }
phase('List')
const r = await agent(`You are completing a Dropbox folder listing, read-only, metadata only. Do not download, fetch, preview, move, share or edit any file; do not post anything; do not run git. Never copy a Dropbox namespace id (ns:<digits>), file id (id:...), link or file name into your final return value; write them only into the TSV named below.

Steps:
1. mkdir -p ${OUT}. Call ToolSearch with query "select:mcp__Dropbox__list_folder" to load the tool.
2. Read the saved cursor string from the file <SCRATCH>/dbxroot/cursor_2026_p3.txt (it is one line). Call mcp__Dropbox__list_folder with exactly {"cursor": "<that string>"} and nothing else. Each response has entries[], cursor and has_more. Keep calling with the new cursor (cursor only) until has_more is false. Count the pages. If a call fails with INVALID_CURSOR, stop and report it in notes (do not restart from the folder path).
3. For every entry in every page append one line to ${OUT}/list_2026_rest.tsv (tab-separated, no header; replace any tab or newline inside a field with a space). Columns: path (the ns_path exactly as returned), name, object_type, size (empty for folders or when absent), client_modified (empty if absent), file_id (empty if absent). Large responses are saved by the harness to a file whose path the tool result names; parse that JSON file with a short python3 -I script rather than retyping, and append the lines from it.
4. Run: wc -l ${OUT}/list_2026_rest.tsv and report the count.

Return JSON: lines_written (the wc -l count), pages (number of list_folder calls made), complete (true only if the last response had has_more false), notes (one or two sentences on errors or retries; no ids, names, links).`, { label: 'list:2026-rest', phase: 'List', schema: SCHEMA, effort: 'low' })
log(r ? `lines ${r.lines_written}, pages ${r.pages}, complete ${r.complete}` : 'agent returned null')
return r