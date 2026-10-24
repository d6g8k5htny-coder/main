export const meta = {
  name: 'dbx-content-hash-retrieval',
  description: 'Retrieve Dropbox content_hash metadata for 222 batches of ns-paths via subagents writing path/size/hash TSVs',
  phases: [{ title: 'Retrieve', detail: 'one agent per 6-batch group; download_link metadata only' }],
}
const S = '<SCRATCH>'
const groups = args.groups
log(`${groups.length} groups`)
const results = await parallel(groups.map(g => () => agent(
`You are retrieving Dropbox file content hashes (metadata only, no file content) for a byte-identity crosswalk. Work only with the Dropbox MCP tool mcp__Dropbox__download_link (load its schema first with ToolSearch query "select:mcp__Dropbox__download_link"). Do NOT fetch, open, preview or HEAD any download_url. Never write a download_url, a Dropbox "id:" value, or a "rev" to any file or into your final answer.

Input: ${S}/dbx/agentbatches/${g}.json — a JSON list of batches; each batch is a list of up to 25 Dropbox ns-paths (strings starting with "ns:"). Read it with cat.

For each batch, in order:
1. Call mcp__Dropbox__download_link with entries = exactly that batch's list of ns-paths (pass each string unchanged, never strip the "ns:<digits>//" prefix) and expiration_in_sec = 60.
2. From the result's entries, for every entry take path (the ns-path), size and content_hash. Compute rel = the part of path after the first "//". Append one line per entry to ${S}/dbx/agentout/${g}.tsv in the form rel<TAB>size<TAB>content_hash (UTF-8; paths contain non-ASCII characters, spaces and quotes, so write them from a Python heredoc with the values as a literal list, never via shell interpolation). No other field.
3. If the call fails (the whole batch fails if any entry is invalid or not allowed), retry the batch split into halves, then further down to single entries; append each ns-path that still fails, then a TAB, then the error text, as one line to ${S}/dbx/agentout/${g}.fail.txt. Do not change or guess paths. A PERMISSION_DENIED / NOT_ALLOWED answer for a single file is final: record it and move on.

Do not read file contents, do not call any other Dropbox tool, do not call fetch. When done, run wc -l on both output files and return exactly one line: "${g}: batches=<n> rows=<n> failures=<n>".`,
  { label: `dbx:${g}`, phase: 'Retrieve', effort: 'low' })))
const ok = results.filter(Boolean)
log(`${ok.length}/${groups.length} agents returned`)
return { returned: ok.length, total: groups.length, lines: ok }