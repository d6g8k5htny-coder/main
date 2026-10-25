export const meta = {
  name: 'dbx-root-hash-metadata',
  description: 'Read Dropbox content_hash from link metadata for the files outside Github/ (33 batches of 25 ns_paths), writing local TSVs; no download',
  phases: [{ title: 'Hash', detail: 'eleven agents, three batches each; download_link metadata only' }],
}
const D = '<SCRATCH>/dbxroot'
const SCHEMA = { type: 'object', properties: { batches_done: { type: 'array', items: { type: 'string' } }, rows_written: { type: 'number' }, failures: { type: 'number' }, notes: { type: 'string' } }, required: ['batches_done', 'rows_written', 'failures', 'notes'] }
const groups = []
for (let i = 0; i < 33; i += 3) groups.push([i, i + 1, i + 2].filter(x => x < 33).map(x => 'b' + String(x).padStart(3, '0')))
phase('Hash')
const results = await parallel(groups.map((g, gi) => () => agent(`You are reading Dropbox file METADATA (content_hash) for ${g.length} batch files, read-only. Do not download, fetch, open, preview or HEAD any download_url; do not move, share or edit any file; do not post anywhere; do not run git. Never copy a Dropbox namespace id (ns:<digits>), file id, link or file name into your final return value; write them only into the TSV files named below.

Steps:
1. Call ToolSearch with query "select:mcp__Dropbox__download_link" to load the tool.
2. For each batch file in this order: ${g.map(b => D + '/batches/' + b + '.json').join(', ')} — read it (a JSON list of up to 25 ns_path strings) and call mcp__Dropbox__download_link once with {"entries": <that list>, "expiration_in_sec": 60}. The response lists entries with path, size and content_hash (and a download_url you must ignore and never request). If the response is too large for inline display the harness saves it to a JSON file whose path the result names; parse that file with a short python3 -I script instead of retyping.
3. For every returned entry append one line to ${D}/parts/hash_<batch>.tsv (tab-separated, no header): path <TAB> size <TAB> content_hash, copied exactly from the response (the path exactly as returned, size as the integer, content_hash as the 64 hex chars). Use one Bash call per batch to write the lines (python3 -I heredoc or printf). Never invent or alter a value.
4. If a batch call fails as a whole (for example a NOT_ALLOWED or not-found error for one entry voids the batch), retry that batch as individual single-entry download_link calls; write the successful ones to the same TSV and append each failing ns_path with the error text to ${D}/parts/hash_<batch>.fail.txt.
5. When done, run: wc -l ${D}/parts/hash_*.tsv for your batches and report the total rows.

Return JSON: batches_done (the batch names you completed), rows_written (total TSV lines you wrote), failures (number of ns_paths recorded in .fail.txt files), notes (one or two sentences: errors, retries, oversized responses; no ids, names or links).`, { label: `hash:${g[0]}-${g[g.length - 1]}`, phase: 'Hash', schema: SCHEMA, effort: 'low' })))
const ok = results.filter(Boolean)
log(`${ok.length}/${groups.length} groups; rows ${ok.reduce((s, r) => s + (r.rows_written || 0), 0)}; failures ${ok.reduce((s, r) => s + (r.failures || 0), 0)}`)
return { groups: ok, missing: groups.filter((g, i) => !results[i]).map(g => g[0]) }