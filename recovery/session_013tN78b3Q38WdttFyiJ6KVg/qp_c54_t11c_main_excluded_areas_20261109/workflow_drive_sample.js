export const meta = {
  name: 't11c-drive-sample',
  description: 'T11c: metadata-only existence check of 116 sampled Drive ids from main drive/ mirror manifests (no content, no owner data)',
  phases: [{ title: 'Drive', detail: 'two agents, 58 ids each' }],
}
const T = '/tmp/claude-0/-home-user-main/d5c985ca-97e5-58d8-a9d4-1e0a19065a22/scratchpad/t11c'
const DRIVE = { type: 'object', properties: { checked: { type: 'array', items: { type: 'object', properties: {
  index: { type: 'integer' }, drive_id: { type: 'string', pattern: '^[A-Za-z0-9_-]{10,80}$' },
  found: { type: 'boolean' }, trashed: { type: 'boolean' }, title_matches_path: { type: 'boolean' },
  mime_type: { type: 'string', pattern: '^[a-z0-9.+/-]{0,80}$' }, file_size: { type: 'string', pattern: '^[0-9]{0,15}$' },
  modified_time: { type: 'string', pattern: '^[0-9T:.Z-]{0,30}$' } },
  required: ['index', 'drive_id', 'found', 'trashed', 'title_matches_path', 'mime_type', 'file_size', 'modified_time'] } } },
  required: ['checked'] }
const prompt = (a, b) => `Read ${T}/drive_sample.json. Its 'sample' list has entries {lane, drive_id, path}. Take the entries with 0-based index ${a} to ${b - 1} inclusive. For each, call the Google Drive MCP tool mcp__Google_Drive__get_file_metadata (load it first with ToolSearch query "select:mcp__Google_Drive__get_file_metadata") with fileId = drive_id, excludeContentSnippets = true and includeRichMetadata = false. Never call any other Drive tool, never download, preview or read file content. Record per entry only: index, drive_id, found (the call returned the file), trashed (as reported; false if not found), title_matches_path (the returned title equals the last path component of 'path' after removing a trailing '.export.txt', '.export.md' or similar export suffix, or equals it exactly), mime_type, file_size (digits only; '' if absent) and modified_time. The tool output contains owner and account data: NEVER copy any owner, email, user, sharing, permission or URL value into your output or anywhere else. Return exactly ${b - a} entries.`
const out = await parallel([[0, 58], [58, 116]].map(([a, b]) => () => agent(prompt(a, b), { label: `drive:${a}-${b - 1}`, phase: 'Drive', schema: DRIVE })))
return { drive: out }
