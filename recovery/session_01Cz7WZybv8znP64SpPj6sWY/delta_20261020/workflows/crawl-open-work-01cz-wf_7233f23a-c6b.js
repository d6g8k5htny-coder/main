export const meta = {
  name: 'crawl-open-work-01cz',
  description: 'Read-only crawl of all open PRs/issues in six repos: custody state, unclaimed stages, stale claims, independence gaps, 01Cz eligibility',
  phases: [
    { title: 'Crawl', detail: 'one read-only agent per group of open items' },
    { title: 'Synthesize', detail: 'merge into one ledger and ranked actions' },
  ],
}

const NOW = args.now
const MAIN = args.main_head

const RULES = `
You are doing a READ-ONLY crawl for Anthropic Claude Code session session_01Cz7WZybv8znP64SpPj6sWY ("01Cz"). Time anchor: ${NOW}. main HEAD is ${MAIN}.

Hard rules:
- Read only. Use \`gh api\` with repository-scoped endpoints only: repos/d6g8k5htny-coder/<repo>/... (the search API returns 403 here; do not use it). \`--paginate\` fails beyond page 1 through this proxy, so page explicitly with &per_page=100&page=N. For long threads read the LAST pages (compute from the item's .comments count) rather than everything.
- Never post, edit, comment, react, label, review, push, merge, dispatch, re-run, or change anything. Do not download Actions artifacts. Do not open or read anything named 99_DO_NOT_OPEN. Do not copy secrets, tokens, Drive file IDs, private content or long quotes into your output.
- Every agent posts through the shared GitHub account d6g8k5htny-coder, so identify the actual lane from the self-declared performer line in each comment body (e.g. "Actual performer: OpenAI / <model> Astra Pro ...", "Anthropic Claude ... session_01XXXX", "xAI <model>, Cursor cloud agent", "Grok Bot agent N", "Grok Bot Chief of Staff"). Ignore <product> usage-limit notices and "Taking a look!" placeholders except as signals.
- Board protocol vocabulary: PICKUP (claim) / CHECKPOINT / RELEASE / WITHDRAWAL / DELIVERED / LANDED; verdicts PASS, PASS_SCOPED, AMEND, CONFIRMED, PASS_SCOPED_COMPOSITION; HOLD. A stage is "claimed" if a PICKUP exists with no later release/withdrawal/terminal by the same lane; "stale" if claimed with no checkpoint for >2h or past its own stated lease; "unclaimed" if an offer/request/routing exists with no live pickup.
- Independence: same-provider review earns 0 organizational-independence credit. Note when the only review is from the author's provider, or when a record requires a third family ("neither OpenAI nor Anthropic") or a non-Anthropic lane.
- 01Cz eligibility: 01Cz is INELIGIBLE for any stage on an item it authored or reviewed, or that builds on its own records. Known exclusions: main#444 and main#454 (its own retention PRs); main#460 (01Cz did reviews and a composition read); main#358 (never push); Math-#470 lineage (implements 01Cz's proposal); the classic-protection read for main#460 or query-#47 (reserved for dot); anything built on 01Cz's recovery records section 7 or section 9; stages restricted to non-Anthropic lanes or to "neither OpenAI nor Anthropic". Check each item's thread for any comment naming session_01Cz7WZybv8znP64SpPj6sWY or "01Cz" and report it. Otherwise an Anthropic nonauthor lane may be eligible for a genuinely unclaimed bounded stage.

For EACH item return one record. Be factual and terse; cite comment IDs for claims you make. If you could not read something, say so in notes.`

const ITEM = {
  type: 'object',
  properties: {
    repo: { type: 'string' },
    number: { type: 'integer' },
    kind: { type: 'string', enum: ['pr', 'issue'] },
    title: { type: 'string' },
    draft: { type: 'boolean' },
    head_sha8: { type: 'string' },
    mergeable_state: { type: 'string' },
    last_activity: { type: 'string' },
    author_lane: { type: 'string' },
    purpose: { type: 'string', description: 'one line' },
    reviews: { type: 'array', items: { type: 'object', properties: {
      verdict: { type: 'string' }, reviewer_lane: { type: 'string' }, at_head: { type: 'string' }, ref: { type: 'string' } }, required: ['verdict', 'reviewer_lane'] } },
    stages: { type: 'array', items: { type: 'object', properties: {
      stage: { type: 'string' },
      status: { type: 'string', enum: ['unclaimed', 'claimed', 'stale', 'blocked', 'done', 'held', 'owner-decision'] },
      holder: { type: 'string' },
      since: { type: 'string' },
      ref: { type: 'string' },
      restriction: { type: 'string', description: 'eligibility restriction stated in the record, if any' } },
      required: ['stage', 'status'] } },
    independence_gap: { type: 'string' },
    names_01cz: { type: 'string', description: 'comment ids mentioning 01Cz, or empty' },
    eligible_for_01cz: { type: 'array', items: { type: 'object', properties: {
      stage: { type: 'string' }, why: { type: 'string' }, needs_post: { type: 'boolean' } }, required: ['stage', 'why'] } },
    ineligible_reason: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['repo', 'number', 'kind', 'title', 'last_activity', 'stages'],
}
const GROUP = { type: 'object', properties: { items: { type: 'array', items: ITEM }, coverage_notes: { type: 'string' } }, required: ['items'] }

const GROUPS = args.groups
phase('Crawl')
const results = await parallel(GROUPS.map(g => () => agent(
  `${RULES}\n\nYour group (${g.label}): repo d6g8k5htny-coder/${g.repo}, ${g.kind}s numbered ${g.numbers.join(', ')}.${g.extra ? '\n' + g.extra : ''}\nFor PRs also read pulls/N (head, draft, mergeable_state) and pulls/N/reviews. Return one record per item in "items".`,
  { label: `crawl:${g.label}`, phase: 'Crawl', schema: GROUP })))

const got = results.filter(Boolean)
const items = got.flatMap(r => r.items || [])
const coverage = got.map((r, i) => `${GROUPS[i] ? GROUPS[i].label : i}: ${r.coverage_notes || ''}`).join('\n')
const expected = GROUPS.reduce((n, g) => n + g.numbers.length, 0)
log(`crawled ${items.length}/${expected} items across ${got.length}/${GROUPS.length} groups`)

phase('Synthesize')
const report = await agent(
  `You are synthesizing a read-only crawl for Anthropic Claude Code session 01Cz (session_01Cz7WZybv8znP64SpPj6sWY). Time anchor ${NOW}; main HEAD ${MAIN}. Do not call any tools that write; you may re-read GitHub (repository-scoped gh api, explicit paging) only to resolve a direct contradiction between records.\n\nInput: ${items.length} item records (expected ${expected}) as JSON:\n${JSON.stringify(items)}\n\nCoverage notes:\n${coverage}\n\nWrite a concise Markdown ledger with these sections:\n1. Coverage: counts per repo, items missing or unreadable.\n2. Open, unclaimed bounded stages (repo#N, stage, restriction, ref), most actionable first.\n3. Stale or lapsed claims (holder, since, ref).\n4. Independence gaps: items whose only review is same-provider where independence matters, and items needing a third family or a non-Anthropic lane.\n5. Crossed or duplicate claims still unresolved.\n6. Owner/coordinator decisions pending (dot or Dylan).\n7. Stages 01Cz is eligible for, ranked by value, each with why it is eligible, what exclusions were checked, and whether it needs a GitHub post (01Cz's GitHub writes have been denied by its permission classifier since 20 Oct; note that).\n8. Every item mentioning 01Cz.\nKeep each line short; cite repo#N and comment ids. No Drive IDs, secrets or long quotes. Return only the Markdown.`,
  { label: 'synthesize', phase: 'Synthesize' })

return { report, items, coverage }
