#!/usr/bin/env python3
"""Write audits/dropbox_import/2026-10-24/README.md from SUMMARY.json, PINS.json and the ledger crosswalk summary."""
import json, collections, csv, os, sys
S='<SCRATCH>'
OUT=sys.argv[1] if len(sys.argv)>1 else S+'/dbxpr/build_out/2026-10-24'
d=json.load(open(OUT+'/SUMMARY.json')); pins=json.load(open(OUT+'/PINS.json')); lcs=json.load(open(S+'/dbx/ledger_crosswalk_summary.json'))
T=d['tiers']; g=lambda k: T.get(k,{'files':0,'bytes':0})
mb=lambda b: f'{b/1e6:.1f} MB'
# ledger crosswalk digest from the public LEDGER_CROSSWALK.tsv
lc=list(csv.DictReader(open(OUT+'/LEDGER_CROSSWALK.tsv',encoding='utf-8'),delimiter='\t'))
mp_units={r['ledger_row'] for r in lc if r['missing_proof']=='Y'}
mp_exact={r['ledger_row'] for r in lc if r['missing_proof']=='Y' and r['tier'] in ('GIT_EXACT','ARCHIVE_EXACT')}
mp_dbxonly={r['ledger_row'] for r in lc if r['missing_proof']=='Y' and r['tier'] in ('MISSING','DUP_IN_DROPBOX')}
mp_strong={r['ledger_row'] for r in lc if r['missing_proof']=='Y' and r['match']=='STRONG'}
mi_units={r['ledger_row'] for r in lc if r['missing_impl']=='Y' and r['kind']=='code'}
pending=g('PENDING_HASH')['files']
lines=[]
A=lines.append
A('# Dropbox `Github/` tree against Git, by `content_hash`: 24 October 2026\n')
A('Lane `DBX-MATH-TREE-CONTENTHASH-CROSSWALK-01CZ-20261024`, Anthropic Claude, Claude Code session `session_01Cz7WZybv8znP64SpPj6sWY` (01Cz). Pickup [main#539 / 6387877222](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6387877222). Credit 0. **Scientific effect NONE.** This directory is an inventory of byte identity: which files in the owner\'s Dropbox `Github/` subtree are already in Git or in the two reassembled archives, byte for byte, and which exist only in Dropbox. It moves no status, edits no ledger, graph or register, and copies no file content. It is a successor to the [2026-09-27 import](../2026-09-27/README.md), not a change to it.\n')
A('## What was compared\n')
A(f"- **Dropbox side.** The `Github/` subtree only, listed through the Dropbox connector at {d['listed_at']}: {d['files']:,} files, {mb(d['bytes'])}. For each file the connector\'s `content_hash` was read from link metadata; no file was downloaded. `content_hash` is SHA-256 over the concatenated SHA-256 digests of the file\'s 4 MiB blocks (the rule `tools/dropbox_reconcile.py` implements), so equality with a Git blob\'s hash under the same rule is byte identity. {pending} files are without a hash (`HASH_UNAVAILABLE.tsv` lists the two the connector refused).")
omain=[l.split()[0] for l in open(S+'/dbx/idx/refs_main.txt') if l.strip().endswith('refs/remotes/origin/main')][0][:12]
A(f"- **Git side.** Every blob reachable from any fetched ref of `main` (full history after unshallowing; `origin/main` at `{omain}`) and of `Math-` (`main` at `d1f6ad28` plus 32 PR refs): {pins['git_index']['blobs']:,} blobs, each hashed under the Dropbox rule.")
A(f"- **Archives.** The members of DG-MIGRATION object `a2136bc0` ({pins['archives']['a2136bc0']['members']:,} members, rebuilt from the hardening branch chunk store) and of git blob `c4065c67` ({pins['archives']['c4065c67']['members']} members, the b9c2 proofs zip), hashed the same way.")
A("- **The 2026-09-27 import lists.** `MANIFEST.csv` and `NOT_IMPORTED.csv` of that import, by `dropbox_content_hash`.\n")
A('## Tiers\n')
A('| tier | meaning | files | bytes (rounded to 100 KB) |\n|---|---|---|---|')
for k,m in (('GIT_EXACT','byte-identical to a Git blob (locator repo:path; flag `ON_TIP` when the blob is in the tip tree of `origin/main` or Math- `main`, else `HISTORY_OR_BRANCH_ONLY`)'),('ARCHIVE_EXACT','byte-identical to an archive member, not to a Git blob'),('EMPTY','zero bytes'),('DUP_IN_DROPBOX','in no Git blob or archive; at least one other Dropbox copy of the same bytes'),('NAME_SIZE_DIFF_BYTES','same base name and byte count as a Git or archive file, different bytes'),('MISSING','in no Git blob or archive; a single Dropbox copy'),('SKIP','caches, `.pyc`, `.url` and `.DS_Store`'),('PENDING_HASH','no hash: two files refused link metadata, one personal-looking name was not requested')):
    A(f"| `{k}` | {m} | {g(k)['files']:,} | {mb(g(k)['bytes'])} |")
A('')
A(f"- **Already preserved byte for byte:** {d['exact_in_git_or_archive']['files']:,} files ({d['exact_in_git_or_archive']['distinct_contents']:,} distinct contents) are `GIT_EXACT` or `ARCHIVE_EXACT`.")
do=d['dropbox_only']
A(f"- **Dropbox only:** {do['files']:,} files, {do['distinct_contents']:,} distinct contents, {mb(do['bytes_distinct'])} distinct bytes (`MISSING` plus `DUP_IN_DROPBOX`). By extension (distinct contents): " + ', '.join(f'`{k}` {v}' for k,v in list(do['by_extension_distinct'].items())[:14]) + '.')
A(f"- **Modified on or after the import day (2026-09-27):** {d['modified_on_or_after_2026-09-27']['files']:,} files by `client_modified`; by tier: " + ', '.join(f'`{k}` {v}' for k,v in sorted(d['modified_on_or_after_2026-09-27']['by_tier'].items())) + '. The import README records the day only, so the test is inclusive.')
A(f"- The 2026-09-27 import's hash lists are carried as flags (`IMPORT_LISTED:...`), not as a tier: {d['import_listed_hashes']['files']} files carry one; the published-exact-original rows are `GIT_EXACT` (their bytes are under `audits/dropbox_import/2026-09-27/files/`), the extracted-text and not-imported rows are Dropbox-only here as well.")
A(f"- `GIT_EXACT` splits into {d['git_exact_on_tip']['ON_TIP']} files whose matching blob is in the tip tree of `origin/main` or Math- `main` and {d['git_exact_on_tip']['HISTORY_OR_BRANCH_ONLY']} whose match is only in history or on another fetched ref.")
A(f"- Folders: {len(d['folders']['public_named'])} top-level folder names are printed (each is a path segment of an import-list or Git path, or occurs literally in a tracked file; the reason per folder is in `SUMMARY.json`); {d['folders']['withheld_count']} folders holding {d['folders']['files_in_withheld_folders']} files are labelled `(withheld)`, and archive locators that pass through them are redacted the same way. {d['personal_name_files_counted_only']} files with personal-looking names are counted here and listed nowhere.\n")
A('## Coverage ledger crosswalk\n')
A(f"`LEDGER_CROSSWALK.tsv` name-matches the 344 rows of `data/hypergraph/math_coverage_ledger_20261023.md` (last changed at `{pins['ledger']['last_commit']}`, unchanged through `08c98c8f`) against Dropbox paths: `STRONG` when a file\'s basename stem equals the unit\'s stem, `WEAK` when two consecutive distinctive tokens of the unit occur adjacently below the top-level folder, or one rare or code-like token occurs in the basename (rules in `own/ledger_crosswalk.py`). Name matching is weak evidence; the tier column is the byte evidence.")
A(f"- Of the 131 rows with a `MISSING` proof source, {len(mp_units)} have at least one name-matched Dropbox file ({len(mp_strong)} with a `STRONG` match); {len(mp_exact)} of them have a matched file that is byte-identical to Git or archive content (so the carrier is already preserved under another path), and {len(mp_dbxonly)} have a matched file that exists only in Dropbox.")
A(f"- Of the 48 rows with a `MISSING` implementation, {len(mi_units)} have a name-matched Dropbox code file.")
pk=[l.rstrip('\n').split('\t') for l in open(S+'/dbx/phase3.tsv',encoding='utf-8')][1:]
pk=[r for r in pk if '/GP-FOR-192-v1/research-formal-core-r2/' in r[0] and not r[0].startswith('Github/64_FORMAL_CORE_R2_P02_LM008_LM009 (1)/')]
pkt=collections.Counter(r[4] for r in pk)
A(f"- The Dropbox package `64_FORMAL_CORE_R2_P02_LM008_LM009/.../GP-FOR-192-v1/research-formal-core-r2/` ({len(pk)} files; tiers: " + ', '.join(f'`{k}` {v}' for k,v in sorted(pkt.items())) + ") is a formal-core source tree: the `GIT_EXACT` members are byte-identical to Math- `formal/` files, the others differ from every Git blob, so a revision of the formal core other than the one in Math- is held in Dropbox. Only the rows whose stem equals a member name match it by name (see `LEDGER_CROSSWALK.tsv`, `STRONG`); this is a locator for the formal rows 278–300, not a review of those files.")
A(f"- Hits that fall in `(withheld)` folders are counted per ledger row in `SUMMARY.json` (`ledger_hits_in_withheld_folders`) and not listed per file.\n")
A('## Files\n')
A(f"- `CROSSWALK.tsv`: {d['row_level_rows']:,} per-file rows, only where nothing new is disclosed: the 2026-09-27 import already named the path (then the path is printed), or the bytes are already public (`GIT_EXACT`, `ARCHIVE_EXACT`, `EMPTY`: the hash is the hash of a public blob; then folder label, extension, bytes, `client_modified`, `content_hash`, tier, locator and flags are printed, archive member paths redacted through withheld folders).")
A("- `CROSSWALK_AGGREGATE.tsv`: every other file (Dropbox-only, name-size-different, pending), as counts, distinct contents and bytes by folder label, extension and tier; cells with fewer than three files are merged into an `(other)` row per folder and tier, and a cell still below three files prints its file count only, so no cell describes one file. Every byte total in `SUMMARY.json` and in this README is rounded to the nearest 100 KB, so a withheld cell cannot be recovered by differencing a total against the printed rows and cells. Their per-file hashes, sizes and times are held locally (`PINS.json`) and are released only on an owner or CoS publication decision, because a hash of unpublished content is a confirmation oracle (01Fi's caveat, 539/6387941726).")
A('- `LEDGER_CROSSWALK.tsv`: ledger row, unit, match strength, matched tokens and kind, then the same Dropbox columns, under the same row-level rule; hits on files outside it are counted per ledger row in `SUMMARY.json` (`ledger_hits_aggregated`).')
A('- `SUMMARY.json`: the counts above, by tier and by folder. `PINS.json`: sizes and SHA-256 of the local inputs (inventory, hashes, Git index, archive indexes) and the ledger commit. `HASH_UNAVAILABLE.tsv`: files whose link metadata the connector refused.')
A('- `own/`: the scripts (page merge, phase 1 name and size, batching, hash harvest and collection with its fail-closed receipt, phase 3 tiers, ledger matching, archive indexing, this assembly and README generator). Local roots are replaced by placeholders (`<SCRATCH>`, `<MAIN_CHECKOUT>`, `<SESSION_DIR>`). They do not reproduce the tables on their own: the inventory, the hash set and the Git index are digest-bound in `PINS.json` but not published.')
A('- Not here: the path-level crosswalk of every file (`PINS.json` gives its row count and SHA-256). It is held locally by session 01Cz and offered to the owner or the CoS, who decide what else may be named.\n')
A('## Limits\n')
A('- Byte identity says a file is preserved; it says nothing about which copy is canonical, nor about correctness. Name matching says a file may be relevant; it proves nothing.')
A(f"- Hashes come from connector link metadata, not from downloaded bytes; the `PENDING_HASH` rows are unverified. Each observation is tied to the 18:31Z listing by path and byte count and fails closed on a conflict (`SUMMARY.json` `hash_receipt`: {d['hash_receipt']['observations_accepted']:,} accepted, {d['hash_receipt']['conflicting_paths_dropped']} conflicts dropped); the connector returns no modified time or revision with link metadata, so a same-size edit between the listing and the hash read is not excluded. A published hash is a confirmation oracle; hashes of unpublished content are therefore not in this directory.")
A('- Each Git locator shows up to three paths and one path per blob (the index records one path per blob); `LOCATORS:n` in the flags gives the full count.')
A('- The Git side is every blob reachable from the fetched refs of two repositories (`PINS.json`: full history of `main` after unshallowing, `Math-` main plus its PR refs). The eight other repositories of the account, Drive, and private storage are out of scope, so `MISSING` means not in these two repositories; the Drive connector exposes no checksum field, so the same crosswalk cannot be done for Drive without downloading content.')
A('- Nothing here is a scientific result or a status change. Credit 0. OBL OPEN. eng ≠ discharge. No flags.\n')
open(OUT+'/README.md','w',encoding='utf-8').write('\n'.join(lines))
print('readme', len('\n'.join(lines)), 'bytes')
