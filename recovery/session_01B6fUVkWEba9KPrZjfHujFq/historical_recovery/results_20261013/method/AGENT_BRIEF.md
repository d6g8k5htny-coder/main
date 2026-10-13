# Brief for a historical-recovery search subagent (session 01B6)

1. Read <SCRATCH>/recovery/RULES.md in full. It is the binding rule set (no invention, read-only, classification A-J, confidence levels, exact quotes only from bytes you read).
2. Your search groups are listed by key in your task. Read their entries (gaps + focus) from <SCRATCH>/recovery/groups.json, and the rows for those gap ids from <SCRATCH>/recovery/gaps.json.
3. The gap list is a snapshot of 2026-09-28. If while searching you see a newer GitHub record (after 2026-09-28) that changes a gap's status, note it in that gap's best_evidence with its locator, but keep your work historical.
4. Tools: the corpus helper (python3 -B <SCRATCH>/corpus/cs.py ..., about 1 s per search), git in /home/user/main and <SCRATCH>/repos/<name>, and GitHub via `gh api repos/d6g8k5htny-coder/<repo>/...` (repository-scoped endpoints only; the search API is unavailable). Live Google Drive / Dropbox only if your group key starts with "live-".
5. NEVER write to GitHub, Drive or Dropbox; never comment, push, edit, move or trash. Never print or copy values from <SCRATCH>/dbx/sensitive_values_*.json or any pii_values file. Do not quote personal data (birth dates, family, IDs, addresses, phone numbers, emails other than the owner's Gmail, finances, health) even if it appears in a corpus file; skip such content.
6. Write ONE JSON file: <SCRATCH>/recovery/results/<your agent key>.json, an object of the SEARCH_SCHEMA shape (gap_assessments, fragments, crosswalk_notes) covering all your groups. Validate it parses (python3 -c 'import json;json.load(open(PATH))').
7. Final reply (short, under 300 words): the file path, counts (gaps assessed by class, fragments by confidence), and your top 3 candidates with locators.
