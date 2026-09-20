#!/usr/bin/env python3
"""Checker for the collision proposals under registers/.

Three dated proposals exist, and none edits an earlier one (CLAUDE.md rule 8):

  registers/collision_proposal.json            (2026-09-18) — 16 records over the 'findings'
      section of registers/KNOWN_FINDINGS.json; source of record is the 2026-09-17 markdown
      export, and every colliding row is quoted as a byte-exact 1-based LINE of that export.
  registers/collision_proposal_2026-09-19.json (2026-09-19) — 7 records over the
      'findings_first_visible_in_2026-09-18_export' section; source of record is the
      2026-09-18 xlsx export, and every colliding row is quoted as the exact list of cell
      strings registers/json/<tab>.json holds (the markdown export has no line for them).
  registers/collision_proposal_2026-09-19b.json (2026-09-19, third) — 14 records over the
      'findings_first_keyed_2026-09-19' section (relations, review_ledger, definitions);
      xlsx mode like the second; successor_of names the second document, which itself
      names the first, so the predecessor chain is two deep.

A proposal is a PROPOSAL: it disambiguates structural defects without repairing anything.
This checker enforces that promise mechanically, for whichever proposal --proposal names:

  1. bijection  — every finding of the named KNOWN_FINDINGS section has exactly one proposal
                  record, and every record names a real finding of that section;
  2. additive   — every executable operation is an append; deletion, merge and overwrite
                  verbs are rejected wherever they appear, and the non-additive follow-ups a
                  full remedy would need are quarantined in a separate, operator-reserved
                  field that carries no executable op;
  3. unchanged  — registers/source, registers/json and registers/csv are byte-identical to
                  git HEAD (the proposal must not have edited exported data), and the export
                  digest recorded in the proposal matches the file on disk;
  4. successors — every proposed successor identifier is unique within the proposal, does not
                  already exist anywhere in registers/json/, and (for a successor proposal)
                  is not one ANY document in its predecessor chain already issued: successor_of
                  may name a document that itself carries a successor_of, the chain is walked
                  to its root, every document on it must match its recorded sha256 and byte
                  count on disk (they are frozen), and a chain that revisits a document fails;
  5. honesty    — independence credit is recorded at 0 with its reason, the
                  independence-requiring gates are stated to remain open, every record states
                  what it does not establish, every quoted row matches its source exactly
                  (line for line, or cell for cell, by source_of_record.kind), and the
                  Markdown companion named by the document's 'companion' field carries the
                  banner '**Nothing has been repaired.**' near its head — anchored, because a
                  companion that quotes register text ending '...; nothing has been repaired.'
                  would satisfy a bare substring search while its own banner said the opposite
                  — and quotes every finding key.  A document may also transcribe each
                  finding's text into its record ('finding_text_as_recorded') and blockquote it
                  in the companion under an explicit attribution to KNOWN_FINDINGS.json; that
                  transcription is then held to its source byte for byte, the companion's copy
                  is held to the record's, the attribution count must equal the number of
                  records transcribing, and a document that transcribes one finding must
                  transcribe every one.  Equality of the strings is all this establishes: the
                  finding text itself may be wrong, and where it is, the records say so and
                  repair neither it nor the register;
  6. bound      — (xlsx mode only) each record is bound to the finding it disambiguates:
                  the finding key '<tab>: duplicate key '<id>' at rows A and B' must name
                  the record's register_tab, its colliding_identifier and exactly the two
                  distinct rows quoted; the key cell of both live rows must equal the
                  colliding identifier; fields_identical_in_both_rows, field_differences
                  and cell_comparison are recomputed from the live cells and must match
                  what the record says; exact_duplicate_row must equal rowA == rowB;
                  both_rows_cite_one_drive_object and the defect class must agree with the
                  identity cells the per-tab rule reads (DRIVE_OBJECT_RULE: 'Source' for the
                  Artifact Index; 'Source URL' and 'Drive ID' for the Evidence Lineage;
                  'Source URL' for the Relation Index and the Definition Registry; 'Exact
                  Object ID' for the Review Independence ledger, which carries no URL); for
                  the Relation Index the class must further say SAME_RELATION exactly when
                  the Source object, Relation type and Target object cells all agree, and
                  target_url_cells_agree, when recorded, must equal what the Target URL cells
                  give; identity_cells_cited, when recorded, must equal the rule's cells;
                  the keeper must be the earlier of the two rows and the successor the
                  later; and the successor id must be
                  '<id>@<TAB-LOCATOR>-R<successor row>' for the record's tab.  Proposed
                  Duplicate Flags cluster ids are held to the same uniqueness rule as
                  successor ids, and a proposal over any section other than 'findings'
                  must name its frozen predecessor in successor_of.  Further, in xlsx mode:
                  row_canonical_bytes must equal the byte length of the live row's canonical
                  serialisation; workbook_tab must be the workbook's name for register_tab
                  (WORKBOOK_TAB, which is itself checked against the sheet-name map of
                  tools/registers_import.py at call time);
                  every number word in a record's materiality ('nine of twelve cells differ',
                  'eight cells agree') must equal the recomputed cell counts; the document-level
                  summary_counts and the classification lists must equal what the records'
                  cells give (nothing at document level is typed, everything is recomputed;
                  the same-object bucket may be named by the legacy key
                  '..._different_status_text' or the neutral key '..._different_cells', and a
                  document whose chain is two or more deep totals the whole chain under
                  'findings_covered_by_the_chain_together' instead of the two-document key);
                  successor_of.successor_ids_issued_by_predecessor must equal the ids the
                  direct predecessor file actually issues; source_of_record.row_locators, the
                  document's own statement of which locator each tab's successor ids carry,
                  must agree with TAB_LOCATOR for every tab (the ids are held to TAB_LOCATOR,
                  so an unchecked declaration would let the document describe itself falsely);
                  and the Markdown companion must quote
                  every live row's canonical JSON, its 'rows[i], canonical N bytes, SHA-256'
                  line, its record's 'Cell count' line and materiality verbatim, and carry
                  exactly one summary-table row per record naming its rows, keeper,
                  reidentified row and successor id.
  7. followups  — (all modes) every APPEND_ROW targets the Duplicate Flags registry, and a
                  REIDENTIFY_KEY_CELL follow-up rewrites exactly the colliding identifier
                  ('from') to an identifier the same record proposes ('to'), so the
                  operator-reserved field can name nothing the proposal did not issue.
  8. prose      — (xlsx mode only) the repository's own prose about these collisions is held
                  to the same live cells the records are: in registers/README.md and
                  docs/FINDINGS_2026-09-18.md, an 'exact duplicate' phrase attributed to a
                  colliding identifier whose record recomputed exact_duplicate_row=false must
                  be a denial ('not an exact duplicate row'), and an 'N of T cells differ/agree'
                  phrase attributed to one must equal that record's recomputed counts.  A
                  phrase is attributed to the nearest colliding identifier named before it in
                  the same sentence; a phrase in a sentence that names none, or before any
                  mention in it, is not attributed and not checked.  This is a contradiction
                  guard over prose, not a proof that the prose is right, and it establishes
                  nothing about any register cell, status or claim.  A prose file that is
                  absent (a sandbox holding only registers/ and tools/) is skipped with a note.

Exit status is nonzero if any check fails.
Run:  python3 tools/collision_proposal_check.py [--proposal PATH] [--section NAME] [-v]
      --proposal defaults to registers/collision_proposal.json (unchanged behaviour).
      --section defaults to the document's own 'findings_source_section', else 'findings';
      naming a section the document does not claim is itself a failure.
Every path is resolved when main() runs, never at import time.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROPOSAL = os.path.join("registers", "collision_proposal.json")
KNOWN_REL = os.path.join("registers", "KNOWN_FINDINGS.json")
JSON_DIR_REL = os.path.join("registers", "json")
MD_EXPORT_REL = os.path.join("registers", "source", "GP-REG-032_v1.2_export_2026-09-17.md")

# Backward-compatible module names (not read by the checks; kept for callers that import them).
JSON_PATH = os.path.join(ROOT, DEFAULT_PROPOSAL)
MD_PATH = os.path.join(ROOT, "registers", "COLLISION_PROPOSAL.md")
KNOWN_PATH = os.path.join(ROOT, KNOWN_REL)
JSON_DIR = os.path.join(ROOT, JSON_DIR_REL)
EXPORT = os.path.join(ROOT, MD_EXPORT_REL)

# Operations the proposal is allowed to contain.  Anything else is a mutation.
ADDITIVE_OPS = {"APPEND_ROW"}
# Follow-ups a full remedy would eventually need.  They are recorded, never proposed
# for execution, and must be flagged operator_reserved.
RESERVED_OPS = {"REIDENTIFY_KEY_CELL", "AMEND_PROTOCOL_TABLE"}
# Verbs that would destroy or silently merge recorded state.  OP-CNS-001 §2 requires
# collisions to be preserved and disambiguated; OP-PROT-019 §6 forbids deletion
# outright ("No permanent deletion in this workflow").
FORBIDDEN_VERBS = ("DELETE", "REMOVE", "MERGE", "PURGE", "OVERWRITE", "REPLACE",
                   "DROP", "TRUNCATE", "CLEAR", "DEDUPE", "DEDUPLICATE", "COLLAPSE")
GUARDED_PATHS = ("registers/source", "registers/json", "registers/csv")
TOKEN_SPLIT = re.compile(r"[\s;,|]+")
VERBATIM_KINDS = {"markdown_export_lines", "xlsx_export_json_rows"}
# The finding keys tools/registers_check.py prints for a duplicate primary key.
DUP_KEY_FINDING = re.compile(r"^(?P<tab>[a-z0-9_]+): duplicate key '(?P<key>[^']+)' at rows (?P<a>\d+) and (?P<b>\d+)$")
# Tab-scoped row-locator abbreviations a successor id must carry ('<id>@<abbr>-R<row>').
TAB_LOCATOR = {"artifact_index": "AIDX", "evidence_lineage": "EVL",
               "relations": "REL", "review_ledger": "RVL", "definitions": "DEF"}
# The workbook's own sheet name for each JSON tab a successor record may quote.  The JSON
# 'tab' field is the importer's machine name; the sheet name is the left column of
# tools/registers_import.py SHEETS, and check_workbook_tab_map() fails if this map and that
# one disagree, so a renamed sheet cannot be quoted under a stale name.
WORKBOOK_TAB = {"artifact_index": "Artifact Index", "evidence_lineage": "Evidence Lineage",
                "relations": "Relation Index", "review_ledger": "Review Independence",
                "definitions": "Definition Registry"}
# The identity cells the drive-object agreement rule reads, per tab.  Two rows 'cite one
# Drive object' exactly when every listed cell agrees whole-cell (no normalisation).  The
# first listed column is the one drive_source_cited_by_both_rows / drive_sources_cited quote
# when it is a URL column (URL_COLUMNS); a tab whose first column is not a URL (the Review
# Independence ledger names its object by Exact Object ID and carries no URL) must record
# drive_source_cited_by_both_rows as null.  For the Relation Index the rule reads the Source
# URL — the Drive object the relation is asserted from — and the Target URL cells are
# compared separately (target_url_cells_agree) and reported, never folded into the flag.
DRIVE_OBJECT_RULE = {
    "artifact_index": ("Source",),
    "evidence_lineage": ("Source URL", "Drive ID"),
    "relations": ("Source URL",),
    "review_ledger": ("Exact Object ID",),
    "definitions": ("Source URL",),
}
URL_COLUMNS = ("Source URL", "Source")
# The cells that identify the relation a Relation Index row describes; their agreement is
# what the SAME_RELATION class token asserts.
RELATION_TRIPLE = ("Source object", "Relation type", "Target object")
# Backward-compatible name (not read by the checks; kept for callers that import it).
SOURCE_COLUMNS = URL_COLUMNS
DUPLICATE_FLAGS_TAB = "duplicate flags"
# Document-level classification lists a successor proposal carries, by key prefix; the three
# list keys are recomputed from the records' cells and must match exactly.  The same-object
# bucket may be named by the legacy key (the 2026-09-19 document, whose same-object pairs
# all differ in Status) or by the neutral key (a document whose same-object pair differs in
# cells other than Status must not call the difference 'status text').
CLASSIFICATION_KEY_PREFIX = "classification_of_the_"
CLASS_SAME = "same_object_different_status_text"
CLASS_SAME_NEUTRAL = "same_object_different_cells"
CLASS_SAME_KEYS = (CLASS_SAME, CLASS_SAME_NEUTRAL)
CLASS_DIFFERENT = "different_objects_one_identifier"
CLASS_EXACT = "exact_duplicate_rows"
SUMMARY_SAME_KEYS = ("same_drive_object_different_status_text", "same_drive_object_different_cells")
SUMMARY_TWO_DOCS_KEY = "findings_covered_by_both_proposals_together"
SUMMARY_CHAIN_KEY = "findings_covered_by_the_chain_together"
# A predecessor chain longer than this is not a chain of frozen proposals but a loop or a
# mistake; the check stops and fails rather than following it.
CHAIN_MAX_DEPTH = 16
# Number words prose may use for cell counts; any other word before 'cells differ/agree'
# is not a count and is not checked.  Hyphenated words above twenty are numbers too.
NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
                "eighteen": 18, "nineteen": 19, "twenty": 20, "twenty-one": 21, "twenty-two": 22,
                "twenty-three": 23, "twenty-four": 24, "twenty-five": 25, "twenty-six": 26,
                "twenty-seven": 27, "twenty-eight": 28, "twenty-nine": 29, "thirty": 30}
CELL_COUNT_PROSE = re.compile(r"\b(?:([\w-]+) of )?([\w-]+) cells (differ|agree)\b", re.IGNORECASE)
# Repository prose (not an export, not the Markdown companion) that discusses these
# collisions.  It is checked, never written, by this tool; a file that is not present is
# skipped.  registers/README.md is the register directory's own description of the keying
# findings; docs/FINDINGS_2026-09-18.md carries the audit narrative.
PROSE_FILES = (os.path.join("registers", "README.md"),
               os.path.join("docs", "FINDINGS_2026-09-18.md"))
# A document may transcribe each finding's text from registers/KNOWN_FINDINGS.json into its
# record ('finding_text_as_recorded') and blockquote it in the companion under an explicit
# attribution.  A transcription presented as a quotation is held to its source byte for byte,
# and a document that transcribes one finding must transcribe every one: a field quietly
# dropped would leave the companion attributing to the register text no record carries.  A
# document that transcribes none (the two frozen predecessors) is skipped with a note.
FINDING_TEXT_FIELD = "finding_text_as_recorded"
COMPANION_FINDING_TEXT_ATTRIBUTION = "Finding text as recorded in `KNOWN_FINDINGS.json`:"
# The companion's nothing-was-repaired banner, required in the emphatic form and near the head
# of the document.  A bare substring search over the whole file is satisfiable by quoted
# material -- every transcribed finding text ends '...; nothing has been repaired.' -- so the
# banner the reader meets first is anchored instead of merely present.
NOTHING_REPAIRED_BANNER = "**Nothing has been repaired.**"
NOTHING_REPAIRED_BANNER_WITHIN_LINES = 40
# Prose is read sentence by sentence over whitespace-flattened text.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
EXACT_DUPLICATE_PROSE = re.compile(r"\bexact duplicate\b", re.IGNORECASE)
# A record with exact_duplicate_row=false licenses only a denial.  Any of these tokens in
# the characters immediately before the phrase makes the mention a denial ('not an exact
# duplicate row', "isn't the exact duplicate the finding text calls it").
PROSE_NEGATORS = ("not ", "n't ", "never ", "no ", "rather than ", "instead of ")
PROSE_NEGATION_WINDOW = 48


class Result:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def check(self, ok: bool, msg: str) -> bool:
        if not ok:
            self.failures.append(msg)
        return ok

    def note(self, msg: str) -> None:
        self.notes.append(msg)


class Paths:
    """Every filesystem location the checks read, resolved from a root at call time."""

    def __init__(self, root: str, proposal_rel: str) -> None:
        self.root = root
        self.proposal = os.path.join(root, proposal_rel)
        self.known = os.path.join(root, KNOWN_REL)
        self.json_dir = os.path.join(root, JSON_DIR_REL)
        self.md_export = os.path.join(root, MD_EXPORT_REL)

    def rel(self, rel: str) -> str:
        return os.path.join(self.root, rel)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def canonical_row_bytes(row: list) -> bytes:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def canonical_row_sha256(row: list) -> str:
    return hashlib.sha256(canonical_row_bytes(row)).hexdigest()


def record_proposed_ids(p) -> set[str]:
    """Every identifier one record proposes: successor.proposed_id and successors[].proposed_id."""
    out: set[str] = set()
    s = p.get("successor")
    if isinstance(s, dict) and s.get("proposed_id"):
        out.add(s["proposed_id"])
    for x in p.get("successors", []) or []:
        if isinstance(x, dict) and x.get("proposed_id"):
            out.add(x["proposed_id"])
    return out


def existing_identifiers(json_dir: str) -> set[str]:
    """Every string that already names something in registers/json/: whole cell values
    and whitespace/semicolon/comma/pipe-delimited tokens inside them, plus headers and
    tab names.  Matching is on exact values, never substrings, so a successor that
    merely *contains* an existing id (e.g. 'TR-P01-006-COLLISION-PROVENANCE' contains
    'TR-P01-006') is correctly treated as new.

    json_dir is required and is never defaulted to the module-level ROOT: this is the one
    function whose job is to prove a proposed identifier is new, and a caller that silently
    read the real repository while checking a sandbox would prove it against the wrong tree
    (CLAUDE.md records exactly that hazard neutering the claims_check mutation tests)."""
    if not json_dir:
        raise ValueError("existing_identifiers() requires an explicit registers/json directory; "
                         "it must never fall back to a path resolved at import time")
    out: set[str] = set()
    for fn in sorted(os.listdir(json_dir)):
        if not fn.endswith(".json"):
            continue
        t = load_json(os.path.join(json_dir, fn))
        out.add(t.get("tab", ""))
        for cell in list(t.get("header", [])) + [c for r in t.get("rows", []) for c in r]:
            cell = (cell or "").strip()
            if not cell:
                continue
            out.add(cell)
            for tok in TOKEN_SPLIT.split(cell):
                tok = tok.strip(" \t'\"()[]{}<>.:")
                if tok:
                    out.add(tok)
    out.discard("")
    return out


def walk_strings(node, path="$"):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk_strings(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk_strings(v, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


def resolve_section(doc, requested: str | None, res: Result) -> str:
    claimed = doc.get("findings_source_section")
    if requested is None:
        return claimed or "findings"
    if claimed is not None:
        res.check(claimed == requested,
                  f"--section {requested!r} disagrees with the document's own "
                  f"findings_source_section {claimed!r}")
    elif requested != "findings":
        res.check(False, f"--section {requested!r} but the document claims no section "
                         f"(it is checked over 'findings')")
    return requested


def check_bijection(doc, paths: Paths, section: str, res: Result) -> None:
    known_doc = load_json(paths.known)
    res.check(section in known_doc, f"KNOWN_FINDINGS.json has no section {section!r}")
    known = set(known_doc.get(section, {}) or {})
    proposed: list[str] = [p.get("finding_key", "") for p in doc.get("proposals", [])]
    seen: set[str] = set()
    for k in proposed:
        res.check(k in known, f"proposal names a finding absent from KNOWN_FINDINGS.json "
                              f"section {section!r}: {k!r}")
        res.check(k not in seen, f"two proposal records claim the same finding key: {k!r}")
        seen.add(k)
    for k in sorted(known - seen):
        res.check(False, f"KNOWN_FINDINGS finding has no proposal record: {k!r}")
    res.check(len(proposed) == len(known),
              f"record count {len(proposed)} != finding count {len(known)} in section {section!r}")


def records_transcribing_finding_text(doc) -> list[dict]:
    """The records that carry a transcription of their finding's text."""
    return [p for p in doc.get("proposals", []) if FINDING_TEXT_FIELD in p]


def check_finding_text_verbatim(doc, paths: Paths, section: str, res: Result) -> None:
    """A record's 'finding_text_as_recorded' is presented to the reader as the register's own
    words, so it is checked against them: every record of a transcribing document must carry
    the field, and each must equal registers/KNOWN_FINDINGS.json's entry for that record's
    finding key byte for byte.  This establishes only that the two strings are equal; it says
    nothing about whether the finding text itself is correct (for REL-EC021-CLS141 the records
    themselves record that it is not, and repair neither the text nor the register)."""
    records = doc.get("proposals", [])
    carrying = records_transcribing_finding_text(doc)
    if not carrying:
        res.note(f"no record carries {FINDING_TEXT_FIELD}: no finding text is transcribed by this "
                 "document, so none is checked")
        return
    known = load_json(paths.known).get(section, {}) or {}
    for p in records:
        rid = p.get("record_id", "?")
        key = p.get("finding_key", "")
        if FINDING_TEXT_FIELD not in p:
            res.check(False, f"{rid}: {FINDING_TEXT_FIELD} is missing although {len(carrying)} of "
                             f"{len(records)} records of this document transcribe the finding text "
                             "(a document that quotes the register quotes it for every record)")
            continue
        want = known.get(key)
        if not res.check(want is not None,
                         f"{rid}: {FINDING_TEXT_FIELD} cannot be checked: finding key {key!r} is not in "
                         f"registers/KNOWN_FINDINGS.json section {section!r}"):
            continue
        res.check(p[FINDING_TEXT_FIELD] == want,
                  f"{rid}: {FINDING_TEXT_FIELD} is not registers/KNOWN_FINDINGS.json section "
                  f"{section!r} entry {key!r} byte for byte")


def check_declared_row_locators(doc, res: Result) -> None:
    """source_of_record.row_locators is the document's own statement of the locator each tab's
    successor identifiers carry.  The identifiers themselves are held to TAB_LOCATOR, so a
    document declaring a different locator would describe itself falsely while still passing;
    the declaration is therefore checked against the same map, for every tab a record names."""
    src = doc.get("source_of_record") or {}
    declared = src.get("row_locators")
    if not isinstance(declared, dict):
        res.note("source_of_record declares no row_locators map: nothing to cross-check")
        return
    for tab in sorted({p.get("register_tab", "") for p in doc.get("proposals", [])}):
        want = TAB_LOCATOR.get(tab)
        if want is None:
            continue
        res.check(declared.get(tab) == want,
                  f"source_of_record.row_locators[{tab!r}] is {declared.get(tab)!r} but the successor "
                  f"identifiers of that tab are checked against {want!r}")
    for tab, got in sorted(declared.items()):
        if tab in TAB_LOCATOR:
            res.check(got == TAB_LOCATOR[tab],
                      f"source_of_record.row_locators[{tab!r}] is {got!r}, not the locator this checker "
                      f"reads for that tab ({TAB_LOCATOR[tab]!r})")


def check_operations(doc, res: Result) -> None:
    for p in doc.get("proposals", []):
        rid = p.get("record_id", "?")
        ops = p.get("operations")
        res.check(isinstance(ops, list) and len(ops) >= 1,
                  f"{rid}: no operations block")
        for i, op in enumerate(ops or []):
            name = op.get("operation", "")
            res.check(name in ADDITIVE_OPS,
                      f"{rid}: operation[{i}] {name!r} is not additive "
                      f"(allowed: {sorted(ADDITIVE_OPS)})")
            res.check(op.get("append_only") is True,
                      f"{rid}: operation[{i}] does not declare append_only")
            res.check(op.get("mutates_existing_rows") is False,
                      f"{rid}: operation[{i}] does not declare mutates_existing_rows=false")
            res.check(isinstance(op.get("row"), list) and len(op["row"]) == len(op.get("header", [])),
                      f"{rid}: operation[{i}] row width does not match its header")
            # The only surface an append may land on is the workbook's collision registry;
            # an append aimed at a content tab would also escape the cluster-id checks.
            res.check(DUPLICATE_FLAGS_TAB in str(op.get("target_tab", "")).lower(),
                      f"{rid}: operation[{i}] APPEND_ROW must target the Duplicate Flags registry, "
                      f"not {op.get('target_tab')!r}")
            for verb in FORBIDDEN_VERBS:
                res.check(verb not in name.upper(),
                          f"{rid}: operation[{i}] name contains forbidden verb {verb}")
        proposed = record_proposed_ids(p)
        ident = p.get("colliding_identifier")
        for i, fu in enumerate(p.get("non_additive_followups", []) or []):
            name = fu.get("op", "")
            res.check(name in RESERVED_OPS,
                      f"{rid}: non_additive_followups[{i}] {name!r} is not a recognised "
                      f"operator-reserved action (allowed: {sorted(RESERVED_OPS)})")
            if name == "REIDENTIFY_KEY_CELL":
                # The reserved rewrite is bound to the record: it renames exactly the colliding
                # identifier, and only to an identifier this record proposes (which the
                # uniqueness checks already cover).
                if ident is not None:
                    res.check(fu.get("from") == ident,
                              f"{rid}: non_additive_followups[{i}] 'from' {fu.get('from')!r} is not the "
                              f"colliding identifier {ident!r}")
                res.check(fu.get("to") in proposed,
                          f"{rid}: non_additive_followups[{i}] 'to' {fu.get('to')!r} is not an identifier "
                          f"this record proposes {sorted(proposed)}")
            res.check(fu.get("operator_reserved") is True,
                      f"{rid}: non_additive_followups[{i}] is not marked operator_reserved")
            res.check(bool(fu.get("why_not_in_operations")),
                      f"{rid}: non_additive_followups[{i}] does not say why it is excluded "
                      f"from the executable operations")
            res.check(name not in ADDITIVE_OPS,
                      f"{rid}: non_additive_followups[{i}] leaks into the additive set")
            for verb in FORBIDDEN_VERBS:
                res.check(verb not in name.upper(),
                          f"{rid}: non_additive_followups[{i}] contains forbidden verb {verb}")
    # A deletion must not hide anywhere in a machine-readable action field.
    for path, val in walk_strings(doc, "$"):
        if path.endswith(".operation") or path.endswith(".op"):
            for verb in FORBIDDEN_VERBS:
                res.check(verb not in val.upper(),
                          f"forbidden verb {verb} in action field {path}: {val!r}")


def check_registers_unchanged(paths: Paths, res: Result) -> None:
    """registers/source, registers/json and registers/csv must be byte-identical to HEAD."""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--"] + list(GUARDED_PATHS),
                             cwd=paths.root, capture_output=True, text=True, timeout=60)
        git_ok = out.returncode == 0
    except (OSError, subprocess.SubprocessError):
        git_ok = False
        out = None
    if git_ok and out is not None:
        dirty = [l for l in out.stdout.splitlines() if l.strip()]
        for l in dirty:
            res.check(False, f"exported register data is not byte-unchanged against git HEAD: {l.strip()}")
        if not dirty:
            res.note("registers/source, registers/json, registers/csv byte-unchanged against git HEAD")
    else:
        res.note("git unavailable: HEAD comparison skipped; falling back to re-import equality")
        try:
            importer = os.path.join(paths.root, "tools", "registers_import.py")
            rc = subprocess.run([sys.executable, importer, "--check"],
                                cwd=paths.root, capture_output=True, text=True, timeout=300)
            res.check(rc.returncode == 0,
                      "registers_import.py --check failed: registers/json or registers/csv "
                      "no longer match registers/source")
        except Exception as exc:  # pragma: no cover - environment dependent
            res.check(False, f"could not verify exported registers are unchanged: {exc}")


def collect_successors(doc) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for p in doc.get("proposals", []):
        rid = p.get("record_id", "?")
        s = p.get("successor")
        if isinstance(s, dict) and s.get("proposed_id"):
            out.append((rid, s["proposed_id"]))
        for x in p.get("successors", []) or []:
            if x.get("proposed_id"):
                out.append((rid, x["proposed_id"]))
    return out


def collect_cluster_ids(doc) -> list[tuple[str, str]]:
    """Every proposed Duplicate Flags cluster id: row[0] of each APPEND_ROW whose target is the
    workbook's collision registry.  They are proposed identifiers like successor ids and are
    held to the same uniqueness rule."""
    out: list[tuple[str, str]] = []
    for p in doc.get("proposals", []):
        rid = p.get("record_id", "?")
        for op in p.get("operations", []) or []:
            if op.get("operation") != "APPEND_ROW":
                continue
            if DUPLICATE_FLAGS_TAB not in str(op.get("target_tab", "")).lower():
                continue
            row = op.get("row")
            if isinstance(row, list) and row and isinstance(row[0], str) and row[0].strip():
                out.append((rid, row[0].strip()))
    return out


def batch_ids(doc) -> dict[str, str]:
    batch = doc.get("batch_level_artifacts_that_would_also_be_appended", {}) or {}
    out: dict[str, str] = {}
    for field in ("no_change_certificate", "correction_record"):
        pid = (batch.get(field) or {}).get("proposed_id")
        if pid:
            out[field] = pid
    return out


def predecessor_chain(doc, paths: Paths, res: Result) -> list[tuple[str, dict]]:
    """Walk successor_of from `doc` to the root of its predecessor chain.  Returns
    [(relative path, parsed document), ...] nearest predecessor first.  Every document on the
    chain is frozen: its recorded sha256 and byte count must match the file on disk, a
    document may not name itself, and a chain that revisits a path (or exceeds
    CHAIN_MAX_DEPTH) is a loop and fails.  A level whose file is missing or whose digest does
    not match ends the walk there, so nothing downstream of a broken link is trusted."""
    out: list[tuple[str, dict]] = []
    seen: set[str] = set()
    own = doc.get("document")
    if isinstance(own, str) and own:
        seen.add(own)
    cur = doc
    while True:
        pred = cur.get("successor_of")
        if not (isinstance(pred, dict) and pred.get("path")):
            return out
        prel = pred["path"]
        if not res.check(prel not in seen,
                         f"predecessor chain revisits {prel!r} (a document cannot be its own predecessor, "
                         "directly or through a loop)"):
            return out
        seen.add(prel)
        if not res.check(len(out) < CHAIN_MAX_DEPTH,
                         f"predecessor chain deeper than {CHAIN_MAX_DEPTH} at {prel!r}; not followed"):
            return out
        ppath = paths.rel(prel)
        if not res.check(os.path.exists(ppath), f"successor_of names a missing predecessor: {prel}"):
            return out
        raw = open(ppath, "rb").read()
        digest_ok = res.check(hashlib.sha256(raw).hexdigest() == pred.get("sha256"),
                              f"successor_of.sha256 does not match {prel} on disk "
                              "(the predecessor is frozen; a changed digest means it was edited)")
        if "bytes" in pred:
            digest_ok = res.check(len(raw) == pred["bytes"],
                                  f"successor_of.bytes {pred['bytes']} != {len(raw)} for {prel}") and digest_ok
        if not digest_ok:
            return out
        try:
            pdoc = json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            res.check(False, f"predecessor {prel} is not valid JSON: {exc}")
            return out
        pdoc_name = pdoc.get("document")
        res.check(pdoc_name == prel,
                  f"predecessor {prel} calls itself {pdoc_name!r}; the chain must name documents by their own path")
        out.append((prel, pdoc))
        cur = pdoc


def check_workbook_tab_map(paths: Paths, res: Result) -> None:
    """WORKBOOK_TAB must agree with the sheet-name map the importer applies to the xlsx
    (tools/registers_import.py SHEETS: (sheet name, machine name) in workbook order), so a
    record's workbook_tab is checked against the importer's knowledge, not this file's.  The
    importer is loaded from the same root the checks read, at call time."""
    importer = os.path.join(paths.root, "tools", "registers_import.py")
    if not res.check(os.path.exists(importer), f"tools/registers_import.py missing at {importer}; "
                                              "workbook tab names cannot be verified"):
        return
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_registers_import_for_check", importer)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        sheets = dict((machine, sheet) for sheet, machine in getattr(mod, "SHEETS"))
    except Exception as exc:  # pragma: no cover - environment dependent
        res.check(False, f"could not read SHEETS from tools/registers_import.py: {exc}")
        return
    for machine, sheet in WORKBOOK_TAB.items():
        res.check(sheets.get(machine) == sheet,
                  f"WORKBOOK_TAB[{machine!r}] = {sheet!r} but tools/registers_import.py maps "
                  f"{machine!r} to {sheets.get(machine)!r}")


def check_successors(doc, paths: Paths, res: Result) -> None:
    existing = existing_identifiers(paths.json_dir)
    succ = collect_successors(doc)
    res.check(len(succ) >= 1, "proposal issues no successor identifiers at all")
    seen: dict[str, str] = {}
    for rid, sid in succ:
        res.check(sid not in seen,
                  f"{rid}: successor id {sid!r} is already issued by {seen.get(sid)}")
        seen[sid] = rid
        res.check(sid not in existing,
                  f"{rid}: successor id {sid!r} already exists in registers/json/")
    for field, pid in batch_ids(doc).items():
        res.check(pid not in existing,
                  f"batch {field} id {pid!r} already exists in registers/json/")
        res.check(pid not in seen, f"batch {field} id {pid!r} collides with a successor id")
    clusters = collect_cluster_ids(doc)
    seen_clusters: dict[str, str] = {}
    for rid, cid in clusters:
        res.check(cid not in seen_clusters,
                  f"{rid}: proposed cluster id {cid!r} is already proposed by {seen_clusters.get(cid)}")
        seen_clusters[cid] = rid
        res.check(cid not in existing,
                  f"{rid}: proposed cluster id {cid!r} already exists in registers/json/ "
                  "(Duplicate Flags or elsewhere)")
        res.check(cid not in seen, f"{rid}: proposed cluster id {cid!r} collides with a successor id")
    # A proposal over any section other than 'findings' is by construction a successor of the
    # 2026-09-18 document and must say so, or the predecessor checks below never run.
    claimed = doc.get("findings_source_section")
    needs_pred = (claimed is not None and claimed != "findings") or ("supersedes" in doc)
    pred = doc.get("successor_of")
    if needs_pred:
        res.check(isinstance(pred, dict) and bool(pred.get("path")),
                  f"a proposal over section {claimed!r} must name its frozen predecessor in "
                  "successor_of (path, sha256, bytes); without it no predecessor check runs")
    # A successor proposal must not reissue anything ANY frozen document in its predecessor
    # chain issued.  The chain is walked to its root; each level is frozen and must match
    # its recorded digest on disk.
    chain = predecessor_chain(doc, paths, res)
    if "predecessor_chain" in doc:
        # A document that lists its whole chain must list exactly what the walk found, by
        # path, digest and byte count, nearest predecessor first.
        walked = [{"path": prel, "sha256": hashlib.sha256(open(paths.rel(prel), "rb").read()).hexdigest(),
                   "bytes": os.path.getsize(paths.rel(prel))} for prel, _ in chain]
        listed = doc.get("predecessor_chain")
        stripped = [{k: x.get(k) for k in ("path", "sha256", "bytes")} for x in listed] \
            if isinstance(listed, list) and all(isinstance(x, dict) for x in listed) else None
        res.check(stripped == walked,
                  f"predecessor_chain lists {[x.get('path') for x in (listed or [])] if isinstance(listed, list) else listed!r} "
                  f"but walking successor_of from this document finds {[p for p, _ in chain]!r} "
                  "(paths, digests and byte counts must all agree)")
    for depth, (prel, pdoc) in enumerate(chain):
        pred_succ = sorted({sid for _, sid in collect_successors(pdoc)})
        if depth == 0:
            recorded = pred.get("successor_ids_issued_by_predecessor")
            res.check(isinstance(recorded, list) and sorted(recorded) == pred_succ,
                      "successor_of.successor_ids_issued_by_predecessor does not equal the successor ids "
                      f"{prel} actually issues ({len(pred_succ)} ids)")
        pred_ids = set(pred_succ) | set(batch_ids(pdoc).values())
        for rid, sid in succ:
            res.check(sid not in pred_ids,
                      f"{rid}: successor id {sid!r} was already issued by the predecessor {prel}")
        for field, pid in batch_ids(doc).items():
            res.check(pid not in pred_ids,
                      f"batch {field} id {pid!r} was already issued by the predecessor {prel}")
        pred_clusters = {cid for _, cid in collect_cluster_ids(pdoc)}
        for rid, cid in clusters:
            res.check(cid not in pred_clusters,
                      f"{rid}: proposed cluster id {cid!r} was already proposed by the predecessor {prel}")
    # Every record that names a colliding key must either issue a successor or say why not.
    for p in doc.get("proposals", []):
        rid = p.get("record_id", "?")
        has = bool(p.get("successors")) or bool((p.get("successor") or {}).get("proposed_id")
                                                if isinstance(p.get("successor"), dict) else False)
        if not has:
            res.check(bool(p.get("successor_not_applicable_reason")),
                      f"{rid}: no successor issued and no successor_not_applicable_reason given")


def check_verbatim_markdown_lines(doc, paths: Paths, res: Result) -> None:
    """Original mode: each quoted row is a 1-based line of the 2026-09-17 markdown export."""
    export = paths.md_export
    if not os.path.exists(export):
        res.check(False, f"source export missing: {export}")
        return
    raw = open(export, encoding="utf-8").read()
    lines = raw.split("\n")
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    rec = (doc.get("source_of_record") or {}).get("sha256")
    res.check(rec == sha, f"recorded export sha256 {rec} != actual {sha}")
    for p in doc.get("proposals", []):
        rid = p.get("record_id", "?")
        rows = p.get("rows") or []
        res.check(len(rows) >= 1, f"{rid}: no colliding rows quoted")
        for r in rows:
            ln = r.get("export_line_number")
            quoted = r.get("verbatim_export_line", "")
            ok = isinstance(ln, int) and 1 <= ln <= len(lines) and lines[ln - 1] == quoted
            res.check(ok, f"{rid}: quoted export line {ln} does not match the export byte for byte")
            res.check(hashlib.sha256(quoted.encode("utf-8")).hexdigest() == r.get("export_line_sha256"),
                      f"{rid}: recorded digest for export line {ln} does not match the quoted bytes")


def check_verbatim_xlsx_json_rows(doc, paths: Paths, section: str, res: Result) -> None:
    """Successor mode: the xlsx export is re-hashed, and each quoted row is compared cell for
    cell with registers/json/<tab>.json rows[register_row_index].  Every record's class is
    recomputed from its cells and the document-level summary is checked against the sum."""
    classes: list[dict] = []
    src = doc.get("source_of_record") or {}
    xlsx_rel = src.get("xlsx_path", "")
    xlsx = paths.rel(xlsx_rel) if xlsx_rel else ""
    if not (xlsx_rel and os.path.exists(xlsx)):
        res.check(False, f"source xlsx export missing: {xlsx_rel or '(no xlsx_path recorded)'}")
        return
    raw = open(xlsx, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    res.check(src.get("xlsx_sha256") == sha,
              f"recorded xlsx sha256 {src.get('xlsx_sha256')} != actual {sha}")
    res.check(src.get("xlsx_bytes") == len(raw),
              f"recorded xlsx byte count {src.get('xlsx_bytes')} != actual {len(raw)}")
    tabs_named = src.get("json_tabs") or []
    res.check(bool(tabs_named), "source_of_record.json_tabs names no JSON tab")
    tabs: dict[str, dict] = {}
    for rel in tabs_named:
        path = paths.rel(rel)
        if res.check(os.path.exists(path), f"named JSON tab missing: {rel}"):
            t = load_json(path)
            tabs[os.path.splitext(os.path.basename(rel))[0]] = t
    for p in doc.get("proposals", []):
        rid = p.get("record_id", "?")
        tab_name = p.get("register_tab", "")
        t = tabs.get(tab_name)
        if not res.check(t is not None,
                         f"{rid}: register_tab {tab_name!r} is not among source_of_record.json_tabs"):
            continue
        header, rows = list(t.get("header", [])), t.get("rows", [])
        quoted_rows = p.get("rows") or []
        res.check(len(quoted_rows) >= 1, f"{rid}: no colliding rows quoted")
        live_ok: list[int] = []
        for r in quoted_rows:
            idx = r.get("register_row_index")
            in_range = isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < len(rows)
            if not res.check(in_range, f"{rid}: register_row_index {idx!r} is out of range "
                                       f"for {tab_name} ({len(rows)} rows)"):
                continue
            verb = r.get("verbatim_row")
            live = rows[idx]
            res.check(isinstance(verb, list) and verb == live,
                      f"{rid}: quoted verbatim_row for {tab_name} row {idx} does not match the live "
                      f"JSON row cell for cell")
            res.check(r.get("row_sha256") == canonical_row_sha256(live),
                      f"{rid}: recorded row_sha256 for {tab_name} row {idx} does not match the canonical "
                      f"digest of the live row")
            res.check(isinstance(verb, list) and r.get("row_sha256") == canonical_row_sha256(verb),
                      f"{rid}: recorded row_sha256 for {tab_name} row {idx} does not match the quoted bytes")
            res.check(r.get("fields") == dict(zip(header, live)),
                      f"{rid}: fields for {tab_name} row {idx} != dict(zip(header, row))")
            nbytes = len(canonical_row_bytes(live))
            res.check(r.get("row_canonical_bytes") == nbytes,
                      f"{rid}: recorded row_canonical_bytes {r.get('row_canonical_bytes')!r} for {tab_name} "
                      f"row {idx} != {nbytes}, the length of the live row's canonical serialisation")
            live_ok.append(idx)
        if len(live_ok) == len(quoted_rows):
            cls = check_record_bound_to_finding(p, tab_name, header, rows, live_ok, res)
            if cls is not None:
                classes.append(cls)
    if len(classes) == len(doc.get("proposals", [])):
        check_document_summary(doc, paths, section, classes, res)
    else:
        res.check(False, "document-level summary not checked: at least one record failed to bind "
                         "to its finding, so its class cannot be recomputed")


def check_record_bound_to_finding(p, tab_name: str, header: list, rows: list,
                                  quoted: list[int], res: Result) -> dict | None:
    """Bind the quoted rows to the finding the record claims to disambiguate, and recompute
    every derived field from the live cells.  A record that names the right finding and
    quotes the wrong rows, the same row twice, or a difference list that was typed rather
    than derived, fails here.  Returns the record's class as the cells give it (for the
    document-level summary), or None when the rows could not be bound."""
    rid = p.get("record_id", "?")
    key = p.get("finding_key", "")
    m = DUP_KEY_FINDING.match(key)
    if not res.check(m is not None,
                     f"{rid}: finding key {key!r} is not of the form "
                     "\"<tab>: duplicate key '<id>' at rows A and B\""):
        return None
    res.check(p.get("workbook_tab") == WORKBOOK_TAB.get(tab_name),
              f"{rid}: workbook_tab {p.get('workbook_tab')!r} is not the workbook's name for "
              f"{tab_name!r} ({WORKBOOK_TAB.get(tab_name)!r})")
    a, b = int(m.group("a")), int(m.group("b"))
    ident = p.get("colliding_identifier")
    res.check(m.group("tab") == tab_name,
              f"{rid}: finding key names tab {m.group('tab')!r} but the record quotes rows of {tab_name!r}")
    res.check(m.group("key") == ident,
              f"{rid}: finding key names {m.group('key')!r} but colliding_identifier is {ident!r}")
    res.check(a != b and len(quoted) == 2 and quoted[0] != quoted[1] and set(quoted) == {a, b},
              f"{rid}: finding key names rows {a} and {b} but the record quotes rows {quoted} "
              "(each of the two rows must be quoted exactly once)")
    if len(quoted) != 2 or quoted[0] == quoted[1]:
        return None
    ia, ib = quoted[0], quoted[1]
    row_a, row_b = rows[ia], rows[ib]
    for idx, row in ((ia, row_a), (ib, row_b)):
        res.check(bool(row) and row[0] == ident,
                  f"{rid}: key cell {header[0] if header else 'column 0'!r} of {tab_name} row {idx} is "
                  f"{(row[0] if row else None)!r}, not the colliding identifier {ident!r}")
    # Derived lists and counts, recomputed from the live cells.
    identical = [h for h, x, y in zip(header, row_a, row_b) if x == y]
    diffs = [{"field": h, f"row_{ia}": x, f"row_{ib}": y}
             for h, x, y in zip(header, row_a, row_b) if x != y]
    res.check(p.get("fields_identical_in_both_rows") == identical,
              f"{rid}: fields_identical_in_both_rows does not equal the list recomputed from "
              f"{tab_name} rows {ia} and {ib}")
    res.check(p.get("field_differences") == diffs,
              f"{rid}: field_differences does not equal the list recomputed from "
              f"{tab_name} rows {ia} and {ib}")
    expected_counts = {"cells_total": len(header), "cells_identical": len(identical),
                       "cells_differing": len(diffs)}
    cc = p.get("cell_comparison")
    got = {k: (cc or {}).get(k) for k in expected_counts} if isinstance(cc, dict) else None
    res.check(got == expected_counts,
              f"{rid}: cell_comparison {got!r} != counts recomputed from the live rows {expected_counts!r}")
    check_prose_cell_counts(p.get("materiality", ""), expected_counts, res,
                            f"{rid}: materiality")
    is_exact = row_a == row_b
    res.check(bool(p.get("exact_duplicate_row", False)) == is_exact,
              f"{rid}: exact_duplicate_row must be {is_exact} (rows {ia} and {ib} "
              f"{'are' if is_exact else 'are not'} cell-for-cell identical)")
    # Same Drive object or not, read from the identity cells the per-tab rule names, must
    # agree with the flag and the class.
    same = False
    rule = DRIVE_OBJECT_RULE.get(tab_name)
    if res.check(rule is not None, f"{rid}: no drive-object agreement rule is defined for tab {tab_name!r}"):
        missing = [c for c in rule if c not in header]
        if res.check(not missing, f"{rid}: {tab_name} lacks the identity column(s) {missing} the rule reads"):
            cols = {c: header.index(c) for c in rule}
            same = all(row_a[i] == row_b[i] for i in cols.values())
            src_col = rule[0]
            si = cols[src_col]
            url_col = src_col in URL_COLUMNS
            cells_word = f"{src_col!r} cells" if len(rule) == 1 else f"{list(rule)} cells"
            flag = p.get("both_rows_cite_one_drive_object")
            res.check(flag is same,
                      f"{rid}: both_rows_cite_one_drive_object is {flag!r} but the {cells_word} of rows "
                      f"{ia} and {ib} {'agree' if same else 'differ'}")
            if "drive_object_rule_cells" in p:
                res.check(p["drive_object_rule_cells"] == list(rule),
                          f"{rid}: drive_object_rule_cells {p['drive_object_rule_cells']!r} is not the rule this "
                          f"checker reads for {tab_name!r} ({list(rule)!r})")
            if "identity_cells_cited" in p:
                want = {c: {f"row_{ia}": row_a[i], f"row_{ib}": row_b[i]} for c, i in cols.items()}
                res.check(p["identity_cells_cited"] == want,
                          f"{rid}: identity_cells_cited does not equal the rule's cells of rows {ia} and {ib}")
            cls = str(p.get("defect_class", ""))
            if same:
                res.check("SAME_DRIVE_OBJECT" in cls and "DIFFERENT_OBJECTS" not in cls,
                          f"{rid}: defect_class {cls!r} does not say SAME_DRIVE_OBJECT although the cells agree")
                if "drive_source_cited_by_both_rows" in p:
                    want_src = row_a[si] if url_col else None
                    res.check(p["drive_source_cited_by_both_rows"] == want_src,
                              f"{rid}: drive_source_cited_by_both_rows is not the shared {src_col!r} cell"
                              if url_col else
                              f"{rid}: drive_source_cited_by_both_rows must be null: {tab_name} carries no URL "
                              f"column (its rule reads {src_col!r})")
            else:
                res.check("DIFFERENT_OBJECTS" in cls and "SAME_DRIVE_OBJECT" not in cls,
                          f"{rid}: defect_class {cls!r} does not say DIFFERENT_OBJECTS although the cells differ")
                if "drive_sources_cited" in p:
                    want_srcs = {f"row_{ia}": row_a[si], f"row_{ib}": row_b[si]} if url_col else None
                    res.check(p["drive_sources_cited"] == want_srcs,
                              f"{rid}: drive_sources_cited does not equal the two {src_col!r} cells"
                              if url_col else
                              f"{rid}: drive_sources_cited must be null: {tab_name} carries no URL column")
                if "drive_source_cited_by_both_rows" in p:
                    res.check(p["drive_source_cited_by_both_rows"] is None,
                              f"{rid}: drive_source_cited_by_both_rows must be null when the identity cells differ")
            if tab_name == "relations":
                # The relation a row describes is its (Source object, Relation type, Target
                # object) triple; the class must say whether that triple agrees.
                tcols = [header.index(c) for c in RELATION_TRIPLE if c in header]
                res.check(len(tcols) == len(RELATION_TRIPLE),
                          f"{rid}: relations lacks one of the triple columns {RELATION_TRIPLE}")
                same_triple = all(row_a[i] == row_b[i] for i in tcols)
                if same_triple:
                    res.check("SAME_RELATION" in cls and "DIFFERENT_RELATIONS" not in cls,
                              f"{rid}: defect_class {cls!r} does not say SAME_RELATION although the "
                              f"{RELATION_TRIPLE} cells of rows {ia} and {ib} all agree")
                else:
                    res.check("SAME_RELATION" not in cls,
                              f"{rid}: defect_class {cls!r} says SAME_RELATION although the "
                              f"{RELATION_TRIPLE} cells of rows {ia} and {ib} differ")
                if "relation_triple_cells_agree" in p:
                    res.check(p["relation_triple_cells_agree"] is same_triple,
                              f"{rid}: relation_triple_cells_agree is {p['relation_triple_cells_agree']!r} but the "
                              f"{RELATION_TRIPLE} cells {'agree' if same_triple else 'differ'}")
                if "Target URL" in header:
                    ti = header.index("Target URL")
                    same_target = row_a[ti] == row_b[ti]
                    if "target_url_cells_agree" in p:
                        res.check(p["target_url_cells_agree"] is same_target,
                                  f"{rid}: target_url_cells_agree is {p['target_url_cells_agree']!r} but the "
                                  f"'Target URL' cells of rows {ia} and {ib} {'agree' if same_target else 'differ'}")
    # Keeper is the earlier row, successor the later; the successor id locates the later row.
    keeper, succ = p.get("keeper") or {}, p.get("successor") or {}
    lo, hi = min(ia, ib), max(ia, ib)
    res.check(keeper.get("register_row_index") == lo,
              f"{rid}: keeper.register_row_index {keeper.get('register_row_index')!r} must be the earlier "
              f"quoted row {lo} (R17 §3 orders contenders by append position)")
    res.check(succ.get("register_row_index") == hi,
              f"{rid}: successor.register_row_index {succ.get('register_row_index')!r} must be the later "
              f"quoted row {hi}")
    if "keeps_identifier" in keeper:
        res.check(keeper["keeps_identifier"] == ident,
                  f"{rid}: keeper.keeps_identifier {keeper['keeps_identifier']!r} != colliding identifier {ident!r}")
    abbr = TAB_LOCATOR.get(tab_name)
    if res.check(abbr is not None, f"{rid}: no row-locator abbreviation is defined for tab {tab_name!r}"):
        want = f"{ident}@{abbr}-R{hi}"
        res.check(succ.get("proposed_id") == want,
                  f"{rid}: successor id {succ.get('proposed_id')!r} must be {want!r} "
                  "(colliding identifier, tab locator, successor row)")
    return {"record_id": rid, "tab": tab_name, "identifier": ident, "rows": (lo, hi),
            "class": CLASS_EXACT if is_exact else (CLASS_SAME if same else CLASS_DIFFERENT),
            "counts": expected_counts, "successor_id": succ.get("proposed_id"),
            "label": f"{ident} (rows {lo} and {hi})"}


def check_prose_cell_counts(text: str, counts: dict, res: Result, where: str) -> None:
    """Every 'N of T cells differ/agree' or 'N cells differ/agree' phrase whose words are
    number words must equal the recomputed counts.  Words that are not numbers ('the
    Dependencies cells differ') are not counts and are left alone."""
    for m in CELL_COUNT_PROSE.finditer(text or ""):
        first, second, verb = m.group(1), m.group(2), m.group(3).lower()
        want = counts["cells_differing"] if verb == "differ" else counts["cells_identical"]
        if first is not None:
            n, t = NUMBER_WORDS.get(first.lower()), NUMBER_WORDS.get(second.lower())
            if n is None or t is None:
                continue
            res.check(n == want and t == counts["cells_total"],
                      f"{where} says {m.group(0)!r} but the live rows give {want} of "
                      f"{counts['cells_total']} cells {verb}")
        else:
            n = NUMBER_WORDS.get(second.lower())
            if n is None:
                continue
            res.check(n == want,
                      f"{where} says {m.group(0)!r} but the live rows give {want} cells {verb}")


def check_document_summary(doc, paths: Paths, section: str, classes: list[dict], res: Result) -> None:
    """The document-level summary_counts and classification lists are recomputed from the
    per-record classes the cells gave and must match exactly; a summary is never typed."""
    known = load_json(paths.known).get(section, {}) or {}
    # Records of every frozen predecessor on the chain (digests re-verified by the walk; a
    # broken link truncates the chain and the total is then not claimed).
    chain = predecessor_chain(doc, paths, Result())
    chain_records = [len(pdoc.get("proposals", [])) for _, pdoc in chain]
    expected = {"findings_in_section": len(known), "proposal_records": len(doc.get("proposals", []))}
    for tab in sorted({c["tab"] for c in classes}):
        expected[f"{tab}_records"] = sum(1 for c in classes if c["tab"] == tab)
    got = doc.get("summary_counts")
    same_keys_used = [k for k in SUMMARY_SAME_KEYS if isinstance(got, dict) and k in got]
    same_key = same_keys_used[0] if len(same_keys_used) == 1 else SUMMARY_SAME_KEYS[0]
    expected[same_key] = sum(1 for c in classes if c["class"] == CLASS_SAME)
    expected["different_objects_one_identifier"] = sum(1 for c in classes if c["class"] == CLASS_DIFFERENT)
    expected["exact_duplicate_rows"] = sum(1 for c in classes if c["class"] == CLASS_EXACT)
    expected["successor_identifiers_proposed"] = len(collect_successors(doc))
    if len(chain) == 1:
        expected[SUMMARY_TWO_DOCS_KEY] = len(known) + chain_records[0]
    elif len(chain) >= 2:
        expected[SUMMARY_CHAIN_KEY] = len(known) + sum(chain_records)
    if res.check(isinstance(got, dict), "summary_counts is missing"):
        res.check(len(same_keys_used) == 1,
                  f"summary_counts must name the same-object bucket by exactly one of {SUMMARY_SAME_KEYS}, "
                  f"found {same_keys_used}")
        for k, v in expected.items():
            res.check(got.get(k) == v,
                      f"summary_counts.{k} is {got.get(k)!r} but the records' cells give {v}")
        for k in sorted(set(got) - set(expected)):
            res.check(False, f"summary_counts.{k} is not a count this checker recomputes; "
                             "a document-level count that is not recomputed is a typed claim")
    keys = [k for k in doc if isinstance(k, str) and k.startswith(CLASSIFICATION_KEY_PREFIX)]
    if res.check(len(keys) == 1,
                 f"expected exactly one document-level '{CLASSIFICATION_KEY_PREFIX}*' block, found {keys}"):
        block = doc.get(keys[0]) or {}
        same_list_keys = [k for k in CLASS_SAME_KEYS if k in block]
        res.check(len(same_list_keys) == 1,
                  f"{keys[0]} must name the same-object list by exactly one of {CLASS_SAME_KEYS}, "
                  f"found {same_list_keys}")
        same_list_key = same_list_keys[0] if len(same_list_keys) == 1 else CLASS_SAME
        for cls, key in ((CLASS_SAME, same_list_key), (CLASS_DIFFERENT, CLASS_DIFFERENT), (CLASS_EXACT, CLASS_EXACT)):
            want = [c["label"] for c in classes if c["class"] == cls]
            res.check(block.get(key) == want,
                      f"{keys[0]}.{key} is {block.get(key)!r} but the records' cells give {want!r}")


def check_verbatim(doc, paths: Paths, section: str, res: Result) -> str:
    kind = (doc.get("source_of_record") or {}).get("kind") or "markdown_export_lines"
    if not res.check(kind in VERBATIM_KINDS,
                     f"unknown source_of_record.kind {kind!r} (known: {sorted(VERBATIM_KINDS)})"):
        return kind
    if kind == "xlsx_export_json_rows":
        check_verbatim_xlsx_json_rows(doc, paths, section, res)
    else:
        check_verbatim_markdown_lines(doc, paths, res)
    return kind


def check_honesty(doc, res: Result) -> None:
    pb = doc.get("prepared_by") or {}
    res.check(pb.get("independence_credit") == 0,
              "prepared_by.independence_credit must be 0")
    res.check(bool(pb.get("independence_credit_reason")),
              "prepared_by.independence_credit_reason is missing")
    res.check("REMAIN" in str(pb.get("independence_requiring_gates_remain_open", "")).upper(),
              "prepared_by must state that independence-requiring gates remain open")
    res.check(doc.get("nothing_repaired") is True, "doc must declare nothing_repaired=true")
    res.check(doc.get("export_remains_faithful") is True,
              "doc must declare export_remains_faithful=true")
    res.check(bool(doc.get("does_not_establish")), "doc-level does_not_establish is missing")
    if "supersedes" in doc:
        res.check(doc.get("supersedes") is None,
                  "a numbered successor must not claim to supersede anything (CLAUDE.md rule 8)")
    for p in doc.get("proposals", []):
        rid = p.get("record_id", "?")
        res.check(bool(p.get("does_not_establish")), f"{rid}: does_not_establish is missing")
        res.check(bool(p.get("governing_clauses")), f"{rid}: governing_clauses is missing")
        res.check(bool(p.get("residual_questions")), f"{rid}: residual_questions is missing")
        if p.get("register_tab") == "quarantine_index":
            opts = p.get("options") or []
            res.check(len(opts) >= 2,
                      f"{rid}: an EXISTING_CONTAINER record must give at least two options")
            for o in opts:
                res.check(bool(o.get("consequences")),
                          f"{rid}: option {o.get('option')!r} has no consequences")
            res.check(bool(p.get("recommendation")), f"{rid}: no recommendation given")
        else:
            res.check(bool((p.get("keeper") or {}).get("reason")),
                      f"{rid}: keeper has no stated reason")


def companion_path(doc, paths: Paths, res: Result) -> str | None:
    rel = doc.get("companion")
    if not res.check(isinstance(rel, str) and bool(rel), "document has no 'companion' field naming its Markdown"):
        return None
    return paths.rel(rel)


def check_markdown(doc, paths: Paths, section: str, res: Result) -> None:
    md_path = companion_path(doc, paths, res)
    if md_path is None:
        return
    if not os.path.exists(md_path):
        res.check(False, f"missing companion document: {md_path}")
        return
    md = open(md_path, encoding="utf-8").read()
    for needle, what in [
        ("requires operator action", "the proposal-requires-operator-action statement"),
        ("export remains faithful", "the export-remains-faithful statement"),
        ("independence_credit = 0", "the zero-independence-credit record"),
    ]:
        res.check(needle.lower() in md.lower(), f"{md_path} does not state {what}")
    # The nothing-was-repaired banner is anchored, not merely present: a companion that quotes
    # register text ending '...; nothing has been repaired.' would otherwise satisfy a bare
    # substring search while its own banner said the opposite.
    head = "\n".join(md.split("\n")[:NOTHING_REPAIRED_BANNER_WITHIN_LINES])
    res.check(NOTHING_REPAIRED_BANNER in head,
              f"{md_path} does not carry the nothing-has-been-repaired banner "
              f"{NOTHING_REPAIRED_BANNER!r} within its first {NOTHING_REPAIRED_BANNER_WITHIN_LINES} lines "
              "(a bare mention further down, or inside quoted material, does not count)")
    check_markdown_finding_text(doc, md, md_path, res)
    # The gate statement is required in the emphatic form; a lowercase 'remains open' in
    # passing prose does not satisfy it.
    res.check("REMAINS OPEN" in md,
              f"{md_path} does not state the independence-gate-remains-open statement (uppercase 'REMAINS OPEN')")
    known = load_json(paths.known).get(section, {}) or {}
    for k in known:
        res.check(k in md, f"{md_path} does not quote the finding key {k!r}")
    if (doc.get("source_of_record") or {}).get("kind") == "xlsx_export_json_rows":
        check_markdown_xlsx_rows(doc, paths, md, md_path, res)


def check_markdown_finding_text(doc, md: str, md_path: str, res: Result) -> None:
    """The companion blockquotes each finding's text under the attribution 'Finding text as
    recorded in `KNOWN_FINDINGS.json`'.  The record's copy is already held to the register byte
    for byte; the companion's copy is held to the record's, and the number of attributions must
    equal the number of records transcribing, so neither half can drift alone."""
    carrying = records_transcribing_finding_text(doc)
    attributions = md.count(COMPANION_FINDING_TEXT_ATTRIBUTION)
    if not carrying:
        res.check(attributions == 0,
                  f"{md_path} attributes text to registers/KNOWN_FINDINGS.json {attributions} time(s) "
                  f"but no record of the document carries {FINDING_TEXT_FIELD}")
        return
    res.check(attributions == len(carrying),
              f"{md_path} carries {attributions} '{COMPANION_FINDING_TEXT_ATTRIBUTION}' attribution(s) "
              f"but {len(carrying)} record(s) transcribe the finding text")
    for p in carrying:
        rid = p.get("record_id", "?")
        text = p.get(FINDING_TEXT_FIELD)
        res.check(isinstance(text, str) and bool(text) and text in md,
                  f"{md_path}: {rid} does not quote its {FINDING_TEXT_FIELD} verbatim (the companion "
                  "attributes the blockquote to registers/KNOWN_FINDINGS.json)")


def check_markdown_xlsx_rows(doc, paths: Paths, md: str, md_path: str, res: Result) -> None:
    """Successor mode: the companion says every quoted row is verbatim and labels each with
    its canonical byte count and digest, so it is held to that: the live row's canonical JSON
    must appear in a fenced block, the 'rows[i], canonical N bytes, SHA-256' line must match
    the live row, each record's 'Cell count' line and materiality must appear verbatim, and
    exactly one summary-table row per record must name its rows, keeper, reidentified row
    and successor id."""
    tabs: dict[str, dict] = {}
    for rel in (doc.get("source_of_record") or {}).get("json_tabs") or []:
        path = paths.rel(rel)
        if os.path.exists(path):
            tabs[os.path.splitext(os.path.basename(rel))[0]] = load_json(path)
    for p in doc.get("proposals", []):
        rid = p.get("record_id", "?")
        tab_name = p.get("register_tab", "")
        t = tabs.get(tab_name)
        if t is None:
            continue
        rows = t.get("rows", [])
        idxs: list[int] = []
        for r in p.get("rows") or []:
            idx = r.get("register_row_index")
            if not (isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < len(rows)):
                continue
            idxs.append(idx)
            live = rows[idx]
            canon = canonical_row_bytes(live).decode("utf-8")
            res.check(f"```json\n{canon}\n```" in md,
                      f"{md_path}: {rid} row {idx} of {tab_name} is not quoted as the canonical JSON of the "
                      "live row inside a fenced block (the companion calls every row block verbatim)")
            label = (f"`registers/json/{tab_name}.json` rows[{idx}], canonical {len(canon.encode('utf-8'))} "
                     f"bytes, SHA-256 `{canonical_row_sha256(live)}`")
            res.check(label in md,
                      f"{md_path}: {rid} row {idx} lacks the line {label!r} matching the live row")
        cc = p.get("cell_comparison") or {}
        if len(idxs) == 2:
            lo, hi = min(idxs), max(idxs)
            line = (f"Cell count: {cc.get('cells_total')} columns compared, {cc.get('cells_identical')} identical, "
                    f"{cc.get('cells_differing')} differing (rows {lo} and {hi} of `registers/json/{tab_name}.json`")
            res.check(line in md, f"{md_path}: {rid} lacks the line {line!r}")
            ident = p.get("colliding_identifier")
            succ_id = (p.get("successor") or {}).get("proposed_id")
            table_rows = [l for l in md.split("\n") if l.startswith(f"| `{ident}` |")]
            if res.check(len(table_rows) == 1,
                         f"{md_path}: {rid} must have exactly one summary-table row starting "
                         f"'| `{ident}` |', found {len(table_rows)}"):
                cells = [c.strip() for c in table_rows[0].strip().strip("|").split("|")]
                res.check(len(cells) == 6 and cells[1] == f"{lo}, {hi}" and cells[3] == str(lo)
                          and cells[4] == str(hi) and cells[5] == f"`{succ_id}`",
                          f"{md_path}: summary-table row for {ident} must read rows '{lo}, {hi}', keeper {lo}, "
                          f"reidentified {hi}, successor '`{succ_id}`'; got {cells!r}")
            res.check(f"`{ident}` → `{succ_id}`" in md,
                      f"{md_path}: {rid} lacks the follow-up line '`{ident}` → `{succ_id}`'")
        mat = p.get("materiality", "")
        res.check(bool(mat) and mat in md,
                  f"{md_path}: {rid} materiality is not quoted verbatim in the companion")


def prose_sentences(text: str) -> list[str]:
    """Whitespace-flattened sentences, so a claim that wraps over several lines is read as
    the one sentence it is."""
    return [x for x in SENTENCE_SPLIT.split(re.sub(r"\s+", " ", text)) if x]


def nearest_identifier(sentence: str, pos: int, idents: list[str]) -> str | None:
    """The colliding identifier named closest before `pos` in `sentence`, or None if none is.
    Attribution is by proximity because that is how apposition reads ('`X`, not an exact
    duplicate row'); a phrase with no identifier before it in its sentence is not attributed
    to any record and is left alone."""
    best: tuple[int, str] | None = None
    for ident in idents:
        start = sentence.rfind(ident, 0, pos)
        if start >= 0 and (best is None or start > best[0]):
            best = (start, ident)
    return best[1] if best else None


def check_repository_prose(doc, paths: Paths, res: Result) -> None:
    """Repository prose must not contradict the cells the records are checked against.

    For every record that recomputes a cell comparison, any 'exact duplicate' phrase or
    'N of T cells differ/agree' phrase attributed to its colliding identifier in
    PROSE_FILES must agree with the live rows: exact_duplicate_row=false licenses only a
    denial, and a count must be the recomputed count.  This establishes nothing about the
    register, the finding text or any status; it only stops the repository asserting two
    different things about one pair of rows in two places.
    """
    by_ident: dict[str, tuple[str, object, dict]] = {}
    for p in doc.get("proposals", []):
        ident = p.get("colliding_identifier")
        cc = p.get("cell_comparison")
        if not isinstance(ident, str) or not ident or not isinstance(cc, dict):
            continue
        if not all(k in cc for k in ("cells_total", "cells_identical", "cells_differing")):
            continue
        by_ident[ident] = (p.get("record_id", "?"), p.get("exact_duplicate_row"), cc)
    if not by_ident:
        return
    for rel in PROSE_FILES:
        path = paths.rel(rel)
        if not os.path.exists(path):
            res.note(f"{rel} is not present; its prose about these collisions is not cross-checked")
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for sentence in prose_sentences(text):
            named = [i for i in by_ident if i in sentence]
            if not named:
                continue
            for m in EXACT_DUPLICATE_PROSE.finditer(sentence):
                ident = nearest_identifier(sentence, m.start(), named)
                if ident is None:
                    continue
                rid, exact, cc = by_ident[ident]
                if exact is not False:
                    continue
                before = sentence[max(0, m.start() - PROSE_NEGATION_WINDOW):m.start()].lower()
                window = sentence[max(0, m.start() - 80):m.end() + 40].strip()
                res.check(any(neg in before for neg in PROSE_NEGATORS),
                          f"{rel} calls {ident} an exact duplicate, but the live rows give "
                          f"{cc['cells_differing']} of {cc['cells_total']} cells differing and record "
                          f"{rid} recomputes exact_duplicate_row=false: ...{window}...")
            for m in CELL_COUNT_PROSE.finditer(sentence):
                ident = nearest_identifier(sentence, m.start(), named)
                if ident is None:
                    continue
                rid, _exact, cc = by_ident[ident]
                check_prose_cell_counts(m.group(0), cc, res,
                                        f"{rel}, of {ident} (record {rid}),")


def parse_args(argv: list[str]) -> tuple[str, str | None, bool]:
    proposal, section, verbose = DEFAULT_PROPOSAL, None, False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-v":
            verbose = True
        elif a == "--proposal" and i + 1 < len(argv):
            proposal = argv[i + 1]
            i += 1
        elif a.startswith("--proposal="):
            proposal = a.split("=", 1)[1]
        elif a == "--section" and i + 1 < len(argv):
            section = argv[i + 1]
            i += 1
        elif a.startswith("--section="):
            section = a.split("=", 1)[1]
        else:
            raise SystemExit(f"usage: collision_proposal_check.py [--proposal PATH] [--section NAME] [-v]; "
                             f"unrecognised argument {a!r}")
        i += 1
    return proposal, section, verbose


def main(argv: list[str]) -> int:
    proposal_rel, section_arg, verbose = parse_args(argv)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = Paths(root, proposal_rel)
    res = Result()
    if not os.path.exists(paths.proposal):
        print(f"FAIL  missing {paths.proposal}")
        print(f"proposal={proposal_rel} records=0 successors=0 failures=1")
        return 1
    doc = load_json(paths.proposal)
    section = resolve_section(doc, section_arg, res)
    check_bijection(doc, paths, section, res)
    check_finding_text_verbatim(doc, paths, section, res)
    check_declared_row_locators(doc, res)
    check_operations(doc, res)
    check_registers_unchanged(paths, res)
    check_workbook_tab_map(paths, res)
    check_successors(doc, paths, res)
    kind = check_verbatim(doc, paths, section, res)
    check_honesty(doc, res)
    check_markdown(doc, paths, section, res)
    if kind == "xlsx_export_json_rows":
        check_repository_prose(doc, paths, res)
    for n in res.notes:
        if verbose:
            print("NOTE  " + n)
    for f in res.failures:
        print("FAIL  " + f)
    n = len(doc.get("proposals", []))
    print(f"proposal={proposal_rel} section={section} verbatim={kind} records={n} "
          f"successors={len(collect_successors(doc))} failures={len(res.failures)}")
    return 1 if res.failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
