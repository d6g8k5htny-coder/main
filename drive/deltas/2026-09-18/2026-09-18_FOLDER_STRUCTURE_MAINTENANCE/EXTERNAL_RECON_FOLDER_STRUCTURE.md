# External reconnaissance — folder structure maintenance
Recorded 2026-09-18T00:09:42.885Z.

Task: assess and apply the owner's proposed Drive naming, triage and template refinements against live metadata.

Public searches and primary-source inspection covered [Drive folder operations](https://developers.google.com/workspace/drive/api/guides/folder), [metadata updates](https://developers.google.com/workspace/drive/api/reference/rest/v3/files/update), and [shortcuts](https://developers.google.com/workspace/drive/api/guides/shortcuts). Searches contained generic API terms, not private research content.

Exact-scope finding: supported folder moves use existing file IDs and explicit parent additions/removals. Metadata updates use patch semantics. Native shortcuts are available in the Drive API, but the exposed create action here does not support the shortcut MIME type; ordinary ID-based links provide the needed navigation without adding a second copy.

Reuse decision: retain the installed R17 central triage and existing registers. Live checks show zero UNKNOWN/ADJACENITY folders, no duplicate 01_ active-package sibling, 15 matching review scaffolds, and populated T4/T5 charters. The proposed wholesale migration and placeholder labeling would repeat completed work or misstate current contents.

Execute bounded refinements: group 13 origin-preserving triage collections under one container; normalize six month folders to YYYY/MM_MONTHNAME with reversible metadata moves; label the old empty migration index as historical; publish current folder paths, naming rules and exact before/after IDs. Keep dated research intake names in ISO day/range form because they identify events, not month-only containers.

Verification: compare names, parents and permissions after writes; check all 15 review scaffold child sets; verify native register additions and current control pointers. Preserve original content and scientific status. This is operational maintenance and earns no mathematical or independent-review credit.

