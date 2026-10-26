# Library current-source search recovery

The owner delegates implementation choices and asks us to continue the finished
reader experience without repeated design questions. This is a small Library-only
increment on Main `f8ea0b530fed1942aa23d79a8fce179a679862f8`.

Current search replaces its list with nothing when no source path matches. Unlike
the dated catalog, it offers neither a recovery hint nor nearby Clear/Link actions.
Add a short no-match message, Clear search, and a native Link to this search.
Keep Save current search in the same compact tool row. Controls appear only after
the existing current snapshot verifies. Clearing dispatches the existing input
event, resets existing pagination, removes only source-q and focuses the input.
The native link preserves unrelated URL parameters, encodes literal query text,
and targets workspace.html#library-current. Explain beside it: the query reopens
against the admitted observation available then; results are not frozen.

Use one small Library helper imported by research-desk.mjs. Leave current.mjs and
the source importer, their admission rules and other pages unchanged. A patch to
current.mjs would overlap the separate source transition work; a second search
engine would duplicate existing filtering. The helper delegates filtering to the
existing input event and observes the list only to maintain the recovery message.
No new storage, network request, dependency, theorem status or source association.

Check literal Unicode/special-character queries, native Back/reload/link opening,
unrelated URL state, show-more then reset, unavailable admission, repeated same-page
fragment navigation, 320px/390px/mobile and desktop light/dark keyboard interaction.
Verify existing saved notes, Undo, source-to-Cite and source refusal remain intact.
Browser and engineering checks do not establish spoken AT or scientific acceptance.
