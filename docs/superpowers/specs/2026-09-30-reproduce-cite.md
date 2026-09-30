# Reproduce and cite the research

A bounded extension of the existing reader experience, authorized by Dylan’s
request to apply the useful parts of an external repository audit.

## Outcome

A new visitor can find one runnable exact calculation, understand what its
output establishes, and construct an immutable reference to the source used.
Home, Explore, Research and Library remain the primary navigation. Reproduce
and Cite become consistent secondary routes, reachable without the work queue.

## Design

- Add `docs/site/reproduce.html`: one pinned Math- SIDE24 example, prerequisites,
  commands, observed output and scope; progressive disclosure links to the
  existing complete reproduction guide and formal package.
- Add `docs/site/cite.html`: repository metadata, source-specific attribution,
  DOI/release status without invented identifiers, and an offline source-link
  builder. Select one of the four scientific repositories; require a full
  40-character commit; optionally enter a relative file path. Produce a permalink
  and plain source reference using text nodes. No network lookup, private data,
  source execution, storage, or implicit proof verdict.
- Keep working links and content without JavaScript. The builder alone requires
  JavaScript; report invalid input and do not retain a stale successful reference.
- Correct main’s CFF description to identify the actual website/reproduction
  repository. Add ORCID only if verified against an existing owner source.
- Publish a concise source-backed audit explaining adopted and rejected advice.
  No bulk repository transfers, visibility changes, blanket relicensing, new
  package facade, proof-source moves or history rewrite.

## Constraints and acceptance

Use existing static HTML/CSS/ES modules and brand palette, no new dependencies.
Preserve all scientific bytes, existing primary nav, source pins, legacy anchors,
required checks and peer ownership. Test malformed references, path encoding,
unsupported repositories, stale output refusal, links and keyboard controls.
Execute the displayed coefficient recipe at its exact source. Review mobile and
desktop layouts. Run focused checks during development, then one complete local
applicable suite on the stable candidate; verify hosted checks and deployed bytes.
