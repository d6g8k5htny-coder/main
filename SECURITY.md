# Security policy

This policy covers the public viewer, intake boundary, repository automation and
lookup tooling. Scientific effect: **NONE**. A security fix or a passing check is
not a mathematical acceptance decision.

## Report a vulnerability privately

Use the [private vulnerability reporting address](https://github.com/d6g8k5htny-coder/main/security/advisories/new)
to contact this repository's owner and maintainers. Private vulnerability reporting
was enabled and read back on 2026-09-26. Do not post credentials, personal data,
private sources, or exploitable details in a public issue or incoming packet.

Include the repository, exact commit and path, affected behavior, a minimal
reproduction, expected impact, and a proposed remedy if available. Use inert
examples; do not access other people's data or test destructive payloads.
No response-time guarantee or bug bounty is implied.

## Scope and supported versions

Report engineering defects against the current default branch and identify any
affected pinned historical version. Historical research artifacts remain immutable
evidence; a repair gets a separately identified successor. There is no supported
production service or theorem-release support guarantee.

Mathematical counterexamples and source-bound proof concerns belong in public
[research issues](https://github.com/d6g8k5htny-coder/main/issues) unless they expose
a security vulnerability. Keep their exact hypotheses and review scope visible.

## Maintainer controls

- Keep the public site static and read-only; no credentials or status-write API.
- Treat submitted packets as untrusted data. The intake job reads trusted
  default-branch code and must not execute submitted code.
- Check trusted workflow/event identity and the exact PR head before merging.
- Preserve nonauthor reviews and source provenance; same-provider review earns
  no organizational-independence credit.
- Secret scanning and push protection are enabled, but neither they nor the
  intake pattern scan prove that all secrets or malicious content are absent.
- Dependabot proposes weekly GitHub Actions updates only. Review changes and
  run existing checks; do not automatically accept new dependencies or approvals.

See [repository setup](docs/PUBLIC_SHOP_SETUP.md) for installed controls and the
owner's account-security steps. No funding or new license terms are introduced.
